import os
import sys
import io
import shutil
import tempfile

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import main
from auth import get_current_user

client = TestClient(main.app)

# ───────────────────────────────
# Override dependencies globally
# ───────────────────────────────

@pytest.fixture(autouse=True)
def override_current_user():
    main.app.dependency_overrides[get_current_user] = lambda: {"username": "tester"}
    yield
    main.app.dependency_overrides = {}

# ───────────────────────────────
# Test /register
# ───────────────────────────────

@patch("main.users_collection.find_one")
@patch("main.users_collection.insert_one")
@patch("main.hash_password")
def test_register_success(mock_hash, mock_insert, mock_find):
    mock_find.return_value = None
    mock_hash.return_value = "hashedpw"
    
    response = client.post(
        "/register",
        data={"username": "user1", "password": "pass123"}
    )
    assert response.status_code == 200
    assert response.json()["message"] == "User registered successfully"
    mock_insert.assert_called_once_with({"username": "user1", "password": "hashedpw"})

@patch("main.users_collection.find_one")
def test_register_user_exists(mock_find):
    mock_find.return_value = {"username": "user1"}
    
    response = client.post(
        "/register",
        data={"username": "user1", "password": "pass123"}
    )
    assert response.status_code == 400
    assert "Username already exists" in response.json()["detail"]

# ───────────────────────────────
# Test /token
# ───────────────────────────────

@patch("main.authenticate_user")
@patch("main.create_access_token")
def test_login_success(mock_create_token, mock_authenticate):
    mock_authenticate.return_value = {"username": "user1"}
    mock_create_token.return_value = "fake.jwt.token"
    
    response = client.post(
        "/token",
        data={"username": "user1", "password": "pass123"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["access_token"] == "fake.jwt.token"
    assert data["token_type"] == "bearer"

@patch("main.authenticate_user")
def test_login_invalid_credentials(mock_authenticate):
    mock_authenticate.return_value = None
    
    response = client.post(
        "/token",
        data={"username": "user1", "password": "wrongpass"}
    )
    assert response.status_code == 401
    assert "Invalid credentials" in response.json()["detail"]

# ───────────────────────────────
# Test /protected
# ───────────────────────────────

def test_protected():
    response = client.get("/protected")
    assert response.status_code == 200
    assert "Yo tester" in response.json()["message"]

# ───────────────────────────────
# Test /upload/train-videos/
# ───────────────────────────────

def test_upload_train_videos(tmp_path):
    test_content = b"dummy video content"
    test_filename = "video1.mp4"
    file_path = tmp_path / test_filename

    with open(file_path, "wb") as f:
        f.write(test_content)

    with open(file_path, "rb") as f:
        response = client.post(
            "/upload/train-videos/",
            files={"files": (test_filename, f, "video/mp4")}
        )

    assert response.status_code == 200
    resp_data = response.json()
    assert "Uploaded" in resp_data["message"]
    assert test_filename in resp_data["videos"]

# ───────────────────────────────
# Test /train/
# ───────────────────────────────

@patch("main.load_or_train_model")
def test_train_success(mock_train_model, tmp_path):
    # Place dummy training file
    video_file = tmp_path / "dummy.mp4"
    video_file.write_bytes(b"video content")
    
    with patch("main.os.listdir", return_value=["dummy.mp4"]), \
         patch("main.TRAIN_DIR", str(tmp_path)):
        response = client.post("/train/")
    
    assert response.status_code == 200
    assert "Training completed successfully" in response.json()["message"]

def test_train_no_videos(tmp_path):
    with patch("main.os.listdir", return_value=[]), \
         patch("main.TRAIN_DIR", str(tmp_path)):
        response = client.post("/train/")
    assert response.status_code == 400
    assert "No training videos" in response.json()["error"]

# ───────────────────────────────
# Test /upload/test/
# ───────────────────────────────

@patch("main.load_or_train_model")
@patch("main.detect_and_crop_anomalies")
@patch("main.convert_to_browser_compatible")
def test_upload_test_video_success(
    mock_convert, mock_detect, mock_train_model, tmp_path
):
    mock_train_model.return_value = MagicMock()
    mock_detect.return_value = {
        "anomalies": ["fake anomaly"],
        "segments": [{"path": str(tmp_path / "segment1.mp4")}],
        "narrations": ["Detected something sus."]
    }

    train_video = tmp_path / "train.mp4"
    train_video.write_bytes(b"video data")

    with patch("main.os.listdir", return_value=["train.mp4"]), \
         patch("main.TRAIN_DIR", str(tmp_path)), \
         patch("main.TEST_DIR", str(tmp_path)):

        file_content = io.BytesIO(b"test video content")

        response = client.post(
            "/upload/test/",
            files={"file": ("testvideo.mp4", file_content, "video/mp4")}
        )

    assert response.status_code == 200
    json_data = response.json()
    assert "anomalies" in json_data
    assert "narrations" in json_data

def test_upload_test_video_requires_training(tmp_path):
    with patch("main.os.listdir", return_value=[]), \
         patch("main.TRAIN_DIR", str(tmp_path)):
        file_content = io.BytesIO(b"test video content")

        response = client.post(
            "/upload/test/",
            files={"file": ("testvideo.mp4", file_content, "video/mp4")}
        )

    assert response.status_code == 400
    assert "Training required" in response.json()["error"]

# ───────────────────────────────
# Utils - convert_to_browser_compatible
# (no test here since we dropped coverage)
