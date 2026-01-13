import os
import sys
import io
import tempfile
import pytest
from fastapi.testclient import TestClient
from fastapi import Request
from unittest.mock import patch, MagicMock

# Ensure backend directory is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import main

# Dependency override to avoid auth errors
main.app.dependency_overrides[main.get_current_user] = lambda: {"username": "tester"}

client = TestClient(main.app)

# ───────────────────────────────
# Exception Tests
# ───────────────────────────────

@patch("main.get_current_user")
@patch("main.load_or_train_model", side_effect=Exception("boom"))
def test_train_raises_exception(mock_train, mock_user, tmp_path):
    mock_user.return_value = {"username": "tester"}
    
    video_file = tmp_path / "dummy.mp4"
    video_file.write_bytes(b"video")

    with patch("main.os.listdir", return_value=["dummy.mp4"]), \
         patch("main.TRAIN_DIR", str(tmp_path)):
        response = client.post("/train/")
    
    assert response.status_code == 500
    assert "boom" in response.json()["error"]


@patch("main.get_current_user")
@patch("main.load_or_train_model", side_effect=Exception("anomaly fail"))
def test_upload_test_video_raises_exception(mock_train, mock_user, tmp_path):
    mock_user.return_value = {"username": "tester"}

    video_file = tmp_path / "dummy.mp4"
    video_file.write_bytes(b"video")

    with patch("main.os.listdir", return_value=["dummy.mp4"]), \
         patch("main.TRAIN_DIR", str(tmp_path)), \
         patch("main.TEST_DIR", str(tmp_path)):
        
        file_content = io.BytesIO(b"test video content")
        response = client.post(
            "/upload/test/",
            files={"file": ("testvideo.mp4", file_content, "video/mp4")}
        )

    assert response.status_code == 500
    assert "anomaly fail" in response.json()["error"]


# ───────────────────────────────
# get_video_stream Tests
# ───────────────────────────────

# Add a test route for video streaming
@main.app.get("/_test_stream")
async def _test_stream(path: str, request: Request):
    return main.get_video_stream(path, request)


def test_get_video_stream_full(tmp_path):
    dummy_video = tmp_path / "test.mp4"
    dummy_video.write_bytes(b"0123456789")

    response = client.get(f"/_test_stream?path={dummy_video}")
    assert response.status_code == 200
    assert response.content == b"0123456789"
    assert response.headers["Content-Length"] == str(len(b"0123456789"))


def test_get_video_stream_range(tmp_path):
    dummy_video = tmp_path / "test.mp4"
    dummy_video.write_bytes(b"0123456789")

    headers = {"Range": "bytes=2-5"}
    response = client.get(f"/_test_stream?path={dummy_video}", headers=headers)

    assert response.status_code == 206
    assert response.content == b"2345"
    assert response.headers["Content-Range"] == f"bytes 2-5/10"
    assert response.headers["Content-Length"] == "4"


def test_get_video_stream_file_missing(tmp_path):
    dummy_video = tmp_path / "missing.mp4"
    request = MagicMock()
    request.headers = {}

    # cause os.path.getsize to fail
    with pytest.raises(FileNotFoundError):
        main.get_video_stream(str(dummy_video), request)
