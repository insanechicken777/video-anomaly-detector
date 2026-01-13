import pytest
import os
from httpx import AsyncClient
from urllib.parse import urlencode

BASE_URL = "http://127.0.0.1:8000"

test_user = {
    "username": "Sandhia",
    "password": "Sandhia1234"
}

@pytest.mark.asyncio
async def test_register_user():
    async with AsyncClient(base_url=BASE_URL) as ac:
        form_encoded = urlencode(test_user)
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        res = await ac.post("/register", content=form_encoded, headers=headers)
        assert res.status_code in [200, 400]  # 400 if already exists in your app

@pytest.mark.asyncio
async def test_token_generation():
    async with AsyncClient(base_url=BASE_URL) as ac:
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        form_encoded = urlencode(test_user)
        res = await ac.post("/token", content=form_encoded, headers=headers)
        assert res.status_code == 200
        assert "access_token" in res.json()

@pytest.mark.asyncio
async def test_upload_train_videos():
    # Step 1: Authenticate
    async with AsyncClient(base_url=BASE_URL) as ac:
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        form_encoded = urlencode(test_user)
        token_response = await ac.post("/token", content=form_encoded, headers=headers)
        assert token_response.status_code == 200
        token = token_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

    # Step 2: Create dummy video
    test_video_dir = "tests"
    test_video_filename = "sample.mp4"
    test_video_path = os.path.join(test_video_dir, test_video_filename)

    os.makedirs(test_video_dir, exist_ok=True)
    with open(test_video_path, "wb") as f:
        f.write(os.urandom(1024))  # 1KB dummy data

    # Step 3: Upload video
    with open(test_video_path, "rb") as video_file:
        files = [("files", (test_video_filename, video_file, "video/mp4"))]
        async with AsyncClient(base_url=BASE_URL, headers=headers) as ac:
            res = await ac.post("/upload/train-videos/", files=files)

    # Step 4: Validate
    assert res.status_code == 200
    assert "Uploaded" in res.json()["message"]
    assert test_video_filename in res.json()["videos"]

    # Step 5: Clean up
    os.remove(test_video_path)
    uploaded_path = os.path.join("videos", "train", test_video_filename)
    if os.path.exists(uploaded_path):
        os.remove(uploaded_path)

@pytest.mark.asyncio
async def test_train_model():
    # Step 1: Get auth token
    async with AsyncClient(base_url=BASE_URL) as ac:
        token_response = await ac.post("/token", data=test_user)
        assert token_response.status_code == 200
        token = token_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

    # Step 2: Call train-model endpoint
    async with AsyncClient(base_url=BASE_URL, headers=headers) as ac:
        response = await ac.post("/train/")

    # Step 3: Assertions
    assert response.status_code == 200
    assert response.json()["message"] == "Training completed successfully"

@pytest.mark.asyncio
async def test_upload_test_video():
    # Step 1: Authenticate and get token
    async with AsyncClient(base_url=BASE_URL) as ac:
        token_response = await ac.post("/token", data=test_user)
        assert token_response.status_code == 200
        token = token_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

    # Step 2: Create dummy test video
    test_video_dir = "tests"
    test_video_filename = "dummy_test.mp4"
    test_video_path = os.path.join(test_video_dir, test_video_filename)
    os.makedirs(test_video_dir, exist_ok=True)
    with open(test_video_path, "wb") as f:
        f.write(os.urandom(1024))  # small dummy file

    # Step 3: Upload test video
    with open(test_video_path, "rb") as video_file:
        files = {"file": (test_video_filename, video_file, "video/mp4")}
        async with AsyncClient(base_url=BASE_URL, headers=headers) as ac:
            response = await ac.post("/upload/test/", files=files)

    # Step 4: Assertions
    assert response.status_code in [200, 400, 500]  # adjust depending on whether model exists
    json_resp = response.json()
    if response.status_code == 200:
        assert "anomalies" in json_resp
    else:
        assert "error" in json_resp

    # Step 5: Cleanup
    os.remove(test_video_path)
    test_upload_path = os.path.join("videos", "test", test_video_filename)
    if os.path.exists(test_upload_path):
        os.remove(test_upload_path)

@pytest.mark.asyncio
async def test_protected_route():
    async with AsyncClient(base_url=BASE_URL) as ac:
        # Without token
        res_no_token = await ac.get("/protected")
        assert res_no_token.status_code == 401

        # With token
        token_res = await ac.post("/token", data=test_user)
        assert token_res.status_code == 200
        token = token_res.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        res_with_token = await ac.get("/protected", headers=headers)
        assert res_with_token.status_code == 200
        assert "authenticated" in res_with_token.json()["message"]
