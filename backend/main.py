from fastapi import Depends
from auth import get_current_user
from fastapi import FastAPI, UploadFile, File, Request, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordRequestForm

from typing import List
import os
import shutil
from pathlib import Path
import subprocess

# Import ML logic
from ml_pipeline import load_or_train_model, detect_and_crop_anomalies

# Import auth
from auth import (
    hash_password,
    authenticate_user,
    create_access_token,
    get_current_user,
    users_collection
)
from datetime import timedelta

app = FastAPI()
app.mount("/videos/crops", StaticFiles(directory="videos/crops"), name="crops")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
TRAIN_DIR = "videos/train"
TEST_DIR = "videos/test"
CROPS_DIR = "videos/crops"
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)
os.makedirs(CROPS_DIR, exist_ok=True)

# ───────────────────────────────
# AUTH ROUTES
# ───────────────────────────────

@app.post("/register")
def register(form_data: OAuth2PasswordRequestForm = Depends()):
    existing_user = users_collection.find_one({"username": form_data.username})
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already exists")
    
    hashed = hash_password(form_data.password)
    users_collection.insert_one({"username": form_data.username, "password": hashed})
    return {"message": "User registered successfully"}

@app.post("/token")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = create_access_token(
        data={"sub": user["username"]},
        expires_delta=timedelta(minutes=30)
    )
    return {"access_token": token, "token_type": "bearer"}

@app.get("/protected")
def protected(user=Depends(get_current_user)):
    return {"message": f"Yo {user['username']}, you authenticated cuh 🔒"}

# ───────────────────────────────
# UTILS
# ───────────────────────────────

def convert_to_browser_compatible(input_path: str, output_path: str = None):
    if output_path is None:
        temp_output = input_path.replace(".mp4", "_browser.mp4")
    else:
        temp_output = output_path

    command = [
        "ffmpeg",
        "-i", input_path,
        "-c:v", "libx264",
        "-c:a", "aac",
        "-movflags", "+faststart",
        "-y",
        temp_output
    ]

    try:
        subprocess.run(command, check=True)
        if output_path is None:
            os.replace(temp_output, input_path)
        print(f"✅ Converted {input_path} to browser-compatible format.")
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg failed on {input_path}: {e}")

def get_video_stream(file_path: str, request: Request):
    file_size = os.path.getsize(file_path)

    def iterfile(start: int = 0, end: int = None):
        if end is None:
            end = file_size - 1
        with open(file_path, 'rb') as f:
            f.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk_size = min(8192, remaining)
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    range_header = request.headers.get('range')
    if range_header:
        try:
            ranges = range_header.replace('bytes=', '').split('-')
            start = int(ranges[0]) if ranges[0] else 0
            end = int(ranges[1]) if len(ranges) > 1 and ranges[1] else file_size - 1
            start = max(0, start)
            end = min(end, file_size - 1)
            content_length = end - start + 1
            return StreamingResponse(
                iterfile(start, end),
                status_code=206,
                headers={
                    'Content-Range': f'bytes {start}-{end}/{file_size}',
                    'Accept-Ranges': 'bytes',
                    'Content-Length': str(content_length),
                    'Content-Type': 'video/mp4',
                }
            )
        except (ValueError, IndexError):
            pass

    return StreamingResponse(
        iterfile(),
        media_type='video/mp4',
        headers={
            'Accept-Ranges': 'bytes',
            'Content-Length': str(file_size)
        }
    )

# ───────────────────────────────
# VIDEO ROUTES
# ───────────────────────────────

@app.post("/upload/train-videos/")
async def upload_train_videos(files: List[UploadFile] = File(...)):
    uploaded = []
    for file in files:
        file_path = os.path.join(TRAIN_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        uploaded.append(file.filename)
    return {"message": f"Uploaded {len(uploaded)} training videos", "videos": uploaded}

@app.post("/train/")
def train(current_user: dict = Depends(get_current_user)):
    try:
        train_paths = [os.path.join(TRAIN_DIR, f) for f in os.listdir(TRAIN_DIR) if f.endswith((".avi", ".mp4"))]
        if not train_paths:
            return JSONResponse(status_code=400, content={"error": "No training videos found in videos/train"})
        load_or_train_model(train_paths=train_paths)
        return {"message": "Training completed successfully"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/upload/test/")
async def upload_test_video( file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)):
    file_path = os.path.join(TEST_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        train_paths = [os.path.join(TRAIN_DIR, f) for f in os.listdir(TRAIN_DIR) if f.endswith((".avi", ".mp4"))]
        if not train_paths:
            return JSONResponse(status_code=400, content={"error": "Training required before testing."})
        model = load_or_train_model(train_paths=train_paths)
        anomalies = detect_and_crop_anomalies(file_path, model)

        for segment in anomalies.get("segments", []):
            segment_path = segment.get("path")
            if segment_path and os.path.exists(segment_path):
                convert_to_browser_compatible(segment_path)

        return JSONResponse(content=anomalies)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
