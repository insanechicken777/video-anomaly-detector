import json
import os
import cv2
import numpy as np
from tensorflow import keras
from keras import layers
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()
# === Browser-Compatible Conversion ===
def convert_to_browser_compatible(input_path):
    output_path = input_path.replace('.mp4', '_web.mp4')
    command = f'ffmpeg -y -i "{input_path}" -c:v libx264 -pix_fmt yuv420p -profile:v baseline -level 3.0 -c:a aac -movflags +faststart "{output_path}"'
    exit_code = os.system(command)
    if exit_code != 0:
        print(f"⚠️ FFmpeg conversion failed for {input_path}")
        return input_path
    print(f"✅ Converted to browser-compatible: {output_path}")
    return output_path

# === Configuration ===
MODEL_SAVE_PATH = 'anomaly_detection_model.keras'
ANOMALY_THRESHOLD = 0.0065
FRAME_SIZE = (64, 64)
SEQUENCE_LENGTH = 10
BATCH_SIZE = 4
EPOCHS = 10
GEMINI_MODEL_NAME = 'gemini-1.5-flash'
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY is None:
    raise ValueError("GEMINI_API_KEY is not set in environment variables.")


genai.configure(api_key=GEMINI_API_KEY)

# === Model ===
def build_convlstm_autoencoder(input_shape=(SEQUENCE_LENGTH, 64, 64, 1)):
    inputs = keras.Input(shape=input_shape)
    x = layers.ConvLSTM2D(32, (3,3), activation='relu', padding='same', return_sequences=True)(inputs)
    x = layers.MaxPooling3D((1,2,2), padding='same')(x)
    x = layers.ConvLSTM2D(16, (3,3), activation='relu', padding='same', return_sequences=True)(x)
    x = layers.MaxPooling3D((1,2,2), padding='same')(x)
    x = layers.ConvLSTM2D(16, (3,3), activation='relu', padding='same', return_sequences=True)(x)
    x = layers.UpSampling3D((1,2,2))(x)
    x = layers.ConvLSTM2D(32, (3,3), activation='relu', padding='same', return_sequences=True)(x)
    x = layers.UpSampling3D((1,2,2))(x)
    decoded = layers.Conv3D(1, (3,3,3), activation='sigmoid', padding='same')(x)
    model = keras.Model(inputs, decoded)
    model.compile(optimizer='adam', loss='mse')
    return model

# === Generator ===
def sequence_generator(video_paths, sequence_length=SEQUENCE_LENGTH, batch_size=BATCH_SIZE, size=FRAME_SIZE):
    buffer = []
    while True:
        for path in video_paths:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise FileNotFoundError(f"Cannot open video file: {path}")
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, size)
                norm = resized.astype('float32') / 255.0
                frames.append(norm)
                if len(frames) == sequence_length:
                    buffer.append(np.array(frames).reshape(sequence_length, size[0], size[1], 1))
                    frames.pop(0)
                    if len(buffer) == batch_size:
                        batch = np.array(buffer)
                        yield batch, batch
                        buffer = []
            cap.release()
# === Narration ===
def narrate_anomaly_with_gemini(video_path):
    try:
        if os.path.getsize(video_path) > (19 * 1024 * 1024):
            return "Video too large for direct upload. GCS upload required."

        with open(video_path, 'rb') as video_file:
            video_bytes = video_file.read()

        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        video_part = {'mime_type': 'video/mp4', 'data': video_bytes}
        prompt = ("Analyze this video segment for unusual activity. Describe the anomaly, its context, and actors. "
                  "If no anomaly, describe normal activity.")
        contents = [prompt, video_part]
        response_stream = model.generate_content(contents, stream=True)

        full_narration = ""
        for chunk in response_stream:
            if chunk.text:
                full_narration += chunk.text
        return full_narration

    except Exception as e:
        return f"Narration failed: {e}"

# === Detection ===
def detect_and_crop_anomalies(video_path, model, threshold=ANOMALY_THRESHOLD, frame_size=FRAME_SIZE, sequence_length=SEQUENCE_LENGTH):
    losses = []
    narrations = []
    anomalous_segments = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    buffer, pre_buffer = [], []
    segment_id = 0
    frame_idx = 0
    out, anomaly_active = None, False
    filename, filepath = "", ""

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pre_buffer.append(frame)
        if len(pre_buffer) > 50:
            pre_buffer.pop(0)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, frame_size)
        norm = resized.astype('float32') / 255.0
        buffer.append(norm)

        if len(buffer) == sequence_length:
            input_seq = np.array(buffer).reshape(1, sequence_length, frame_size[0], frame_size[1], 1)
            reconstructed = model.predict(input_seq, verbose=0)
            loss = np.mean((input_seq - reconstructed)**2)
            print(f"Frame {frame_idx}: Reconstruction Loss = {loss:.6f}")
            losses.append(loss)

            if loss > threshold:
                if not anomaly_active:
                    anomaly_active = True
                    segment_id += 1
                    filename = f"anomaly_segment_{segment_id}.mp4"
                    filepath = os.path.join("videos", "crops", filename)
                    os.makedirs(os.path.dirname(filepath), exist_ok=True)
                    out = cv2.VideoWriter(filepath, fourcc, fps, (frame.shape[1], frame.shape[0]))
                    for f in pre_buffer:
                        out.write(f)
                out.write(frame)
                normal_count = 0
            elif anomaly_active:
                normal_count += 1
                out.write(frame)
                if normal_count >= 50:
                    out.release()
                    converted_path = convert_to_browser_compatible(filepath)
                    converted_filename = os.path.basename(converted_path)
                    anomalous_segments.append(converted_path)
                    narration = narrate_anomaly_with_gemini(converted_path)
                    narrations.append({"segment": converted_filename, "narration": narration})
                    print(f"Anomaly saved: {converted_filename}")
                    print(narration)
                    anomaly_active = False
                    out = None
                    pre_buffer.clear()
            buffer.pop(0)
        frame_idx += 1

    if anomaly_active and out:
        out.release()
        converted_path = convert_to_browser_compatible(filepath)
        converted_filename = os.path.basename(converted_path)
        anomalous_segments.append(converted_path)
        narration = narrate_anomaly_with_gemini(converted_path)
        narrations.append({"segment": converted_filename, "narration": narration})

    cap.release()
    if losses:
        print(f"Suggested 95th percentile threshold: {np.percentile(losses, 95):.6f}")
    os.makedirs("videos", exist_ok=True)
    with open("videos/narrations.json", "w") as f:
        json.dump(narrations, f, indent=2)

    return {
        "anomalies": anomalous_segments,
        "narrations": narrations,
    }
# === Optional Display ===
def play_video_with_anomalies(video_path, model, threshold=ANOMALY_THRESHOLD, frame_size=FRAME_SIZE, sequence_length=SEQUENCE_LENGTH):
    cap = cv2.VideoCapture(video_path)
    buffer = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, frame_size)
        norm = resized.astype('float32') / 255.0
        buffer.append(norm)
        if len(buffer) == sequence_length:
            input_seq = np.array(buffer).reshape(1, sequence_length, frame_size[0], frame_size[1], 1)
            reconstructed = model.predict(input_seq, verbose=0)
            loss = np.mean((input_seq - reconstructed)**2)
            text = f"Anomaly: {'YES' if loss > threshold else 'NO'} | Loss: {loss:.4f}"
            color = (0, 0, 255) if loss > threshold else (0, 255, 0)
            cv2.putText(frame, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.imshow('Anomaly Detection', frame)
            buffer.pop(0)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# === Load/Train ===
def load_or_train_model(train_paths=None, force_retrain=False):
    if os.path.exists(MODEL_SAVE_PATH) and not force_retrain:
        print("Model found. Loading existing model.")
        return keras.models.load_model(MODEL_SAVE_PATH)
    if not train_paths:
        print("No training data provided.")
        return None
    print("Training model...")
    model = build_convlstm_autoencoder()
    gen = sequence_generator(train_paths)
    model.fit(gen, epochs=EPOCHS, steps_per_epoch=100,
              callbacks=[keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)])
    model.save(MODEL_SAVE_PATH)
    print("Model saved.")
    return model

# === Run ===
def get_video_paths(prompt):
    print(prompt)
    paths = input("Enter comma-separated video paths: ").split(',')
    return [p.strip() for p in paths if os.path.exists(p.strip())]

def main():
    print("Anomaly Detection Pipeline")
    print("1. Train\n2. Detect\n3. Both")
    choice = input("Choose mode: ").strip()
    train_paths, test_paths = [], []

    if choice in ['1', '3']:
        train_paths = get_video_paths("Enter training video paths:")
    model = load_or_train_model(train_paths if choice in ['1', '3'] else None)
    if not model:
        print("Error: Model not available.")
        return
    if choice in ['2', '3']:
        test_paths = get_video_paths("Enter test video paths:")
        for path in test_paths:
            print(f"\nDetecting anomalies in {path}...")
            segments = detect_and_crop_anomalies(path, model)
            print("Segments:", segments)
            play_video_with_anomalies(path, model)

if __name__ == '__main__':
    main()
