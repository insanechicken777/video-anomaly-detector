import torch
import openai # Keeping this if you still use it elsewhere, but not for narration here
import base64
import io
from PIL import Image # Keeping this for potential future image processing, not strictly needed for Gemini video input
import cv2
import numpy as np
import os
from tensorflow import keras
from keras import layers

# Import Gemini specific libraries
import google.generativeai as genai

genai.configure(api_key="AIzaSyAgVrbHnOMtnrg7afkS_j6Cf6Xz-A91Q7I")


# Remove previous anomaly clips
for f in os.listdir():
    if f.startswith("anomaly_segment_") and f.endswith(".avi"):
        os.remove(f)

# === Parameters ===
FRAME_SIZE = (64, 64)
SEQUENCE_LENGTH = 10
BATCH_SIZE = 4
EPOCHS = 10
ANOMALY_THRESHOLD = 0.01
MERGE_BUFFER_FRAMES = 50
GEMINI_MODEL_NAME = 'gemini-1.5-flash'
MODEL_SAVE_PATH = 'anomaly_detection_model.keras'  # Path to save/load the model

# === 1. ConvLSTM Autoencoder ===
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

# === 2. Frame Sequence Generator ===
def sequence_generator(video_paths, sequence_length=SEQUENCE_LENGTH, batch_size=BATCH_SIZE, size=FRAME_SIZE):
    buffer = []
    while True:
        for path in video_paths:
            cap = cv2.VideoCapture(path)
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

# === 3. Load or Train Model ===
def load_or_train_model():
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"Loading existing model from {MODEL_SAVE_PATH}...")
        try:
            autoencoder = keras.models.load_model(MODEL_SAVE_PATH)
            print("Model loaded successfully!")
            return autoencoder
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Training new model...")
    else:
        print("No existing model found. Training new model...")
    
    # Train new model
    autoencoder = build_convlstm_autoencoder()
    normal_videos = [f'videos/v{i}.avi' for i in range(1, 17)]
    
    # Check if training videos exist
    existing_videos = [v for v in normal_videos if os.path.exists(v)]
    if not existing_videos:
        print("Error: No training videos found. Please check the 'videos' directory.")
        return None
    
    print(f"Found {len(existing_videos)} training videos. Starting training...")
    train_gen = sequence_generator(existing_videos)
    
    # Train the model
    autoencoder.fit(train_gen, epochs=EPOCHS, steps_per_epoch=100, verbose=1,
                    callbacks=[keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)])
    
    # Save the trained model
    try:
        autoencoder.save(MODEL_SAVE_PATH)
        print(f"Model saved successfully to {MODEL_SAVE_PATH}")
    except Exception as e:
        print(f"Error saving model: {e}")
    
    print("Training complete!")
    return autoencoder

# Load or train the model
autoencoder = load_or_train_model()

if autoencoder is None:
    print("Failed to load or train model. Exiting.")
    exit()

# --- Gemini Narration Function ---
def narrate_anomaly_with_gemini(video_path):
    print(f"Narrating anomaly clip: {video_path}")
    try:
        # Check file size. If over ~18-19MB (to leave room for prompt tokens),
        # you might need to upload to GCS first.
        # For simplicity, we'll try direct upload here.
        if os.path.getsize(video_path) > (19 * 1024 * 1024): # Approx 19MB limit
            print(f"Warning: Video file '{video_path}' is larger than 19MB. "
                  "Direct upload might fail. Consider implementing GCS upload.")
            # Fallback (or raise error) if GCS is not implemented
            return "Video too large for direct upload. GCS upload required."

        with open(video_path, 'rb') as video_file:
            video_bytes = video_file.read()

        model = genai.GenerativeModel(GEMINI_MODEL_NAME)

        # Create a Part from the video bytes
        video_part={
                     'mime_type': 'video/avi', # Ensure this matches your video format (e.g., "video/mp4", "video/x-msvideo")
                    'data': video_bytes
        }

        # Craft the prompt for Gemini
        prompt = (
            "Analyze this video segment for any unusual or anomalous activities. "
            "Describe the anomalous event in detail, including what is happening, "
            "who or what is involved, and the general context. "
            "Focus on the 'why' if possible, or observable deviations from normal. "
            "If no obvious anomaly, describe the main activity. "
            "Keep the narration concise and focused on the unusual elements."
        )

        contents = [prompt, video_part]

        response_stream = model.generate_content(contents, stream=True)

        full_narration = ""
        for chunk in response_stream:
            if chunk.text:
                full_narration += chunk.text
        return full_narration

    except Exception as e:
        print(f"Error during Gemini narration for {video_path}: {e}")
        return f"Narration failed due to an error: {e}"


def detect_and_crop_anomalies(video_path, model, threshold=ANOMALY_THRESHOLD, frame_size=FRAME_SIZE, sequence_length=SEQUENCE_LENGTH):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    original_frames_buffer = []
    pre_anomaly_buffer = []
    buffer = []

    segment_id = 0
    out = None
    frame_index = 0
    normal_frame_count = 0
    anomaly_active = False
    anomalous_segments = []

    current_segment_start_frame = -1
    PRE_ANOMALY_FRAMES = 50 # Number of frames before anomaly to include
    POST_ANOMALY_FRAMES = MERGE_BUFFER_FRAMES  # Already defined earlier

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        original_frames_buffer.append(frame)
        pre_anomaly_buffer.append(frame)
        if len(pre_anomaly_buffer) > PRE_ANOMALY_FRAMES:
            pre_anomaly_buffer.pop(0)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, frame_size)
        norm = resized.astype('float32') / 255.0
        buffer.append(norm)

        if len(buffer) == sequence_length:
            input_seq = np.array(buffer).reshape(1, sequence_length, frame_size[0], frame_size[1], 1)
            reconstructed = model.predict(input_seq, verbose=0)
            loss = np.mean((input_seq - reconstructed)**2)

            if loss > threshold:
                normal_frame_count = 0
                if not anomaly_active:
                    anomaly_active = True
                    segment_id += 1
                    current_segment_start_frame = frame_index - sequence_length + 1
                    filename = f"anomaly_segment_{segment_id}.avi"
                    out = cv2.VideoWriter(filename, fourcc, fps, (frame.shape[1], frame.shape[0]))
                    print(f"Anomaly started at frame {current_segment_start_frame} (writing to {filename})")

                    # Write pre-anomaly buffer
                    for f_in_pre in pre_anomaly_buffer:
                        out.write(f_in_pre)
                out.write(frame)
            else:
                if anomaly_active:
                    normal_frame_count += 1
                    out.write(frame)
                    if normal_frame_count >= POST_ANOMALY_FRAMES:
                        out.release()
                        print(f"Anomaly segment {segment_id} ended at frame {frame_index}. Saved to {filename}")
                        anomalous_segments.append(filename)

                        # Narrate
                        narration = narrate_anomaly_with_gemini(filename)
                        print(f"Narration for {filename}:\n{narration}\n")

                        anomaly_active = False
                        out = None
                        current_segment_start_frame = -1
                        pre_anomaly_buffer.clear()
                else:
                    if len(original_frames_buffer) > sequence_length:
                        original_frames_buffer.pop(0)

            buffer.pop(0)
        frame_index += 1

    # Handle anomaly active till end
    if anomaly_active and out:
        out.release()
        filename = f"anomaly_segment_{segment_id}.avi"
        anomalous_segments.append(filename)
        print(f"Anomaly segment {segment_id} ended at video end. Saved to {filename}")

        narration = narrate_anomaly_with_gemini(filename)
        print(f"Narration for {filename}:\n{narration}\n")

    cap.release()
    if not anomalous_segments:
        print("No anomalies detected.")
    else:
        print(f"Detected {len(anomalous_segments)} anomaly segments.")
    return anomalous_segments

# === 5. Run Detection ===
test_video = 'testvid/t14.avi' # Ensure this path is correct
if not os.path.exists(test_video):
    print(f"Error: Test video '{test_video}' not found. Please check the path.")
else:
    print("Running anomaly detection...")
    anomaly_files = detect_and_crop_anomalies(test_video, autoencoder)
    print("Detection completed. Anomaly files:", anomaly_files)

# === 6. Optional Visualization ===
def play_video_with_anomalies(video_path, model, threshold=ANOMALY_THRESHOLD, frame_size=FRAME_SIZE, sequence_length=SEQUENCE_LENGTH):
    cap = cv2.VideoCapture(video_path)
    buffer = []
    frame_index = 0
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
        frame_index += 1
    cap.release()
    cv2.destroyAllWindows()

# Run visualization if the test video exists
if os.path.exists(test_video):
    play_video_with_anomalies(test_video, autoencoder)

print("Script complete.")