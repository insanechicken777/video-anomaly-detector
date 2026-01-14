# Video Anomaly Detector

A deep learning–based system that detects anomalous events in videos by learning normal behavior patterns and flagging deviations.  
Designed for applications such as **CCTV surveillance**, **industrial monitoring**, and **security analytics**.

---

## Features

- Upload and process video files
- Learns *normal* motion patterns using deep learning
- Detects anomalies based on reconstruction error
- Automatically extracts anomalous video segments
- Generates textual narration/description of detected anomalies
- Frontend–backend architecture for end-to-end usage

---

## Model Architecture

- **ConvLSTM Autoencoder**
  - Captures both spatial and temporal features
  - Trained only on *normal* video data
- **Anomaly Detection**
  - High reconstruction error → anomaly
- **Narration Module**
  - Generates natural language explanations for detected events

---

## Tech Stack

### Backend
- Python
- FastAPI
- OpenCV
- TensorFlow / PyTorch
- ConvLSTM Autoencoder

### Frontend
- React (Vite)
- Axios
- Modern UI with progress tracking

### Other
- SSH-based GitHub workflow
- Modular pipeline design

---

## Project Structure
video-anomaly-detector/
│
├── backend/
│ ├── models/
│ ├── routes/
│ ├── utils/
│ └── main.py
│
├── frontend/
│ ├── src/
│ ├── components/
│ └── pages/
│
├── data/
│ ├── train/
│ └── test/
│
└── README.md


---

##  How It Works

1. Train the model using videos containing **only normal behavior**
2. Upload a test video
3. System analyzes frame sequences
4. Anomalies are detected based on reconstruction error
5. Anomalous segments are extracted and narrated

---

##  Use Cases

- Smart CCTV surveillance
- Industrial safety monitoring
- Traffic anomaly detection
- Behavioral analysis
- Security systems

---

##  Future Improvements

- Real-time video stream support
- Multi-camera anomaly fusion
- Improved narration using larger vision-language models
- Dashboard analytics & alerting
- Cloud deployment (AWS/GCP)

---

##  Author

**insanechicken777**

---

##  License

This project is for educational and research purposes.


