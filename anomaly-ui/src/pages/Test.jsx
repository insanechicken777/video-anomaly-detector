import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import AnomalySegment from "../components/AnomalySegment";
import "../App.css";

export default function Test() {
  const [file, setFile] = useState(null);
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const navigate = useNavigate();

  // 🔐 Redirect if no token
  useEffect(() => {
    const token = localStorage.getItem("token");
    if (!token) {
      navigate("/login");
    }
  }, [navigate]);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setError("");
  };

  const handleUpload = async () => {
    if (!file) {
      setError("Please select a video file first.");
      return;
    }

    setLoading(true);
    setError("");
    const formData = new FormData();
    formData.append("file", file);

    try {
      const token = localStorage.getItem("token");

      const res = await axios.post("http://localhost:8000/upload/test/", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
          Authorization: `Bearer ${token}`, // 🔐 Include token
        },
      });

      setResponse(res.data);
    } catch (err) {
      console.error("Upload failed", err);
      setError("❌ Failed to detect anomalies. Please check the server.");
      setResponse(null);
    }

    setLoading(false);
  };

  return (
    <div className="train-container">
      <h1 className="home-title" style={{ marginBottom: "30px" }}>
        <span style={{ fontWeight: "normal" }}>Test</span>{" "}
        <span className="glow">AnomalyVision</span>
      </h1>

      <div style={{ marginBottom: "25px" }}>
        <input
          type="file"
          accept="video/*"
          onChange={handleFileChange}
          id="video-test-upload"
          style={{ display: "none" }}
        />
        <label htmlFor="video-test-upload" className="neon-button">
          {file ? file.name : "Choose Test Video"}
        </label>
      </div>

      <div style={{ marginBottom: "30px" }}>
        <button
          onClick={handleUpload}
          disabled={loading}
          className="neon-button-alt"
          style={{
            opacity: loading ? 0.6 : 1,
            cursor: loading ? "not-allowed" : "pointer",
          }}
        >
          {loading ? "Detecting..." : "Detect Anomalies"}
        </button>
      </div>

      {error && (
        <div style={{ textAlign: "center", marginBottom: "20px" }}>
          <p
            style={{
              color: "#ff6b6b",
              fontSize: "16px",
              padding: "12px 20px",
              backgroundColor: "rgba(255,107,107,0.1)",
              borderRadius: "8px",
              border: "1px solid #ff6b6b",
              maxWidth: "600px",
              margin: "auto",
            }}
          >
            {error}
          </p>
        </div>
      )}

      {response && response.narrations.length > 0 && (
        <div style={{ marginTop: "40px" }}>
          <h2
            style={{
              fontSize: "24px",
              fontWeight: "600",
              marginBottom: "20px",
              textAlign: "center",
              color: "#4ecdc4",
            }}
          >
            Detected Anomalies
          </h2>
          {response.narrations.map((entry, i) => (
            <AnomalySegment key={i} segment={entry.segment} narration={entry.narration} />
          ))}
        </div>
      )}
    </div>
  );
}
