import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import "../App.css";

export default function Train() {
  const [videos, setVideos] = useState([]);
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  // 🔒 Check for token and redirect if not logged in
  useEffect(() => {
    const token = localStorage.getItem("token");
    if (!token) {
      navigate("/login");
    }
  }, [navigate]);

  const handleUpload = async () => {
    if (videos.length === 0) {
      setMessage("Please select videos to upload.");
      return;
    }

    const formData = new FormData();
    for (let i = 0; i < videos.length; i++) {
      formData.append("files", videos[i]);
    }

    try {
      setLoading(true);
      setMessage("Uploading...");
      const token = localStorage.getItem("token");

      const res = await fetch("http://localhost:8000/upload/train-videos/", {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
        },
        body: formData,
      });

      const data = await res.json();
      setMessage(data.message || "Upload complete.");
    } catch (err) {
      setMessage("❌ Error uploading videos.");
    } finally {
      setLoading(false);
    }
  };

  const handleTrain = async () => {
    try {
      setLoading(true);
      setMessage("Training started...");
      const token = localStorage.getItem("token");

      const res = await fetch("http://localhost:8000/train/", {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      const data = await res.json();
      setMessage(data.message || data.error || "Training complete.");
    } catch (err) {
      setMessage("❌ Error starting training.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="train-container">
      <h1 className="train-title">
        <span className="plain-title">Train</span> <span className="glow">AnomalyVision</span>
      </h1>

      <div style={{ margin: "20px 0" }}>
        <input
          type="file"
          multiple
          accept="video/*"
          onChange={(e) => setVideos(e.target.files)}
          id="video-upload"
          style={{ display: "none" }}
        />
        <label htmlFor="video-upload" className="neon-button">
          {videos.length > 0 ? `${videos.length} files selected` : "Choose Files"}
        </label>
      </div>

      <div style={{ display: "flex", gap: "15px", flexWrap: "wrap", justifyContent: "center" }}>
        <button onClick={handleUpload} className="neon-button">
          Upload
        </button>
        <button onClick={handleTrain} className="neon-button-alt">
          Start Training
        </button>
      </div>

      {message && (
        <div style={{ marginTop: "30px", textAlign: "center" }}>
          <p
            style={{
              fontSize: "16px",
              color: message.startsWith("❌") ? "#ff6b6b" : "#4ecdc4",
              padding: "12px 20px",
              backgroundColor: "rgba(255,255,255,0.05)",
              borderRadius: "8px",
              border: `1px solid ${message.startsWith("❌") ? "#ff6b6b" : "#4ecdc4"}`,
              maxWidth: "600px",
              margin: "auto",
            }}
          >
            {loading ? "⏳ " : message.startsWith("❌") ? "" : "✅ "}
            {message}
          </p>
        </div>
      )}
    </div>
  );
}
