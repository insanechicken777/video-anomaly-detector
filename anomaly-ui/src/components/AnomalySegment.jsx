import React from "react";

const styles = {
  card: {
    backgroundColor: "#1e1e1e",
    padding: "16px",
    borderRadius: "10px",
    boxShadow: "0 0 10px rgba(255,0,0,0.3)",
    marginBottom: "20px",
    color: "white",
  },
  title: {
    fontSize: "1.2rem",
    marginBottom: "10px",
  },
  video: {
    borderRadius: "6px",
    marginBottom: "10px",
  },
  narration: {
    fontStyle: "italic",
    fontSize: "0.95rem",
  },
};

export default function AnomalySegment({ segment, narration }) {
  console.log("Segment value:", segment);
  
  // Extract just the filename from the segment path
  const filename = segment.split("/").pop();
  
  // Use the backend static file serving endpoint
  const videoUrl = `http://localhost:8000/videos/crops/${filename}`;
  
  console.log("Video URL:", videoUrl);
  
  return (
    <div style={styles.card}>
      <h3 style={styles.title}>{filename}</h3>
      <video
        controls
        style={styles.video}
        src={videoUrl}
        width="100%"
        preload="metadata"
      >
        Your browser does not support the video tag.
      </video>
      <p style={styles.narration}>
        <strong>Narration:</strong> {narration}
      </p>
    </div>
  );
}