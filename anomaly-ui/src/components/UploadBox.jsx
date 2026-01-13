import React from "react";

const styles = {
  container: {
    backgroundColor: "#1a1a1a",
    border: "2px dashed #ff1a1a",
    borderRadius: "10px",
    padding: "20px",
    textAlign: "center",
    color: "white",
    marginBottom: "20px",
    boxShadow: "0 0 15px rgba(255, 0, 0, 0.3)",
  },
  input: {
    display: "none",
  },
  label: {
    display: "inline-block",
    padding: "10px 20px",
    backgroundColor: "#ff1a1a",
    color: "white",
    fontWeight: "bold",
    borderRadius: "5px",
    cursor: "pointer",
    boxShadow: "0 0 8px rgba(255, 0, 0, 0.6)",
    transition: "background-color 0.3s",
  },
};

export default function UploadBox({ onFileSelect }) {
  const handleChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      onFileSelect(e.target.files[0]);
    }
  };

  return (
    <div style={styles.container}>
      <input
        type="file"
        id="upload-input"
        accept="video/*"
        onChange={handleChange}
        style={styles.input}
      />
      <label htmlFor="upload-input" style={styles.label}>
        Upload Video
      </label>
    </div>
  );
}
