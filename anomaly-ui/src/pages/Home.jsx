import React, { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import "../App.css";

export default function Home() {
  const navigate = useNavigate();

  useEffect(() => {
    const token = localStorage.getItem("token");
    if (!token) {
      navigate("/login");
    }
  }, [navigate]);

  return (
    <div
      className="home-container"
      style={{
        textAlign: "center",
        display: "flex",
        flexDirection: "column",
        justifyContent: "center",
        minHeight: "70vh",
        padding: "20px",
      }}
    >
      <h1 className="home-title">
        Welcome to <span className="glow">AnomalyVision</span>
      </h1>

      <p
        className="home-description"
        style={{
          fontSize: "18px",
          marginTop: "20px",
          color: "#ccc",
          maxWidth: "700px",
          marginLeft: "auto",
          marginRight: "auto",
        }}
      >
        AnomalyVision helps detect unusual activity in surveillance footage using AI-powered video anomaly detection and narration.
        Train models, detect anomalies, and view segments with intelligent narrations.
      </p>
    </div>
  );
}
