import React, { useState } from "react";
import axios from "axios";
import { useNavigate, Link } from "react-router-dom";

export default function Login() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const navigate = useNavigate();

  const handleLogin = async (e) => {
    e.preventDefault();
    try {
      const formData = new URLSearchParams();
      formData.append("username", username);
      formData.append("password", password);

      const response = await axios.post("http://127.0.0.1:8000/token", formData, {
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
      });

      const token = response.data.access_token;
      localStorage.setItem("token", token);
      setError("");
      navigate("/");
    } catch (err) {
      setError("Invalid username or password");
    }
  };

  return (
    <div
      className="train-container"
      style={{
        textAlign: "center",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        minHeight: "70vh",
        justifyContent: "center",
        padding: "20px",
      }}
    >
      <h1 className="home-title">
        <span style={{ fontWeight: "normal" }}>Login to </span>
        <span className="glow">AnomalyVision</span>
      </h1>

      <form
        onSubmit={handleLogin}
        style={{
          display: "flex",
          flexDirection: "column",
          gap: "20px",
          marginTop: "30px",
          alignItems: "center",
        }}
      >
        <input
          type="text"
          placeholder="Username"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          className="padded-input"
        />
        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          className="padded-input"
        />
        <button type="submit" className="neon-button-alt">
          Login
        </button>
        {error && <p style={{ color: "#ff6b6b" }}>{error}</p>}
      </form>

      <p style={{ color: "#ccc", marginTop: "20px" }}>
        Don’t have an account?{" "}
        <Link to="/signup" style={{ color: "#ff66cc" }}>
          Sign Up
        </Link>
      </p>
    </div>
  );
}
