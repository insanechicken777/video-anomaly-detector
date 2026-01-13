import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";

export default function Signup() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [message, setMessage] = useState("");
  const navigate = useNavigate();

  const handleSignup = async () => {
    try {
      const res = await fetch("http://localhost:8000/register", {
        method: "POST",
        headers: {
          "Content-Type": "application/x-www-form-urlencoded",
        },
        body: new URLSearchParams({
          username,
          password,
        }),
      });

      if (res.ok) {
        setMessage("✅ Registered! Redirecting to login...");
        setTimeout(() => navigate("/login"), 1500);
      } else {
        const data = await res.json();
        setMessage(`❌ ${data.detail || "Registration failed"}`);
      }
    } catch (err) {
      setMessage("❌ Registration error.");
    }
  };

  return (
    <div
      className="login-container"
      style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "20px" }}
    >
      <h1 className="home-title">
        Sign Up to <span className="glow">AnomalyVision</span>
      </h1>
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
      <button onClick={handleSignup} className="neon-button">
        Sign Up
      </button>
      <p style={{ color: "#ccc" }}>
        Already have an account? <Link to="/login" style={{ color: "#ff66cc" }}>Login</Link>
      </p>
      {message && <p style={{ color: "#ccc" }}>{message}</p>}
    </div>
  );
}
