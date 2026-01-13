import React from "react";

const containerStyle = {
  width: "100%",
  backgroundColor: "#2c2c2c",
  borderRadius: "8px",
  overflow: "hidden",
  boxShadow: "0 0 10px rgba(255, 0, 0, 0.4)",
  marginBottom: "20px",
};

const fillerBase = {
  height: "16px",
  transition: "width 0.4s ease-in-out",
};

const labelStyle = {
  color: "white",
  fontSize: "0.85rem",
  textAlign: "right",
  paddingRight: "5px",
  marginTop: "4px",
};

export default function ProgressBar({ progress = 0 }) {
  const fillerStyle = {
    ...fillerBase,
    width: `${progress}%`,
    background: "linear-gradient(90deg, #ff0033, #ff3300)",
    boxShadow: "0 0 10px rgba(255, 0, 0, 0.6)",
  };

  return (
    <div>
      <div style={containerStyle}>
        <div style={fillerStyle}></div>
      </div>
      <div style={labelStyle}>{progress}%</div>
    </div>
  );
}
