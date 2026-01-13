import { useEffect, useState } from "react";

export default function Narration() {
  const [narrations, setNarrations] = useState([]);
  const [error, setError] = useState("");

  useEffect(() => {
    const fetchNarrations = async () => {
      try {
        const res = await fetch("http://localhost:8000/videos/narrations.json");
        if (!res.ok) {
          throw new Error("Failed to load narrations.");
        }
        const data = await res.json();
        setNarrations(data);
      } catch (err) {
        setError("Could not fetch narration data.");
      }
    };

    fetchNarrations();
  }, []);

  return (
    <div style={{ paddingTop: "80px", paddingLeft: "20px", color: "white" }}>
      <h1 style={{ fontSize: "28px", fontWeight: "bold" }}>Narration Log</h1>
      {error && <p style={{ color: "red" }}>{error}</p>}

      {narrations.length === 0 ? (
        <p>No narrations available.</p>
      ) : (
        narrations.map((item, index) => (
          <div key={index} style={{ marginBottom: "20px" }}>
            <h3>Segment: {item.segment}</h3>
            <p style={{ fontStyle: "italic" }}>{item.narration}</p>
            <hr style={{ marginTop: "10px" }} />
          </div>
        ))
      )}
    </div>
  );
}
