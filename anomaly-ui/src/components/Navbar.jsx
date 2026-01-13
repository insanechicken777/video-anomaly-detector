import { Link, useNavigate } from "react-router-dom";
import "./Navbar.css";

export default function Navbar() {
  const navigate = useNavigate();
  const isLoggedIn = !!localStorage.getItem("token");

  const handleLogout = () => {
    localStorage.removeItem("token");
    navigate("/login");
  };

  return (
    <nav className="navbar">
      <h1 className="navbar-title">AnomalyVision</h1>

      <div className="navbar-links">
        {isLoggedIn && (
          <>
            <Link to="/">Home</Link>
            <Link to="/train">Train</Link>
            <Link to="/test">Test</Link>
            <button onClick={handleLogout} className="logout-button">
              Logout
            </button>
          </>
        )}
      </div>
    </nav>
  );
}
