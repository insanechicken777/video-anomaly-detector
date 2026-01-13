import PrivateRoute from "./components/PrivateRoute";
import { BrowserRouter as Router, Routes, Route, useLocation } from "react-router-dom";
import Navbar from "./components/Navbar";
import Home from "./pages/Home";
import Train from "./pages/Train";
import Test from "./pages/Test";
import Login from "./pages/Login"; 
import Signup from "./pages/Signup";
import { useEffect, useState } from "react";

// Custom wrapper to use hooks outside Router
function AppWrapper() {
  return (
    <Router>
      <Navbar />
      <AppContent />
    </Router>
  );
}

function AppContent() {
  const location = useLocation();
  const isHome = location.pathname === "/";

  return (
    <div
      style={{
        paddingTop: "80px",
        minHeight: "100vh",
        backgroundColor: "#111",
        color: "#fff",
        display: "flex",
        justifyContent: "center",
        alignItems: isHome ? "center" : "flex-start",
      }}
    >
      <div
        style={{
          width: "100%",
          padding: "20px",
        }}
      >
        <Routes>
  <Route
    path="/"
    element={
      <PrivateRoute>
        <Home />
      </PrivateRoute>
    }
  />
  <Route path="/signup" element={<Signup />} />
  <Route path="/login" element={<Login />} />
  <Route path="/train" element={<Train />} />
  <Route path="/test" element={<Test />} />
</Routes>

      </div>
    </div>
  );
}

export default AppWrapper;
