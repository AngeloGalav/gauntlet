import React, { useState } from "react";
import "./NavBar.css"; // Import the CSS file

function Navbar() {
  return (
    <nav className="navbar">
      <div className="navbar-logo gloock-regular">Gauntlet xAI</div>
      <div className="navbar-button">
        About
      </div>
    </nav>
  );
}

export default Navbar;
