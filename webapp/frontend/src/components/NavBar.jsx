import React from "react";
import "./NavBar.css"; // Import the CSS file
import Logo from "../logo.png" 

function Navbar({ onAboutClick }) {
  return (
    <nav className="navbar">
      <div className="navbar-logo gloock-regular">
          <img src={Logo} className="logo" alt="logo" />
          Gauntlet xAI
      </div>
      <div className="navbar-button" onClick={onAboutClick}>
        About
      </div>
    </nav>
  );
}

export default Navbar;
