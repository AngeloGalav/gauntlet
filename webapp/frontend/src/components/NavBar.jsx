import React, { useState } from "react";
import "./NavBar.css"; // Import the CSS file
import Logo from "../logo.png" 

function Navbar() {
  return (
    <nav className="navbar">
      <div class="navbar-logo gloock-regular">
          <img src={Logo} className="logo" />
          Gauntlet xAI
      </div>
      <div className="navbar-button">
        About
      </div>
    </nav>
  );
}

export default Navbar;
