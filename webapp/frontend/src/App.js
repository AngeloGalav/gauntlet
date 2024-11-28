import 'bootstrap/dist/css/bootstrap.css';
import MyFooter from './components/Footer';
import ImageProcessingPage from './components/ImageProcessingPage';
import AboutModal from './components/AboutModal';
import Navbar from './components/NavBar';
import './App.css';
import React, { useState } from "react";

function App() {

  const [showAboutModal, setShowAboutModal] = useState(false);

  const handleShowAboutModal = () => {setShowAboutModal(true);}
  const handleCloseAboutModal = () => setShowAboutModal(false);

  return (
    <div>
    <Navbar onAboutClick={handleShowAboutModal} />
    <div
    className='min-vh-90' style={{minHeight:"100%", maxWidth:"100%"}}
    >
      <ImageProcessingPage />
      <MyFooter />
    </div>

    <AboutModal show={showAboutModal} handleClose={handleCloseAboutModal} />
    </div>
  );
}

export default App;
