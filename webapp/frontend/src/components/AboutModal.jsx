import React from "react";
import { Modal, Button } from "react-bootstrap";

function AboutModal({ show, handleClose }) {
  return (
    <Modal
      show={show}
      onHide={handleClose}
      centered
      animation={true}
      dialogClassName="custom-modal"
    >
      <Modal.Header>
      <h4>About</h4>
      </Modal.Header>
      <Modal.Body>
        <p>
        GAUNTLET is an explainable model for detecting AI-generated visual
content. By examining the finer characteristics of AI generated images (such as noise patterns), it can distinguish AI-generated images from human-crafted ones. In addition, it outputs a heatmap which highlights the features that activated the model.
<br />
Our aim is not just to provide a tool for people to use, but also to educate them into learning the characteristics of an AI generated image, so that they can detect and avoid malicious content without the necessity of this app. In a sense, the app should train the user so that it won't rely on this app. Kinda crazy if you think about it.
        </p>
      </Modal.Body>
    </Modal>
  );
}

export default AboutModal;
