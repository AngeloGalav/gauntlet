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
          Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus
          lacinia odio vitae vestibulum vestibulum. Cras venenatis euismod
          malesuada.
        </p>
      </Modal.Body>
    </Modal>
  );
}

export default AboutModal;
