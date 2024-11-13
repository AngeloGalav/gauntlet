// ToastNotification.js
import React from 'react';
import { Toast, ToastContainer } from 'react-bootstrap';
import 'bootstrap/dist/css/bootstrap.min.css';

const ToastNotification = ({ show, onClose, message, variant }) => {
  const bgColor = variant === 'error' ? 'bg-danger text-white' : 'bg-success text-white';

  return (
    <ToastContainer position="top-end" className="p-3">
      <Toast onClose={onClose} show={show} delay={3000} autohide>
        <Toast.Header className={bgColor}>
          <strong className="me-auto">{variant === 'error' ? 'Error' : 'Success'}</strong>
        </Toast.Header>
        <Toast.Body className='text-black'>{message}</Toast.Body>
      </Toast>
    </ToastContainer>
  );
};

export default ToastNotification;
