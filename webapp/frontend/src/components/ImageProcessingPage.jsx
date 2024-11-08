import 'bootstrap/dist/css/bootstrap.css';
import './ImageProcessingPage.css'
import React, { useState } from "react";

function ImageProcessingPage() {
  const [selectedModel, setSelectedModel] = useState("");
  const [incomingImage, setIncomingImage] = useState(null);
  const [uploadedImage, setUploadedImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);

  // Dummy model list for the dropdown
  const modelList = ["Model A", "Model B", "Model C"];

  const handleModelChange = (e) => {
    setSelectedModel(e.target.value);
    // TODO: Add server logic for setting model
  };

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setUploadedImage(file);
      // TODO: Add server logic with axios here for uploading the image
    
    // Image preview
    const reader = new FileReader();
    reader.onloadend = () => {
        setPreviewUrl(reader.result);
        };
        reader.readAsDataURL(file);
    }
  };

  // Function to retrieve image from the server (replace with actual fetch logic)
  const fetchIncomingImage = async () => {
    // Example: Fetch the image URL from your server
    // const response = await fetch('your-server-url');
    // const imageUrl = await response.json();
    const imageUrl = "https://placehold.co/600x400"; // Placeholder image URL
    setIncomingImage(imageUrl);
  };

  return (
    <div 
    className="page-container"
    style={{ maxWidth: "", margin: "auto" }}
    id='grad2' >
      <div className='images-container'>
      {/* Image Upload Section */}
      <div>
        <h3 className='font-cool'>Upload an Image</h3>
        <input type="file" accept="image/*" onChange={handleImageUpload} />
        {uploadedImage &&
        <div>
            <img
            src={previewUrl}
            alt="Uploaded Preview"
            style={{ width: "100%", maxHeight: "300px", objectFit: "contain"}}
            />
            {console.log(uploadedImage)}
            Uploaded image: {uploadedImage.name}
        </div>
        }
      </div>
    
          {/* Incoming Image Display */}
      <div style={{ marginBottom: "20px" }}>
          <h3 className='font-cool'>Incoming Image from Server</h3>
          {incomingImage ? (
            <img
            src={incomingImage}
            alt="Incoming from server"
            style={{ width: "100%", maxHeight: "300px", objectFit: "contain" }}
            />
          ) : (
            <p>No image loaded. <button onClick={fetchIncomingImage}>Load Image</button></p>
          )}
      </div>
    </div>
    <div className='buttons-container'>
          {/* Model Selection Dropdown */}
        <div style={{ marginBottom: "20px" }}>
        <h3 className='font-cool'>Select Model</h3>
        <select value={selectedModel} onChange={handleModelChange}>
          <option value="">Select a model</option>
          {modelList.map((model) => (
            <option key={model} value={model}>
              {model}
            </option>
          ))}
        </select>
        </div>
        {/* Upload button */}
        <div style={{ marginBottom: "20px" }}>
        <h3 className='font-cool'>...and finally</h3>
        <button class="button">Scan image!</button>
        </div>
      </div>
    </div>
  );
}

export default ImageProcessingPage;
