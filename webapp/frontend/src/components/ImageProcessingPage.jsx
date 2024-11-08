import 'bootstrap/dist/css/bootstrap.css';
import './ImageProcessingPage.css'
import React, { useState } from "react";
import axios from 'axios';

function ImageProcessingPage() {
  const [selectedModel, setSelectedModel] = useState("RVAA_FTRes50");
  const [incomingImage, setIncomingImage] = useState(null);
  const [loadedImage, setLoadedImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isLoading, setIsLoading] = useState(false);  // Loading state
  const [serverURL, setServerURL] = useState("http://localhost:5000");  // Loading state

  // TODO: add other models later
  const models = [
    {name: 'RVAA_FTRes50' },
    {name: 'RVAA_FTRes50_beefy' }
  ];

  const handleModelChange = (e) => {
    setSelectedModel(e.target.value);
    // TODO: Add server logic for setting model
  };

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setLoadedImage(file);

    // Image preview
    const reader = new FileReader();
    reader.onloadend = () => {
        setPreviewUrl(reader.result);
        };
        reader.readAsDataURL(file);
    }
  };

  const handleServerUpload = async (e) => {
    if (loadedImage) {
      // Upload the image to the server
      const formData = new FormData();
      formData.append('image', loadedImage);
      
      setIsLoading(true);

      try {
        const response = await axios.post(serverURL + '/process-image', formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          },
          responseType: 'blob'  // Expect a blob response
        });

        console.log(response)

        if (response.status === 200) {
          // If the response is successful, set the incoming image to the server's response
          const imageBlob = response.data;
          console.log(response.data)
          const imageUrl = URL.createObjectURL(response.data);
          console.log(imageUrl)
          setIsLoading(false);
          setIncomingImage(imageUrl);
        }
      } catch (error) {
        console.error("Error uploading image:", error);
        alert("An error occurred while uploading the image.");
      }
    } else {
      alert("Upload an image first!")
    }
  };

  return (
    <div
    className="page-container"
    style={{ maxWidth: "", margin: "auto" }}
    >
      <div className='images-container'>
      {/* Image Upload Section */}
      <div>
        <h3 className='font-cool'>Upload an Image</h3>
        <input type="file" accept="image/*" onChange={handleImageUpload} />
        {loadedImage &&
        <div>
            <img
            src={previewUrl}
            className='image-place'
            alt="Uploaded Preview"
            />
            <p>
            Uploaded image: {loadedImage.name}
            </p>
        </div>
        }
      </div>
    
      {/* Incoming Image Display */}
      {(isLoading || incomingImage) &&
        (<div style={{ marginBottom: "20px" }}>
          <h3 className='font-cool'>Model Output</h3>
          {(incomingImage && !isLoading) ? (
            <img
            src={incomingImage}
            alt="Incoming from server"
            style={{ width: "100%", maxHeight: "500px", objectFit: "contain" }}
            />
          ) : (
            null
          )}

          {
          (isLoading) &&
            (<div class="loader"></div>)
          }
      </div>)}
    </div>
    <div className='buttons-container'>
          {/* Model Selection Dropdown */}
        <div style={{ marginBottom: "20px" }}>
        <h3 className='font-cool'>Select Model</h3>
        <select
                className="form-select"
                value={selectedModel}
                onChange={handleModelChange}
            >
                {models.map((model) => (
                    <option key={model.id} value={model.id}>{model.name}</option>
                ))}
            </select>
        </div>

        {/* server input */}
        <div>
        <h3 className='font-cool'>Server URL</h3>
        <form>
          <div className="form-group">
            <input className="form-control" id="serverl_url" placeholder="http://localhost:5000" />
          </div>
        </form>
        </div>

        {/* backend radio */}
        <div className='xai-radio'>
        <h3 className='font-cool'>Select xAI backend</h3>
          <div className="form-check">
            <input className="form-check-input" type="radio" name="flexRadioDefault" id="flexRadioDefault1" />
            <label className="form-check-label" for="flexRadioDefault1">
              AblationCAM
            </label>
          </div>
          <div className="form-check">
            <input className="form-check-input" type="radio" name="flexRadioDefault" id="flexRadioDefault2" />
            <label className="form-check-label" for="flexRadioDefault2">
            ScoreCAM
            </label>
          </div>
          <div className="form-check">
            <input className="form-check-input" type="radio" name="flexRadioDefault" id="flexRadioDefault3" />
            <label className="form-check-label" for="flexRadioDefault3">
              LIME
            </label>
          </div>
        </div>

        {/* Upload button */}
        <div style={{ marginBottom: "20px" }}>
        <h3 className='font-cool'>...and finally</h3>
        <button className="button" onClick={handleServerUpload}>Scan image!</button>
        </div>
        
        {/* Server status */}
        <div className='server-status'>
          SERVER STATUS: <span id="server-status">OK</span>
        </div>
      </div>
    </div>
  );
}

export default ImageProcessingPage;
