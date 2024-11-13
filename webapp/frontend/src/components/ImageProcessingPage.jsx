import 'bootstrap/dist/css/bootstrap.css';
import './ImageProcessingPage.css'
import ToastNotification from "./ToastNotification.jsx";
import React, { useState, useEffect} from "react";
import axios from 'axios';
import { BiRefresh } from "react-icons/bi";

function ImageProcessingPage() {
  const [selectedModel, setSelectedModel] = useState("RVAA_FTRes50");
  const [incomingImage, setIncomingImage] = useState(null);
  const [loadedImage, setLoadedImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isLoading, setIsLoading] = useState(false);  // Loading state
  const [serverURL, setServerURL] = useState("http://localhost:5000");
  const [selectedBackend, setSelectedBackend] = useState("ScoreCAM");
  const [models, setModels] = useState([{name : "Loading models"}]);

   // Toast state
   const [toastMessage, setToastMessage] = useState("");
   const [showToast, setShowToast] = useState(false);
   const [toastVariant, setToastVariant] = useState("success");

  // dynamic fetchModels
  const fetchModels = async () => {
    try {
      const response = await axios.get(serverURL + "/get-models");
      if (response.status === 200) {
        const modelData = response.data.models.map(modelName => ({ name: modelName }));
        setModels(modelData);
        setToastMessage("Models loaded successfully!");
        setToastVariant("success");
      }
    } catch (error) {
      console.error("Error fetching models:", error);
      setToastMessage("Error fetching models. Please try again.");
      setToastVariant("error");
      setModels([])
    }
    setShowToast(true);
  };

  // Run fetchModels on component mount
  useEffect(() => {
    fetchModels();
  }, []);

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
            'Content-Type': 'multipart/form-data',
            'model': selectedModel,
            'xai-backend': selectedBackend,
          },
          responseType: 'blob'  // Expect a blob response
        });

        console.log(response)

        if (response.status === 200) {
          // If the response is successful, set the incoming image to the server's response
          const imageUrl = URL.createObjectURL(response.data);
          console.log(imageUrl)
          setIsLoading(false);
          setIncomingImage(imageUrl);
        }
      } catch (error) {
        console.error("Error uploading image:", error);
        setToastMessage("An error occurred while uploading the image.");
        setToastVariant("error");
        setIsLoading(false);
        setShowToast(true)
      }
    } else {
      setToastMessage("Upload an image first!");
      setToastVariant("error");
      setShowToast(true);
    }
  };

  return (
    <div
    className="page-container"
    style={{ maxWidth: "", margin: "auto" }}
    >

      {/* Toast Notification */}
      <ToastNotification
        show={showToast}
        onClose={() => setShowToast(false)}
        message={toastMessage}
        variant={toastVariant}
      />

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
      onChange={(e) => setSelectedModel(e.target.value)}
    >
            {models.map((model) => (
                <option key={model.id} value={model.id}>{model.name}</option>
            ))}
        </select>
    </div>

    {/* server input */}
    <div>
    <h3 className='font-cool'>Server URL <span onClick={fetchModels} style={{ cursor: 'pointer' }}><BiRefresh /></span></h3>
    <form>
      <div className="form-group">
        <input className="form-control" 
          onChange={(e) => setServerURL(e.target.value)}
          value={serverURL}
          id="serverl_url" />
      </div>
    </form>
    </div>


    {/* backend radio */}
    <div className='xai-radio'>
    <h3 className='font-cool'>Select xAI backend</h3>
      <div className="form-check">
        <input className="form-check-input" checked={selectedBackend === "AblationCAM"}
          value="AblationCAM"
          onChange={() => setSelectedBackend("AblationCAM")}
          type="radio" name="flexRadioDefault" id="flexRadioDefault1" />
        <label className="form-check-label" for="flexRadioDefault1">
          AblationCAM
        </label>
      </div>
      <div className="form-check">
        <input className="form-check-input" checked={selectedBackend === "ScoreCAM"}
          value="ScoreCAM"
          onChange={() => setSelectedBackend("ScoreCAM")}
          type="radio" name="flexRadioDefault" id="flexRadioDefault2" />
        <label className="form-check-label" for="flexRadioDefault2">
        ScoreCAM
        </label>
      </div>
      <div className="form-check">
        <input className="form-check-input"
          checked={selectedBackend === "LIME"}
          value="LIME"
          onChange={() => setSelectedBackend("LIME")}
          type="radio" name="flexRadioDefault" id="flexRadioDefault3" />
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

    </div>
  </div>
  );
}

export default ImageProcessingPage;
