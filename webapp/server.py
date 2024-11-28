import sys
import os
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import matplotlib
from flask import Flask, request, send_file, jsonify
from torch import nn

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

# project specific library imports
from data_utils import pad_to_square
from model_utils import get_device
from xai import webapp_gradcam, webapp_lime
from flask_cors import CORS

import time

print("Server is starting...")

app = Flask(__name__)
FLASK_APP_PORT = int(os.getenv("FLASK_APP_PORT", 5000))
CORS(app) # Enable CORS for all routes

# non-interactive backend
matplotlib.use('Agg')

# setup stuff
device = get_device()
current_model = "RVAA_FTRes50"
model = resnet50(weights='IMAGENET1K_V1')
model.fc = nn.Linear(model.fc.in_features, 2)

# transformation pipeline
IMG_SIZE = 224
transform = transforms.Compose([
    transforms.Lambda(lambda image: image.convert('RGB')),
    transforms.Resize(IMG_SIZE),                  # Resize the shorter side to IMG_SIZE while keeping aspect ratio
    transforms.Lambda(pad_to_square),  # Apply the dynamic padding to make the image square
    transforms.Resize((IMG_SIZE, IMG_SIZE)),      # Ensure the final image is exactly IMG_SIZExIMG_SIZE
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizing "should" help with Pretrained nets
])


def load_model(model_name):
    print(f"Loading model {model_name}")
    weight_path = f"checkpoints/best_{model_name}.pth"
    try:
        model.load_state_dict(torch.load(weight_path, weights_only=True))
        print("Weights loaded successfully from", weight_path)
    except Exception as e:
        print("Failed to load weights:", e)
        exit()
    start_time = time.time()
    model.to(device)
    end_time = time.time()
    model.eval()

    duration = end_time - start_time
    print(f"Model loaded in {duration} seconds")
    print(f'Model {model_name} is ready.')

load_model(current_model)

# had to write this endpoint bc of react...
@app.route('/get-models', methods=['GET'])
def get_models():
    checkpoints_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../checkpoints'))
    try:
        # List all files in the directory
        files = os.listdir(checkpoints_dir)

        # Filter and format model names
        model_names = [
            file.replace("best_", "").replace(".pth", "")
            for file in files
            if file.startswith("best_") and file.endswith(".pth")
        ]

        # Send the model names as JSON
        return jsonify({"models": model_names}), 200

    except Exception as e:
        print("Error loading models:", e)
        return jsonify({"error": "Could not load models"}), 500

@app.route('/process-image', methods=['POST'])
def process_image():
    global current_model
    if 'image' not in request.files:
        return jsonify({"error": "No image file found in request"}), 400

    # Load image from the request
    image_file = request.files['image']
    image = Image.open(image_file).convert("RGB")

    # header info
    model_name = request.headers.get('Model')
    backend = request.headers.get('Xai-Backend')

    print(f"got model: {model_name}, backend: {backend}")

    # Apply the transformation
    image = transform(image)
    image = image.to(device)

    if model_name != current_model :
        current_model = model_name
        load_model(model_name)

    if backend != "LIME":
        mapper = "ac" if backend == "AblationCAM" else "sc"
        image_result_buf = webapp_gradcam(image,
                        model,
                        target_layers=[model.layer2, model.layer3, model.layer4],
                        mapper=mapper)
    else:
        image_result_buf = webapp_lime(image,
                model)

    return send_file(image_result_buf, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True, threaded=True, host="0.0.0.0", port=FLASK_APP_PORT)
