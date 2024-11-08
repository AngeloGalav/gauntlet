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
from data_utils import pad_to_square
from model_utils import get_device
from xai import webapp_gradcam
from flask_cors import CORS


app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# non-interactive backend
matplotlib.use('Agg')

# setup stuff
device = get_device()
model = resnet50(weights='IMAGENET1K_V1')
model.fc = nn.Linear(model.fc.in_features, 1)

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


# TODO: change it so it can be changed on request
def load_model(model_name):
    weight_path = f"checkpoints/best_{model_name}.pth"
    try:
        model.load_state_dict(torch.load(weight_path, weights_only=True))
        print("Weights loaded successfully from", weight_path)
    except Exception as e:
        print("Failed to load weights:", e)
        exit()
    model.to(device)
    model.eval()
    print(f'Model {model_name} is ready.')


load_model("RVAA_FTRes50")

@app.route('/process-image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file found in request"}), 400

    # Load image from the request
    image_file = request.files['image']
    image = Image.open(image_file).convert("RGB")

    # Apply the transformation
    image = transform(image)
    image = image.to(device)

    image_result_buf = webapp_gradcam(image,
                       model,
                       target_layers=[model.layer2, model.layer3, model.layer4],
                       mapper="sc")

    return send_file(image_result_buf, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True, threaded=True, host="0.0.0.0", port=5000)
