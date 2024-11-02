import io
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import matplotlib.pyplot as plt
from flask import Flask, request, send_file, jsonify
from torch import nn
import data_utils
import xai
import model_utils


app = Flask(__name__)

device = model_utils.get_device()

# Load a pre-trained model (replace with your own model)
class DummyModel(nn.Module):
    def forward(self, x):
        return x.mean(dim=1, keepdim=True)  # Dummy model to demonstrate

model_name = "RVAA_FTRes50_beefy"
weight_path = f"checkpoints/best_{model_name}.pth"


model = resnet50(weights='IMAGENET1K_V1')
model.fc = nn.Linear(model.fc.in_features, 1)
try:
    model.load_state_dict(torch.load(weight_path, weights_only=True))
    print("Weights loaded successfully from", weight_path)
except Exception as e:
    print("Failed to load weights:", e)

model.to(device)

model.eval()

# transformation pipeline
IMG_SIZE = 224
transform = transforms.Compose([
    transforms.Lambda(lambda image: image.convert('RGB')),
    transforms.Resize(IMG_SIZE),                  # Resize the shorter side to IMG_SIZE while keeping aspect ratio
    transforms.Lambda(data_utils.pad_to_square),  # Apply the dynamic padding to make the image square
    transforms.Resize((IMG_SIZE, IMG_SIZE)),      # Ensure the final image is exactly IMG_SIZExIMG_SIZE
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizing "should" help with Pretrained nets
])


@app.route('/process-image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file found in request"}), 400
    
    # Load image from the request
    image_file = request.files['image']
    image = Image.open(image_file).convert("RGB")
    
    # Apply the transformation
    transformed_image = transform(image).unsqueeze(0)  # Add batch dimension
    
    # TODO: gradcam code here!

    # Run the model on the transformed image
    with torch.no_grad():
        output = model(transformed_image)
    
    # Convert the output back to an image format suitable for plotting
    output_image = output.squeeze(0).permute(1, 2, 0).numpy()  # Convert to HxWxC format

    # Plot the original and transformed images
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(image)
    ax[0].set_title("Original Image")
    ax[1].imshow(output_image, cmap="gray")
    ax[1].set_title("Model Output")
    
    for axis in ax:
        axis.axis("off")  # Remove axes for cleaner display
    
    # Save the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    
    # Send the image back as a response
    return send_file(buf, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
