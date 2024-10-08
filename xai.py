# scripts for xai stuff
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import numpy as np
import os, json

import torch
import torch.nn.functional as F
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn.functional as F
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import numpy as np

from pytorch_grad_cam import AblationCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

device = 'cpu'

# LIME predict function, returns class probabilities
def batch_predict(images, model):
    # Convert list of numpy arrays to tensor
    batch = torch.stack([torch.tensor(i).permute(2, 0, 1) for i in images]).to(device)
    logits = model(batch)
    probs = torch.sigmoid(logits)
    return probs.detach().cpu().numpy()


def explain_lime_single_image(dataloader, model):
    model.eval()
    for images, _ in dataloader:
        # Select the first image in the batch
        image = images[0]  # Tensor format

        # place data in correct channels
        numpy_image = image.permute(1, 2, 0).cpu().numpy()

        # Initialize LIME Image Explainer
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            numpy_image,  # Image as numpy array
            lambda x: batch_predict(x, model),  # Prediction function
        )

        # Get the image and mask for the top predicted class
        img, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=False,
            hide_rest=True
        )

        fig = plt.figure(figsize=(15,5))

        ax = fig.add_subplot(141)
        ax.imshow(numpy_image, cmap="gray");
        ax.set_title("Original Image")
        ax = fig.add_subplot(142)
        ax.imshow(img);
        ax.set_title("Image")
        ax = fig.add_subplot(143)
        ax.imshow(mask);
        ax.set_title("Mask")
        ax = fig.add_subplot(144)
        ax.imshow(mark_boundaries(img, mask, color=(0,1,0)));
        ax.set_title("Image+Mask Combined");

        break  # Only run for the first batch

#TODO: change function so that it takes image index
def explain_gradcam_single_image(dataloader, model, target_layers):
    model.eval()
    for images, _ in dataloader:

        image = images[0]
        numpy_image = image.permute(1, 2, 0).cpu().numpy()
        am = AblationCAM(model=model, target_layers=target_layers)

        def image_mapper(image):
            grayscale_am = am(
                input_tensor=image.unsqueeze(0),
                targets=[ClassifierOutputTarget(0)],
            )[0]

            return show_cam_on_image(
                image.permute(1, 2, 0).numpy(),
                grayscale_am,
                use_rgb=True,
            )

        plt.imshow(image_mapper(image))

        break  # Only run for the first batch


# def explain_gradcam_batch(
#     dataloader,
#     batch_size,
#     show_label=True,
#     batch_indices=lambda data_loader: [0, len(data_loader) - 2],
#     columns=32,
#     img_size=2
# ):
#     nrows = (batch_size // columns)
#     batch_index = -1

#     for images, labels in dataloader:
#         batch_index += 1

#         plt.figure(figsize=(columns * img_size, nrows * img_size), dpi=300)

#         if show_label:
#             plt.subplots_adjust(hspace=0.6)

#         for i, (image, label) in enumerate(zip(images, labels)):
#             plt.subplot(nrows, columns, i + 1)
#             plt.axis("off")

#             if show_label:
#                 plt.title(label)

#             plt.imshow(image_mapper(image))

#         plt.show()

#         print(f"Visualized batch #{batch_index + 1}!")


def show_batches(
    dataloader,
    batch_size,
    image_mapper=lambda image: image.permute(1, 2, 0),
    show_label=True,
    batch_indices=lambda data_loader: [0, len(data_loader) - 2],
    columns=32,
    img_size=2
):
    nrows = (batch_size // columns)
    batch_index = -1

    for images, labels in dataloader:
        batch_index += 1

        plt.figure(figsize=(columns * img_size, nrows * img_size), dpi=300)

        if show_label:
            plt.subplots_adjust(hspace=0.6)

        for i, (image, label) in enumerate(zip(images, labels)):
            plt.subplot(nrows, columns, i + 1)
            plt.axis("off")

            if show_label:
                plt.title(label)

            plt.imshow(image_mapper(image))

        plt.show()

        print(f"Visualized batch #{batch_index + 1}!")