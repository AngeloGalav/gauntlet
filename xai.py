import torch
import torchvision.transforms as transforms
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import math
import numpy as np

from pytorch_grad_cam import AblationCAM, ScoreCAM, GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import os
import io

device = 'cuda'
save_results=True

def set_device(device_to_set):
    global device
    device = device_to_set

# LIME predict function, returns class probabilities
def batch_predict(images, model):
    global device
    # Convert list of numpy arrays to tensor
    batch = torch.stack([torch.tensor(i).permute(2, 0, 1) for i in images]).to(device)
    logits = model(batch)
    probs = torch.sigmoid(logits)
    return probs.detach().cpu().numpy()

def get_predicted_label(labels, predictions, idx):
    """
    Get label, prediction and title color for pretty print of results.
    """
    label = "REAL" if labels[idx].item() == 0 else "FAKE"
    predicted = "REAL" if predictions[idx].cpu().item() == 0 else "FAKE"
    title_color = "green" if labels[idx].item() == predictions[idx].cpu().item() else "red"

    return label, predicted, title_color

def explain_lime_single_image(dataloader, model, model_name=None, dataset_name=None, index=0):
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            # Select the first image in the batch

            images_dev  = images.to(device)
            pred = model(images_dev)
            probabilities = torch.sigmoid(pred)
            predictions = (probabilities > 0.5).float()
            image = images[index]

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
            
            probability = probabilities[index].item()  # Probability of being "fake"
            fake_prob = probability * 100
            real_prob = (1 - probability) * 100

            label, predicted, title_color = get_predicted_label(labels, predictions, index)

            fig = plt.figure(figsize=(15,5))
            numpy_image = (invert_normalization(image)).permute(1, 2, 0).cpu().numpy()
            # TEMPORARY FIX FOR LIME TITLE
            plt.tight_layout(pad=0.5)  # Adjust 'pad' to control overall spacing
            fig.subplots_adjust(top=0.85)  # Adjust top margin to move the title closer to the images

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
            ax.imshow(mark_boundaries(img, mask, color=(0,0,1)));
            ax.set_title("Image+Mask Combined");
            fig.suptitle(f"Labelled {label}, Predicted {predicted}\n"
                        f"Fake Probability: {fake_prob:.2f}%, Real Probability: {real_prob:.2f}%",
                        x=0.5, y=1.02, ha="center", fontsize=15, color=title_color)
            
            if model_name != None and dataset_name != None:
                if not os.path.exists(f'outputs/{model_name}/lime/{dataset_name}/'):
                    os.makedirs(f'outputs/{model_name}/lime/{dataset_name}/')

                plt.savefig(f"outputs/{model_name}/lime/{dataset_name}/lime_example_{index}.png", bbox_inches='tight')
            plt.show()

            break  # Only run for the first batch

def explain_gradcam_single_image(dataloader, model, target_layers, model_name=None, dataset_name=None, index=0, mapper="ac"):
    global device
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images_dev = images.to(device)

            # Get raw model output
            raw_output = model(images_dev)
            probabilities = torch.sigmoid(raw_output)

            predictions = (probabilities > 0.5).float()
            
            image = images[index]
            probability = probabilities[index].item()
            label, predicted, title_color = get_predicted_label(labels, predictions, index)
            
            # Setup for GradCAM
            image_mapper = get_gradcam_mapper(model, target_layers, mapper=mapper)
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))

            # Prepare original image for display
            image = invert_normalization(image)
            original_image = image.permute(1, 2, 0).numpy()
            ax[0].imshow(original_image)
            ax[0].set_title("Original Image")

            # Display GradCAM overlay
            ax[1].imshow(image_mapper(image))
            ax[1].set_title("GradCAM")

            # Show probability on the title
            fake_prob = probability * 100
            real_prob = (1 - probability) * 100
            fig.suptitle(f"Labelled: {label}, Predicted: {predicted}\n"
                         f"Fake Probability: {fake_prob:.2f}%, Real Probability: {real_prob:.2f}%",
                         x=0.5, y=1.02, ha="center", fontsize=15, color=title_color)

            # Save the figure if model and dataset names are provided
            if model_name and dataset_name:
                save_path = f'outputs/{model_name}/grad_cam/{dataset_name}/'
                os.makedirs(save_path, exist_ok=True)
                plt.savefig(f"{save_path}/grad_example_{index}.png", bbox_inches='tight')

            plt.show()
            break


def get_gradcam_mapper(model, target_layers, mapper="ac") :
    if (mapper == "ac") :
        # removes ports of the input to see which contributes more to the score
        am = AblationCAM(model=model, target_layers=target_layers)
    elif (mapper == "sc") :
        # perturbates input using the activation from each feature map and scores them
        # based on much they affect the target layer
        am = ScoreCAM(model=model, target_layers=target_layers)
    else :
        # gradients of the target class flowing into the targer layers (FASTEST)
        # am = GradCAM(model=model, target_layers=target_layers) # only available in training mode
        am = AblationCAM(model=model, target_layers=target_layers)

    def image_mapper(image):
        image = invert_normalization(image)
        image = image.to(device)
        image_np = image.permute(1, 2, 0).cpu().numpy().astype(np.float32)
        grayscale_am = am(
            input_tensor=image.unsqueeze(0),
            targets=[ClassifierOutputTarget(0)],
        )[0]

        return show_cam_on_image(
            image_np / np.max(image_np),
            grayscale_am,
            use_rgb=True,
        )

    return image_mapper


def explain_gradcam_batch(dataloader, batch_size, model, target_layers, show_label=True, columns=32, img_size=2, mapper="ac"):
    # N.B.: AblationCAM and ScoreCAM have batched implementations.
    # so it should be easy to transfer the batch representation to them.
    model.eval()

    image_mapper = get_gradcam_mapper(model, target_layers, mapper=mapper)

    columns = math.ceil(math.sqrt(batch_size))
    nrows = math.ceil(batch_size / columns)

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
                label_title = "REAL" if label == 0 else "FAKE"
                plt.title(label_title)

            plt.imshow(image_mapper(image))

        # TODO: create a suitable dir to save stuff
        if not os.path.exists(f'TEMP'):
            os.makedirs(f'TEMP')

        plt.savefig(f"TEMP/gradCam_batch_{batch_index}.png", bbox_inches='tight')
        plt.close()
        #plt.show()

        print(f"Visualized batch #{batch_index + 1}!")
        break # shows a single batch for now

def invert_normalization(tensor):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    inv_normalize = transforms.Normalize(
        mean=[-m/s for m, s in zip(mean, std)],
        std=[1/s for s in std]
    )
    tensor = inv_normalize(tensor)

    return tensor

# TODO: change it so that it is compatible with lime as well with a single function
def webapp_gradcam(image, model, target_layers, mapper="sc"):
    global device
    model.eval()
    with torch.no_grad():
        image_dev = image.to(device)
        # add batch dimension for single image compat
        numpy_image = image_dev.unsqueeze(0)
        raw_output = model(numpy_image)

        probabilities = torch.sigmoid(raw_output)
        probability = probabilities.item()

        image_mapper = get_gradcam_mapper(model, target_layers, mapper=mapper)
        fig, ax = plt.subplots(figsize=(6, 6))

        gradcam_image = image_mapper(image_dev)
        ax.imshow(gradcam_image)

        # pretty title stuff
        predicted = "AI GENERATED" if probability > 0.5 else "REAL"
        fake_prob = probability * 100
        real_prob = (1 - probability) * 100
        fig.suptitle(f"Predicted: {predicted}\n"
                     f"Fake Probability: {fake_prob:.2f}%, Real Probability: {real_prob:.2f}%",
                     x=0.5, y=0.95, ha="center", fontsize=12)
        plt.axis("off")
        
        # save image to buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        plt.close(fig)  # close the figure to avoid displaying it in Jupyter

        # move the buffer's position to the start
        buf.seek(0)

        return buf


# def show_batches(
#     dataloader,
#     batch_size,
#     image_mapper=lambda image: image.permute(1, 2, 0),
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