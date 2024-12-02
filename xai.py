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

BATCHSIZE_TO_COLUMNS = {1: 1,
                        2: 2,
                        4: 4,
                        8: 4,
                        16: 4,
                        32: 4,
                        64: 8,
                        128: 16
}


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
            pred_logits = model(images_dev)
            
            probabilities = torch.softmax(pred_logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            image = images[index]

            # place data in correct channels
            # invert_normalization removes warning
            numpy_image = invert_normalization(image).permute(1, 2, 0).cpu().numpy()


            # Initialize LIME Image Explainer
            explainer = lime_image.LimeImageExplainer()
            explanation = explainer.explain_instance(
                numpy_image,  # Image as numpy array
                lambda x: batch_predict(x, model),  # Prediction function
                num_samples=1000,
                random_seed=42  # Set a seed for reproducibility
            )

            # Get the image and mask for the top predicted class
            img, mask = explanation.get_image_and_mask(
                explanation.top_labels[0], # the top predicted class
                positive_only=False,
            )

            fake_probability = probabilities[index][1]  # Probability of being "fake"
            real_probability = probabilities[index][0]
            fake_prob = fake_probability * 100
            real_prob = real_probability * 100

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


def explain_lime_batch(dataloader, batch_size, model, model_name, dataset_name, batches_to_show = 1, show_label=True, columns=32, img_size=2):
    model.eval()

    columns = BATCHSIZE_TO_COLUMNS[batch_size]
    nrows = math.ceil(batch_size / columns)

    batch_index = 0

    if not os.path.exists(f'outputs/{model_name}/lime/{dataset_name}/batch_visualization'):
        os.makedirs(f'outputs/{model_name}/lime/{dataset_name}/batch_visualization')

    with torch.no_grad():
        for images, labels in dataloader:
            batch_index += 1

            images_dev  = images.to(device)
            pred_logits = model(images_dev)
            
            probabilities = torch.softmax(pred_logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            # place data in correct channels

            plt.figure(figsize=(columns * img_size, nrows * img_size), dpi=300)

            # loop for each image in the batch
            for i, (image, label) in enumerate(zip(images, labels)):
                # inverting normalization to remove warning
                image = invert_normalization(image)
                numpy_image = (image).permute(1, 2, 0).cpu().numpy()
                # Initialize LIME Image Explainer
                explainer = lime_image.LimeImageExplainer()
                explanation = explainer.explain_instance(
                    numpy_image,  # Image as numpy array
                    lambda x: batch_predict(x, model),  # Prediction function
                    hide_color=0,
                    num_samples=1000,
                    random_seed=42,  # Set a seed for reproducibility
                )

                # Get the image and mask for the top predicted class
                img, mask = explanation.get_image_and_mask(
                    explanation.top_labels[0],
                    positive_only=False,
                )
                mask_img = mark_boundaries(img, mask, color=(0,0,1))
                plt.subplot(nrows, columns, i + 1)
                plt.axis("off")
                if show_label:
                    _, predicted, title_color = get_predicted_label(labels, predictions, i)
                    plt.title(f"PRED: {predicted}", color=title_color)

                plt.imshow(mask_img)

            plt.savefig(f"outputs/{model_name}/lime/{dataset_name}/batch_visualization/lime_batch_{batch_index}.png", bbox_inches='tight')
            plt.close()

            print(f"Visualized batch #{batch_index}!")

                # stop visualization at the given batch
            if batches_to_show == batch_index:
                break


def explain_gradcam_single_image(dataloader, model, target_layers, model_name=None, dataset_name=None, index=0, mapper="ac"):
    global device
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images_dev = images.to(device)

            # Get raw model output
            pred_logits = model(images_dev)
            probabilities = torch.softmax(pred_logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

            image = images[index]
            fake_probability = probabilities[index][1] # prob of being fake
            real_probability = probabilities[index][0]
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
            fake_prob = fake_probability * 100
            real_prob = real_probability * 100
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
        # create grayscale activation map
        grayscale_am = am(
            input_tensor=image.unsqueeze(0),
            targets=None, # if it's none, the target its the highest scoring category
        )[0]

        return show_cam_on_image(
            image_np / np.max(image_np),
            grayscale_am,
            use_rgb=True,
        )

    return image_mapper


def explain_gradcam_batch(dataloader, batch_size, model, target_layers, model_name, dataset_name, batches_to_show = 1, show_label=True, columns=32, img_size=2, mapper="ac"):
    # N.B.: AblationCAM and ScoreCAM have batched implementations.
    # so it should be easy to transfer the batch representation to them.
    model.eval()

    image_mapper = get_gradcam_mapper(model, target_layers, mapper=mapper)

    columns = BATCHSIZE_TO_COLUMNS[batch_size]
    
    nrows = math.ceil(batch_size / columns)

    batch_index = 0

    for images, labels in dataloader:

        images_dev = images.to(device)
        # Get raw model output
        pred_logits = model(images_dev)
        probabilities = torch.softmax(pred_logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)


        batch_index += 1

        plt.figure(figsize=(columns * img_size, nrows * img_size), dpi=300)

        if show_label:
            plt.subplots_adjust(hspace=0.6)

        for i, (image, _) in enumerate(zip(images, labels)):
            plt.subplot(nrows, columns, i + 1)
            plt.axis("off")
            if show_label:
                _, predicted, title_color = get_predicted_label(labels, predictions, i)
                plt.title(f"PRED: {predicted}", color=title_color)

            plt.imshow(image_mapper(image))

        if not os.path.exists(f'outputs/{model_name}/grad_cam/{dataset_name}/batch_visualization/{mapper}'):
            os.makedirs(f'outputs/{model_name}/grad_cam/{dataset_name}/batch_visualization/{mapper}')

        plt.savefig(f"outputs/{model_name}/grad_cam/{dataset_name}/batch_visualization/{mapper}/gradCam_batch_{batch_index}.png", bbox_inches='tight')
        plt.close()

        print(f"Visualized batch #{batch_index}!")

        # stop visualization at the given batch
        if batches_to_show == batch_index:
            break

def invert_normalization(tensor):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    inv_normalize = transforms.Normalize(
        mean=[-m/s for m, s in zip(mean, std)],
        std=[1/s for s in std]
    )
    tensor = inv_normalize(tensor)

    return tensor

def webapp_lime(image, model):
    global device
    model.eval()
    with torch.no_grad():
        image_dev = image.to(device)
        # add batch dimension
        numpy_image = image_dev.unsqueeze(0)

        pred_logits = model(numpy_image)
        probabilities = torch.softmax(pred_logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)

        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            invert_normalization(image).permute(1, 2, 0).cpu().numpy(),
            lambda x: batch_predict(x, model),
            num_samples=1000,
            random_seed=42
        )

        # get the image and mask for the top predicted class
        img, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=False,
            hide_rest=False
        )

        positive_regions = [region for region, _ in explanation.local_exp[explanation.top_labels[0]]]
        negative_regions = [region for region, _ in explanation.local_exp[explanation.top_labels[1]]]

        original_image = invert_normalization(image).permute(1, 2, 0).cpu().numpy()
        colored_image = original_image.copy()

        # apply green for positive contributions and red for negative contributions
        alpha = 0.5  # transparency level for the mask
        for region in positive_regions:
            colored_image[mask == region] = colored_image[mask == region] * (1 - alpha) + alpha * np.array([0, 1, 0])  # Green
        for region in negative_regions:
            colored_image[mask == region] = colored_image[mask == region] * (1 - alpha) + alpha * np.array([1, 0, 0])  # Red

        # pretty print information for title
        predicted = "AI GENERATED" if predictions == 1 else "REAL"
        
        fake_probability = probabilities[0][1] # prob of being fake
        real_probability = probabilities[0][0]
        fake_prob = fake_probability * 100
        real_prob = real_probability * 100

        # plot only the image with the colored mask applied on the original background
        fig, ax = plt.subplots(figsize=(6, 6))

        # display the original image with the colored mask
        ax.imshow(colored_image)
        ax.set_title(f"Predicted: {predicted}\n"
                     f"AI Probability: {fake_prob:.2f}%, Real Probability: {real_prob:.2f}%",
                     fontsize=12)
        ax.axis("off")

        # save image to buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)  # close the figure to avoid display issues in Jupyter

        # move the buffer's position to the start
        buf.seek(0)

        return buf

def webapp_gradcam(image, model, target_layers, mapper="sc"):
    """
    Executes gradcam on a single image, and returns a buffer
    """
    global device
    model.eval()
    with torch.no_grad():
        image_dev = image.to(device)
        # add batch dimension for single image compat
        numpy_image = image_dev.unsqueeze(0)
        pred_logits = model(numpy_image)

        probabilities = torch.softmax(pred_logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)

        image_mapper = get_gradcam_mapper(model, target_layers, mapper=mapper)
        fig, ax = plt.subplots(figsize=(6, 6))

        gradcam_image = image_mapper(image_dev)
        ax.imshow(gradcam_image)

        fake_probability = probabilities[0][1] # prob of being fake
        real_probability = probabilities[0][0]

        # pretty title stuff
        predicted = "AI GENERATED" if predictions == 1 else "REAL"
        fake_prob = fake_probability * 100
        real_prob = real_probability * 100
        fig.suptitle(f"Predicted: {predicted}\n"
                     f"AI Probability: {fake_prob:.2f}%, Real Probability: {real_prob:.2f}%",
                     x=0.5, y=0.95, ha="center", fontsize=12)
        plt.axis("off")

        # save image to buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        plt.close(fig)  # close the figure to avoid displaying it in Jupyter

        # move the buffer's position to the start
        buf.seek(0)

        return buf