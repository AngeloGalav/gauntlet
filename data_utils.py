# data preprocessing stuff here

from pathlib import Path
from PIL import Image, ImageOps
from torch import Tensor
from torchvision.transforms.functional import to_tensor
from torch.utils.data import Dataset, random_split
from typing import List, Tuple
from torch import nn
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
from torch.utils.data import DataLoader
import torchvision.transforms as transforms



# TODO: make this more generic once we have the other dataset
class CIFAKEDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            split (string): 'train' or 'test' to select the dataset split.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        # Create the list of image paths and their corresponding labels
        self.image_paths, self.labels = self.read_file()

    def read_file(self) -> Tuple[List[str], List[int]]:
        image_paths = []
        labels = []

        fake_dir = os.path.join(self.root_dir, self.split, "FAKE")
        real_dir = os.path.join(self.root_dir, self.split, "REAL")

        # Load REAL images
        for file_name in os.listdir(real_dir):
            if file_name.endswith(('.png', '.jpg', '.jpeg')):  # Add supported image extensions
                image_paths.append(os.path.join(real_dir, file_name))
                labels.append(0)  # Label for REAL images

        # Load FAKE images
        for file_name in os.listdir(fake_dir):
            if file_name.endswith(('.png', '.jpg', '.jpeg')):  # Add supported image extensions
                image_paths.append(os.path.join(fake_dir, file_name))
                labels.append(1)  # Label for FAKE images

        return image_paths, labels

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx)  -> Tuple[Tensor, int] :
        """Returns an image and its label."""
        img_path = self.image_paths[idx]
        image = Image.open(img_path)

        label = self.labels[idx]

        # apply image transformations (if specified)
        if self.transform:
            image = self.transform(image)

        return to_tensor(image), label


class RVAADataset(Dataset):
    def __init__(self, root_dir, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform

        # Create the list of image paths and their corresponding labels
        self.image_paths, self.labels = self.read_file()

    def read_file(self) -> Tuple[List[str], List[int]]:
        image_paths = []
        labels = []

        fake_dir = os.path.join(self.root_dir, "AiArtData", "AiArtData")
        real_dir = os.path.join(self.root_dir, "RealArt", "RealArt")

        # Load REAL images
        for file_name in os.listdir(real_dir):
            if file_name.endswith(('.png', '.jpg', '.jpeg')):  # Add supported image extensions
                image_paths.append(os.path.join(real_dir, file_name))
                labels.append(0)  # Label for REAL images

        # Load FAKE images
        for file_name in os.listdir(fake_dir):
            if file_name.endswith(('.png', '.jpg', '.jpeg')):  # Add supported image extensions
                image_paths.append(os.path.join(fake_dir, file_name))
                labels.append(1)  # Label for FAKE images

        return image_paths, labels

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx) :
        """Returns an image and its label."""
        img_path = self.image_paths[idx]
        image = Image.open(img_path)

        label = self.labels[idx]

        # apply image transformations (if specified)
        if self.transform:
            image = self.transform(image)

        return image, label


def get_dataset_stats(image_array):
    sizes = []
    for el in image_array:
        sizes.append(el[0].size)
    widths, heights = zip(*sizes)

    stats = {
        'min_width': np.min(widths),
        'max_width': np.max(widths),
        'mean_width': np.mean(widths),
        'std_width': np.std(widths),
        'min_height': np.min(heights),
        'max_height': np.max(heights),
        'mean_height': np.mean(heights),
        'std_height': np.std(heights),
    }
    return stats

# Function to pad the image to a square (only when needed)
def pad_to_square(image):
    width, height = image.size
    max_dim = max(width, height)

    # Calculate padding (add padding equally to both sides)
    padding_left = (max_dim - width) // 2
    padding_right = max_dim - width - padding_left
    padding_top = (max_dim - height) // 2
    padding_bottom = max_dim - height - padding_top

    # Pad the image
    padding = (padding_left, padding_top, padding_right, padding_bottom)
    return transforms.functional.pad(image, padding, fill=0)  # Add padding with black (fill=0)

# Does train/val/test split
def train_test_split(split_ratio, dataset):
    train_ratio, val_ratio = split_ratio
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    return train_dataset, val_dataset, test_dataset
