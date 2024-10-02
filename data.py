# data preprocessing stuff here

from pathlib import Path
from PIL import Image, ImageOps
from torch import Tensor
from torchvision.transforms.functional import to_tensor
from torch.utils.data import Dataset
from typing import List, Tuple
from torch import nn
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
from torch.utils.data import DataLoader


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
