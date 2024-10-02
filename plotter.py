# Plotter functions/wrappers

import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import random

def display_result_plot():
    ...

def plot_losses(losses):
    plt.plot(losses)
    plt.show()

# stupid function but is more readable
def display_image(image):
    plt.imshow(image);
