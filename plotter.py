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

def plot_metrics(values, metric: str):
    train_val, test_val = values
    train_accs_cpu = [t.cpu() for t in train_val if t.is_cuda]
    test_accs_cpu = [t.cpu() for t in test_val if t.is_cuda]
    plt.plot(train_accs_cpu, label = metric)
    plt.plot(test_accs_cpu, label='val_' + metric)
    plt.xlabel('Epochs')
    plt.ylabel(metric)
    plt.legend()
    plt.show()