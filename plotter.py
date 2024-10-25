# Plotter functions/wrappers

import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import random
import os

def display_result_plot():
    ...

def plot_losses(losses):
    plt.plot(losses)
    plt.show()

# stupid function but is more readable
def display_image(image):
    plt.imshow(image);

def plot_metrics(values, metric: str, model_name):
    if not os.path.exists(f'{model_name}'):
        os.makedirs(f'{model_name}')

    train_val, test_val = values
    train_accs_cpu = [t.cpu() for t in train_val if t.is_cuda]
    test_accs_cpu = [t.cpu() for t in test_val if t.is_cuda]
    plt.plot(train_accs_cpu, label = metric)
    plt.plot(test_accs_cpu, label='val_' + metric)
    plt.xlabel('Epochs')
    plt.ylabel(metric)
    plt.legend()
    plt.savefig(f'{model_name}/accuracies.png')
    plt.show()

def plot_precision_recall_curve(precisions, recalls, model_name):
    '''
    Plots precision-recall curve, and also precision curve and recall curve
    '''
    
    if not os.path.exists(f'{model_name}'):
        os.makedirs(f'{model_name}')

    assert len(precisions) == len(recalls), "PRECISIONS AND RECALLS MUST HAVE THE SAME SIZE!"
    epochs = [i+1 for i in range(len(precisions))]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, precisions, label='Precision', color='blue', marker='o')
    plt.plot(epochs, recalls, label='Recall', color='orange', marker='o')
    plt.title('Precision and Recall over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.xticks(epochs)  # Show each epoch on the x-axis
    plt.legend(loc='best')
    plt.savefig(f'{model_name}/pr_over_epochs.png', format='png')
    plt.show()

    prec_rec_map = {}
    for i in range(len(precisions)):
        prec_rec_map[recalls[i]] = precisions[i]

    recalls.sort()
    sorted_precisions = []
    for i in range(len(recalls)):
        sorted_precisions.append(prec_rec_map[recalls[i]])

    plt.plot(recalls, sorted_precisions, marker='o', color='blue')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig(f'{model_name}/pr_curve.png', format='png')
    plt.show()