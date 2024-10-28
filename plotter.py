# Plotter functions/wrappers

import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import random
import os

save_plots = True

# TODO: delete this function if it not used anywhere
def plot_losses(losses):
    plt.plot(losses)
    plt.show()

# TODO: test for loss as well
def plot_generic_metrics(values, metric: str, model_name=None):
    """
    plot function used to compare the metrics between the training and
    testing phase. It supports both loss and accuracy metrics.
    """
    global save_plots

    train_metrics, val_metrics = values
    train_metrics_cpu = [t.cpu() for t in train_metrics if t.is_cuda]
    val_metrics_cpu = [t.cpu() for t in val_metrics if t.is_cuda]
    plt.plot(train_metrics_cpu, marker='o', label=f"train_{metric}")
    plt.plot(val_metrics_cpu, marker='o', label=f"val_{metric}")
    plt.xlabel('Epochs')
    plt.ylabel(metric)
    plt.legend()
    if save_plots and model_name is not None:
        if not os.path.exists(f'outputs/{model_name}'):
            os.makedirs(f'outputs/{model_name}')
        plt.savefig(f'outputs/{model_name}/{metric}.png')
    plt.show()

def plot_precision_recall_curve(precisions, recalls, model_name=None):
    '''
    Plots precision-recall curve, and also precision curve and recall curve
    '''
    # precision curve and recall curve over epochs
    global save_plots

    assert len(precisions) == len(recalls), "PRECISIONS AND RECALLS MUST HAVE THE SAME SIZE!"
    #epochs = [i+1 for i in range(len(precisions))]

    plt.figure(figsize=(10, 6))
    plt.plot(precisions, label='Precision', marker='o')
    plt.plot(recalls, label='Recall', marker='o')
    plt.title('Precision and Recall over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.legend(loc='best')

    if save_plots and model_name is not None:
        if not os.path.exists(f'outputs/{model_name}'):
            os.makedirs(f'outputs/{model_name}')
        plt.savefig(f'outputs/{model_name}/pr_over_epochs.png', format='png')

    plt.show()

    # display (and save) precision/recall curve
    prec_rec_map = {}
    for i in range(len(precisions)):
        prec_rec_map[recalls[i]] = precisions[i]

    recalls.sort()
    sorted_precisions = []
    for i in range(len(recalls)):
        sorted_precisions.append(prec_rec_map[recalls[i]])

    plt.plot(recalls, sorted_precisions, marker='o', color='green')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')

    if save_plots and model_name is not None:
        if not os.path.exists(f'outputs/{model_name}'):
            os.makedirs(f'outputs/{model_name}')
        plt.savefig(f'outputs/{model_name}/pr_curve.png', format='png')

    plt.show()