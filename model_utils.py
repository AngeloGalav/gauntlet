# In this file we define the models used here in the project
# as well as other utils used (i.e. training function, set_seed etc)
# in order to keep the notebook slim

import torch
import torchvision.models as tmodels
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

def get_device():
    '''Sets device for operation
    '''
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    return device

def fix_random(seed: int) -> None:
    """Fix all the possible sources of randomness.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# TODO: test this model
class CNN(nn.Module):
    def __init__(self, num_classes=2):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        # reduce computational complexity
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # fully connected layers (after flattening)
        self.fc1 = nn.Linear(128 * 32 * 32 // 8, 512)  # Adjust based on input size
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Connecting feature extractor
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # falttening
        x = x.view(x.size(0), -1)  # Flattening the tensor

        # fc
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

def prepare_for_ft(model, num_classes=2) -> None:
    # we can change it later if we find better models
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)


# modified resnet with more fc
class ModifiedResNet(nn.Module):
    def __init__(self, original_resnet, use_dropout=False, num_classes=1, hidden_size1=1024, hidden_size2=512):
        super(ModifiedResNet, self).__init__()

        # Use the original ResNet up to the global average pooling layer
        self.resnet = nn.Sequential(*list(original_resnet.children())[:-1])

        in_features = original_resnet.fc.in_features
        # additional linear layers
        self.fc1 = nn.Linear(in_features, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, num_classes)

        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout1 = nn.Dropout(0.5)  # First dropout layer
            self.dropout2 = nn.Dropout(0.5)  # Second dropout laye

    def prepare_for_ft(self):
        for param in self.resnet.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.resnet(x)  # Pass through the ResNet layers
        x = torch.flatten(x, 1)  # Flatten the output

        x = torch.relu(self.fc1(x))
        if self.use_dropout:      # apply dropout if enabled
            x = self.dropout1(x)

        x = torch.relu(self.fc2(x))
        if self.use_dropout:
            x = self.dropout2(x)
        x = self.fc3(x)

        return x


def train(dataloaders, loss_fn, optimizer, model, model_name, batch_size, epochs, loss_thresh=2.5, force_train=True):
    '''
    Trains model on multiple epochs
    '''
    import os
    device = get_device()
    train_dataloader, val_dataloader = dataloaders
    weight_filename = f"models/best_{model_name}.pth"
    losses = []
    train_accs = []
    val_accs = []
    precisions = []
    recalls = []
    
    if not os.path.exists('models'):
        os.makedirs('models')

    # trains only if filename exists
    if not os.path.isfile(weight_filename) or force_train :
        best_loss = float('inf')
        for t in range(epochs):
            print(f"Epoch {t+1}/{epochs}\n-------------------------------")

            #train step
            train_accuracy = train_loop(train_dataloader, model, loss_fn, optimizer, batch_size, device)
            train_accs.append(train_accuracy)

            #eval step
            curr_loss, val_accuracy, prec, rec = test(val_dataloader, model, loss_fn, device, validation=True)

            precisions.append(prec)
            recalls.append(rec)

            if curr_loss < best_loss:
                best_loss = curr_loss
                print("New best model found! (based on lowest loss)")
                if best_loss < loss_thresh:
                    torch.save(model.state_dict(), weight_filename)
                    print('...and saved.')

            losses.append(curr_loss)
            val_accs.append(val_accuracy)
            print("\n")

    model.load_state_dict(torch.load(weight_filename))
    if losses != []:
        return losses, train_accs, val_accs, precisions, recalls


def train_loop(dataloader, model, loss_fn, optimizer, batch_size, device):
    '''
    Does a single train epoch step (i.e. perfoming backprop on all samples).
    '''

    model.train()
    size = len(dataloader.dataset)
    correct = 0

    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        # Compute prediction and loss (fw)
        pred = model(X)
        y = y.unsqueeze(1)
        loss = loss_fn(pred, y.float())
        # Zero the gradients (to prevent gradient accomulation)
        optimizer.zero_grad()

        # Backpropagation (bw)
        loss.backward()
        optimizer.step()

        probabilities = torch.sigmoid(pred)
        predictions = (probabilities > 0.5).float()

        correct += (predictions == y).float().sum()

        # acc = correct/size
        acc = (correct/((batch+1) * batch_size))
        if (batch + 1) % 40 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"Training loss: {loss:>7f}, train accuracy: {(100*acc):>0.2f}%  [{current:>5d}/{size:>5d}]")

    return acc

def confusion_matrix_computation(predictions, ground_truth):
    '''
    Computes the confusion matrix for the given predictions.

    Args:
        - predictions: array of predictions of the network (either 0 or 1).
        - ground_truth: array with the ground truths (either 0 or 1).

    Returns:
        - cm: the confusion matrix, which contains:
            - tp: true positives (pred == 1 & gt == 1).
            - fp: false positives (pred == 1 & gt == 0).
            - fn: false negatives (pred == 0 & gt == 1).
            - tn: true negatives (pred == 0 & gt == 0).
    '''
    assert len(predictions) == len(ground_truth), "PREDICTIONS AND GROUND TRUTHS SHOULD HAVE THE SAME SIZE!"

    #save cm for later visualization
    cm = confusion_matrix(ground_truth, predictions)

    return cm

def metrics_computation(tp, fp, fn):
    '''
    Computes precision, recall and f1.
    '''
    precision = tp / (tp + fp) if (tp+fp) != 0 else 0
    recall = tp / (tp + fn) if (tp+fn) != 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return precision, recall, f1_score

def test(dataloader, model, loss_fn, device, validation:bool=False, model_name:str='',visualize:bool=False):
    '''
    Computes loss on the test/val set (without backprop) on all samples.

    Returns the losses list.
    '''
    # Set the model to evaluation mode
    model.eval()
    size = len(dataloader.dataset)
    # num_batches = len(dataloader)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    total_predictions = []
    total_gt = []

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            logits = model(X)
            y = y.unsqueeze(1)
            test_loss += loss_fn(logits, y.float()).item()

            probabilities = torch.sigmoid(logits)
            predictions = (probabilities > 0.5).float()
            correct += (predictions == y).float().sum()

            total_predictions.extend(predictions.detach().cpu().numpy())
            total_gt.extend(y.detach().cpu().numpy())

    test_loss /= num_batches
    acc = correct / size
    print(f"{'Validation' if validation else 'Test'} Error:\nAccuracy: {(100*acc):>0.1f}%, Avg loss: {test_loss:>8f}")
    
    if not validation:
        cm = confusion_matrix_computation(total_predictions, total_gt)
        tp, fp, fn, tn = cm.ravel()
        print(f"Confusion matrix report, tp: {tp}, fp: {fp}, fn: {fn}, tn:{tn}")

        if visualize:
            #create dir if not created yet
            if not os.path.exists(f'{model_name}'):
                os.makedirs(f'{model_name}')

            disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=[0, 1])
            disp.plot().figure_.savefig(f'{model_name}/confusion_matrix.png')
            plt.show()

        prec, rec, f1 = metrics_computation(tp, fp, fn)
        print(f"Precision: {prec:>0.2f}, Recall: {rec:>0.2f}, F1-Score: {f1:>0.2f}")
    else:
        tp, fp, fn, _ = confusion_matrix_computation(total_predictions, total_gt).ravel()
        prec, rec, _ = metrics_computation(tp, fp, fn)
        return test_loss, acc, prec, rec

    return test_loss, acc

def test_single_image(model, dataloader, index, device, plt) :
    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            image = X[index]
            label = y[index]

            print(f"this is the image and has label {label}")
            plt.imshow(image.permute(1, 2, 0))

            image = image.to(device)
            logits = model(image.unsqueeze(0))
            logits = logits.squeeze()
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities > 0.5).float()
            print("it was predicted as :", predictions)
            break
