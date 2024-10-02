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

class FTModel():
    def __init__(self, model, num_classes=2) -> None:
        # we can change it later if we find better models
        self.model = model

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)


def train(dataloaders, loss_fn, optimizer, model, model_name, batch_size, epochs, loss_thresh=2.5, force_train=True):
    '''
    Trains model on multiple epochs
    '''
    import os
    device = get_device()
    train_dataloader, val_dataloader = dataloaders
    weight_filename = f"best_{model_name}.pth"

    # trains only if filename exists
    if not os.path.isfile(weight_filename) or force_train :
        best_loss = float('inf')
        losses = []
        for t in range(epochs):
            print(f"Epoch {t+1}/{epochs}\n-------------------------------")
            #train step
            train_loop(train_dataloader, model, loss_fn, optimizer, batch_size, device)
            #eval step
            curr_loss = test(val_dataloader, model, loss_fn, device, validation=True)
            if curr_loss < best_loss:
                best_loss = curr_loss
                print('new best model found')
                if best_loss < loss_thresh:
                    torch.save(model, weight_filename)
                    print('best model saved')
            losses.append(curr_loss)

    model=torch.load(weight_filename)


def train_loop(dataloader, model, loss_fn, optimizer, batch_size, device):
    '''
    Does a single train epoch step (i.e. perfoming backprop on all samples).
    '''
    size = len(dataloader.dataset)
    # Set the model to training mode
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        # Compute prediction and loss (fw)
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation (bw)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (batch + 1) % 40 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn, device, validation:bool=False):
    '''
    Computes loss on the test/val set (without backprop) on all samples.

    Returns the losses list.
    '''
    # Set the model to evaluation mode
    model.eval()
    size = len(dataloader)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = np.array(y).to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    print(f" {'Validation' if validation else 'Test'} Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss

