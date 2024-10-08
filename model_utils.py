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

def prepare_for_ft(model, num_classes=2) -> None:
    # we can change it later if we find better models
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)



def train(dataloaders, loss_fn, optimizer, model, model_name, batch_size, epochs, loss_thresh=2.5, force_train=True):
    '''
    Trains model on multiple epochs
    '''
    import os
    device = get_device()
    train_dataloader, val_dataloader = dataloaders
    weight_filename = f"best_{model_name}.pth"
    losses = []
    train_accs = []
    test_accs = []

    # trains only if filename exists
    if not os.path.isfile(weight_filename) or force_train :
        best_loss = float('inf')
        for t in range(epochs):
            print(f"Epoch {t+1}/{epochs}\n-------------------------------")

            #train step
            train_accuracy = train_loop(train_dataloader, model, loss_fn, optimizer, batch_size, device)
            train_accs.append(train_accuracy)

            #eval step
            curr_loss, test_accuracy = test(val_dataloader, model, loss_fn, device, validation=True)

            if curr_loss < best_loss:
                best_loss = curr_loss
                print('new best model found')
                if best_loss < loss_thresh:
                    torch.save(model.state_dict(), weight_filename)
                    print('best model saved')

            losses.append(curr_loss)
            test_accs.append(test_accuracy)

    model.load_state_dict(torch.load(weight_filename))
    if losses != []:
        return losses, train_accs, test_accs


def train_loop(dataloader, model, loss_fn, optimizer, batch_size, device):
    '''
    Does a single train epoch step (i.e. perfoming backprop on all samples).
    '''
    size = len(dataloader.dataset)
    model.train()
    correct = 0

    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        # Compute prediction and loss (fw)
        pred = model(X)
        pred = pred.squeeze()
        loss = loss_fn(pred, y.float())
        # Zero the gradients (to prevent gradient accomulation)
        optimizer.zero_grad()

        # Backpropagation (bw)
        loss.backward()
        optimizer.step()

        # correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        probabilities = torch.sigmoid(pred)
        predictions = (probabilities > 0.5).float()
        correct += (predictions == y).float().sum()
        acc = correct/size

        if (batch + 1) % 40 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"training loss: {loss:>7f}, train accuracy: {(100*acc):>0.2f}  [{current:>5d}/{size:>5d}]")
    return acc


def test(dataloader, model, loss_fn, device, validation:bool=False):
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

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            logits = model(X)
            logits = logits.squeeze()
            test_loss += loss_fn(logits, y.float()).item()
            # correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities > 0.5).float()
            correct += (predictions == y).float().sum()

    test_loss /= num_batches
    acc = correct / size

    print(f"{'Validation' if validation else 'Test'} Error:\nAccuracy: {(100*acc):>0.1f}%, Avg loss: {test_loss:>8f} \n")
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
