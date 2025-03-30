from model import EquityModel
from data_loader import EquityDataset

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import os
import pandas as pd
from datetime import datetime

import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.utils import make_grid

def show_tensor_image(tensor, title=None):
    """
    Show a single image tensor.
    """
    # Convert tensor to numpy array
    img = tensor.cpu().numpy()

    # Normalize to [0, 1]
    # img = (img - img.min()) / (img.max() - img.min())
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.show()

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
    
def get_model_accuracy(model, data_loader):
    device = get_device()
    model.to(device)

    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        # Assuming the model outputs are probabilities
        predicted = torch.argmax(outputs, dim=1)
        correct = (predicted == targets).sum().item()
        accuracy = correct / len(targets)

def train_model(model, train_loader, val_loader, num_epochs=10, lr=0.0001, save_interval=10):
    device = get_device()
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    training_losses = []
    validation_losses = []
    training_accuracies = []
    validation_accuracies = []

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            # Only do loss on the first 4 channels (price, volume, etc.)
            loss = criterion(outputs[:, :, :4], targets[:, :, :4])
            # loss = criterion(outputs[:, :, 3], targets[:, :, 3])

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        training_losses.append(running_loss / len(train_loader))
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        validation_losses.append(val_loss / len(val_loader))

        if validation_losses[-1] < best_val_loss:
            best_val_loss = validation_losses[-1]
            # Save the model if validation loss improves
            if not os.path.exists("./cached_models"):
                os.makedirs("./cached_models")
            torch.save(model.state_dict(), "./cached_models/best_equitymodel.pth")
            print(f"Model saved at epoch {epoch+1} with validation loss: {best_val_loss:.4f}")
        
        # Print statistics
        print(f"Epoch {epoch+1} | Training Loss: {training_losses[-1]:.4f} | Validation Loss: {validation_losses[-1]:.4f}")

    return training_losses, validation_losses, training_accuracies, validation_accuracies
        

def main():
    # note that the input height muse be 256 times the output height

    convolutional_layers = [4, 16, 64, 256, 32, 16, 4, 2]
    o_ht = 1
    i_ht = o_ht * (2**len(convolutional_layers))
    eqds = EquityDataset(input_height=i_ht, output_height=o_ht, include_industry_specific=True, normalize=False)
    train_loader, val_loader, test_loader = eqds.construct_data_loaders(industry="technology", sample_stride=4, batch_size=32)
    model = EquityModel(width_k_bar=17, kernel_size=3, input_height=i_ht, output_height=o_ht, conv_out_channel_list=convolutional_layers, pool_type='avg')
    
    # print(train_loader.dataset[0][0].shape)
    # show_tensor_image(train_loader.dataset[0][0], title="First Image in Training Set")
    # print(val_loader.dataset[0][0].shape)
    # show_tensor_image(val_loader.dataset[0][0], title="First Image in Validation Set")
    # print(test_loader.dataset[0][0].shape)
    # show_tensor_image(test_loader.dataset[0][0], title="First Image in Test Set")
    # print("Training set size:", len(train_loader.dataset))
    # print("Validation set size:", len(val_loader.dataset))
    # print("Test set size:", len(test_loader.dataset))

    tl, vl, ta, va = train_model(model, train_loader, val_loader, num_epochs=30, lr=0.0001)
    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(tl, label='Training Loss', color='blue')
    plt.plot(vl, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid()
    plt.show()



if __name__ == "__main__":
    main()