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

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

def train_model(model, train_loader, val_loader, num_epochs=10, lr=0.001, batch_size=32, save_interval=5):
    device = get_device()
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    training_losses = []
    validation_losses = []
    training_accuracies = []
    validation_accuracies = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        training_losses.append(running_loss / len(train_loader))
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
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
        print(f"Validation Loss: {val_loss/len(val_loader):.4f}")
        # Save the model
        if (epoch + 1) % save_interval == 0:
            torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")
            print(f"Model saved at epoch {epoch+1}")
        

def main():
    i_wdth = 256
    o_wdth = 16
    eqds = EquityDataset(input_width=i_wdth, output_width=o_wdth, include_industry_specific=True, normalize=False)
    train_loader, val_loader, test_loader = eqds.construct_data_loaders(industry="technology", sample_stride=1)
    model = EquityModel(width_k_bar=18, kernel_size=2, input_height=i_wdth, output_height=o_wdth, conv_out_channel_list=[4, 16, 64, 256, 32, 1])
    print(train_loader.dataset.data.shape)
    print(val_loader.dataset.data.shape)
    print(test_loader.dataset.data.shape)
    # tl, vl = train_model(model, train_loader, val_loader, num_epochs=10, lr=0.001, batch_size=32)

if __name__ == "__main__":
    main()