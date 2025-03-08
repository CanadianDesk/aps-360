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

def train_model(model, train_loader, val_loader, num_epochs=10, lr=0.001, batch_size=32):
    device = get_device()
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

