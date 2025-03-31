from model import EquityModel, MCEWithDirectionPenalty
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

eqds = None

def show_prediction(model, o_ht=1, ticker="AAPL"):
    # Do another quick test on the last sample window days of a stock
    data, _max, _min = eqds.get_recent_input_tensror_for_ticker(ticker)

    model.to(device)
    prediction = model(data[0].unsqueeze(0).to(device))

    # un-normalize the prediction and actual values
    input_tensor = data[0].cpu().detach().numpy()[:,:4]
    input_tensor = input_tensor * (_max - _min) + _min
    prediction = prediction.cpu().detach().numpy().squeeze(0)[:,:4]
    prediction = prediction * (_max - _min) + _min
    truth = data[1].cpu().detach().numpy()[:, :4]
    truth = truth * (_max - _min) + _min

    prediction = np.concatenate((input_tensor, prediction), axis=0)
    input_tensor = np.concatenate((input_tensor, truth), axis=0)

    # Show the prediction vs truth in 4 subplots corresponding to the 4 channels, with predicion as large dots
    _, axs = plt.subplots(1, 1, figsize=(10, 10))
    # axs[0, 0].plot(prediction[:,0], label='Prediction', color='red')
    # axs[0, 0].plot(input_tensor[:,0], label='Actual', color='indigo')
    # axs[0, 0].plot(input_tensor[:len(input_tensor)-1,0], label='Actual (Past)', color='blue')

    # axs[0, 1].plot(prediction[:,1], label='Prediction', color='red')
    # axs[0, 1].plot(input_tensor[:,1], label='Actual', color='indigo')
    # axs[0, 1].plot(input_tensor[:len(input_tensor)-1,1], label='Actual (Past)', color='blue')

    # axs[1, 0].plot(prediction[:,2], label='Prediction', color='red')
    # axs[1, 0].plot(input_tensor[:,2], label='Actual', color='indigo')
    # axs[1, 0].plot(input_tensor[:len(input_tensor)-1,2], label='Actual (Past)', color='blue')

    axs.plot(prediction[:,3], label='Prediction', color='red')
    axs.plot(input_tensor[:,3], label='Actual', color='fuchsia')
    axs.plot(input_tensor[:len(input_tensor)-o_ht,3], label='Actual (Past)', color='blue')

    # axs[0, 0].set_title('1st Channel')
    # axs[0, 0].set_xlabel('Time')
    # axs[0, 0].set_ylabel('Price')
    
    # axs[0, 1].set_title('2nd Channel')
    # axs[0, 1].set_xlabel('Time')
    # axs[0, 1].set_ylabel('Price')
    
    # axs[1, 0].set_title('3rd Channel')
    # axs[1, 0].set_xlabel('Time')
    # axs[1, 0].set_ylabel('Price')
    
    axs.set_title(f'{ticker} Price Prediction')
    axs.set_xlabel('Time')
    axs.set_ylabel('Price')
    
    # axs[0, 0].grid()
    # axs[0, 1].grid()
    # axs[1, 0].grid()
    axs.grid()

    # axs[0, 0].legend()
    # axs[0, 1].legend()
    # axs[1, 0].legend()
    axs.legend()
    plt.show()

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
        print("Using Apple Silicon GPU")
        return "mps"
    else:
        return "cpu"
device = get_device()


def avishit_model_accuracy(model, data_loader, output_height=1):
    return 0
    
def get_model_accuracy(model, data_loader, output_height=1):
    model.to(device)
    total_return = 0.0
    total_trades = 0
    correct_trades = 0
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        with torch.no_grad():
            outputs = model(inputs)

        for i, output in enumerate(outputs):
            for day in range(output_height):
                todays_price = inputs[i][-1, 3].squeeze(0)
                tomorrows_price = output[day, 3].squeeze(0)
                if day != 0:
                    todays_price = targets[i][day-1, 3].squeeze(0)
                true_price = targets[i][day, 3].squeeze(0)
                # average price of the first 4 channels
                
                if tomorrows_price > todays_price and true_price > todays_price:
                    correct_trades += 1
                elif tomorrows_price < todays_price and true_price < todays_price:
                    correct_trades += 1
                elif tomorrows_price == true_price:
                    correct_trades += 1
                elif true_price == todays_price:
                    correct_trades += 1
                
                total_trades += 1
        if total_trades > 10000: 
            break
                
    return (correct_trades / total_trades)

def train_model(model, train_loader, val_loader, num_epochs=10, lr=0.0001, output_height=1, _pf=0.1):
    model.to(device)
    # criterion = nn.MSELoss()
    criterion = MCEWithDirectionPenalty(penalty_factor=_pf)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    training_losses = []
    validation_losses = []
    validation_accuracies = []
    training_accuracies = []

    best_val_loss = float('inf')
    best_val_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            # Only do loss on the first 4 channels (price, volume, etc.)
            # loss = criterion(outputs[:, :, 3], targets[:, :, 3])
            loss = criterion(outputs, targets, inputs)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        training_losses.append(running_loss / len(train_loader))
        training_accuracies.append(get_model_accuracy(model, train_loader, output_height=output_height))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                # loss = criterion(outputs[:, :, 3], targets[:, :, 3])
                loss = criterion(outputs, targets, inputs)
                val_loss += loss.item()
        validation_losses.append(val_loss / len(val_loader))
        validation_accuracies.append(get_model_accuracy(model, val_loader, output_height=output_height))

        if validation_losses[-1] < best_val_loss:
            best_val_loss = validation_losses[-1]
            # Save the model if validation loss improves
            if not os.path.exists("./cached_models"):
                os.makedirs("./cached_models")
            torch.save(model.state_dict(), "./cached_models/best_loss_equitymodel.pth")
        if validation_accuracies[-1] > best_val_accuracy:
            best_val_accuracy = validation_accuracies[-1]
            # Save the model if validation accuracy improves
            if not os.path.exists("./cached_models"):
                os.makedirs("./cached_models")
            torch.save(model.state_dict(), "./cached_models/best_accuracy_equitymodel.pth")
        
        # Print statistics
        print(f"Epoch [{epoch+1}] | Training Loss: {training_losses[-1]:.4f} | Validation Loss: {validation_losses[-1]:.4f} | Validation Accuracy: {validation_accuracies[-1]:.4f} | Training Accuracy: {training_accuracies[-1]:.4f}")

    return training_losses, validation_losses, validation_accuracies, training_accuracies
        

def main(train=True, tickers=None):
    # note that the input height muse be 256 times the output height
    torch.manual_seed(42)
    np.random.seed(42)
    global eqds
    # convolutional_layers = [4, 16, 64, 256, 32, 16, 8, 4] # this works well

    # if the below are changed the cached data loaders must be cleared manually rn
    convolutional_layers = [4, 8, 32, 128, 256, 64, 16, 4]
    o_ht = 1
    i_ht = o_ht * (2**len(convolutional_layers))
    # below this should be ok

    eqds = EquityDataset(input_height=i_ht, output_height=o_ht, include_industry_specific=True, normalize=False)
    train_loader, val_loader, test_loader = eqds.construct_data_loaders(industry="technology", sample_stride=4, batch_size=32)
    model = EquityModel(width_k_bar=18, kernel_size=5, input_height=i_ht, output_height=o_ht, conv_out_channel_list=convolutional_layers, pool_type='avg')
    
    # print(train_loader.dataset[0][0].shape)
    # show_tensor_image(train_loader.dataset[0][0], title="First Image in Training Set")
    # print(val_loader.dataset[0][0].shape)
    # show_tensor_image(val_loader.dataset[0][0], title="First Image in Validation Set")
    # print(test_loader.dataset[0][0].shape)
    # show_tensor_image(test_loader.dataset[0][0], title="First Image in Test Set")
    # print("Training set size:", len(train_loader.dataset))
    # print("Validation set size:", len(val_loader.dataset))
    # print("Test set size:", len(test_loader.dataset))

    if train:
        tl, vl, va, ta = train_model(model, train_loader, val_loader, num_epochs=10, lr=0.0001, output_height=o_ht, _pf=0.075)
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
        # Plot training and validation accuracy
        plt.figure(figsize=(10, 5))
        plt.plot(va, label='Validation Accuracy', color='red')
        plt.plot(ta, label='Training Accuracy', color='blue')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.grid()
        plt.show()

    best_model = torch.load("./cached_models/best_loss_equitymodel.pth")
    model.load_state_dict(best_model)

    for ticker in tickers:
        print(f"Testing {ticker}...")
        show_prediction(model, o_ht, ticker)
        # Uncomment below to get accuracy on test set
        # test_accuracy = get_model_accuracy(model, test_loader, output_height=o_ht)
        # print(f"Test Accuracy: {100*test_accuracy}%")

    # test_accuracy = get_model_accuracy(model, test_loader, output_height=o_ht)
    # print(f"Test Accuracy: {100*test_accuracy}%")

    

if __name__ == "__main__":
    tickers_to_test = [
        "AAPL", "ADBE", "ADI", "AMAT", "AMD", "AMZN", "AVGO", "AXP", 
        "BAC", "BLK", "BX", "C", "CB", "CRM", "CSCO", "GOOG", "GS", 
        "HDB", "HSBC", "INTU", "JPM", "KKR", "META", "MMC", "MS", 
        "MSFT", "MU", "MUFG", "NOW", "NVDA", "ORCL", "PGR", "PLD", 
        "RY", "SCHW", "SMFG", "TD", "TSLA", "TXN", "UBS", "VRN.TO", "WFC"
    ]
    main(False, tickers_to_test)