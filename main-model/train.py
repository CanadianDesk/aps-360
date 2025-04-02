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
import copy

import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.utils import make_grid

eqds = None

def renormalize_tensor(tensor):
    """
    Renormalize a tensor to the range [0, 1].
    """
    min_val = tensor.min()
    max_val = tensor.max()
    return (tensor - min_val) / (max_val - min_val)

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


# def complex_model_accuracy(model, data_loader, num_days=7, output_height=1):
#     # Simulates the performance of the model if it were to be used to
#     # trade a portfolio of the stocks in data_loader over the next num_days

#     MAX_NUM_DAYS = 30

#     if num_days > MAX_NUM_DAYS:
#         print(f"num_days cannot be greater than {MAX_NUM_DAYS}. Setting num_days to {MAX_NUM_DAYS}")
#         num_days = MAX_NUM_DAYS

#     model.to(device)
#     num_stocks = len(data_loader.dataset) * 32
#     print(f"DEBUG: num_stocks: {num_stocks}")

#     copy_data_loader = copy.deepcopy(data_loader)
#     portfolio = [1 / num_stocks] * num_stocks
#     print(f"DEBUG: This should be 1: {sum(portfolio)}")
#     sold_stocks = [0] * num_stocks

#     # cash_in_hand = 0

#     for day in range(num_days): # loop over num_days
#         print(f"Simulating day {day+1}/{num_days}")
        
#         # Create a list to store updated batch data
#         updated_data = []
        
#         for inputs, targets in copy_data_loader: # looping through batches
#             inputs, targets = inputs.to(device), targets.to(device)
#             with torch.no_grad():
#                 outputs = model(inputs)


#             for (i, output) in enumerate(outputs): # looping through individual stocks
#                 # Skip stocks that have already been sold
#                 if sold_stocks[i] == 1:
#                     continue

#                 # See if the max price over the next min(num_days, output_height) days is greater than the current price
#                 todays_price = inputs[i][-1, 3].squeeze(0)  # Assuming index 3 is the Close price
#                 future_prices = []

#                 for future_day in range(min(num_days-day, output_height)):
#                     future_prices.append(output[future_day, 3].squeeze(0))

#                 # Check if max price is greater than today's price
#                 if max(future_prices) > todays_price:
#                     # Hold
#                     # Change value in portfolio to scale by the price increase of one day
#                     portfolio[i] *= (future_prices[0] / todays_price)
#                     print(f"DEBUG: todays_price: {todays_price}")
#                 else:
#                     # Sell
#                     # Freezes the value of the stock in the portfolio
#                     sold_stocks[i] = 1
#                     # TODO: add to cash_in_hand (this requires inversing the normalization of the data)

#             # After processing this batch, store the updated inputs/targets for the next day
#             for j in range(inputs.size(0)):
#                 # Create new input by dropping oldest timepoint and adding first target timepoint
#                 new_input = inputs[j, 1:, :].clone()  # Remove first day
#                 first_target = targets[j, 0, :].unsqueeze(0)  # Get first target day
#                 new_input = torch.cat([new_input, first_target], dim=0)  # Combine
                
#                 # Create new target by dropping the used day
#                 # Since targets is always 30 days, we can safely remove the first day
#                 new_target = targets[j, 1:, :].clone()
                
#                 updated_data.append((new_input, new_target))
        
#         #debug print
#         print(f"DEBUG: Portfolio Sum: {sum(portfolio)}") 
#         # print(f"DEBUG: Portfolio: {portfolio}")

#         # Replace dataloader with updated data if this is not the last day
#         if day < num_days - 1:
#             # Create a new dataset from updated data
#             copy_data_loader = DataLoader(
#                 updated_data,
#                 batch_size=copy_data_loader.batch_size,
#                 shuffle=False  # Important: maintain original order
#             )

#     # Find the total return of the portfolio
#     total = sum(portfolio)

#     # Simulate for a 10k portfolio
#     total_balance = 10000 + 10000 * total
#     print(f"Simulated portfolio for {num_days} days:")
#     print(f"Starting balance: 10k\nResulting balance: {total_balance}\n")
#     print(f"Total return: {total_balance - 10000} for {total_balance/100:.2f}% return.")
    
#     return total


def run_simulations(model, ticker_list, output_height, week_sim_count, day_sim_count, month_sim_count):
    print(f"Running {week_sim_count} week simulations, {day_sim_count} day simulations, and {month_sim_count} month simulations.")

    total_week_returns = 0
    total_day_returns = 0
    total_month_returns = 0
    average_week_returns = 0
    average_day_returns = 0
    average_month_returns = 0

    num_day_gains = 0
    num_month_gains = 0
    num_week_gains = 0
    


    for day in range(day_sim_count):
        # pick a random number between 5 and len(ticker_list)
        print(f"Running day simulation {day+1}/{day_sim_count}")
        num_stocks = torch.randint(5, len(ticker_list), (1,)).item()
        indices = torch.randperm(len(ticker_list))[:num_stocks]
        tickers_to_sim = [ticker_list [i] for i in indices]

        trial_return = complex_model_accuracy_v2(model, tickers_to_sim, num_days=1, output_height=output_height)
        total_day_returns += trial_return        

        if trial_return > 0:
            num_day_gains += 1
        
    average_day_returns = total_day_returns / day_sim_count
    print(f"Average daily return: {average_day_returns:.2f}")

    for week in range(week_sim_count):
        # pick a random number between 5 and len(ticker_list)
        print(f"Running week simulation {week+1}/{week_sim_count}")
        num_stocks = torch.randint(5, len(ticker_list), (1,)).item()
        indices = torch.randperm(len(ticker_list))[:num_stocks]
        tickers_to_sim = [ticker_list [i] for i in indices]

        trial_return = complex_model_accuracy_v2(model, tickers_to_sim, num_days=7, output_height=output_height)
        total_week_returns += trial_return

        if trial_return > 0:
            num_week_gains += 1

    average_week_returns = total_week_returns / week_sim_count
    print(f"Average weekly return: {average_week_returns:.2f}")

    for month in range(month_sim_count):
        # pick a random number between 5 and len(ticker_list)
        print(f"Running month simulation {month+1}/{month_sim_count}")
        num_stocks = torch.randint(5, len(ticker_list), (1,)).item()
        indices = torch.randperm(len(ticker_list))[:num_stocks]
        tickers_to_sim = [ticker_list [i] for i in indices]

        trial_return = complex_model_accuracy_v2(model, tickers_to_sim, num_days=28, output_height=output_height)
        total_month_returns += trial_return

        if trial_return > 0:
            num_month_gains += 1

    average_month_returns = total_month_returns / month_sim_count
    print(f"Average monthly return: {average_month_returns:.2f}")

    def annualize(returns, period_in_days):
        return (1 + returns) ** (365/period_in_days) - 1
    
    annualized_daily_returns = annualize(average_day_returns, period_in_days=1)
    print(f"Annualized average daily return: {annualized_daily_returns * 100:.2f}%")
    annualized_weekly_returns = annualize(average_week_returns, period_in_days=7)
    print(f"Annualized averageweekly return: {annualized_weekly_returns * 100:.2f}%")
    annualized_monthly_returns = annualize(average_month_returns, period_in_days=28)
    print(f"Annualized average monthly return: {annualized_monthly_returns * 100:.2f}%")

    print(f"Percent positive gains in daily sims: {num_day_gains/day_sim_count * 100:.2f}%")
    print(f"Percent positive gains in weekly sims: {num_week_gains/week_sim_count * 100:.2f}%")
    print(f"Percent positive gains in monthly sims: {num_month_gains/month_sim_count * 100:.2f}%")

    return annualized_daily_returns, annualized_weekly_returns, annualized_monthly_returns

def complex_model_accuracy_v2(model, ticker_list, num_days=7, output_height=1):
    MAX_NUM_DAYS = 29
    PORTFOLIO_STARTING_VALUE = 10000
    if num_days > MAX_NUM_DAYS:
        print(f"num_days cannot be greater than {MAX_NUM_DAYS}. Setting num_days to {MAX_NUM_DAYS}")
        num_days = MAX_NUM_DAYS
    
    model.to(device)
    num_stocks = len(ticker_list)
    portfolio = [1 / num_stocks * PORTFOLIO_STARTING_VALUE] * num_stocks
    sold_stocks = set()
    cash_in_hand = 0

    ticker_to_data = {}
    max_vals = {}
    min_vals = {}
    for ticker in ticker_list:
        ticker_to_data[ticker], max_vals[ticker], min_vals[ticker] = eqds.get_recent_input_tensror_for_ticker(ticker, target_window=MAX_NUM_DAYS)
        ticker_to_data[ticker] = list(ticker_to_data[ticker])

    for day in range(num_days): # loop over num_days
        # print(f"Simulating day {day+1}/{num_days}")

        for ticker in ticker_list:
            
            todays_price = ticker_to_data[ticker][0][-1, 3]
            tomorrows_price = ticker_to_data[ticker][1][0, 3]
            future_prices_inferred = []

            # run the inference
            with torch.no_grad():
                outputs = model(ticker_to_data[ticker][0].unsqueeze(0).to(device))

            # show_tensor_image(ticker_to_data[ticker][0], title=f"{ticker}")
            # show_tensor_image(ticker_to_data[ticker][0], title=f"{ticker}")

            # populate future_prices with the prices for the next min(num_days, output_height) days
            # from the inference
            for future_day in range(min(num_days-day, output_height)):
                future_prices_inferred.append(outputs[0, future_day, 3].squeeze(0))

            # check if max price is greater than today's price
            todays_price_denormalized = todays_price * (max_vals[ticker] - min_vals[ticker]) + min_vals[ticker]
            tomorrows_price_denormalized = tomorrows_price * (max_vals[ticker] - min_vals[ticker]) + min_vals[ticker]

            if max(future_prices_inferred) > todays_price:

                if ticker not in sold_stocks:
                    # hold
                    # get the price increase percentage of one day
                    change_ratio = tomorrows_price_denormalized / todays_price_denormalized
                    #increase portfolio value
                    portfolio[ticker_list.index(ticker)] *= change_ratio

                #see if buying is good
                #check if there is cash in hand
                if cash_in_hand > 0:
                    #divide the cash in hand by the number of stocks sold
                    cash_to_spend = cash_in_hand / len(sold_stocks)
                    portfolio[ticker_list.index(ticker)] += cash_to_spend
                    cash_in_hand -= cash_to_spend
                    if ticker in sold_stocks:
                        sold_stocks.remove(ticker)
            else:
                # sell
                sold_stocks.add(ticker)
                cash_in_hand += portfolio[ticker_list.index(ticker)]
                portfolio[ticker_list.index(ticker)] = 0


        # after processing, need to update the ticker_to_data dictionary to
        # shift data by one day
        for ticker in ticker_list:

            # print(f"INITIAL DATA: {ticker_to_data[ticker][0]}")
            # drop the first day in the input
            ticker_to_data[ticker][0] = ticker_to_data[ticker][0][1:, :]

            # print(f"FIRST DAY DROPPED DATA: {ticker_to_data[ticker][0]}")
            # add the first day in the target to the end of the input
            ticker_to_data[ticker][0] = torch.cat((ticker_to_data[ticker][0], ticker_to_data[ticker][1][0].unsqueeze(0)), dim=0)
            # drop the first day in the target
            ticker_to_data[ticker][1] = ticker_to_data[ticker][1][1:].squeeze(0)
            # print(f"UPDATED DATA: {ticker_to_data[ticker][0]}")
    
    total_balance = sum(portfolio) + cash_in_hand
    print(f"Simulated portfolio for {num_days} days:")
    print(f"Starting balance: ${PORTFOLIO_STARTING_VALUE:.2f}\nResulting balance: ${total_balance:.2f}\n")
    print(f"Total in stocks: ${sum(portfolio):.2f}. Total in cash: ${cash_in_hand:.2f}")
    print(f"Total return: ${(total_balance - PORTFOLIO_STARTING_VALUE):.2f} for {((total_balance - PORTFOLIO_STARTING_VALUE) / PORTFOLIO_STARTING_VALUE) * 100:.2f}% return.")
    return (total_balance - PORTFOLIO_STARTING_VALUE) / PORTFOLIO_STARTING_VALUE



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

def train_model(model, train_loader, val_loader, num_epochs=10, lr=0.0001, output_height=1, _pf=0.3):
    model.to(device)
    # criterion = nn.MSELoss()
    criterion = MCEWithDirectionPenalty(penalty_factor=_pf)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.00001)

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
        

def main(train=True, tickers=None, sim_tickers=None):
    # note that the input height muse be 256 times the output height
    torch.manual_seed(0)
    np.random.seed(0)
    global eqds
    # convolutional_layers = [4, 16, 64, 256, 32, 16, 8, 4] # this works well

    # if the below are changed the cached data loaders must be cleared manually rn
    convolutional_layers = [4, 16, 32, 128, 256, 64, 32, 4]
    # convolutional_layers = [16, 64, 128, 64, 16]
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
        tl, vl, va, ta = train_model(model, train_loader, val_loader, num_epochs=10, lr=0.0001, output_height=o_ht, _pf=0.05)
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

    best_model = torch.load("./cached_models/best_accuracy_equitymodel.pth")
    model.load_state_dict(best_model)

    # test_accuracy = complex_model_accuracy(model, test_loader, num_days=7, output_height=o_ht)
    # test_accuracy = complex_model_accuracy_v2(model, tickers, num_days=7, output_height=o_ht)
    run_simulations(model, sim_tickers, output_height=o_ht, week_sim_count=20, day_sim_count=20, month_sim_count=20)

    for ticker in tickers:
        print(f"Testing {ticker}...")
        show_prediction(model, o_ht, ticker)
        # Uncomment below to get accuracy on test set
        # test_accuracy = get_model_accuracy(model, test_loader, output_height=o_ht)
        # print(f"Test Accuracy: {100*test_accuracy}%")

    # test_accuracy = get_model_accuracy(model, test_loader, output_height=o_ht)
    # print(f"Test Accuracy: {100*test_accuracy}%")

    

if __name__ == "__main__":
    charles_portfolio = [
        "AAPL", "ADBE", "ADI", "AMD", "AMZN",
        "BX","GOOG",
        "HSBC", "KKR", "META", 
        "MSFT", "NVDA", "ORCL",
        "TD", "TXN"
    ]
    tickers_to_test = [
        "AAPL", "ADBE", "ADI", "AMAT", "AMD", "AMZN", "AVGO", "AXP", 
        "BAC", "BLK", "BX", "C", "CB", "CRM", "CSCO", "GOOG", "GS", 
        "HDB", "HSBC", "INTU", "JPM", "KKR", "META", "MMC", "MS", 
        "MSFT", "MU", "MUFG", "NOW", "NVDA", "ORCL", "PGR", "PLD", 
        "RY", "SCHW", "SMFG", "TD", "TSLA", "TXN", "UBS", "WFC"
    ]    
    main(False, tickers=charles_portfolio, sim_tickers=tickers_to_test)