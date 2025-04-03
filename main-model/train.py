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
    data, _max, _min = eqds.get_validation_input_tensor_for_ticker(ticker)

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


def run_simulations(model, ticker_list, output_height, week_sim_count, day_sim_count, month_sim_count, return_acc=False, industry="technology"):
    print(f"Running {week_sim_count} week simulations, {day_sim_count} day simulations, and {month_sim_count} month simulations.")

    total_week_returns = 0
    total_day_returns = 0
    total_month_returns = 0

    average_week_returns = 0
    average_day_returns = 0
    average_month_returns = 0

    avg_day_returns_list = []
    avg_week_returns_list = []
    avg_month_returns_list = []

    num_day_gains = 0
    num_month_gains = 0
    num_week_gains = 0

    total_day_cash_in_hand = 0
    total_week_cash_in_hand = 0
    total_month_cash_in_hand = 0

    average_day_cash_in_hand = 0
    average_week_cash_in_hand = 0    
    average_month_cash_in_hand = 0

    day_pos_gains = 0
    week_pos_gains = 0
    month_pos_gains = 0

    if day_sim_count > 0:
        for day in range(day_sim_count):
            # pick a random number between 5 and len(ticker_list)
            print(f"Running day simulation {day+1}/{day_sim_count}")
            num_stocks = torch.randint(5, len(ticker_list), (1,)).item()
            indices = torch.randperm(len(ticker_list))[:num_stocks]
            tickers_to_sim = [ticker_list [i] for i in indices]

            trial_return, cash_in_hand = complex_model_accuracy(model, tickers_to_sim, num_days=1, output_height=output_height, industry=industry)
            total_day_returns += trial_return      
            avg_day_returns_list.append(trial_return)  
            total_day_cash_in_hand += cash_in_hand

            if trial_return > 0:
                num_day_gains += 1
            
        average_day_returns = total_day_returns / day_sim_count
        print(f"Average return after one day: {average_day_returns:.2f}")
        average_day_cash_in_hand = total_day_cash_in_hand / day_sim_count
        print(f"Average cash in hand after one day: {average_day_cash_in_hand:.2f}")

    if week_sim_count > 0:
        for week in range(week_sim_count):
            # pick a random number between 5 and len(ticker_list)
            print(f"Running week simulation {week+1}/{week_sim_count}")
            num_stocks = torch.randint(5, len(ticker_list), (1,)).item()
            indices = torch.randperm(len(ticker_list))[:num_stocks]
            tickers_to_sim = [ticker_list [i] for i in indices]

            trial_return, cash_in_hand = complex_model_accuracy(model, tickers_to_sim, num_days=7, output_height=output_height, industry=industry)
            total_week_returns += trial_return
            avg_week_returns_list.append(trial_return)
            total_week_cash_in_hand += cash_in_hand

            if trial_return > 0:
                num_week_gains += 1

        average_week_returns = total_week_returns / week_sim_count
        print(f"Average return after one week: {average_week_returns:.2f}")
        average_week_cash_in_hand = total_week_cash_in_hand / week_sim_count
        print(f"Average cash in hand after one week: {average_week_cash_in_hand:.2f}")

    if month_sim_count > 0:
        for month in range(month_sim_count):
            # pick a random number between 5 and len(ticker_list)
            print(f"Running month simulation {month+1}/{month_sim_count}")
            num_stocks = torch.randint(5, len(ticker_list), (1,)).item()
            indices = torch.randperm(len(ticker_list))[:num_stocks]
            tickers_to_sim = [ticker_list [i] for i in indices]

            trial_return, cash_in_hand = complex_model_accuracy(model, tickers_to_sim, num_days=28, output_height=output_height, industry=industry)
            total_month_returns += trial_return
            avg_month_returns_list.append(trial_return)
            total_month_cash_in_hand += cash_in_hand

            if trial_return > 0:
                num_month_gains += 1

        average_month_returns = total_month_returns / month_sim_count
        print(f"Average return after one month: {average_month_returns:.2f}")
        average_month_cash_in_hand = total_month_cash_in_hand / month_sim_count
        print(f"Average cash in hand after one month: {average_month_cash_in_hand:.2f}")

    def annualize(returns, period_in_days):
        return (1 + returns) ** (365/period_in_days) - 1
    

    # PRINT RESULTS
    print("\n\n" + "=" * 25 + " RESULTS for output height " + str(output_height) + " " + "=" * 25 + "\n\n")

    # PRINT NON-ANNUALIZED RESULTS
    if day_sim_count > 0:
        print(f"Average return day-long simulations: {average_day_returns * 100:.2f}%")
    if week_sim_count > 0:
        print(f"Average return week-long simulations: {average_week_returns * 100:.2f}%")
    if month_sim_count > 0:
        print(f"Average return month-long simulations: {average_month_returns * 100:.2f}%")

    # PRINT ANNUALIZED RESULTS
    if day_sim_count > 0:
        annualized_daily_returns = annualize(average_day_returns, period_in_days=1)
        print(f"Annualized average return on day-long simulations: {annualized_daily_returns * 100:.2f}%")
    if week_sim_count > 0:
        annualized_weekly_returns = annualize(average_week_returns, period_in_days=7)
        print(f"Annualized average return on week-long simulations: {annualized_weekly_returns * 100:.2f}%")
    if month_sim_count > 0:
        annualized_monthly_returns = annualize(average_month_returns, period_in_days=28)
        print(f"Annualized average return on month-long simulations: {annualized_monthly_returns * 100:.2f}%")

    # PRINT POSITIVE GAINS
    if day_sim_count > 0:
        day_pos_gains = num_day_gains/day_sim_count * 100
        print(f"Percent positive gains in day-long sims: {day_pos_gains:.2f}% ({num_day_gains}/{day_sim_count})")
    if week_sim_count > 0:
        week_pos_gains = num_week_gains/week_sim_count * 100
        print(f"Percent positive gains in week-long sims: {week_pos_gains:.2f}% ({num_week_gains}/{week_sim_count})")
    if month_sim_count > 0:
        month_pos_gains = num_month_gains/month_sim_count * 100
        print(f"Percent positive gains in month-long sims: {month_pos_gains:.2f}% ({num_month_gains}/{month_sim_count})")

    print("\n\n" + "=" * 25 + " END RESULTS " + "=" * 25)


    num_figs = 0
    if day_sim_count > 0:
        num_figs += 1
    if week_sim_count > 0:
        num_figs += 1
    if month_sim_count > 0:
        num_figs += 1

    plt.figure(figsize=(5*num_figs, 5))

    if day_sim_count > 0:
        plt.subplot(1, 3, 1)
        plt.hist(np.array(avg_day_returns_list) * 100, bins=20, alpha=0.7, color='blue')
        plt.axvline(average_day_returns * 100, color='red', linestyle='dashed', linewidth=2)
        plt.title(f'Daily Returns (Avg: {average_day_returns * 100:.2f}%)')
        plt.xlabel('Return (%)')
        plt.ylabel('Frequency')

    if week_sim_count > 0:
        plt.subplot(1, 3, 2)
        plt.hist(np.array(avg_week_returns_list) * 100, bins=20, alpha=0.7, color='green')
        plt.axvline(average_week_returns * 100, color='red', linestyle='dashed', linewidth=2)
        plt.title(f'Weekly Returns (Avg: {average_week_returns * 100:.2f}%)')
        plt.xlabel('Return (%)')

    if month_sim_count > 0:
        plt.subplot(1, 3, 3)
        plt.hist(np.array(avg_month_returns_list) * 100, bins=20, alpha=0.7, color='purple')
        plt.axvline(average_month_returns * 100, color='red', linestyle='dashed', linewidth=2)
        plt.title(f'Monthly Returns (Avg: {average_month_returns * 100:.2f}%)')
        plt.xlabel('Return (%)')

    plt.tight_layout()
    plt.savefig(f'returns_histogram_output_height_{output_height}.png')
    plt.show()    

    if return_acc:
        return day_pos_gains, week_pos_gains, month_pos_gains
    return annualized_daily_returns, annualized_weekly_returns, annualized_monthly_returns

def complex_model_accuracy(model, ticker_list, num_days=7, output_height=1, industry="technology"):
    MAX_NUM_DAYS = 28
    PORTFOLIO_STARTING_VALUE = 10000
    CONFIDENCE_THRESHOLD = 0.005  # Only sell if predicted to drop more than 0.5%
    MAX_ALLOCATION_PER_STOCK = 0.25  # Maximum 25% allocation to any stock
    BUY_RESERVE_RATIO = 0.2  # Keep 20% of cash in reserve
    
    if num_days > MAX_NUM_DAYS:
        print(f"num_days cannot be greater than {MAX_NUM_DAYS}. Setting num_days to {MAX_NUM_DAYS}")
        num_days = MAX_NUM_DAYS
    
    model.to(device)
    num_stocks = len(ticker_list)
    portfolio = [1 / num_stocks * PORTFOLIO_STARTING_VALUE] * num_stocks
    stock_positions = {ticker: portfolio[i] for i, ticker in enumerate(ticker_list)}
    cash_in_hand = 0
    
    # Dictionary to store predicted returns for better allocation decisions
    predicted_returns = {ticker: 0 for ticker in ticker_list}

    ticker_to_data = {}
    max_vals = {}
    min_vals = {}
    for ticker in ticker_list:
        ticker_to_data[ticker], max_vals[ticker], min_vals[ticker] = eqds.get_random_validation_day_input_tensor_for_ticker(ticker, target_window=MAX_NUM_DAYS, industry=industry)
        #showimage
        # show_tensor_image(ticker_to_data[ticker][0], title=ticker)
        ticker_to_data[ticker] = list(ticker_to_data[ticker])
        
    # Track daily portfolio values for performance metrics
    daily_portfolio_values = [PORTFOLIO_STARTING_VALUE]
    benchmark_values = [PORTFOLIO_STARTING_VALUE]  # Buy and hold benchmark

    for day in range(num_days):
        # Calculate benchmark (buy and hold) performance
        if day > 0:
            benchmark_total = 0
            for i, ticker in enumerate(ticker_list):
                initial_position = PORTFOLIO_STARTING_VALUE / num_stocks
                current_price = ticker_to_data[ticker][0][-1, 3]
                initial_price = ticker_to_data[ticker][0][-(day+1), 3]
                if initial_price > 0:
                    benchmark_total += initial_position * (current_price / initial_price)
            benchmark_values.append(benchmark_total)

        for ticker in ticker_list:
            # Get current price information
            todays_price = ticker_to_data[ticker][0][-1, 3]
            tomorrows_price = ticker_to_data[ticker][1][0, 3]
            
            # Denormalize prices
            todays_price_denormalized = todays_price * (max_vals[ticker] - min_vals[ticker]) + min_vals[ticker]
            tomorrows_price_denormalized = tomorrows_price * (max_vals[ticker] - min_vals[ticker]) + min_vals[ticker]

            # Run the inference for future prices
            with torch.no_grad():
                outputs = model(ticker_to_data[ticker][0].unsqueeze(0).to(device))

            # Calculate weighted average of predicted future prices
            future_prices_denormalized = []
            weights = []
            # Assign decreasing weights to predictions further in the future
            for future_day in range(min(num_days-day, output_height)):
                predicted_price = outputs[0, future_day, 3].item()
                predicted_price_denormalized = predicted_price * (max_vals[ticker] - min_vals[ticker]) + min_vals[ticker]
                future_prices_denormalized.append(predicted_price_denormalized)
                weights.append(1.0 / (future_day + 1))  # Higher weight for nearer predictions
            
            if not future_prices_denormalized:
                continue
                
            # Calculate weighted average prediction
            weighted_future_price = sum(p * w for p, w in zip(future_prices_denormalized, weights)) / sum(weights)
            
            # Calculate expected return
            expected_return = (weighted_future_price / todays_price_denormalized) - 1
            predicted_returns[ticker] = expected_return
            
            # Decision making with improved logic
            if expected_return > CONFIDENCE_THRESHOLD:
                # Buy or hold decision
                change_ratio = tomorrows_price_denormalized / todays_price_denormalized
                
                # Update existing position if we have one
                if stock_positions[ticker] > 0:
                    stock_positions[ticker] *= change_ratio
            else:
                # Sell decision - if we expect return below threshold, sell the position
                if stock_positions[ticker] > 0:
                    cash_in_hand += stock_positions[ticker]
                    stock_positions[ticker] = 0

        # Allocate cash to stocks with positive predicted returns
        if cash_in_hand > 0:
            # Sort stocks by expected return (highest first)
            buy_candidates = [(t, r) for t, r in predicted_returns.items() if r > CONFIDENCE_THRESHOLD]
            buy_candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Calculate total portfolio value for allocation constraints
            total_portfolio_value = sum(stock_positions.values()) + cash_in_hand
            
            # Keep some cash in reserve
            cash_to_deploy = cash_in_hand * (1 - BUY_RESERVE_RATIO)
            cash_in_hand -= cash_to_deploy
            
            for ticker, _ in buy_candidates:
                # Check maximum allocation constraint
                max_allowed = total_portfolio_value * MAX_ALLOCATION_PER_STOCK
                current_allocation = stock_positions[ticker]
                
                # Skip if already at max allocation
                if current_allocation >= max_allowed:
                    continue
                
                # Calculate how much we can add to this position
                available_allocation = max_allowed - current_allocation
                amount_to_buy = min(cash_to_deploy, available_allocation)
                
                if amount_to_buy > 0:
                    stock_positions[ticker] += amount_to_buy
                    cash_to_deploy -= amount_to_buy
                    
                # If we've deployed all available cash, break
                if cash_to_deploy <= 0:
                    break
            
            # Return any unallocated cash to our reserve
            cash_in_hand += cash_to_deploy

        # Update portfolio dictionary from stock_positions
        portfolio = [stock_positions[ticker] for ticker in ticker_list]
            
        # Track daily portfolio value
        daily_portfolio_values.append(sum(portfolio) + cash_in_hand)

        # After processing, update the ticker_to_data dictionary to shift data by one day
        for ticker in ticker_list:
            # Drop the first day in the input
            ticker_to_data[ticker][0] = ticker_to_data[ticker][0][1:, :]
            # Add the first day in the target to the end of the input
            ticker_to_data[ticker][0] = torch.cat((ticker_to_data[ticker][0], ticker_to_data[ticker][1][0].unsqueeze(0)), dim=0)
            # Drop the first day in the target
            ticker_to_data[ticker][1] = ticker_to_data[ticker][1][1:, :]
            # Handle case when target becomes empty
            if ticker_to_data[ticker][1].size(0) == 0:
                ticker_to_data[ticker][1] = torch.zeros((0, ticker_to_data[ticker][0].size(1)))
    
    # Calculate final results
    total_balance = sum(portfolio) + cash_in_hand
    
    # Calculate Sharpe ratio (simplified)
    if len(daily_portfolio_values) > 1:
        daily_returns = [(daily_portfolio_values[i] / daily_portfolio_values[i-1]) - 1 for i in range(1, len(daily_portfolio_values))]
        avg_daily_return = sum(daily_returns) / len(daily_returns)
        std_daily_return = (sum((r - avg_daily_return) ** 2 for r in daily_returns) / len(daily_returns)) ** 0.5
        sharpe_ratio = (avg_daily_return / std_daily_return) * (252 ** 0.5) if std_daily_return > 0 else 0
    else:
        sharpe_ratio = 0
    
    # Print results
    print(f"Simulated portfolio for {num_days} days:")
    print(f"Starting balance: ${PORTFOLIO_STARTING_VALUE:.2f}\nResulting balance: ${total_balance:.2f}\n")
    print(f"Total in stocks: ${sum(portfolio):.2f}. Total in cash: ${cash_in_hand:.2f}")
    print(f"Total return: ${(total_balance - PORTFOLIO_STARTING_VALUE):.2f} for {((total_balance - PORTFOLIO_STARTING_VALUE) / PORTFOLIO_STARTING_VALUE) * 100:.2f}% return.")
    
    # Compare against buy-and-hold benchmark
    if benchmark_values[-1] > 0:
        benchmark_return = (benchmark_values[-1] - PORTFOLIO_STARTING_VALUE) / PORTFOLIO_STARTING_VALUE
        print(f"Buy & Hold return: {benchmark_return * 100:.2f}%")
        print(f"Strategy outperformance: {((total_balance/PORTFOLIO_STARTING_VALUE) - (benchmark_values[-1]/PORTFOLIO_STARTING_VALUE)) * 100:.2f}%")
    
    print(f"Sharpe ratio: {sharpe_ratio:.2f}")
    
    return ((total_balance - PORTFOLIO_STARTING_VALUE) / PORTFOLIO_STARTING_VALUE), cash_in_hand

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

def train_model(model, train_loader, val_loader, num_epochs=10, lr=0.0001, output_height=1, _pf=0.3, ticker_list=None, industry="technology"):
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
        # validation_acc, _, _ = run_simulations(model, ticker_list, output_height=output_height, day_sim_count=100, week_sim_count=0, month_sim_count=0, return_acc=True)
        # validation_accuracies.append(validation_acc)

        if validation_losses[-1] < best_val_loss:
            best_val_loss = validation_losses[-1]
            # Save the model if validation loss improves
            if not os.path.exists("./cached_models"):
                os.makedirs("./cached_models")
            torch.save(model.state_dict(), f"./cached_models/best_loss_equitymodel_{industry}.pth")
        if validation_accuracies[-1] > best_val_accuracy:
            best_val_accuracy = validation_accuracies[-1]
            # Save the model if validation accuracy improves
            if not os.path.exists("./cached_models"):
                os.makedirs("./cached_models")
            torch.save(model.state_dict(), f"./cached_models/best_accuracy_equitymodel_{industry}.pth")
        
        # Print statistics
        print(f"Epoch [{epoch+1}] | Training Loss: {training_losses[-1]:.4f} | Validation Loss: {validation_losses[-1]:.4f} | Validation Accuracy: {validation_accuracies[-1]:.4f} | Training Accuracy: {training_accuracies[-1]:.4f}")

    return training_losses, validation_losses, validation_accuracies, training_accuracies
        

def main(train=True, show_predictions=True, tickers=None, sim_tickers=None):
    # note that the input height muse be 256 times the output height
    torch.manual_seed(0)
    np.random.seed(0)
    global eqds
    # convolutional_layers = [4, 16, 64, 256, 32, 16, 8, 4] # this works well

    # if the below are changed the cached data loaders must be cleared manually rn
    # convolutional_layers = [4, 16, 32, 128, 256, 64, 32, 4] #output height = 1
    convolutional_layers = [32, 128, 256, 128, 32] #output height = 8
    o_ht = 8
    i_ht = o_ht * (2**len(convolutional_layers))

    # below this should be ok
    industry = "energy"
    eqds = EquityDataset(input_height=i_ht, output_height=o_ht, include_industry_specific=True, normalize=False)
    train_loader, val_loader, test_loader = eqds.construct_data_loaders(industry=industry, sample_stride=4, batch_size=32)
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
        tl, vl, va, ta = train_model(model, train_loader, val_loader, num_epochs=10, lr=0.0001, output_height=o_ht, _pf=0.05, ticker_list=sim_tickers, industry=industry)
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

    best_model = torch.load(f"./cached_models/best_accuracy_equitymodel_{industry}.pth")
    model.load_state_dict(best_model)

    # test_accuracy = complex_model_accuracy(model, test_loader, num_days=7, output_height=o_ht)
    # test_accuracy = complex_model_accuracy_v2(model, tickers, num_days=7, output_height=o_ht)

    run_simulations(model, sim_tickers, output_height=o_ht, week_sim_count=0, day_sim_count=0, month_sim_count=150, industry=industry)

    if show_predictions:
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
    temp = [
        "BX", "META", "HSBC", "JPM", "AXP", "CSCO", "NOW", "TXN"
    ]

    energy_tickers = [
        "COP", "CVX", "DVN", "EOG", "HAL", "KMI", "MPC", "OKE", "OXY", "PSX", "SLB", 
        "VLO", "WMB", "XOM", "BKR"
    ]
    main(train=False, show_predictions=False, tickers=charles_portfolio, sim_tickers=energy_tickers)