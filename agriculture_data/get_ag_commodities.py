import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import os

def fetch_commodity_data(tickers, start_date, end_date, output_file):
    """
    Fetch historical commodity price data including open, close, and volume.
    Save to CSV with columns for each data type per commodity.
    
    Parameters:
    tickers (dict): Dictionary mapping commodity names to their Yahoo Finance tickers
    start_date (str): Start date in YYYY-MM-DD format
    end_date (str): End date in YYYY-MM-DD format
    output_file (str): Path to output CSV file
    """
    print(f"Fetching data for {len(tickers)} commodities from {start_date} to {end_date}...")
    
    # Create empty DataFrame to store all commodity data
    all_data = pd.DataFrame()
    
    # Fetch data for each commodity
    for commodity_name, ticker in tickers.items():
        print(f"Downloading data for {commodity_name} ({ticker})...")
        
        try:
            # Create a Ticker object first
            ticker_obj = yf.Ticker(ticker)
            
            # Then get historical data using the history method
            data = ticker_obj.history(start=start_date, end=end_date)
            
            if not data.empty:
                # Add columns for opening price, closing price and volume with commodity name prefix
                if all_data.empty:
                    # If this is the first commodity, create DataFrame with Date index
                    all_data = pd.DataFrame(index=data.index)
                
                # Add open, close, and volume data with commodity name prefix
                all_data[f'{commodity_name}_Open'] = data['Open']
                all_data[f'{commodity_name}_Close'] = data['Close']
                all_data[f'{commodity_name}_Volume'] = data['Volume']
                
                print(f"Successfully retrieved {len(data)} days of data for {commodity_name}")
            else:
                print(f"No data available for {commodity_name}")
        
        except Exception as e:
            print(f"Error retrieving data for {commodity_name}: {str(e)}")
    
    # Reset index to make date a column
    if not all_data.empty:
        all_data.reset_index(inplace=True)
        all_data.rename(columns={'index': 'Date'}, inplace=True)
        
        # Save to CSV
        all_data.to_csv(output_file, index=False)
        print(f"Data successfully saved to {output_file}")
        return all_data
    else:
        print("No data was retrieved for any commodity")
        return None

def main():
    # Define major North American oil and gas commodities with their Yahoo Finance tickers
    commodities = {
        'Corn': 'ZC=F',        # Corn

    }
    
    # Calculate date range (default: last 5 years to today)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=35*365)).strftime('%Y-%m-%d')
    
    # Define output file
    output_dir = 'commodity_data'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'corn_prices_{start_date}_to_{end_date}.csv')
    
    # Fetch and save the data
    data = fetch_commodity_data(commodities, start_date, end_date, output_file)
    
    if data is not None:
        # Display sample of the data
        print("\nSample of the collected data:")
        print(data.head())
        
        # Display data statistics
        print("\nData statistics:")
        print(f"Date range: {data['Date'].min()} to {data['Date'].max()}")
        print(f"Total days: {len(data)}")
        
        # Display column information
        print("\nCSV Columns:")
        for column in data.columns:
            print(f"- {column}")
        
        # Display completeness information
        print("\nData completeness (%):")
        completeness = (data.count() / len(data) * 100).round(2)
        print(completeness)


if __name__ == "__main__":
    main()