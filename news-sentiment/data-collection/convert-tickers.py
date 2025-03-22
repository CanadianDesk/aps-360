import yfinance as yf
import json
import os

def get_company_name(ticker):
  try:
    stock = yf.Ticker(ticker)
    company_info = stock.info
    
    display_name = company_info.get('displayName', None)
    if display_name:
      print(f"{ticker} -> {display_name} (displayName)")
      return display_name
    
    short_name = company_info.get('shortName', None)
    if short_name:
      print(f"{ticker} -> {short_name} (shortName)")
      return short_name
    
    long_name = company_info.get('longName', None)
    if long_name:
      print(f"{ticker} -> {long_name} (longName)")
      return long_name
    
    return None
  
  except Exception as e:
    print(f"Error retrieving company name for {ticker}: {e}")
    return None

def process_tickers(num_tickers=1):
    # Load tickers from JSON
    try:
        with open('./tickers.json', 'r') as file:
            tickers = json.load(file)
    except FileNotFoundError:
        print("Error: tickers.json file not found.")
        return
    except json.JSONDecodeError:
        print("Error: tickers.json is not a valid JSON file.")
        return
    
    # Limit the number of tickers if specified
    if num_tickers is not None:
        tickers_to_process = tickers[:num_tickers]
    else:
        tickers_to_process = tickers.copy()
    
    # Process each ticker and get company names
    names = []
    failed_tickers = []
    
    for ticker in tickers_to_process:
        print(f"Processing {ticker}...")
        company_name = get_company_name(ticker)
        if company_name:
            names.append(company_name)
        else:
            print(f"Could not retrieve name for {ticker}")
            failed_tickers.append(ticker)
    
    # Save names to JSON
    try:
        with open('./names.json', 'w') as file:
            json.dump(names, file, indent=2)
        print(f"Successfully saved {len(names)} company names to names.json")
    except Exception as e:
        print(f"Error saving names to file: {e}")
    
    # Remove failed tickers from tickers.json
    if failed_tickers:
        try:
            # Remove failed tickers from the list
            updated_tickers = [t for t in tickers if t not in failed_tickers]
            
            # Save the updated list back to tickers.json
            with open('./tickers.json', 'w') as file:
                json.dump(updated_tickers, file, indent=2)
            
            print(f"Removed {len(failed_tickers)} failed tickers from tickers.json")
        except Exception as e:
            print(f"Error updating tickers.json: {e}")



if __name__ == "__main__":

  # empty names file
  # with open('names.json', 'w') as file:
  #   json.dump([], file)
  # print("Successfully emptied names.json")

  num_tickers = None
  process_tickers(num_tickers)

  # empty tickers file
  # with open('tickers.json', 'w') as file:
  #   json.dump([], file)
  # print("Successfully emptied tickers.json")