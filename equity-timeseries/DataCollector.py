import requests
import os
from matplotlib import pyplot as plt

import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd

import torch

class EquityDataCollector:

    def __init__(self):
        self.data = []
        self.yf_data = pd.DataFrame()

    def collect(self, ticker):
        # replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
        key = os.environ.get('ALPHAVANTAGE_API_KEY')
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={key}'
        r = requests.get(url)
        self.data = r.json()
        print(self.data)

    def collect_with_yfinance(self, ticker, days_=(28*13+30)): # 13 months of data
        
        """
        Fetch stock data for a given ticker and time period using yfinance
        :param ticker: str
        :param days_: int
        :return: None
        """
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_)
        
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_str, end=end_str)
        
        self.yf_data = data

    def save(self, filename, yf=True):
        if not yf:
            with open(filename, 'w') as f:
                f.write(str(self.data))
        self.yf_data.to_csv(filename)
    
    def plot(self, ticker, yf=True):
        if yf:
            plt.plot(self.yf_data['Close'])
        else:
            plt.plot([float(x) for x in self.data['Time Series (Daily)'].keys()])
        plt.title(f'{ticker} Price')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.show()

    def get(self, yf=True):
        return self.data if not yf else self.yf_data    
