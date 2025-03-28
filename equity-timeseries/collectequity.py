
from DataCollector import EquityDataCollector

def collect_and_plot_from_list(list_=['META']):
    for ticker in list_:
        eqdc = EquityDataCollector()
        eqdc.collect_with_yfinance(ticker, days_=365*20)
        eqdc.save(f'./collected_data/{ticker}.csv')
        # eqdc.plot(ticker)

def get_tickers_with_sentiment(path):
    import os
    import pandas as pd

    files = os.listdir(path)
    tickers = []
    for file in files:
        if file.endswith(".csv"):
            tickers.append(file.split(".")[0])
    return tickers

if __name__ == "__main__":

    available_tickers = get_tickers_with_sentiment("../news-sentiment/output/")
        
    tech = [
            'AAPL',
            'MSFT',
            'GOOG',
            'AMZN',
            'TSLA',
            'NVDA',
            'BB',
            'AMD',
            'INTC',
            'IBM',
            'ORCL',
            'CRM',
            'ADBE',
            'CSCO',
            'QCOM',
            'TXN',
            'MU',
            'NOW',
            'SNOW',
            'ZM',
            'DOCU',
            'BBA'
        ]
    collect_and_plot_from_list(
        available_tickers
    )