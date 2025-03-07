
from DataCollector import EquityDataCollector

def collect_and_plot_from_list(list_=['META']):
    for ticker in list_:
        eqdc = EquityDataCollector()
        eqdc.collect_with_yfinance(ticker, days_=365*20)
        eqdc.save(f'./collected_data/{ticker}.csv')
        # eqdc.plot(ticker)

if __name__ == "__main__":
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
        tech
    )