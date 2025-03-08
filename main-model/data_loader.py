import pandas as pd
import numpy as np
import torch
import os
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader

class EquityDataset(Dataset):
    def __init__(self, root_paths=["../news-sentiment/output/", "../equity-timeseries/collected_data/"], input_width=512, output_width=32, include_industry_specific=False, industry_spec_len=0):

        self.dataframes_dict = {}
        self.macroeconomics = pd.read_csv("./constant_input/macroeconomics.csv")

        allpaths = []
        for root_path in root_paths:
            allpaths += [x for x in os.listdir(root_path) if x.endswith(".csv")]

        self.holistic_cols = ["Date", "Open", "High", "Low", "Close", "Volume", "Sentiment"]
        if include_industry_specific:
            self.holistic_cols += ["Industry Specific {}".format(i) for i in range(industry_spec_len)]

        # News sentiment and equity data only
        self.ticker_to_paths = {}
        for path in allpaths:
            ticker = path.split(".")[0]
            if not ticker in self.ticker_to_paths:
                self.ticker_to_paths[ticker] = []
            self.ticker_to_paths[ticker].append(path)

        self.ticker_to_paths = {k: v for k, v in sorted(self.ticker_to_paths.items(), key=lambda item: item[0]) if len(v) == 2}
        for ticker, paths in self.ticker_to_paths.items():
            df_a = pd.read_csv(root_paths[0] + paths[0])
            df_b = pd.read_csv(root_paths[1] + paths[1])

            if "Open" not in df_a.columns:
                news_df = df_a.dropna()
                equity_df = df_b.dropna()
            else:
                news_df = df_b.dropna()
                equity_df = df_a.dropna()

            # Enforce that the dataframes are the same length (duration wise)
            oldest_date = max(min(equity_df["Date"]), min(news_df["Date"]))
            newest_date = min(max(equity_df["Date"]), max(news_df["Date"]))
            news_df = news_df[(news_df["Date"] >= oldest_date) & (news_df["Date"] <= newest_date)]
            equity_df = equity_df[(equity_df["Date"] >= oldest_date) & (equity_df["Date"] <= newest_date)]

            # enforce that both "Date" colums are of the same format (without time) YYYY-MM-DD
            print(equity_df)
            print(news_df)
            news_df["Date"] = pd.to_datetime(news_df["Date"], utc=True).dt.strftime("%Y-%m-%d")
            equity_df["Date"] = pd.to_datetime(equity_df["Date"], utc=True).dt.strftime("%Y-%m-%d")

            # if the dataframes are still not the same length, interpolate the news dataframe
            prev_date = min(news_df["Date"])
            prev_sentiment = 0.0
            
            for date in equity_df["Date"]:
                if not date in news_df["Date"].values:
                    # interpolate by holding the value from the most recent day before the current that is present in the news dataframe
                    news_df = news_df._append(pd.DataFrame({"Date": [date], "Sentiment": [prev_sentiment]}))
                else:
                    prev_date = date
                    prev_sentiment = news_df[news_df["Date"] == prev_date]["Sentiment"].values[0]

            news_df = news_df.sort_values(by="Date").reset_index(drop=True)

            # join the dataframes, aligning on the date column
            df = pd.merge(equity_df, news_df, on="Date", how="inner")
            df = pd.merge(df, self.macroeconomics, on="Date", how="inner")

            self.dataframes_dict[ticker] = df

        self.input_width = input_width
        self.output_width = output_width
        self.sample_window = input_width + output_width

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def sample_dataframes(self, sample_step=1):
        sampled_dataframes = {}
        for ticker, df in self.dataframes_dict.items():
            sampled_dataframes[ticker] = df.iloc[::sample_step]
        return sampled_dataframes

    def construct_data_loaders(self):
        train_data = []
        val_data = []
        test_data = []

        for ticker, df in self.dataframes_dict.items():
            # split the data into train, val, and test
            train_df = df[:int(0.7*len(df))]
            val_df = df[int(0.7*len(df)):int(0.85*len(df))]
            test_df = df[int(0.85*len(df)):]

            train_data += self._split_df_into_samples(train_df)
            val_data += self._split_df_into_samples(val_df)
            test_data += self._split_df_into_samples(test_df)

        self.train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(val_data, batch_size=32, shuffle=True)
        self.test_loader = DataLoader(test_data, batch_size=32, shuffle=True)

    def _split_df_into_samples(self, df):
        data = []
        for i in range(len(df) - self.sample_window):
            sample = df.iloc[i:i+self.sample_window]
            data.append(sample)
        return data
    
    def plot_and_save_image(self, ticker, save_path="./dataset_plots"):

        if ticker not in self.dataframes_dict:
            raise ValueError("Ticker not found in dataset")
        df = self.dataframes_dict[ticker]
        
        fig, ax1 = plt.subplots()
        color = 'tab:blue'
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Closing Price', color=color)
        ax1.plot(df["Date"], df["Close"], color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:green'
        ax2.set_ylabel('Sentiment', color=color)
        ax2.plot(df["Date"], df["Sentiment"], color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()
        plt.title(ticker)
        # plt.show()
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, "{}.png".format(ticker)))
        plt.close()
    
    def __len__(self):
        return len(self.dataframes_dict)
    
if __name__ == "__main__":
    eqds = EquityDataset()
    eqds.plot_and_save_image("TSLA")