import pandas as pd
import numpy as np
import torch
import os
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader

class EquityDataset(Dataset):
    def __init__(self, root_paths=["../news-sentiment/output/", "../equity-timeseries/collected_data/"], input_width=512, output_width=32, include_industry_specific=True, normalize=False):

        self.energy_tickers = [
            "CNQ.TO", "COP", "CVE.TO", "CVX", "DVN", "ENB.TO", "EOG", "HAL", 
            "IMO.TO", "KMI", "MPC", "OKE", "OXY", "PPL.TO", "PSX", "SLB", 
            "SU.TO", "VLO", "WMB", "XOM", "BKR"
        ]
        self.technology_tickers = [
            "AAPL", "ADBE", "ADI", "AMAT", "AMD", "AMZN", "AVGO", "AXP", 
            "BAC", "BLK", "BX", "C", "CB", "CRM", "CSCO", "GOOG", "GS", 
            "HDB", "HSBC", "INTU", "JPM", "KKR", "META", "MMC", "MS", 
            "MSFT", "MU", "MUFG", "NOW", "NVDA", "ORCL", "PGR", "PLD", 
            "RY", "SCHW", "SMFG", "TD", "TSLA", "TXN", "UBS", "VRN.TO", "WFC"
        ]
        self.agriculture_tickers = [
            "ADM", "BG", "CTVA", "MOS"
        ]

        self.dataframes_dict = {}
        self.macroeconomics = pd.read_csv("./constant_input/macroeconomics.csv")

        allpaths = []
        for root_path in root_paths:
            allpaths += [x for x in os.listdir(root_path) if x.endswith(".csv")]

        # News sentiment and equity data only
        self.ticker_to_paths = {}
        for path in allpaths:
            ticker = path.split(".")[0]
            if ticker not in self.technology_tickers and not include_industry_specific:
                continue
            if not ticker in self.ticker_to_paths:
                self.ticker_to_paths[ticker] = []
            self.ticker_to_paths[ticker].append(path)

        self.ticker_to_paths = {k: v for k, v in sorted(self.ticker_to_paths.items(), key=lambda item: item[0]) if len(v) == 2}
        total = len(self.ticker_to_paths)
        cnt = 0
        for ticker, paths in self.ticker_to_paths.items():

            cnt += 1
            print(f"Processing {ticker}... ({cnt}/{total})")

            industry = "technology"
            if include_industry_specific:
                if ticker in self.energy_tickers:
                    industry = "energy"
                elif ticker in self.agriculture_tickers:
                    industry = "agriculture"
            if ticker not in self.technology_tickers and industry == "technology":
                continue

            if (os.path.exists(f"./cached_images/{industry}/{ticker}.csv")):
                self.dataframes_dict[ticker] = pd.read_csv(f"./cached_images/{industry}/{ticker}.csv")
                continue

            df_a = pd.read_csv(root_paths[0] + paths[0])
            df_b = pd.read_csv(root_paths[1] + paths[1])

            if "Open" not in df_a.columns:
                news_df = df_a.dropna()
                equity_df = df_b.dropna()
            else:
                news_df = df_b.dropna()
                equity_df = df_a.dropna()
            equity_df = equity_df.drop(columns=["Dividends","Stock Splits"])

            # Capitalize the column names
            news_df.columns = [col.capitalize() for col in news_df.columns]
            equity_df.columns = [col.capitalize() for col in equity_df.columns]

            # Enforce that the dataframes are the same length (duration wise)
            oldest_date = max(min(equity_df["Date"]), min(news_df["Date"]))
            newest_date = min(max(equity_df["Date"]), max(news_df["Date"]))
            news_df = news_df[(news_df["Date"] >= oldest_date) & (news_df["Date"] <= newest_date)]
            equity_df = equity_df[(equity_df["Date"] >= oldest_date) & (equity_df["Date"] <= newest_date)]

            # enforce that both "Date" colums are of the same format (without time) YYYY-MM-DD
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
            if include_industry_specific and industry != "technology":
                # merge with the industry specific data
                if industry == "agriculture":
                    df = pd.merge(df, pd.read_csv(f"./constant_input/agriculture/agriculture.csv"), on="Date", how="inner")
                elif industry == "energy":
                    df = pd.merge(df, pd.read_csv(f"./constant_input/energy/energy.csv"), on="Date", how="inner")

            if normalize:
                # Store mins and maxes before normalization
                min_max_df = pd.DataFrame(columns=["ID", "Min", "Max"])
                
                rf = df.copy()
                
                for col in rf.columns:
                    min_val = rf[col].min()
                    max_val = rf[col].max()
                    min_max_df.loc[col] = [col, min_val, max_val]
                    
                    # Check for division by zero to avoid NaN results
                    if max_val - min_val != 0:
                        df[col] = (df[col] - min_val) / (max_val - min_val)
                    else:
                        # Handle constant columns - set to 0 or 0.5 or keep as is
                        df[col] = 0.5  # or another strategy for constant columns
                
                min_max_df.to_csv(f"./minmax_images/{industry}/{ticker}.csv", index=False)

            # go through the df day by day and make sure it is consecutive, if not, fill in the missing days with the last known value
            for i in range(len(df) - 1):
                if df["Date"].iloc[i] != df["Date"].iloc[i + 1]:
                    # fill in the missing days with the last known value
                    missing_days = pd.date_range(start=df["Date"].iloc[i], end=df["Date"].iloc[i + 1])
                    for day in missing_days:
                        if day not in df["Date"].values:
                            df = df._append({"Date": day}, ignore_index=True)
                        if day in equity_df["Date"].values:
                            # borrow the value from the equity dataframe
                            for col in ["Open", "High", "Low", "Close", "Volume"]:
                                if col in df.columns:
                                    df.loc[df["Date"] == day, col] = equity_df[equity_df["Date"] == day][col].values[0]
                        if day in news_df["Date"].values:
                            # borrow the value from the news dataframe
                            for col in ["Sentiment"]:
                                if col in df.columns:
                                    df.loc[df["Date"] == day, col] = news_df[news_df["Date"] == day][col].values[0]
                        if day in self.macroeconomics["Date"].values:
                            # borrow the value from the macroeconomics dataframe
                            for col in self.macroeconomics.columns:
                                if col != "Date" and col in df.columns:
                                    df.loc[df["Date"] == day, col] = self.macroeconomics[self.macroeconomics["Date"] == day][col].values[0]
            # sort the dataframe by date, but ensure that the date column is in datetime format
            df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
            # sort the dataframe by date
            df = df.sort_values(by="Date")
            # fill in the missing values with the last known value
            df = df.fillna(method="ffill")
            # drop the first row if it is empty
            if df.iloc[0].isnull().any():
                df = df.drop(index=0).reset_index(drop=True)
            # drop the last row if it is empty
            if df.iloc[-1].isnull().any():
                df = df.drop(index=-1).reset_index(drop=True)
            # drop records with the same date
            df = df.drop_duplicates(subset=["Date"], keep="last")

            self.dataframes_dict[ticker] = df
            # cache the df for later use
            df.to_csv(f"./cached_images/{industry}/{ticker}.csv", index=False)

        self.input_width = input_width
        self.output_width = output_width
        self.sample_window = input_width + output_width

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def construct_data_loaders(self, industry="technology", sample_stride=1):
        if industry not in ["technology", "energy", "agriculture"]:
            raise ValueError("Invalid industry specified. Choose from 'technology', 'energy', or 'agriculture'.")
        train_data = []
        val_data = []
        test_data = []

        designee_tickers = self.technology_tickers if industry == "technology" else self.energy_tickers if industry == "energy" else self.agriculture_tickers

        total = len(self.dataframes_dict)
        cnt = 0
        for ticker, df in self.dataframes_dict.items():
            
            cnt += 1
            print(f"Loading {ticker}... ({cnt}/{total})")

            # Check if the ticker is in the specified industry
            if ticker not in designee_tickers: continue
            # Split the dataframe into samples
            samples = self.split_df_into_samples(df, sample_window=self.sample_window, sample_stride=sample_stride)
            num_samples = len(samples)
            num_train_samples = int(num_samples * 0.8)
            num_val_samples = int(num_samples * 0.1)
            num_test_samples = int(num_samples * 0.1)

            # Split the samples into train, val, and test sets by random stratified sampling
            np.random.shuffle(samples)
            train_data += samples[:num_train_samples]
            val_data += samples[num_train_samples:num_train_samples + num_val_samples]
            test_data += samples[num_train_samples + num_val_samples:num_train_samples + num_val_samples + num_test_samples]

        # Create DataLoader objects
        self.train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
        self.test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
        print(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}, Test samples: {len(test_data)}")
        return self.train_loader, self.val_loader, self.test_loader

    def split_df_into_samples(self, df, sample_window=None, sample_stride=1, normalize=True):
        if sample_window is None:
            sample_window = self.sample_window
        if len(df) < sample_window:
            raise ValueError("Dataframe is too short to split into samples")
        if len(df) == sample_window:
            return [df]
        # Ensure the dataframe is sorted by date
        df = df.sort_values(by="Date").reset_index(drop=True)
        samples = []
        for i in range(0, len(df) - sample_window + 1, sample_stride):
            sample = df.iloc[i:i + sample_window]
            # Drop the date column
            sample = sample.drop(columns=["Date"])
            if normalize:
                # normalize the sample based on the first input_width values
                for col in sample.columns:
                    min_val = sample[col][:self.input_width].min()
                    max_val = sample[col][:self.input_width].max()
                    if max_val - min_val != 0:  # Avoid division by zero
                        sample.loc[:, col] = (sample[col] - min_val) / (max_val - min_val)
                    else:
                        sample.loc[:, col] = 0.5  # Handle constant columns
            # convert the sample to a tensor
            sample = torch.tensor(sample.values, dtype=torch.float32)
            # make it an input, target pair
            input_sample = sample[:self.input_width, 1:]
            target_sample = sample[self.input_width:, 1:]
            # add the sample to the list
            samples.append((input_sample, target_sample))
        # return the samples as a list of tuples
        return samples
        
    
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
        plt.show()
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, "{}.png".format(ticker)))
        plt.close()
    
    def __len__(self):
        return len(self.dataframes_dict)
    
if __name__ == "__main__":
    eqds = EquityDataset()