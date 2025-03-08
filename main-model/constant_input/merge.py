import pandas as pd
import numpy as np
import torch
import os
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader

def merge_dataframes(mc1, mc2):

    # mc1 has date format "YYYY-MM-DD" and mc2 has date format "MM/DD/YYYY". make them both YYYY-MM-DD with no time data, just the date
    mc2["Date"] = pd.to_datetime(mc2["Date"]).dt.strftime("%Y-%m-%d")
    mc1["Date"] = pd.to_datetime(mc1["Date"]).dt.strftime("%Y-%m-%d")

    print(mc1)
    print(mc2)
    
    # join the dataframes, aligning on the date column
    df = pd.merge(mc1, mc2, on="Date", how="inner")
    # remove the 4 month column
    df = df.drop(columns=["4 Mo"])
    
    print(df)
    df.to_csv("./merged.csv", index=False)

if __name__ == "__main__":
    merge_dataframes(pd.read_csv("./mc1.csv"), pd.read_csv("./mc2.csv"))