import pandas as pd

def main():
    corn_prices = pd.read_csv(f'./corn_prices_1990-03-16_to_2025-03-07.csv')
    weather_devs = pd.read_csv(f'./weather_deviations.csv')
    output_file = "./agriculture.csv"

    # enforce a regular capitalization of the column names and a consistent date format
    corn_prices.columns = [col.capitalize() for col in corn_prices.columns]
    weather_devs.columns = [col.capitalize() for col in weather_devs.columns]
    corn_prices["Date"] = pd.to_datetime(corn_prices["Date"], utc=True).dt.strftime("%Y-%m-%d")
    weather_devs["Date"] = pd.to_datetime(weather_devs["Date"], utc=True).dt.strftime("%Y-%m-%d")

    # perform a merge by date
    df = pd.merge(corn_prices, weather_devs, on="Date", how="inner")
    # drop columns with >90% missing values
    df = df.loc[:, df.isnull().mean() < 0.9]
    # drop null records
    df = df.dropna()
    # drop duplicates
    df = df.drop_duplicates()
    # drop the "Month-Day" column
    df = df.drop(columns=["Month_day"])
    # write out
    df.to_csv(output_file, index=False)
    print(f"Saved merged data to {output_file}")

if __name__ == "__main__":
    main()