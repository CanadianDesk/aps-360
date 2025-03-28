import pandas as pd

def main():
    og_p_df = pd.read_csv(f'./oil_gas_prices_1986-03-16_to_2025-03-06.csv')
    output_file = "./energy.csv"

    # enforce regular capilitalization of the column names and a consistent date format
    og_p_df.columns = [col.capitalize() for col in og_p_df.columns]
    # drop columns with >90% missing values
    og_p_df = og_p_df.loc[:, og_p_df.isnull().mean() < 0.9]

    # drop null records
    og_p_df = og_p_df.dropna()
    # drop duplicates
    og_p_df = og_p_df.drop_duplicates()
    og_p_df["Date"] = pd.to_datetime(og_p_df["Date"], utc=True).dt.strftime("%Y-%m-%d")

    # save
    og_p_df.to_csv(output_file, index=False)
    print(f"Saved merged data to {output_file}")

if __name__ == "__main__":
    main()