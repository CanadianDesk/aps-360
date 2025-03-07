import pandas as pd
import numpy as np
from datetime import datetime
import csv

def calculate_daily_averages(input_file, output_file):
    """
    Calculate the average values for each day of the year across all years in the dataset.
    
    Parameters:
    input_file (str): Path to the input CSV file with weather data
    output_file (str): Path to save the output CSV file with daily averages
    """
    print(f"Reading data from {input_file}...")
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Convert DATE column to datetime
    df['DATE'] = pd.to_datetime(df['DATE'])
    
    # Extract month and day for grouping
    df['MONTH_DAY'] = df['DATE'].dt.strftime('%m-%d')
    
    # Create a column for leap year day (Feb 29)
    is_leap_day = (df['DATE'].dt.month == 2) & (df['DATE'].dt.day == 29)
    
    print(f"Loaded {len(df)} days of weather data")
    
    # Create a list to hold our results
    results = []
    
    # Group by month and day, excluding leap days from standard calculations
    print("Calculating averages for regular days...")
    regular_days = df[~is_leap_day]
    daily_averages = regular_days.groupby('MONTH_DAY').agg({
        'TMIN': 'mean',
        'TMAX': 'mean',
        'AWND': 'mean',
        'PRCP': 'mean'
    }).reset_index()
    
    # Add a date column for sorting (using a non-leap year)
    daily_averages['SORT_DATE'] = pd.to_datetime('2001-' + daily_averages['MONTH_DAY'])
    
    # Handle leap day separately
    print("Handling February 29th (leap day)...")
    leap_day_data = df[is_leap_day]
    if not leap_day_data.empty:
        leap_day_avg = leap_day_data.agg({
            'TMIN': 'mean',
            'TMAX': 'mean',
            'AWND': 'mean',
            'PRCP': 'mean'
        }).to_dict()
        
        # Add leap day to results
        leap_day_row = {
            'MONTH_DAY': '02-29',
            'SORT_DATE': pd.to_datetime('2000-02-29'),  # Use leap year for sort date
            'TMIN': leap_day_avg['TMIN'],
            'TMAX': leap_day_avg['TMAX'],
            'AWND': leap_day_avg['AWND'],
            'PRCP': leap_day_avg['PRCP']
        }
        
        # Add leap day to the dataframe
        daily_averages = pd.concat([
            daily_averages, 
            pd.DataFrame([leap_day_row])
        ], ignore_index=True)
    
    # Sort by month and day
    daily_averages = daily_averages.sort_values('SORT_DATE').reset_index(drop=True)
    
    # Count how many years of data we have for each day
    day_counts = regular_days.groupby('MONTH_DAY').size().reset_index(name='YEARS_COUNT')
    daily_averages = daily_averages.merge(day_counts, on='MONTH_DAY', how='left')
    
    # Add leap day count
    if not leap_day_data.empty:
        leap_day_index = daily_averages[daily_averages['MONTH_DAY'] == '02-29'].index
        if not leap_day_index.empty:
            daily_averages.loc[leap_day_index, 'YEARS_COUNT'] = len(leap_day_data)
    
    # Columns for the output CSV
    output_columns = ['MONTH_DAY', 'TMIN', 'TMAX', 'AWND', 'PRCP', 'YEARS_COUNT']
    
    # Write to CSV
    print(f"Writing results to {output_file}...")
    daily_averages[output_columns].to_csv(output_file, index=False)
    
    print(f"Successfully calculated and saved daily averages across {daily_averages['YEARS_COUNT'].max()} years!")
    return daily_averages

def main():
    # Configuration
    input_file = "result/weather_data_1990_2024.csv"
    output_file = "result/daily_averages.csv"
    
    try:
        # Calculate daily averages
        daily_averages = calculate_daily_averages(input_file, output_file)
        
        # Display a preview of the results
        print("\nPreview of the daily averages:")
        preview = daily_averages.head(10)
        for _, row in preview.iterrows():
            print(f"{row['MONTH_DAY']}: TMIN={row['TMIN']:.2f}, TMAX={row['TMAX']:.2f}, AWND={row['AWND']:.2f}, PRCP={row['PRCP']:.2f} (based on {int(row['YEARS_COUNT'])} years)")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()