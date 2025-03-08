"""
Takes in the daily averages from 1990-2025 from Sioux City Iowa, and then the Averages from thate given timeline
outputes weather_deviations in ./results where the value is how much it deviates from the averge to the 
+ or - side.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import os

def calculate_daily_deviations(weather_data_file, daily_averages_file, output_file):
    """
    Calculate the deviation of each day's weather data from the long-term daily averages.
    
    Parameters:
    weather_data_file (str): Path to the CSV file with daily weather data over 35 years
    daily_averages_file (str): Path to the CSV file with daily averages for each calendar day
    output_file (str): Path to save the output CSV file with deviation data
    """
    print(f"Reading weather data from {weather_data_file}...")
    weather_df = pd.read_csv(weather_data_file)
    
    print(f"Reading daily averages from {daily_averages_file}...")
    averages_df = pd.read_csv(daily_averages_file)
    
    # Convert DATE column to datetime
    weather_df['DATE'] = pd.to_datetime(weather_df['DATE'])
    
    # Extract month and day for joining with averages
    weather_df['MONTH_DAY'] = weather_df['DATE'].dt.strftime('%m-%d')
    
    print(f"Joining datasets to calculate deviations...")
    # Merge the weather data with the averages data
    merged_df = pd.merge(
        weather_df,
        averages_df,
        how='left',
        on='MONTH_DAY',
        suffixes=('', '_AVG')
    )
    
    # Calculate actual deviations (not absolute)
    print("Calculating actual deviations for each day...")
    for column in ['TMIN', 'TMAX', 'AWND', 'PRCP']:
        merged_df[f'{column}_DEV'] = merged_df[column] - merged_df[f'{column}_AVG']
    
    # Calculate a combined deviation metric (optional)
    # Normalize each deviation by its typical range to make them comparable
    print("Calculating combined deviation metric...")
    
    # Normalize the deviations (z-score approach)
    for column in ['TMIN', 'TMAX', 'AWND', 'PRCP']:
        dev_column = f'{column}_DEV'
        std = merged_df[dev_column].std()
        if std > 0:  # Avoid division by zero
            merged_df[f'{column}_DEV_NORM'] = merged_df[dev_column] / std
        else:
            merged_df[f'{column}_DEV_NORM'] = 0
    
    # Combined deviation score (average of normalized deviations)
    # Note: Using absolute values for the combined deviation to measure total deviation magnitude
    merged_df['COMBINED_DEV'] = merged_df[[
        'TMIN_DEV_NORM', 'TMAX_DEV_NORM', 'AWND_DEV_NORM', 'PRCP_DEV_NORM'
    ]].abs().mean(axis=1)
    
    # Select columns for the output
    output_columns = [
        'DATE', 'MONTH_DAY',
        'TMIN', 'TMAX', 'AWND', 'PRCP',
        'TMIN_AVG', 'TMAX_AVG', 'AWND_AVG', 'PRCP_AVG',
        'TMIN_DEV', 'TMAX_DEV', 'AWND_DEV', 'PRCP_DEV',
        'COMBINED_DEV'
    ]
    
    result_df = merged_df[output_columns].sort_values('DATE').reset_index(drop=True)
    
    # Write to CSV
    print(f"Writing results to {output_file}...")
    result_df.to_csv(output_file, index=False)
    
    # Return the most extreme days for reporting
    extreme_days = find_extreme_days(result_df)
    
    print(f"Successfully calculated deviations for {len(result_df)} days!")
    return result_df, extreme_days

def find_extreme_days(result_df):
    """Find the most extreme days in the dataset based on deviation"""
    # Find days with the highest combined deviation
    extreme_days = {}
    
    # Most extreme day overall (by magnitude of combined deviation)
    extreme_days['highest_combined'] = result_df.loc[result_df['COMBINED_DEV'].idxmax()]
    
    # Most extreme days for each metric (both highest positive and negative deviations)
    for column in ['TMIN', 'TMAX', 'AWND', 'PRCP']:
        extreme_days[f'highest_{column}'] = result_df.loc[result_df[f'{column}_DEV'].idxmax()]
        extreme_days[f'lowest_{column}'] = result_df.loc[result_df[f'{column}_DEV'].idxmin()]
    
    return extreme_days

def main():
    # Configuration
    weather_data_file = "result/weather_data_1990_2024.csv"
    daily_averages_file = "result/daily_averages.csv"
    output_file = "weather_deviations.csv"
    
    try:
        # Calculate daily deviations
        result_df, extreme_days = calculate_daily_deviations(
            weather_data_file, 
            daily_averages_file, 
            output_file
        )
        
        # Display a preview of the results
        print("\nPreview of the deviation data:")
        preview = result_df.head(5)
        for _, row in preview.iterrows():
            date_str = row['DATE'].split()[0] if isinstance(row['DATE'], str) else row['DATE'].strftime('%Y-%m-%d')
            print(f"{date_str}: TMIN_DEV={row['TMIN_DEV']:.2f}, TMAX_DEV={row['TMAX_DEV']:.2f}, " + 
                  f"AWND_DEV={row['AWND_DEV']:.2f}, PRCP_DEV={row['PRCP_DEV']:.2f}, " +
                  f"COMBINED_DEV={row['COMBINED_DEV']:.2f}")
        
        # Report extreme days
        print("\nMost extreme weather days:")
        for key, day in extreme_days.items():
            date_str = day['DATE'].split()[0] if isinstance(day['DATE'], str) else day['DATE'].strftime('%Y-%m-%d')
            dev_metric = key.split('_')[1] + '_DEV'
            if key.startswith('lowest_'):
                print(f"Lowest {key.split('_')[1]}: {date_str} with deviation of {day[dev_metric]:.2f}")
            else:
                print(f"Highest {key.split('_')[1]}: {date_str} with deviation of {day[dev_metric]:.2f}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()