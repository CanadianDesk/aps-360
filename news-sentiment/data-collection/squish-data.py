import pandas as pd
import csv

def process_csv(input_file, output_file):
    # Read the CSV file with more robust parsing options
    df = pd.read_csv(
        input_file,
        engine='python',  # More flexible parser
        quotechar='"',    # Specify quote character
        escapechar='\\',  # Handle escaped characters
        on_bad_lines='warn'  # Warn about problematic lines
    )
    
    # Convert date column to datetime for proper sorting
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort by date
    df = df.sort_values('date')
    
    # Drop the headline column
    df = df.drop('headline', axis=1)
    
    # Group by date and sum the labels
    df = df.groupby('date').sum().reset_index()
    
    # Convert date back to string format YYYY-MM-DD
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    
    # Save to a new CSV file
    df.to_csv(output_file, index=False)
    
    print(f"Processed data saved to {output_file}")

# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python script.py input_file.csv output_file.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    process_csv(input_file, output_file)