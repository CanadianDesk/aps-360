import pandas as pd

def process_csv(input_file, output_file):
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Convert date column to datetime for proper sorting
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort by date
    df = df.sort_values('date')
    
    # Drop the headline column
    df = df.drop('headline', axis=1)
    
    # Group by date and average the labels
    df = df.groupby('date').mean().reset_index()
    
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