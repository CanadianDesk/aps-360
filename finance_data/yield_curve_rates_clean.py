import pandas as pd

# Define file paths
input_file = "yield-curve-rates-1990-2024.csv"
output_file = "yield-curve-rates-1990-2024-cleaned.csv"

# Load CSV into a DataFrame
df = pd.read_csv(input_file)

# Remove the '2 Mo' column if it exists
if "2 Mo" in df.columns:
    df.drop(columns=["2 Mo"], inplace=True)

# Save the modified DataFrame back to a CSV file
df.to_csv(output_file, index=False)

print(f"File saved without '2 Mo' column as {output_file}")
