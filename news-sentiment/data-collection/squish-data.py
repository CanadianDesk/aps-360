import pandas as pd
import csv
import os

SUM_OUTPUT_DIRECTORY = "../output-sum/" 
AVG_OUTPUT_DIRECTORY = "../output-avg/"
SRC_DIRECTORY = "./labeled-articles/"

def process_CSVs(max_count=1, sum=True):
  count = 0

  output_dir = SUM_OUTPUT_DIRECTORY if sum else AVG_OUTPUT_DIRECTORY

  for filename in os.listdir(SRC_DIRECTORY):
    count += 1
    if count > max_count:
      print(f"Max count reached: {count - 1}")
      break

    if (not filename.endswith(".csv")):
      print(f"Skipping {filename}")
      continue

    df = pd.read_csv(
      os.path.join(SRC_DIRECTORY, filename),
      engine='python',  
      quotechar='"',    
      escapechar='\\',  
      on_bad_lines='warn'
    )

    df['date'] = pd.to_datetime(df['date'])
    
    df = df.sort_values('date')
    
    df = df.drop('headline', axis=1)
    
    if sum:
      df = df.groupby('date').sum().reset_index()
    else:
      df = df.groupby('date').mean().reset_index()

    df['date'] = df['date'].dt.strftime('%Y-%m-%d')

    # rename "label" column to "sentiment"
    df = df.rename(columns={'label': 'sentiment'})
  
    output_file = os.path.join(output_dir, filename)
    df.to_csv(output_file, index=False)
    
    print(f"Processed data saved to {output_file}")

  print(f"Total articles processed: {count - 1}")

if __name__ == "__main__":
  
  process_CSVs(100, True)
  process_CSVs(100, False)