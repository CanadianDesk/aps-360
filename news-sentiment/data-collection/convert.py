import csv

# Path to input and output files
input_file = './tsla.csv'
output_file = './tsla_stripped.csv'

# Read the input CSV and write to the output CSV with empty labels
with open(input_file, 'r', newline='', encoding='utf-8') as infile, \
     open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    
    # Create CSV reader and writer
    reader = csv.DictReader(infile)
    writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
    
    # Write the header row
    writer.writeheader()
    
    # Process each row
    for row in reader:
        # Empty the "label" column
        row['label'] = ''
        # Write the modified row
        writer.writerow(row)

print(f"Successfully created {output_file} with empty labels.")