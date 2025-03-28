import os
import cv2
import pandas as pd
import numpy as np

def csv_to_grayscale_image(csv_path, output_dir):
    """
    Reads a CSV file, drops the first column (assumed to be date),
    and converts the remaining numeric data into a grayscale image using OpenCV.
    """
    # Read the CSV
    df = pd.read_csv(csv_path)

    # Drop the first column (e.g., a date column)
    # Using 'errors=ignore' to safely drop even if it doesn't exist
    df_numeric = df.drop(columns=df.columns[0], errors='ignore')

    # Convert the dataframe to a numpy array (float32)
    data = df_numeric.to_numpy(dtype=np.float32)

    # Handle the case where the data might be 1D (single numeric column)
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    # Clip values to [0, 1] to avoid out-of-range data
    data = np.clip(data, 0, 1)

    # Scale to [0..255] and convert to uint8 for grayscale image
    data = (data * 255).astype(np.uint8)

    # Construct the output file path
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    output_file = os.path.join(output_dir, base_name + ".png")

    # Write the image using OpenCV
    try:
        cv2.imwrite(output_file, data)
    except Exception as e:
        print(f"Error saving image {output_file}: {e}")
        return
    print(f"Saved grayscale image: {output_file}")

def process_directory(input_dir, output_dir):
    """
    Recursively scans 'input_dir' for CSV files and converts each one
    into a grayscale image stored in 'output_dir'.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.csv'):
                csv_path = os.path.join(root, file)
                csv_to_grayscale_image(csv_path, output_dir)

if __name__ == "__main__":
    input_directory = "./cached_images"  # Change this to your input directory
    output_directory = "./data_images"  # Change this to your output directory
    process_directory(input_directory, output_directory)
