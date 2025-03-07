import os
import csv
import json
import time
import random
from dotenv import load_dotenv
import google.generativeai as genai
from google.ai.generativelanguage_v1beta.types import content

def load_prompt():
    """Load the prompt template from the markdown file."""
    with open("./prompt.md", "r") as f:
        return f.read()

def read_csv_in_batches(filepath, batch_size=150):
    """Generator to read CSV file in batches."""
    with open(filepath, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        
        batch = []
        for row in reader:
            batch.append(row)
            if len(batch) >= batch_size:
                yield [header] + batch
                batch = []
        
        # Yield the last batch if it exists
        if batch:
            yield [header] + batch

def csv_to_string(csv_data):
    """Convert CSV data to a string format."""
    return "\n".join([",".join(row) for row in csv_data])

def write_results_to_csv(results, output_file, append=False):
    """Write the Gemini results to a CSV file."""
    mode = "a" if append else "w"
    with open(output_file, mode, newline='') as f:
        writer = csv.writer(f)
        
        # Write headers if it's a new file
        if not append:
            writer.writerow(["date", "headline", "label"])
        
        # Write data rows
        for item in results:
            writer.writerow([item["date"], item["headline"], item["label"]])

def generate_with_backoff(prompt, max_retries=5, initial_delay=1):
    """Generate content with exponential backoff for failures."""
    load_dotenv()  # Load environment variables from .env file
    
    # Configure Gemini
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    
    # Define schema for response
    schema = content.Schema(
        type=content.Type.ARRAY,
        items=content.Schema(
            type=content.Type.OBJECT,
            required=["headline", "label", "date"],
            properties={
                "headline": content.Schema(
                    type=content.Type.STRING,
                ),
                "label": content.Schema(
                    type=content.Type.NUMBER,
                ),
                "date": content.Schema(
                    type=content.Type.STRING,
                ),
            },
        ),
    )
    
    # Create generation config
    generation_config = {
        "temperature": 0.3,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_schema": schema,
        "response_mime_type": "application/json",
    }
    
    # Create the model
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        generation_config=generation_config,
    )
    
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            print(f"Sending request (attempt {attempt + 1}/{max_retries})...")
            
            # Generate content
            response = model.generate_content(prompt)
            
            # Check if the response was generated successfully
            if not response.candidates:
                raise Exception(f"No response generated. Safety ratings: {response.prompt_feedback}")
            
            # Get the response text
            response_text = response.candidates[0].content.parts[0].text
            
            # Parse the JSON response
            result = json.loads(response_text)
            print(f"Successfully received response with {len(result)} items")
            return result
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed with error: {e}")
            if attempt < max_retries - 1:
                sleep_time = delay * (random.uniform(0.8, 1.2))  # Add jitter
                print(f"Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
                delay *= 2  # Exponential backoff
            else:
                print("Max retries reached. Giving up.")
                raise
    
    return None

def process_csv():
    """Process the CSV file in batches."""
    prompt_template = load_prompt()
    batch_num = 0
    
    for batch in read_csv_in_batches("./tsla_stripped.csv", batch_size=150):
        # Convert the batch to a string and replace <csv data> in the prompt
        csv_string = csv_to_string(batch)
        current_prompt = prompt_template.replace("<csv data>", csv_string)
        
        # Generate results
        print(f"Processing batch {batch_num + 1} with {len(batch) - 1} rows of data...")
        results = generate_with_backoff(current_prompt)
        
        # Write results to CSV
        write_results_to_csv(results, "./tsla_gemini.csv", append=(batch_num > 0))
        
        batch_num += 1
    
    print(f"CSV processing complete. Processed {batch_num} batches.")

if __name__ == "__main__":
    process_csv()