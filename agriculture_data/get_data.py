import requests
import json
import csv
import time
from datetime import datetime, timedelta
from collections import defaultdict

def fetch_weather_data(start_date, end_date, station_id, datatypes, token, offset=1, limit=1000):
    """
    Fetch weather data from NCEI API with pagination
    """
    base_url = "https://www.ncei.noaa.gov/cdo-web/api/v2/data"
    
    # Build URL with parameters
    url = f"{base_url}?datasetid=GHCND&stationid={station_id}&startdate={start_date}&enddate={end_date}&limit={limit}&offset={offset}"
    
    # Add each datatype to the URL
    for datatype in datatypes:
        url += f"&datatypeid={datatype}"
    
    headers = {
        'token': token
    }
    
    print(f"Requesting data from: {url}")
    response = requests.request("GET", url, headers=headers, data={})
    
    if response.status_code != 200:
        raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
    
    return response.json()

def fetch_all_data_for_year(start_date, end_date, station_id, datatypes, token):
    """
    Fetch all data for a year using pagination
    """
    all_results = []
    offset = 1
    limit = 1000  # API's maximum limit
    
    while True:
        # Fetch a batch of data
        json_data = fetch_weather_data(start_date, end_date, station_id, datatypes, token, offset, limit)
        
        # Get results and metadata
        results = json_data.get("results", [])
        metadata = json_data.get("metadata", {}).get("resultset", {})
        
        # Add results to our collection
        all_results.extend(results)
        
        print(f"Retrieved {len(results)} data points (offset: {offset})")
        
        # Check if we've reached the end
        count = metadata.get("count", 0)
        if not results or len(all_results) >= metadata.get("count", 0) or len(results) < limit:
            print(f"Completed fetching data. Total data points: {len(all_results)}")
            break
        
        # Increment offset for next batch
        offset += limit
        
        # Pause to avoid rate limiting
        time.sleep(0.5)
    
    # Create a combined JSON response
    return {"results": all_results}

def process_data(json_data):
    """
    Process JSON data into a format suitable for CSV
    """
    # Create a dictionary to hold data organized by date
    data_by_date = defaultdict(dict)
    
    # Process each result
    results = json_data.get("results", [])
    print(f"Processing {len(results)} data points")
    
    # Group data points by date
    unique_dates = set()
    for item in results:
        # Extract date (remove time component)
        date_str = item["date"].split("T")[0]
        unique_dates.add(date_str)
        datatype = item["datatype"]
        value = item["value"]*.05
        value = round(value, 4)
        
        # Store data by date and datatype
        data_by_date[date_str][datatype] = value
    
    print(f"Found data for {len(unique_dates)} unique dates")
    return data_by_date

def write_to_csv(data_by_date, output_file, datatypes, append=False):
    """
    Write processed data to CSV file
    """
    # Determine whether to write or append
    mode = 'a' if append else 'w'
    
    with open(output_file, mode, newline='') as csvfile:
        # Create CSV writer
        writer = csv.writer(csvfile)
        
        # Write header row only if we're not appending
        if not append:
            header = ["DATE"] + datatypes
            writer.writerow(header)
        
        # Write data rows
        for date in sorted(data_by_date.keys()):
            row = [date]
            for datatype in datatypes:
                row.append(data_by_date[date].get(datatype, ""))
            writer.writerow(row)
    
    return len(data_by_date)

def process_year(year, station_id, datatypes, token, output_file, append=False):
    """
    Process an entire year of data
    """
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    
    print(f"Processing year {year}: {start_date} to {end_date}")
    
    # Fetch all data for this year using pagination
    json_data = fetch_all_data_for_year(start_date, end_date, station_id, datatypes, token)
    
    # Process the data
    data_by_date = process_data(json_data)
    
    # Write to CSV
    days_written = write_to_csv(data_by_date, output_file, datatypes, append=append)
    
    print(f"Successfully wrote {days_written} days of data for {year}")
    return days_written

def main():
    # Configuration
    start_year = 1990
    end_year = 2024
    station_id = "GHCND:USW00014944"
    datatypes = ["TMIN", "TMAX", "AWND", "PRCP"]
    token = "qeVHxrWdKHdyPAIjohZjtIaqzZdphqde"
    output_file = "weather_data_1990_2024.csv"
    
    total_days = 0
    
    try:
        print(f"Fetching weather data from {start_year} to {end_year}")
        
        # Process each year
        for i, year in enumerate(range(start_year, end_year + 1)):
            try:
                # Process this year and write to CSV
                days_written = process_year(year, station_id, datatypes, token, output_file, append=(i > 0))
                total_days += days_written
            except Exception as e:
                print(f"Error processing year {year}: {e}")
                print("Continuing with next year...")
            
            # Add delay between years to avoid rate limiting
            if year < end_year:
                print(f"Waiting before processing next year...")
                time.sleep(2)
        
        print(f"Successfully created {output_file} with {total_days} days of weather data.")
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()