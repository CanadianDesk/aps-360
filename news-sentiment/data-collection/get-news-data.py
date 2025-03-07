import json
import requests
import time
import os
import csv
import re
from datetime import datetime, timedelta
from urllib.parse import urlencode

# DOCUMENTATION:
# https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/

# API Endpoint
GDELT_URL = "http://api.gdeltproject.org/api/v2/doc/doc"

# Base parameters - will update the query for each name
BASE_QUERY_PARAMS = {
  "query": "{name} sourcelang:eng toneabs<10",
  "mode": "artlist",
  "maxrecords": 160,
  "startdatetime": "{startdate}",
  "enddatetime": "{enddate}",
  "format": "json",
  "sort": "hybdridrel"
}

# Base tone parameters - will update for each name
BASE_TONE_PARAMS = {
  "query": "{name} sourcelang:eng",
  "mode": "tonechart",
  "maxrecords": 160,
  "startdatetime": "{startdate}",
  "enddatetime": "{enddate}",
  "format": "json",
  "sort": "hybdridrel"
}

# Directory path for article files
ARTICLES_DIR = "./articles/"
DATA_CSV = "./tsla.csv"

def load_names():
    """Load names from names.json file."""
    try:
        with open("./names.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error loading names: {e}")
        return []

def ensure_directory_exists(directory):
    """Make sure the directory exists, create if not."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def ensure_csv_exists():
    """Create data.csv if it doesn't exist."""
    if not os.path.exists(DATA_CSV):
        with open(DATA_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(['date','headline', 'label']) #modify back
        print(f"Created CSV file: {DATA_CSV}")

def split_date_range(start_date, end_date, segments=12):
    """Split date range into segments."""
    start_dt = datetime.strptime(start_date, "%Y%m%d%H%M%S")
    end_dt = datetime.strptime(end_date, "%Y%m%d%H%M%S")
    
    # Calculate total duration and segment duration
    total_duration = end_dt - start_dt
    segment_duration = total_duration / segments
    
    date_ranges = []
    for i in range(segments):
        segment_start = start_dt + (segment_duration * i)
        segment_end = start_dt + (segment_duration * (i + 1))
        
        # Convert back to string format
        segment_start_str = segment_start.strftime("%Y%m%d%H%M%S")
        segment_end_str = segment_end.strftime("%Y%m%d%H%M%S")
        
        date_ranges.append((segment_start_str, segment_end_str))
    
    return date_ranges

def fetch_news(name, start_date, end_date):
    """Fetch news data with rate-limiting and retries across multiple date segments."""
    # Split the date range into 12 segments
    date_segments = split_date_range(start_date, end_date, 98)
    all_articles = []
    
    for i, (segment_start, segment_end) in enumerate(date_segments):
        print(f"Fetching segment {i+1}/98: {segment_start} to {segment_end}")
        
        max_retries = 5
        wait_time = 2  # Initial wait time (in seconds)
        
        # Create query params for this specific name and date segment
        query_params = BASE_QUERY_PARAMS.copy()
        query_params["query"] = query_params["query"].format(name=name)
        query_params["startdatetime"] = segment_start
        query_params["enddatetime"] = segment_end

        full_url = f"{GDELT_URL}?{urlencode(query_params)}"
        print(f"Fetching articles for '{name}' from: {full_url}")

        for attempt in range(max_retries):
            try:
                headers = {"User-Agent": "Mozilla/5.0"}
                response = requests.get(full_url, headers=headers, timeout=10)

                if response.status_code == 429:
                    print(f"ERROR 429: RATE LIMIT OR USER AGENT ERROR. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    wait_time *= 2  # Exponential backoff
                    continue
                
                response.raise_for_status()  # Raise error for non-200 responses
                segment_articles = response.json().get("articles", [])
                all_articles.extend(segment_articles)
                
                # Add a sleep to avoid hitting rate limits
                time.sleep(1)
                break  # Exit the retry loop on success

            except requests.RequestException as e:
                print(f"Request failed (Attempt {attempt + 1}): {e}")
                time.sleep(wait_time)
                wait_time *= 2  # Exponential backoff
        else:  # This executes if the for loop completes without breaking
            print(f"Max retries reached for segment {i+1}. Skipping ticker.")
            return [], True  # Return empty list and error flag
    
    print(f"Total articles fetched across all segments: {len(all_articles)}")
    return all_articles, False  # Return articles and no error flag

def load_existing_news(name):
    """Load existing news from file if it exists."""
    file_path = os.path.join(ARTICLES_DIR, f"{name}.json")
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []

def get_tones(name, start_date, end_date):
    """Attempts to get the tones for each article across all date segments."""
    # Split the date range into 12 segments
    date_segments = split_date_range(start_date, end_date, 12)
    all_tone_data = {"tonechart": []}
    
    for i, (segment_start, segment_end) in enumerate(date_segments):
        print(f"Fetching tones for segment {i+1}/12: {segment_start} to {segment_end}")
        
        max_retries = 5
        wait_time = 2  # Initial wait time (in seconds)
        
        # Create tone params for this specific name and date segment
        tone_params = BASE_TONE_PARAMS.copy()
        tone_params["query"] = tone_params["query"].format(name=name)
        tone_params["startdatetime"] = segment_start
        tone_params["enddatetime"] = segment_end

        full_url = f"{GDELT_URL}?{urlencode(tone_params)}"
        print(f"Fetching tones for '{name}' from: {full_url}")

        for attempt in range(max_retries):
            try:
                headers = {"User-Agent": "Mozilla/5.0"}
                response = requests.get(full_url, headers=headers, timeout=10)

                if response.status_code == 429:
                    print(f"ERROR 429: RATE LIMIT OR USER AGENT ERROR. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    wait_time *= 2  # Exponential backoff
                    continue
                
                response.raise_for_status()
                segment_tone_data = response.json()
                
                # Merge the tone chart data
                if segment_tone_data.get("tonechart"):
                    all_tone_data["tonechart"].extend(segment_tone_data["tonechart"])
                
                # Add a sleep to avoid hitting rate limits
                time.sleep(1)
                break  # Exit the retry loop on success
                
            except requests.RequestException as e:
                print(f"Request failed (Attempt {attempt + 1}): {e}")
                time.sleep(wait_time)
                wait_time *= 2  # Exponential backoff
        else:  # This executes if the for loop completes without breaking
            print(f"Max retries reached for tone segment {i+1}. Skipping ticker.")
            return {"tonechart": []}, True  # Return empty tone data and error flag
    
    return all_tone_data, False  # Return tone data and no error flag

def clean_date(date_str):
    """Convert GDELT date format to YYYY-MM-DD."""
    # GDELT typically uses YYYYMMDDhhmmss format
    if len(date_str) >= 8:  # At least has YYYYMMDD
        year = date_str[0:4]
        month = date_str[4:6]
        day = date_str[6:8]
        return f"{year}-{month}-{day}"
    return date_str  # Return unchanged if format not recognized

def scale_tone(tone_value):
    """
    Clamp tone value to [-10, 10] range and scale to [-1, 1].
    """
    # Clamp to [-10, 10] range
    clamped_tone = max(-10, min(10, tone_value))
    
    # Scale to [-1, 1] range
    scaled_tone = clamped_tone / 10.0
    
    return scaled_tone

def fix_text_formatting(text):
    """Fix common formatting issues in article headlines."""
    # Fix spaces around parentheses
    text = re.sub(r'\s*\(\s*', ' (', text)
    text = re.sub(r'\s*\)\s*', ') ', text)
    
    # Fix spaces around colons
    text = re.sub(r'\s*:\s*', ': ', text)
    
    # Fix spaces around periods in URLs and abbreviations
    text = re.sub(r'(\w)\s*\.\s*(\w)', r'\1.\2', text)  # e.g., "KTBB . com" -> "KTBB.com"
    
    # Fix spaces around commas
    text = re.sub(r'\s*,\s*', ', ', text)
    
    # Fix spaces around dashes
    text = re.sub(r'\s*-\s*', ' - ', text)
    
    # Fix spaces around percentage signs
    text = re.sub(r'(\d)\s*%', r'\1%', text)
    
    # Fix spaces around dollar signs
    text = re.sub(r'\$\s*(\d)', r'$\1', text)
    
    # Fix multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def clean_articles(articles):
    """Clean up articles by removing duplicates and fixing titles."""
    # Clean up titles (replace double spaces with single spaces)
    cleaned_articles = []
    seen_titles = set()
    
    for article in articles:
        # Clean the title with regex formatting fixes
        clean_title = fix_text_formatting(article["headline"])
        article["headline"] = clean_title
        
        # Add only if title is not a duplicate
        if clean_title not in seen_titles:
            seen_titles.add(clean_title)
            cleaned_articles.append(article)
    
    return cleaned_articles

def add_tones_to_articles(articles, tones_data):
    """Match articles with tone data and add tone field."""
    # Create a deep copy of articles to avoid modifying the original
    articles_with_tones = [article.copy() for article in articles]
    
    # Create lookup table by URL for our articles
    article_indices_by_url = {article["url"]: i for i, article in enumerate(articles_with_tones)}
    
    # Track which articles have tones
    articles_with_tone_values = []
    
    # Go through each tone bin
    for tone_bin in tones_data.get("tonechart", []):
        bin_value = tone_bin.get("bin")
        
        # Go through each article in this tone bin
        for tone_article in tone_bin.get("toparts", []):
            tone_url = tone_article.get("url", "")
            
            # Try to match by URL
            if tone_url in article_indices_by_url:
                index = article_indices_by_url[tone_url]
                
                # Store only scaled tone
                articles_with_tones[index]["tone"] = scale_tone(bin_value)
                articles_with_tone_values.append(articles_with_tones[index])
    
    return articles_with_tones, articles_with_tone_values

def append_to_data_csv(articles_with_tones):
    """Append articles with tones to data.csv."""
    with open(DATA_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        for article in articles_with_tones:
            writer.writerow([article["date"], article["headline"], article["tone"]]) #modify back

def save_news(ticker, news_list):
    """Save news data to file."""
    file_path = os.path.join(ARTICLES_DIR, f"{ticker}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(news_list, f, indent=2, ensure_ascii=False)

def process_name(name, ticker, start_date, end_date):
    """Process a single name - fetch articles, get tones, save results."""
    print(f"\n{'='*50}\nProcessing articles for: {name}\n{'='*50}")

    # Load existing news for this name
    existing_news = load_existing_news(ticker)

    # Fetch new data across all segments
    new_articles, news_error = fetch_news(name, start_date, end_date)
    
    # If we encountered an error, skip this ticker
    if news_error:
        print(f"Skipping '{name}' due to persistent errors in fetching news.")
        return 0, 0

    # Format new articles with cleaned dates
    formatted_articles = [
        {
            "url": article["url"],
            "headline": fix_text_formatting(article["title"]),  # Apply formatting at creation
            "date": clean_date(article["seendate"]),
        }
        for article in new_articles
    ]

    if not formatted_articles:
        print(f"No articles found for '{name}'. Skipping...")
        return 0, 0

    # Clean up articles (remove duplicates, fix titles)
    cleaned_articles = clean_articles(formatted_articles)
    removed_count = len(formatted_articles) - len(cleaned_articles)
    print(f"Cleaned {removed_count} duplicate articles. Remaining: {len(cleaned_articles)}")
    
    # Get tone data across all segments
    tones_data, tones_error = get_tones(name, start_date, end_date)
    
    # If we encountered an error, skip this ticker
    if tones_error:
        print(f"Skipping '{name}' due to persistent errors in fetching tones.")
        return 0, 0
    
    # Add tone information to articles
    all_articles_with_tones, articles_with_tone_values = add_tones_to_articles(cleaned_articles, tones_data)
    
    # Count how many articles got tones
    toned_count = len(articles_with_tone_values)
    
    # Append articles with tones to data.csv
    if toned_count > 0:
        append_to_data_csv(articles_with_tone_values)
        print(f"Added {toned_count} articles with tones to data.csv")
    
    # Sort articles by date (oldest first)
    all_articles_with_tones.sort(key=lambda x: x["date"], reverse=False)
    
    # Append and sort the combined list by date
    updated_news = existing_news + all_articles_with_tones
    updated_news.sort(key=lambda x: x["date"], reverse=False)
    
    # Save sorted news
    save_news(ticker, updated_news)

    print(f"Found tones for {toned_count}/{len(cleaned_articles)} articles ({toned_count/len(cleaned_articles)*100:.1f}% if available).")
    print(f"Saved {len(cleaned_articles)} new articles for '{name}'. Total in database: {len(updated_news)}")
    
    return len(cleaned_articles), len(updated_news)

def main(start_date="20240101000000", end_date="20250101000000"):
    """Main function to fetch and store news for all names within the date range."""
    # Make sure the articles directory exists
    ensure_directory_exists(ARTICLES_DIR)
    ensure_csv_exists()
    
    # Load names
    names = load_names()
    if not names:
        print("No names found. Please check names.json file.")
        return
    
    print(f"Found {len(names)} names to process.")
    print(f"Processing data from {start_date} to {end_date}")
    
    # Track statistics
    total_new_articles = 0
    total_articles = 0
    skipped_tickers = []

    have_tickers = False
    tickers = []
    try:
        with open("./tickers.json", "r", encoding="utf-8") as f:
            tickers = json.load(f)
            have_tickers = True
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error loading tickers: {e}")
          
    # Process each name
    for index, name in enumerate(names):
        ticker = tickers[index] if have_tickers and index < len(tickers) else name
        cleaned_name = name.replace(".com", "").replace("Inc", "")
        
        try:
            new_articles, total_name_articles = process_name(cleaned_name, ticker, start_date, end_date)
            
            # If both returned 0, it might have been skipped due to errors
            if new_articles == 0 and total_name_articles == 0:
                skipped_tickers.append(ticker)
            
            total_new_articles += new_articles
            total_articles += total_name_articles
        except Exception as e:
            print(f"Unexpected error processing {ticker}: {e}")
            skipped_tickers.append(ticker)
    
    print(f"\n{'='*50}")
    print(f"Processing complete. Added {total_new_articles} new articles. Total across all names: {total_articles}")
    if skipped_tickers:
        print(f"Skipped {len(skipped_tickers)} tickers due to errors: {', '.join(skipped_tickers)}")
    print(f"Article files stored in: {ARTICLES_DIR}")
    print(f"Articles with tones saved to: {DATA_CSV}")

if __name__ == "__main__":
    # Call main with default date range or customize as needed
    main(start_date="20170101000000", end_date="20250301000000")