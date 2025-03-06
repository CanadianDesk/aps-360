import json
import requests
import time
import os
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
  "startdatetime": "20250101000000",
  "enddatetime": "20250131000000",
  "format": "json",
  "sort": "hybdridrel"
}

# Base tone parameters - will update for each name
BASE_TONE_PARAMS = {
  "query": "{name} sourcelang:eng",
  "mode": "tonechart",
  "maxrecords": 160,
  "startdatetime": "20250101000000",
  "enddatetime": "20250131000000",
  "format": "json",
  "sort": "hybdridrel"
}

# Directory path for article files
ARTICLES_DIR = "./articles/"

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

def fetch_news(name):
  """Fetch news data with rate-limiting and retries."""
  max_retries = 5
  wait_time = 2  # Initial wait time (in seconds)
  
  # Create query params for this specific name
  query_params = BASE_QUERY_PARAMS.copy()
  query_params["query"] = query_params["query"].format(name=name)

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
      return response.json().get("articles", [])

    except requests.RequestException as e:
      print(f"Request failed (Attempt {attempt + 1}): {e}")
      time.sleep(wait_time)
      wait_time *= 2  # Exponential backoff

  print("Max retries reached. Could not fetch news.")
  return []

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

def get_tones(name):
  """Attempts to get the tones for each article."""
  max_retries = 5
  wait_time = 2  # Initial wait time (in seconds)
  
  # Create tone params for this specific name
  tone_params = BASE_TONE_PARAMS.copy()
  tone_params["query"] = tone_params["query"].format(name=name)

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
      return response.json()
    except requests.RequestException as e:
      print(f"Request failed (Attempt {attempt + 1}): {e}")
      time.sleep(wait_time)
      wait_time *= 2  # Exponential backoff
  
  print("Max retries reached. Could not fetch tone data.")
  return {"tonechart": []}  # Return empty tone chart if failed

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

def clean_articles(articles):
  """Clean up articles by removing duplicates and fixing titles."""
  # Clean up titles (replace double spaces with single spaces)
  cleaned_articles = []
  seen_titles = set()
  
  for article in articles:
    # Clean the title by replacing double spaces with single spaces
    clean_title = article["title"]
    while "  " in clean_title:  # Keep replacing until no double spaces left
      clean_title = clean_title.replace("  ", " ")

    # Clean the title by replacing spaces around commas
    while " , " in clean_title:  
      clean_title = clean_title.replace(" , ", ", ")
    
    article["title"] = clean_title
    
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
  
  return articles_with_tones

def save_news(ticker, news_list):
  """Save news data to file."""
  file_path = os.path.join(ARTICLES_DIR, f"{ticker}.json")
  with open(file_path, "w", encoding="utf-8") as f:
    json.dump(news_list, f, indent=2, ensure_ascii=False)

def process_name(name, ticker):
  """Process a single name - fetch articles, get tones, save results."""
  print(f"\n{'='*50}\nProcessing articles for: {name}\n{'='*50}")

  # Load existing news for this name
  existing_news = load_existing_news(name)

  # Fetch new data
  new_articles = fetch_news(name)

  # Format new articles with cleaned dates
  formatted_articles = [
    {
      "url": article["url"],
      "headline": article["title"],
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
  
  # Get tone data
  tones_data = get_tones(name)
  
  # Add tone information to articles
  all_articles_with_tones = add_tones_to_articles(cleaned_articles, tones_data)
  
  # Filter to only include articles that have a tone
  articles_with_tones = [article for article in all_articles_with_tones if "tone" in article]
  
  # Count how many articles were excluded due to missing tones
  excluded_count = len(all_articles_with_tones) - len(articles_with_tones)
  
  # Sort articles by date (oldest first)
  articles_with_tones.sort(key=lambda x: x["date"], reverse=False)
  
  # Append and sort the combined list by date
  updated_news = existing_news + articles_with_tones
  updated_news.sort(key=lambda x: x["date"], reverse=False)
  
  # Save sorted news
  save_news(ticker, updated_news)

  print(f"Found tones for {len(articles_with_tones)}/{len(cleaned_articles)} articles ({len(articles_with_tones)/len(cleaned_articles)*100:.1f}% if available).")
  print(f"Excluded {excluded_count} articles without tones.")
  print(f"Saved {len(articles_with_tones)} new articles with tones for '{name}'. Total in database: {len(updated_news)}")
  
  return len(articles_with_tones), len(updated_news)

def main():
  """Main function to fetch and store news for all names."""
  # Make sure the articles directory exists
  ensure_directory_exists(ARTICLES_DIR)
  
  # Load names
  names = load_names()
  if not names:
    print("No names found. Please check names.json file.")
    return
  
  print(f"Found {len(names)} names to process.")
  
  # Track statistics
  total_new_articles = 0
  total_articles = 0


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
    ticker = tickers[index] if have_tickers else name
    cleaned_name = name.replace(".com", "").replace("Inc", "")
    new_articles, total_name_articles = process_name(cleaned_name, ticker)
    total_new_articles += new_articles
    total_articles += total_name_articles
  
  print(f"\n{'='*50}")
  print(f"Processing complete. Added {total_new_articles} new articles. Total across all names: {total_articles}")
  print(f"Article files stored in: {ARTICLES_DIR}")

if __name__ == "__main__":
  main()