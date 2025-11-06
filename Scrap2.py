# ... all your other imports (pandas, numpy, etc.) ...
import os
from datetime import datetime
import time  # <-- Add time
import requests # <-- Add requests
import Rate  # <-- Make sure Rate.py is available

# --- Add your API Key ---
API_KEY = os.environ.get('API_KEY')

def fetch_movie_data_for_processing(title_query):
    """
    Searches for a movie and returns all data needed for processing.
    This one function replaces Scrap.Search().
    Returns a tuple: (original_title, popularity, runtime, date_str, rate, overview, lang, genres)
    Returns (None, None, ...) on failure.
    """
    print(f"API: Searching for '{title_query}'...")
    
    # 1. Search for the movie to get its ID
    search_url = f'https://api.themoviedb.org/3/search/movie'
    search_params = {'api_key': API_KEY, 'query': title_query}
    
    try:
        search_resp = requests.get(search_url, params=search_params)
        search_resp.raise_for_status()
        search_data = search_resp.json()
        
        if not search_data['results']:
            print(f"API: No results found for '{title_query}'")
            
        movie_id = search_data['results'][0]['id']
        
        # 2. Get the full details for that movie ID
        details_url = f'https://api.themoviedb.org/3/movie/{movie_id}'
        details_params = {'api_key': API_KEY}
        
        details_resp = requests.get(details_url, params=details_params)
        details_resp.raise_for_status()
        data = details_resp.json()

        # 3. Extract all the data you need for your model
        original_title = data.get('title')
        popularity = data.get('popularity', 0)
        runtime = data.get('runtime', 0)
        release_date = data.get('release_date', '2000-01-01') # YYYY-MM-DD
        overview = data.get('overview', '')
        original_language = data.get('original_language', 'en')
        
        # Get genres as a list of names
        genres = [g['name'] for g in data.get('genres', [])]
        
        # Calculate rate using your custom module
        rate = Rate.rate(data.get('vote_count', 0), data.get('vote_average', 0))
        
        print(f"API: Found '{original_title}'")
        return [(original_title),(genres),(original_language),(overview),(popularity),(release_date),(runtime),(rate)]

    except Exception as e:
        print(f"API Error: {e}")
    


# Scrap2.py

# ... (all other code, including fetch_movie_data_for_processing, stays the same) ...

# --- THIS FUNCTION IS REPLACED ---
def fetch_movie_data_for_display(title_query, dataset_release_year=None, needs_poster=True, needs_rating=True, needs_release_date=True):
    """
    Searches for a movie and returns *only* the details that are needed.
    If dataset_release_year is provided, it will try to find a movie matching that year.
    """
    if not API_KEY:
        print("API: Skipping fetch, TMDB_API_KEY is not set.")
        return {}
        
    print(f"API: Getting specific details for '{title_query}' (Target Year: {dataset_release_year})...")
    
    time.sleep(0.25) 
    
    search_url = f'https://api.themoviedb.org/3/search/movie'
    search_params = {'api_key': API_KEY, 'query': title_query}
    
    try:
        search_resp = requests.get(search_url, params=search_params)
        search_resp.raise_for_status()
        search_data = search_resp.json()
        
        if not search_data['results']:
            print(f"API: No results for '{title_query}'")
            return {}
            
        # --- NEW FILTERING LOGIC ---
        target_result = None
        
        # If no year is provided, just use the first result (old behavior)
        if not dataset_release_year:
            print("API: No target year provided, using first result.")
            target_result = search_data['results'][0]
        else:
            print(f"API: Searching for exact year {dataset_release_year}...")
            top_results = search_data['results'][:6] # Get top results once

            for result in top_results:
                api_date = result.get('release_date', '') # e.g., '1995-12-15'
                
                # Check if the API date string contains our target year
                if api_date and str(dataset_release_year) in api_date:
                    print(f"API: Found year match for '{title_query}' ({dataset_release_year})")
                    target_result = result
                    break # Stop on the first match

            # --- Secondary Search: +/- 1 Year (Runs ONLY if no exact match was found) ---
            if not target_result:
                year_before = str(dataset_release_year - 1)
                year_after = str(dataset_release_year + 1)
                print(f"API: No exact match. Checking for {year_before} or {year_after}...")
                
                for result in top_results: # Loop through the same top results again
                    api_date = result.get('release_date', '')
                    
                    # Check for the year before OR the year after
                    if api_date and (year_before in api_date or year_after in api_date):
                        matched_year = year_before if year_before in api_date else year_after
                        print(f"API: Found close year match for '{title_query}' (found {matched_year})")
                        target_result = result
                        break # Stop on the first close match
            
            if not target_result:
                print(f"API: No result for '{title_query}' matched year {dataset_release_year}. Aborting fetch.")
                return {} # Return nothing if no year match is found
        
        # --- END NEW LOGIC ---

        # This is the data we will return
        fetched_data = {
            "title": target_result.get('title', title_query) 
        }

        # --- ORIENTED FETCHING (using target_result) ---
        if needs_poster:
            poster_path = target_result.get('poster_path')
            poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else "https://via.placeholder.com/500x750.png?text=No+Image"
            fetched_data["poster_url"] = poster_url
        
        if needs_release_date:
            date_string = target_result.get('release_date', '') # e.g., "2002-05-01"

            if date_string and '-' in date_string:
                year = date_string.split('-')[0] # Get just "2002"
                fetched_data["release_date"] = year
            else:
                fetched_data["release_date"] = "N/A"
        
        if needs_rating:
            rating = Rate.rate(target_result.get('vote_count', 0), target_result.get('vote_average', 0))
            rating = round(rating, 1) if rating else "N/A"
            fetched_data["rating"] = rating

        return fetched_data
        
    except Exception as e:
        print(f"API Error fetching display details: {e}")
        return {}