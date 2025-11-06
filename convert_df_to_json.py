import pandas as pd
import json
import numpy as np

# --- 1. CONFIGURATION: UPDATE THESE VALUES ---

# UPDATE THIS: The path to your CSV or Excel file that has all the movie data.
DATA_SOURCE_FILE = 'dataframe_backup.csv' # 👈 e.g., 'movies_metadata.csv'

# UPDATE THESE: The exact column names from your file.
TITLE_COLUMN = 'original_title'           # 👈 Or 'original_title', 'name', etc.
POSTER_COLUMN = 'poster_path'    # 👈 The name of your poster path column
RATING_COLUMN = 'rate'           # 👈 The name of your rating column
RELEASE_DATE_COLUMN = 'date' # 👈 The name of your release date column

# UPDATE THIS: The base URL for your poster images.
# If your poster_path is just a fragment like '/xyz.jpg', this is needed.
# If your column already has the full URL, set this to ""
POSTER_BASE_URL = "https://image.tmdb.org/t/p/w500"

# This is the final file your Streamlit app will use.
OUTPUT_JSON_FILE = 'movie_details.json'

# --- 2. SCRIPT (No need to edit below) ---

print(f"Loading data from '{DATA_SOURCE_FILE}'...")
try:
    if DATA_SOURCE_FILE.endswith('.csv'):
        df = pd.read_csv(DATA_SOURCE_FILE)
    elif DATA_SOURCE_FILE.endswith('.xlsx'):
        df = pd.read_excel(DATA_SOURCE_FILE)
    else:
        raise ValueError("Unsupported file type. Please use .csv or .xlsx")
except FileNotFoundError:
    print(f"❌ ERROR: File not found: '{DATA_SOURCE_FILE}'.")
    print("Please update the DATA_SOURCE_FILE variable at the top of the script.")
    exit()
except Exception as e:
    print(f"❌ ERROR loading file: {e}")
    exit()

print(f"Successfully loaded {len(df)} rows.")
df = df.replace({np.nan: None}) # Replace pandas 'nan' with standard 'None'

# Create the main dictionary
movie_details_map = {}

print("Processing rows and building the details map...")
for index, row in df.iterrows():
    try:
        # Get the data from the row
        title = str(row[TITLE_COLUMN])
        poster_path = str(row[POSTER_COLUMN]) if row[POSTER_COLUMN] is not None else ""
        rating = row[RATING_COLUMN]
        release_date = str(row[RELEASE_DATE_COLUMN]) if row[RELEASE_DATE_COLUMN] is not None else ""

        # --- Build the full poster URL ---
        if not poster_path:
            full_poster_url = None # Or a placeholder image URL
        elif poster_path.startswith('http'):
            full_poster_url = poster_path # It's already a full URL
        elif POSTER_BASE_URL:
            full_poster_url = f"{POSTER_BASE_URL}{poster_path}" # Prepend the base URL
        else:
            full_poster_url = poster_path # Use as-is

        # Create the sub-dictionary in the format our app needs
        details = {
            "poster_url": full_poster_url,
            "rating": rating,
            "release_date": release_date
        }
        
        # Add it to the main map, keyed by the movie title
        movie_details_map[title] = details

    except KeyError as e:
        print(f"\n❌ ERROR: A column name is wrong. Could not find column: {e}")
        print("Please update the_COLUMN variables at the top of the script.")
        exit()
    except Exception as e:
        print(f"\n⚠️ Warning: Skipping row {index} (Title: {title}) due to error: {e}")

# --- 3. Save the complete dictionary to the JSON file ---
try:
    with open(OUTPUT_JSON_FILE, 'w', encoding='utf-8') as f:
        json.dump(movie_details_map, f, indent=4, ensure_ascii=False)
    print(f"\n✅ Success! All details saved to '{OUTPUT_JSON_FILE}'.")
    print(f"Total movies in file: {len(movie_details_map)}")
except Exception as e:
    print(f"\n❌ ERROR: Could not save the final file. Reason: {e}")