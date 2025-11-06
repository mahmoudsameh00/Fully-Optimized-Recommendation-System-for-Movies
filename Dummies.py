# Dummies2.py (Optimized for Real-Time)

import json
import numpy as np

# --- 1. Load pre-computed assets ONCE at import time ---
try:
    with open('language_columns.json', 'r') as f:
        LANGUAGE_COLUMNS_LIST = json.load(f)
    
    with open('genre_columns.json', 'r') as f:
        GENRE_COLUMNS_LIST = json.load(f)

except FileNotFoundError:
    print("FATAL ERROR: 'language_columns.json' or 'genre_columns.json' not found.")
    print("Please run the 'build_preprocessing_assets.py' script first.")
    # Set to empty lists to avoid crashing the import, but functions will fail
    LANGUAGE_COLUMNS_LIST = []
    GENRE_COLUMNS_LIST = []


# --- 2. Create efficient lookups (dictionaries) ONCE at import time ---

# This maps 'original_language_en' -> 0, 'original_language_fr' -> 1, etc.
LANGUAGE_TO_INDEX = {col_name: i for i, col_name in enumerate(LANGUAGE_COLUMNS_LIST)}

# This maps 'genre_Action' -> 0, 'genre_Adventure' -> 1, etc.
GENRE_TO_INDEX = {col_name: i for i, col_name in enumerate(GENRE_COLUMNS_LIST)}

NUM_LANGUAGES = len(LANGUAGE_COLUMNS_LIST)
NUM_GENRES = len(GENRE_COLUMNS_LIST)


# --- 3. Define the optimized functions ---

def selectdummyLanguage(InputLanguage):
    """
    Generates a one-hot encoded language vector using pre-computed lookups.
    Returns a numpy array.
    """
    # Create a zero array
    lang_vector = np.zeros(NUM_LANGUAGES, dtype='float32')
    
    # Format the input key
    InputLanguageKey = 'original_language_' + InputLanguage
    
    # Find the index for this language
    if InputLanguageKey in LANGUAGE_TO_INDEX:
        idx = LANGUAGE_TO_INDEX[InputLanguageKey]
        lang_vector[idx] = 1.0
            
    return lang_vector # Return the numpy array directly


def selectdummyGenre(InputGenres):
    """
    Generates a multi-hot encoded genre vector using pre-computed lookups.
    Returns a numpy array.
    """
    # Create a zero array
    genre_vector = np.zeros(NUM_GENRES, dtype='float32')
    
    # Iterate over the input list (e.g., ['Action', 'Thriller'])
    for genre in InputGenres:
        InputGenreKey = 'genre_' + genre
        
        # Find the index for this genre
        if InputGenreKey in GENRE_TO_INDEX:
            idx = GENRE_TO_INDEX[InputGenreKey]
            genre_vector[idx] = 1.0
                
    return genre_vector # Return the numpy array directly