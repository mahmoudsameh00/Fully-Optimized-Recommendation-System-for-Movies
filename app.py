import json
import os
import threading
import requests
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import faiss
import streamlit as st
from streamlit_tags import st_tags
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

# --- Custom Modules ---
import Dummies
import Scrap2

# --- 1. RESOURCE LOADING ---

@st.cache_resource
def load_resources():
    """Loads models, FAISS index, and dataset. Ensures synchronization between CSV and Index."""
    Scrap2.API_KEY = os.environ.get('API_KEY')
    encoder = tf.keras.models.load_model("encoder_model_last.keras")
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    index = faiss.read_index('prebuilt_index.faiss')
    preprocessor = joblib.load('preprocessor.pkl')
    
    # Load movie details for display
    try:
        with open('movie_details.json', 'r', encoding='utf-8') as f:
            all_movie_details = json.load(f)
    except FileNotFoundError:
        st.error("FATAL ERROR: 'movie_details.json' not found.")
        all_movie_details = {}

    # Load embeddings data
    try:
        df_embed = pd.read_csv('embeddings.csv')
    except FileNotFoundError:
        st.error("FATAL ERROR: 'embeddings.csv' not found.")
        df_embed = pd.DataFrame(columns=['Title'])

    # Generate title lists and maps to ensure index alignment
    all_titles = list(df_embed['Title'])
    all_titles_lower = set(t.lower() for t in all_titles)
    lower_to_canonical_title = {t.lower(): t for t in all_titles}
    
    # Sanity check: FAISS index size must match DataFrame length
    if index.ntotal != len(all_titles):
        st.error(f"FATAL SYNC ERROR: Index has {index.ntotal} items, CSV has {len(all_titles)} items.")
    
    # Optimize lookup: Create a dictionary for title -> embedding
    title_to_embedding = {}
    all_embedding_cols = list(df_embed.columns[1:])
    
    for _, row in df_embed.iterrows():
        title_lower = row['Title'].lower()
        # Ensure unique entries (mimics .head(1) behavior)
        if title_lower not in title_to_embedding:
            embedding = row[all_embedding_cols].values.astype('float32').reshape(1, -1)
            title_to_embedding[title_lower] = embedding

    return encoder, sbert_model, index, all_titles, preprocessor, all_movie_details, df_embed, all_titles_lower, title_to_embedding, lower_to_canonical_title

# --- 2. HELPER FUNCTIONS ---

def is_url_valid(url):
    """Checks if a URL returns a 200 OK status and is an image."""
    if not url or not url.startswith('http'):
        return False
    try:
        response = requests.head(url, timeout=3, allow_redirects=True)
        if response.status_code == 200 and 'image' in response.headers.get('Content-Type', ''):
            return True
        else:
            print(f"URL check failed (Status: {response.status_code}): {url}")
            return False
    except requests.exceptions.RequestException:
        return False

def generate_colored_star_rating(rating, max_stars=5):
    """Generates a string of colored HTML stars based on rating."""
    try:
        rating_out_of_5 = float(rating) / 2.0
    except (ValueError, TypeError):
        return "N/A"

    full_stars = round(rating_out_of_5)
    empty_stars = max_stars - full_stars
    gold_star = "<span style='color: gold;'>★</span>"
    gray_star = "<span style='color: lightgray;'>☆</span>"
    star_string = (gold_star * full_stars) + (gray_star * empty_stars)
    
    return star_string

def save_updates_to_disk(original_title, df_embed, all_movie_details, index,
                         csv_filename, json_filename, index_filename):
    """Background thread function to save memory state to disk."""
    print(f"BACKGROUND: Saving '{original_title}' to disk...")
    try:
        df_embed.to_csv(csv_filename, index=False, encoding='utf-8')
        print(f"BACKGROUND: Saved {csv_filename}")

        faiss.write_index(index, index_filename)
        print(f"BACKGROUND: Saved {index_filename}")

        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(all_movie_details, f, indent=4)
        print(f"BACKGROUND: Saved {json_filename}")

    except Exception as e:
         print(f"BACKGROUND ERROR saving updates: {e}")


# --- 3. CORE LOGIC ---

def auto_processing_for_movie(movie_title, encoder, sbert_model):
    """Generates the combined feature embedding for a single movie title."""
    try:
        d = movie_title
        original_title, genres, lang, overview, popularity, date_str, runtime, rate = d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]
        
        # Define constraints
        POPULARITY_LIMIT = 12.0
        RUNTIME_MIN = 60.0
        RUNTIME_MAX = 200.0

        # Clip numerical values
        if popularity > POPULARITY_LIMIT: popularity = POPULARITY_LIMIT
        if runtime < RUNTIME_MIN: runtime = RUNTIME_MIN
        elif runtime > RUNTIME_MAX: runtime = RUNTIME_MAX

        # Feature Engineering
        def dateSTR(x): return 'Old' if x < 2000 else 'Modern' if 2000 < x < 2010 else 'New'
        date_val = int(date_str.split('-')[0])
        date_enc = np.array(dateSTR(date_val)).reshape(1, -1)
        
        lang_enc = Dummies.selectdummyLanguage(lang).reshape(1, -1)
        genres_enc = Dummies.selectdummyGenre(genres).reshape(1, -1)
        popularity = np.array(popularity).reshape(1, -1).astype('float32')
        runtime = np.array(runtime).reshape(1, -1).astype('float32')
        rate = np.array(rate).reshape(1, -1).astype('float32')

        # Prepare Numerical/Categorical Input
        arrvalues = np.hstack([popularity, runtime, rate, date_enc, lang_enc, genres_enc])
        cols = ['popularity', 'runtime', 'rate', 'date'] + Dummies.LANGUAGE_COLUMNS_LIST + Dummies.GENRE_COLUMNS_LIST
        prepared_data = pd.DataFrame(arrvalues, columns=cols)
        
        # Generate Embeddings
        preprocessed_data = preprocessor.transform(prepared_data).astype('float32')
        inval_emb = encoder.predict(preprocessed_data)
        inval_emb = normalize(inval_emb, norm='l2', axis=1) * 3.5  # Weighting
        
        intxt_emb = sbert_model.encode(overview).reshape(1, -1) * 1.0 # Weighting
        input_combined_embedding = np.hstack([intxt_emb, inval_emb])
        
        return original_title, input_combined_embedding
    except Exception as e:
        print(f"CRITICAL: Failed to process '{original_title}'. Error: {e}")
        return None, None

def auto_updating_for_database(original_title, input_combined_embedding,
                               all_movie_details, all_titles, new_display_data, df_embed, index,
                               csv_filename="embeddings.csv", json_filename="movie_details.json", 
                               index_filename="prebuilt_index.faiss"):
    """Updates IN-MEMORY resources and spawns a background thread to save to disk."""
    if original_title in df_embed['Title'].values:
        print(f"'{original_title}' already exists. Skipping update.")
        return df_embed

    # Prepare new row
    new_embedding_values = input_combined_embedding.flatten().tolist()
    new_row_data = [original_title] + new_embedding_values

    # Update In-Memory Data
    try:
        df_embed.loc[len(df_embed)] = new_row_data
        
        new_embedding_normalized = input_combined_embedding.copy().astype('float32')
        faiss.normalize_L2(new_embedding_normalized)
        index.add(new_embedding_normalized)
        
        all_movie_details[original_title] = new_display_data
        all_titles.append(original_title)
        print(f"Successfully added '{original_title}' to memory.")
    except Exception as e:
        st.error(f"Failed to update in-memory database: {e}")
        return df_embed

    # Spawn Background Save Thread
    save_thread = threading.Thread(
        target=save_updates_to_disk,
        args=(original_title, df_embed.copy(), all_movie_details.copy(), index, csv_filename, json_filename, index_filename)
    )
    save_thread.start()
    return df_embed

def get_recommendations(movie_title, input_combined_embedding, index, all_titles, all_movie_details):
    """Performs FAISS search and returns recommended movies with display details."""
    print(f"Searching recommendations for: {movie_title}")
    
    k = 6 
    faiss.normalize_L2(input_combined_embedding)
    distances, indices = index.search(input_combined_embedding, k)
    cosine_similarities = 1 - distances / 2.0
    
    recommended_movies = []
    for i, similarity in zip(indices[0], cosine_similarities[0]):
        title = all_titles[i]
        
        if title.lower() != movie_title.lower():
            movie_details = all_movie_details.get(title, {}).copy()
            movie_details['title'] = title 

            # Check for missing display data
            poster_url = movie_details.get("poster_url")
            rating_val = movie_details.get("rating")
            date_val = movie_details.get("release_date")
            
            placeholder_img = "https://via.placeholder.com/500x750.png?text=No+Image"
            needs_poster = (poster_url == placeholder_img or not is_url_valid(poster_url))
            needs_rating = (rating_val is None or rating_val == "N/A")
            needs_release_date = (date_val is None or date_val == "N/A" or not date_val)
            
            # Extract year hint for scraper
            dataset_year_hint = None
            if date_val and isinstance(date_val, str):
                try:
                    dataset_year_hint = int(date_val) 
                except (ValueError, IndexError, TypeError):
                    dataset_year_hint = None

            # Patch missing data if necessary
            if needs_poster or needs_rating or needs_release_date:
                print(f"Patching missing data for '{title}'...")
                try:
                    fetched_data = Scrap2.fetch_movie_data_for_display(
                        title, 
                        dataset_release_year=dataset_year_hint,
                        needs_poster=needs_poster, 
                        needs_rating=needs_rating, 
                        needs_release_date=needs_release_date
                    )
                    if fetched_data:
                        movie_details.update(fetched_data)
                        all_movie_details[title].update(fetched_data)
                except Exception as e:
                    print(f"Error patching data for '{title}': {e}")
            
            recommended_movies.append(movie_details)
            
    return recommended_movies[:5]

# --- 4. STREAMLIT UI ---

st.set_page_config(layout="wide")
st.title("🎬 AI Movie Recommendation System")

# Load Resources
encoder, sbert_model, index, all_titles, preprocessor, all_movie_details, df_embed, all_titles_lower, title_to_embedding, lower_to_canonical_title = load_resources()

# Session State
if 'last_submitted_movie' not in st.session_state:
    st.session_state.last_submitted_movie = []

# Search Input
selected_movie_list = st_tags(
    label='Enter a Movie Title and press Enter to search:',
    text=" ",
    value=[],
    suggestions=all_titles,
    maxtags=1
)

input_has_changed = (st.session_state.last_submitted_movie != selected_movie_list)

if input_has_changed:
    st.session_state.last_submitted_movie = selected_movie_list

    if selected_movie_list:
        selected_movie = selected_movie_list[0]
        selected_movie_lower = selected_movie.lower()
        
        st.write("---")
        st.subheader(f"Recommendations for '{selected_movie}'")
        
        existing_title = None
        input_combined_embedding = None

        # --- BRANCH 1: EXISTING MOVIE ---
        if selected_movie_lower in all_titles_lower:
            st.info("Found movie in database...")
            existing_title = lower_to_canonical_title.get(selected_movie_lower)
            input_combined_embedding = title_to_embedding.get(selected_movie_lower)
            
            if input_combined_embedding is None:
                 st.error(f"Sync Error: '{existing_title}' found in titles but missing in embeddings.")

        # --- BRANCH 2: NEW MOVIE ---
        else:
            search_results = Scrap2.fetch_movie_data_for_processing(selected_movie)
            
            if search_results:
                st.info(f"'{selected_movie}' is new! Fetching data and processing...")
                existing_title, input_combined_embedding = auto_processing_for_movie(search_results, encoder, sbert_model)
                
                # Extract year hint
                dataset_year_hint = None
                try:
                    date_val = search_results[5].split('-')[0]
                    dataset_year_hint = int(date_val) if date_val else None
                except (ValueError, IndexError, TypeError):
                    dataset_year_hint = None

                if input_combined_embedding is not None:
                    # Fetch display data
                    new_display_data = Scrap2.fetch_movie_data_for_display(
                        existing_title,
                        dataset_release_year=dataset_year_hint,
                        needs_poster=True, needs_rating=True, needs_release_date=True
                    )
                    
                    # Add to database
                    df_embed = auto_updating_for_database(
                        original_title=existing_title,  
                        input_combined_embedding=input_combined_embedding,
                        all_movie_details=all_movie_details,
                        all_titles=all_titles,
                        new_display_data=new_display_data,
                        df_embed=df_embed,
                        index=index
                    )
                    
                    # Update live maps
                    all_titles_lower.add(existing_title.lower())
                    lower_to_canonical_title[existing_title.lower()] = existing_title
                    title_to_embedding[existing_title.lower()] = input_combined_embedding.astype('float32')
                    st.success(f"'{existing_title}' added to databases!")
                else:
                    st.error(f"Could not process new movie '{selected_movie}'.")
            else:
                 st.error(f"Sorry, we couldn't find a movie called '{selected_movie}'.")

        # --- DISPLAY RECOMMENDATIONS ---
        if input_combined_embedding is not None and existing_title is not None:
            # Get display details for input movie
            input_movie_details = all_movie_details.get(existing_title, {}).copy()
            
            # Patch input movie details if necessary
            if input_movie_details:
                poster_url = input_movie_details.get("poster_url")
                rating_val = input_movie_details.get("rating")
                date_val = input_movie_details.get("release_date")
                
                needs_poster = (not poster_url or not is_url_valid(poster_url))
                needs_rating = (rating_val is None or rating_val == "N/A")
                needs_release_date = (not date_val or date_val == "N/A")

                dataset_year_hint = None
                if date_val and isinstance(date_val, str):
                    try:
                        dataset_year_hint = int(date_val)
                    except (ValueError, TypeError):
                        dataset_year_hint = None

                if needs_poster or needs_rating or needs_release_date:
                    st.info(f"Patching missing display data for '{existing_title}'...")
                    fetched_data = Scrap2.fetch_movie_data_for_display(
                        existing_title,
                        dataset_release_year=dataset_year_hint,
                        needs_poster=needs_poster, needs_rating=needs_rating, needs_release_date=needs_release_date
                    )
                    if fetched_data:
                        input_movie_details.update(fetched_data)
                        all_movie_details[existing_title].update(fetched_data)
                        print(f"Successfully patched INPUT '{existing_title}'.")
            
            if 'title' not in input_movie_details:
                input_movie_details['title'] = existing_title

            # Generate Recommendations
            with st.spinner(f'Analyzing "{existing_title}"...'):
                recommendations = get_recommendations(existing_title, input_combined_embedding, index, all_titles, all_movie_details)
            
            if recommendations:
                st.write("---")
                col1, col2 = st.columns([1, 3])
                
                # Column 1: Selected Movie
                with col1:
                    if input_movie_details:
                        st.markdown("##### Your Selection:")
                        st.markdown(f"**{input_movie_details.get('title', existing_title)}**")
                        poster_url = input_movie_details.get('poster_url') or "https://via.placeholder.com/500x750.png?text=No+Image"
                        st.image(poster_url)
                        st.markdown(generate_colored_star_rating(input_movie_details.get('rating', 'N_A')), unsafe_allow_html=True)
                        st.caption(f"**Released:** {input_movie_details.get('release_date', 'N/A')}")
                
                # Column 2: Recommendations
                with col2:
                    st.success(f"Here are 5 movies similar to **{existing_title}**:")
                    cols = st.columns(5)
                    for i, movie in enumerate(recommendations):
                        with cols[i]:
                            st.markdown(f"##### {movie['title']}")
                            st.image(movie.get('poster_url') or "https://via.placeholder.com/500x750.png?text=No+Image")
                            st.markdown(generate_colored_star_rating(movie.get('rating', 'N_A')), unsafe_allow_html=True)
                            st.caption(f"**Released:** {movie.get('release_date', 'N/A')}")
            else:
                st.warning("Could not find any similar movies.")
        elif not existing_title:
             pass # Error handled above
        else:
            st.error(f"Could not generate embedding for '{selected_movie}'.")
