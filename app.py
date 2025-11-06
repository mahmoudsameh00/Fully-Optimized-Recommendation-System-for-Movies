import streamlit as st

from streamlit_tags import st_tags

import joblib

import numpy as np

import pandas as pd

import tensorflow as tf

import faiss

import threading

import json

import requests

from sentence_transformers import SentenceTransformer

from sklearn.preprocessing import normalize





# --- IMPORTANT: Make sure your custom modules can be imported ---

import Dummies

import Scrap2

import os

# --- 1. RESOURCE LOADING: Cached for performance ---

@st.cache_resource

def load_resources():

    """Loads all models, data, and builds the FAISS index."""

    st.info("Loading resources... This may take a moment.")

    Scrap2.API_KEY = os.environ.get('API_KEY')

    encoder = tf.keras.models.load_model("encoder_model_last.keras")

    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

    index = faiss.read_index('prebuilt_index.faiss')

    preprocessor = joblib.load('preprocessor.pkl')

   

    # Load movie details for displaying

    try:

        with open('movie_details.json', 'r', encoding='utf-8') as f:

            all_movie_details = json.load(f)

    except FileNotFoundError:

        st.error("FATAL ERROR: 'movie_details.json' not found. Please run the data extraction script.")

        all_movie_details = {}



    # REQ 1: Load embeddings.csv once at startup

    try:

        df_embed = pd.read_csv('embeddings.csv')

    except FileNotFoundError:

        st.error("FATAL ERROR: 'embeddings.csv' not found. The app cannot function without it.")

        df_embed = pd.DataFrame(columns=['Title'])



    # --- FIX 1: (REQ 3 Corrected) ---

    # Generate all_titles from df_embed['Title'].

    # This guarantees all_titles has the same length and order as the FAISS index.

    all_titles = list(df_embed['Title'])



    all_titles_lower = set(t.lower() for t in all_titles)



    lower_to_canonical_title = {t.lower(): t for t in all_titles}

   

    # --- Add a sanity check to prevent future errors ---

    if index.ntotal != len(all_titles):

        st.error(f"FATAL SYNC ERROR: FAISS index has {index.ntotal} items, but embeddings.csv has {len(all_titles)} titles. Please rebuild the FAISS index.")

        # This will likely still crash, but the error message is now clear.

   

    title_to_embedding = {}

    all_embedding_cols = list(df_embed.columns[1:])

   

    # --- THIS LOOP IS THE FIX ---

    for _, row in df_embed.iterrows():

        title_lower = row['Title'].lower()

       

        # Only add the embedding if this title has NOT been seen yet.

        # This perfectly mimics the .head(1) behavior of the old code.

        if title_lower not in title_to_embedding:

            embedding = row[all_embedding_cols].values.astype('float32').reshape(1, -1)

            title_to_embedding[title_lower] = embedding

    # --- End of fix ---



    st.success("Resources loaded successfully!")

    return encoder, sbert_model, index, all_titles, preprocessor, all_movie_details, df_embed, all_titles_lower, title_to_embedding, lower_to_canonical_title



# if st.button("Clear ALL resource caches"):

#     st.cache_resource.clear()

#     st.rerun()



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





def auto_processing_for_movie(movie_title, encoder, sbert_model):

    """

    Generates the combined feature embedding for a single movie title.

    """

    try:

        d = movie_title

        original_title, genres, lang, overview, popularity, date_str, runtime, rate = d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]

        # Define the same limits you used for training

        POPULARITY_LIMIT = 12.0

        RUNTIME_MIN = 60.0

        RUNTIME_MAX = 200.0



        # 1. Clip Popularity

        if popularity > POPULARITY_LIMIT:

            print(f"Clipping popularity for '{original_title}' from {popularity} to {POPULARITY_LIMIT}")

            popularity = POPULARITY_LIMIT

       

        # 2. Clip Runtime

        if runtime < RUNTIME_MIN:

            print(f"Clipping runtime for '{original_title}' from {runtime} to {RUNTIME_MIN}")

            runtime = RUNTIME_MIN

        elif runtime > RUNTIME_MAX:

            print(f"Clipping runtime for '{original_title}' from {runtime} to {RUNTIME_MAX}")

            runtime = RUNTIME_MAX





        def dateSTR(x): return 'Old' if x < 2000 else 'Modern' if 2000 < x < 2010 else 'New'

        date_val = int(date_str.split('-')[0])

        date_enc = np.array(dateSTR(date_val)).reshape(1, -1)

       

        lang_enc = Dummies.selectdummyLanguage(lang).reshape(1, -1)

        genres_enc = Dummies.selectdummyGenre(genres).reshape(1, -1)



        popularity = np.array(popularity).reshape(1, -1).astype('float32')

        runtime = np.array(runtime).reshape(1, -1).astype('float32')

        rate = np.array(rate).reshape(1, -1).astype('float32')



        arrvalues = np.hstack([popularity, runtime, rate, date_enc, lang_enc, genres_enc])



   



        language_cols = Dummies.LANGUAGE_COLUMNS_LIST

        genre_cols = Dummies.GENRE_COLUMNS_LIST



        cols = ['popularity', 'runtime', 'rate', 'date'] + language_cols + genre_cols



        prepared_data = pd.DataFrame(arrvalues, columns=cols)

       

        preprocessed_data = preprocessor.transform(prepared_data).astype('float32')



        inval_emb = encoder.predict(preprocessed_data)

        inval_emb = normalize(inval_emb, norm='l2', axis=1)

        inval_emb = inval_emb * 3.5  # <--- THIS IS THE FIX

        intxt_emb = sbert_model.encode(overview).reshape(1, -1)

        intxt_emb = intxt_emb * 1.0

        input_combined_embedding = np.hstack([intxt_emb, inval_emb])

       

        return original_title, input_combined_embedding

   

    except Exception as e:

        print(f"CRITICAL: Failed to process '{original_title}'. Error: {e}")

        return None, None









def save_updates_to_disk(original_title, df_embed, all_movie_details, index,

                         csv_filename, json_filename, index_filename):

    """

    This function runs in a background thread to save updates to disk.

    """

    print(f"BACKGROUND: Saving '{original_title}' to disk...")

    try:

        # 1. Save embeddings CSV

        df_embed.to_csv(csv_filename, index=False, encoding='utf-8')

        print(f"BACKGROUND: Saved {csv_filename}")

       

        # 2. Save FAISS index

        faiss.write_index(index, index_filename)

        print(f"BACKGROUND: Saved {index_filename}")



        # 3. Save movie details

        with open(json_filename, 'w', encoding='utf-8') as f:

            json.dump(all_movie_details, f, indent=4)

        print(f"BACKGROUND: Saved {json_filename}")

           

    except Exception as e:

        # Log errors from the background thread

        print(f"BACKGROUND ERROR saving updates: {e}")





# --- FIX 2: Function updated to accept and modify the 'index' object ---

def auto_updating_for_database(original_title, input_combined_embedding,

                               all_movie_details, all_titles, new_display_data, df_embed, index,

                               csv_filename="embeddings.csv",

                               json_filename="movie_details.json",

                               index_filename="prebuilt_index.faiss"): # <-- Add index filename

    """

    Updates all IN-MEMORY resources and spawns a background thread to save to disk.

    Modifies 'all_movie_details', 'all_titles', 'df_embed', and 'index' objects.

    """



    # 1. Check if movie already exists in the DataFrame

    if original_title in df_embed['Title'].values:

        print(f"'{original_title}' already exists in the database. Skipping update.")

        return df_embed



    # 2. Prepare new row for embeddings DataFrame

    new_embedding_values = input_combined_embedding.flatten().tolist()

    new_row_data = [original_title] + new_embedding_values



    try:

        # 3. Update IN-MEMORY DataFrame

        df_embed.loc[len(df_embed)] = new_row_data

        print(f"Successfully added '{original_title}' to in-memory DataFrame.")

    except Exception as e:

        print(f"Error adding row with .loc: {e}")

        st.error(f"Failed to update in-memory DataFrame: {e}")

        return df_embed # Return original, unmodified df



    # 4. Update the IN-MEMORY FAISS INDEX

    try:

        new_embedding_normalized = input_combined_embedding.copy().astype('float32')

        faiss.normalize_L2(new_embedding_normalized)

        index.add(new_embedding_normalized)

        print(f"Successfully added '{original_title}' to in-memory FAISS index. New total: {index.ntotal}")

    except Exception as e:

        print(f"CRITICAL: Failed to add new vector to FAISS index: {e}")

        st.error("Failed to update live index. Recommendations may be stale until restart.")



    # 5. Update IN-MEMORY movie_details

    all_movie_details[original_title] = new_display_data

   

    # 6. Update IN-MEMORY titles list

    all_titles.append(original_title)

   

    # --- 7. START BACKGROUND THREAD TO SAVE TO DISK ---

    # We pass copies of the objects to the thread

    save_thread = threading.Thread(

        target=save_updates_to_disk,

        args=(

            original_title,

            df_embed.copy(),              # Pass a copy

            all_movie_details.copy(),     # Pass a copy

            index,                        # Pass the index object (it's thread-safe for writing)

            csv_filename,

            json_filename,

            index_filename

        )

    )

    save_thread.start()

    print(f"Main thread continuing while '{original_title}' saves in background.")

   

    # 8. Return the updated DataFrame for in-memory use IMMEDIATELY

    return df_embed





# --- 2. RECOMMENDATION LOGIC ---
def get_recommendations(movie_title, input_combined_embedding, index, all_titles, all_movie_details):
    """
    Takes a movie title and returns a list of recommended movies with their details.
    """
    
    print("--- NOTEBOOK OUTPUT ---")
    print(f"Final Embedding Shape: {input_combined_embedding.shape}")

    # --- B. Search for Similar Movies in FAISS ---
    k = 6 
    faiss.normalize_L2(input_combined_embedding)
    
    distances, indices = index.search(input_combined_embedding, k)
    cosine_similarities = 1 - distances / 2.0
    
    recommended_movies = []
    for i, similarity in zip(indices[0], cosine_similarities[0]):
        title = all_titles[i] 
        print(f"\nCandidate: {title} (Similarity: {similarity:.4f})")
        
        if title.lower() != movie_title.lower():
            movie_details = all_movie_details.get(title, {}).copy()
            movie_details['title'] = title 

            poster_url = movie_details.get("poster_url")
            rating_val = movie_details.get("rating")
            
            # Get the release_date (full string) FROM OUR DATASET
            date_val = movie_details.get("release_date") 

            placeholder_img = "https://via.placeholder.com/500x750.png?text=No+Image"

            if poster_url == placeholder_img or not is_url_valid(poster_url):
                needs_poster = True
            else:
                needs_poster = False

            needs_rating = (rating_val is None or rating_val == "N/A")
            
            # Check if date is missing (e.g., None, "N/A", or empty string)
            needs_release_date = (date_val is None or date_val == "N/A" or not date_val)
            
            # --- THIS IS THE FIX ---
            # We must extract the YEAR from the date string to pass to the scraper
            dataset_year_hint = None
            if date_val and isinstance(date_val, str):
                try:
                    # Get the integer year, e.g., 1997
                    dataset_year_hint = int(date_val) 
                except (ValueError, IndexError, TypeError):
                    dataset_year_hint = None # Use None if format is bad
            # --- END OF FIX ---

            if needs_poster or needs_rating or needs_release_date:
                print(f"Patching missing data for '{title}' from API...")
                try:
                    # --- MODIFICATION IS HERE ---
                    # We now pass the 'dataset_year_hint' (our dataset's year) to the scraper
                    fetched_data = Scrap2.fetch_movie_data_for_display(
                        title, 
                        dataset_release_year=dataset_year_hint,  # <-- PASS THE YEAR HINT
                        needs_poster=needs_poster, 
                        needs_rating=needs_rating, 
                        needs_release_date=needs_release_date
                    )
                    # --- END MODIFICATION ---
                    
                    if fetched_data:
                        movie_details.update(fetched_data)
                        # Also update the master list so we don't fetch it again next time
                        all_movie_details[title].update(fetched_data)
                        
                except Exception as e:
                    print(f"Error patching data for '{title}': {e}")
            
            recommended_movies.append(movie_details)
            
    return recommended_movies[:5]






def generate_colored_star_rating(rating, max_stars=5):

    """

    Generates a string of colored HTML stars.

    """

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





# --- 3. STREAMLIT USER INTERFACE ---



st.set_page_config(layout="wide")

st.title("🎬 AI Movie Recommendation System")



# Load all resources. df_embed and all_titles are now correctly synced with index.

encoder, sbert_model, index, all_titles, preprocessor, all_movie_details, df_embed, all_titles_lower, title_to_embedding, lower_to_canonical_title = load_resources()





if 'last_submitted_movie' not in st.session_state:

    st.session_state.last_submitted_movie = []



selected_movie_list = st_tags(

    label='Enter a Movie Title and press Enter to search:',

    text=" ",

    value=[],

    suggestions= all_titles, # This list is now from embeddings.csv

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



        # --- LOGIC REFACTOR (Required) ---

        # 1. CHECK IF MOVIE EXISTS IN OUR DATABASE *FIRST*

        if selected_movie_lower in all_titles_lower:

            # --- EXISTING MOVIE LOGIC ---

            st.info("Found movie in database...")

           

            # Get the *canonical* title (e.g., "Titanic") from our map

            existing_title = lower_to_canonical_title.get(selected_movie_lower)

           

            # Get the pre-calculated embedding

            input_combined_embedding = title_to_embedding.get(selected_movie_lower)

           

            if input_combined_embedding is None:

                 st.error(f"Data sync error: '{existing_title}' is in title list but not in embeddings. Cannot proceed.")



        else:

            # --- NEW MOVIE LOGIC ---
           
            # 1. Fetch data for processing

            search_results = Scrap2.fetch_movie_data_for_processing(selected_movie)
           
            if search_results:

                # 2. Generate embedding

                existing_title, input_combined_embedding = auto_processing_for_movie(search_results, encoder, sbert_model)

                st.info(f"'{selected_movie}' is new! Fetching data and processing...")

                dataset_year_hint = None
                date_val = search_results[5].split('-')[0]

                if date_val and isinstance(date_val, str):

                    try:

                        # Get the integer year, e.g., 1997

                        dataset_year_hint = int(date_val)

                    except (ValueError, IndexError, TypeError):

                        dataset_year_hint = None # Use None if format is bad

                if input_combined_embedding is not None:

                    # 3. Fetch display data

                    new_display_data = Scrap2.fetch_movie_data_for_display(

                        existing_title,

                        dataset_release_year=dataset_year_hint,  # <-- PASS THE YEAR HINT

                        needs_poster=True,

                        needs_rating=True,

                        needs_release_date=True

                    )

                   
                    # 4. Add to database (in-memory and in background)

                    df_embed = auto_updating_for_database(

                        original_title=existing_title,  

                        input_combined_embedding=input_combined_embedding,

                        all_movie_details=all_movie_details,

                        all_titles=all_titles,

                        new_display_data=new_display_data,

                        df_embed=df_embed,

                        index=index

                    )

                   

                    # 5. Update live maps

                    all_titles_lower.add(existing_title.lower())

                    lower_to_canonical_title[existing_title.lower()] = existing_title

                    title_to_embedding[existing_title.lower()] = input_combined_embedding.astype('float32')

                    st.success(f"'{existing_title}' added to databases!")

                else:

                    st.error(f"Could not process new movie '{selected_movie}'.")

            else:

                 st.error(f"Sorry, we couldn't find a movie called '{selected_movie}'. Please check the title and try again.")

        # --- END LOGIC REFACTOR ---





        # --- PROCEED TO RECOMMENDATIONS (This block is now shared by both paths) ---

        if input_combined_embedding is not None and existing_title is not None:

           

            # --- Get details for the INPUT movie for display ---

            input_movie_details = all_movie_details.get(existing_title, {}).copy()

           

            # --- PATCHING LOGIC (for existing movies with bad/missing data) ---

            if input_movie_details:

                poster_url = input_movie_details.get("poster_url")

                rating_val = input_movie_details.get("rating")

                date_val = input_movie_details.get("release_date") # This is the full string "1997-12-19"

                placeholder_img = "https://via.placeholder.com/500x750.png?text=No+Image"



                needs_poster = (poster_url == placeholder_img or not is_url_valid(poster_url))

                needs_rating = (rating_val is None or rating_val == "N/A")

                needs_release_date = (date_val is None or date_val == "N/A" or not date_val)



                # --- THIS IS THE FIX ---

                # We must extract the YEAR from the date string to pass to the scraper

                dataset_year_hint = None

                if date_val and isinstance(date_val, str):

                    try:

                        # Get the integer year, e.g., 1997

                        dataset_year_hint = int(date_val)

                    except (ValueError, IndexError, TypeError):

                        dataset_year_hint = None # Use None if format is bad

                # --- END OF FIX ---



                if needs_poster or needs_rating or needs_release_date:

                    st.info(f"Patching missing display data for '{existing_title}'...")

                    try:

                        fetched_data = Scrap2.fetch_movie_data_for_display(

                            existing_title,

                            dataset_release_year=dataset_year_hint,  # <-- PASS THE YEAR HINT

                            needs_poster=needs_poster,

                            needs_rating=needs_rating,

                            needs_release_date=needs_release_date

                        )

                        if fetched_data:

                            input_movie_details.update(fetched_data)

                            all_movie_details[existing_title].update(fetched_data) # Update master list

                            print(f"Successfully patched INPUT '{existing_title}'.")

                    except Exception as e:

                        print(f"Error patching data for INPUT '{existing_title}': {e}")

           

            if 'title' not in input_movie_details:

                input_movie_details['title'] = existing_title

            # --- END PATCHING ---



            with st.spinner(f'Analyzing "{existing_title}" and finding recommendations...'):

                recommendations = get_recommendations(existing_title, input_combined_embedding, index, all_titles, all_movie_details)

           

            if recommendations:

               

                st.write("---") # Add a separator

                col1, col2 = st.columns([1, 3])

               

                # --- Column 1: Input Movie ---

                with col1:

                    if input_movie_details:

                        st.markdown(f"##### Your Selection:")

                        st.markdown(f"**{input_movie_details.get('title', existing_title)}**")

                       

                        poster_url = input_movie_details.get('poster_url')

                        if not poster_url or not is_url_valid(poster_url):

                            poster_url = "https://via.placeholder.com/500x750.png?text=No+Image"

                       

                        st.image(poster_url)

                       

                        rating_value = input_movie_details.get('rating', 'N_A')

                        stars = generate_colored_star_rating(rating_value)

                        st.markdown(f"{stars}", unsafe_allow_html=True)

                        st.caption(f"**Released:** {input_movie_details.get('release_date', 'N/A')}")

                        st.caption(f"**Rating:** {input_movie_details.get('rating', 'N/A')}")

                    else:

                        st.warning(f"Could not load display details for '{existing_title}'.")

               

                # --- Column 2: Recommendations ---

                with col2:

                    st.success(f"Here are 5 movies similar to **{existing_title}**:")

                   

                    cols = st.columns(5)

                    for i, movie in enumerate(recommendations):

                        with cols[i]:

                            st.markdown(f"##### {movie['title']}")

                            if movie.get('poster_url'):

                                st.image(movie['poster_url'])

                            rating_value = movie.get('rating', 'N_A')

                            stars = generate_colored_star_rating(rating_value)

                            st.markdown(f"{stars}", unsafe_allow_html=True)

                            st.caption(f"**Released:** {movie.get('release_date', 'N/A')}")

                            st.caption(f"**Rating:** {movie.get('rating', 'N/A')}")

           

            else:

                st.warning("Could not find any similar movies.")

       

        elif not existing_title:

             # This handles the case where a new movie search failed

             pass # Error was already shown

        else:

            st.error(f"Could not find or generate an embedding for '{selected_movie}'. Cannot proceed.")