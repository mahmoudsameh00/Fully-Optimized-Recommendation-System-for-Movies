# Rate.py
import pandas as pd
import numpy as np

# --- Start of Notebook Cleaning Pipeline (from Proj (1).ipynb) ---
try:
    df=pd.read_csv('movies_metadata.csv', low_memory=False)

    # Cells 128, 133
    df=df.drop(index=[29503, 19730, 35687, 35587], errors='ignore') 
    
    # Cell 130
    df.at[19574,'original_language']='en'
    df.at[21602,'original_language']='en'
    df.at[22832,'original_language']='en'
    df.at[32141,'original_language']='en'
    df.at[37407,'original_language']='cs'
    df.at[41047,'original_language']='ur'
    df.at[41872,'original_language']='xx'
    df.at[44057,'original_language']='fr'
    df.at[44410,'original_language']='sv'
    df.at[44576,'original_language']='de'
    df.at[44655,'original_language']='xx'
    
    # Cell 141
    df=df[df['status'].isin(['Released','Post Production'])]
    
    # Cell 143
    df=df.dropna(subset=['release_date'])
    
    # Cell 151
    df=df.drop_duplicates()
    
    # Cell 154: Filter by vote_count
    m_filter_quantile = df['vote_count'].quantile(0.2)
    df=df.loc[df['vote_count'] >= m_filter_quantile]
    
    # Cell 178: Filter by runtime (Hard-coded limits from notebook)
    df['runtime'] = pd.to_numeric(df['runtime'], errors='coerce')
    df = df.dropna(subset=['runtime']) 
    df = df.loc[(df['runtime'] >= 60) & (df['runtime'] <= 200)]
    
    # Cell 180: Filter by popularity (Hard-coded limits from notebook)
    df['popularity']=pd.to_numeric(df['popularity'], errors='coerce')
    df = df.dropna(subset=['popularity']) 
    df = df.loc[df['popularity'] <= 12]
    
    # --- End of Filtering ---
    
    # Cell 155: Calculate c and m from the *filtered* dataframe
    c = df['vote_average'].mean()
    m = df['vote_count'].quantile(0.2) # This is the 'm' used for the rating formula

except FileNotFoundError:
    print("WARNING: 'movies_metadata.csv' not found. Using fallback values for c and m.")
    c = 6.07 # Fallback from your notebook's `df.describe()`
    m = 25.0   # Fallback (approx. from notebook)
except Exception as e:
    print(f"WARNING: Error processing 'movies_metadata.csv': {e}. Using fallback values.")
    c = 6.07
    m = 25.0

# --- End of Notebook Cleaning Pipeline ---


def rate(x, z):
    # x = vote_count, z = vote_average
    # These c and m values are now based on the *filtered* training set
    rate=round((x/(x+m)*z) + (m/(m+x)*c),1)
    return rate