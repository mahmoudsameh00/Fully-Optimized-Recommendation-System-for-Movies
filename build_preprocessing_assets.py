# build_preprocessing_assets.py
import pandas as pd
from ast import literal_eval
from sklearn.preprocessing import MultiLabelBinarizer
import json
import numpy as np

print("Starting one-time preprocessing...")

try:
    df=pd.read_csv('movies_metadata.csv', low_memory=False)
except FileNotFoundError:
    print("FATAL ERROR: 'movies_metadata.csv' not found.")
    exit()
    
# --- Start of Notebook Cleaning Pipeline (from Proj (1).ipynb) ---

df=df.drop(index=[29503,19730,35587]) # Was [29503,19730,35687]

# (Proj Cell 7)
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
# *** THIS IS THE FIX ***
df['runtime'] = pd.to_numeric(df['runtime'], errors='coerce')
df = df.loc[(df['runtime'] >= 60) & (df['runtime'] <= 200)]

# Cell 180: Filter by popularity (Hard-coded limits from notebook)
# *** THIS IS THE FIX ***
df['popularity']=pd.to_numeric(df['popularity'], errors='coerce')
df = df.loc[df['popularity'] <= 12]

# Drop duplicates based on the movie title, keeping the first one
df.drop_duplicates(subset=['original_title'], keep='first', inplace=True)

# *** THIS STEP IS CRITICAL ***
# Reset the index so that df, df1, and df2 can be concatenated correctly in Cell 60
df.reset_index(drop=True, inplace=True)

# --- End of Filtering ---

# --- Genre processing (Cell 153) ---
def safe_literal_eval(x):
    try:
        if isinstance(x, str) and x.startswith('['):
            return [i['name'] for i in literal_eval(x)]
    except:
        pass
    return [] # Return empty list on failure

df['genres'] = df['genres'].apply(safe_literal_eval)

mlb=MultiLabelBinarizer()
x=mlb.fit_transform(df['genres'])


# --- SAVE ASSETS ---

# 1. Save the Language Columns
print("Saving language columns...")
lanlist = pd.get_dummies(df[['original_language']], columns=["original_language"], drop_first=True, dtype=float)
final_language_columns = list(lanlist.columns)

with open('language_columns.json', 'w', encoding='utf-8') as f:
    json.dump(final_language_columns, f)
print(f"Saved {len(final_language_columns)} language columns to language_columns.json")


# 2. Save the Genre Columns
print("Saving genre columns...")
final_genre_columns = ["genre_{}".format(c) for c in mlb.classes_]

with open('genre_columns.json', 'w', encoding='utf-8') as f:
    json.dump(final_genre_columns, f)
print(f"Saved {len(final_genre_columns)} genre columns to genre_columns.json")

print("--- Preprocessing complete. Assets saved. ---")