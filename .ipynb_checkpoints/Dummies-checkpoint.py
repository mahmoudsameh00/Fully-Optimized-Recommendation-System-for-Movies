import pandas as pd
df=pd.read_csv('movies_metadata.csv')
df=df.drop(index=[29503,19730,35687])
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
df=df.drop(index=35587)
df=df[df['status'].isin(['Released','Post Production'])]
df=df.dropna(subset=['release_date'])
df=df.drop_duplicates()
df=df.loc[df['vote_count']>=df['vote_count'].quantile(0.2)]



Q1= df['runtime'].quantile(0.97)
Q3= df['runtime'].quantile(0.50)
iqr=Q3-Q1
lower_limit = Q1 - (1.5 * iqr)
upper_limit = Q3 + (1.5 * iqr)
y=df[['runtime']].loc[(df['runtime']>upper_limit) ^ (df['runtime']<lower_limit)]
df= df.drop(labels=y.index, axis=0)



df['popularity']=df['popularity'].astype(float)
Q1= df['popularity'].quantile(0.99)
Q3= df['popularity'].quantile(0.01)
iqr=Q3-Q1
lower_limit = Q1 - (1.5 * iqr)
upper_limit = Q3 + (1.5 * iqr)
y=df[['popularity']].loc[(df['popularity']>upper_limit) ^ (df['popularity']<lower_limit)]
df= df.drop(labels=y.index, axis=0)





def selectdummyLanguage(InputLanguage):
    lanlist=pd.get_dummies(df[['original_language']],columns=["original_language"],drop_first=True)
    lanlist=list(lanlist.columns)
    languages= pd.DataFrame(columns = lanlist)
    languages.loc[0] = 0  # Set all values to 0 initially

    InputLanguage='original_language_'+InputLanguage
    if InputLanguage in lanlist:
        languages.loc[0, InputLanguage] = 1

    # Return the values as a list
    return list(languages.loc[0].values)
            
        
        
        
from ast import literal_eval
df['genres'] = df['genres'].apply(lambda x: [i['name'] for i in literal_eval(x)])
from sklearn.preprocessing import MultiLabelBinarizer
mlb=MultiLabelBinarizer()
x=mlb.fit_transform(df['genres'])
cols=["genre_{}".format(c) for c in mlb.classes_]
df2=pd.DataFrame(data=x,columns=cols)





def selectdummyGenre(InputGenres):
    genlist = list(df2.columns)
    genres = pd.DataFrame(columns=genlist)
    genres.loc[0] = 0  # Set all values to 0 initially

    # Update the relevant genre columns
    for genre in InputGenres:
        InputGenre = 'genre_' + genre
        if InputGenre in genlist:
            genres.loc[0, InputGenre] = 1

    # Return the values as a list
    return list(genres.loc[0].values)

