import numpy as np 
import pandas as pd
import ast
import os
import pickle
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("📦 Loading datasets...")

movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')
movies = movies.merge(credits, on='title')

movies = movies[['genres', 'movie_id', 'title', 'overview', 'keywords', 'cast', 'crew']]
movies.dropna(inplace=True)

# convert functions
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

def convertCast(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L

def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

# apply functions
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convertCast)
movies['crew'] = movies['crew'].apply(fetch_director)
movies['overview'] = movies['overview'].apply(lambda x: x.split())

# remove spaces in multi-word names
def removespace(l):
    return [i.replace(" ", "") for i in l]

movies['genres'] = movies['genres'].apply(removespace)
movies['keywords'] = movies['keywords'].apply(removespace)
movies['cast'] = movies['cast'].apply(removespace)
movies['crew'] = movies['crew'].apply(removespace)

# make tags
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
new_df = movies[['movie_id', 'title', 'tags']]

new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))

# stemming
ps = PorterStemmer()
def stem(text):
    return " ".join([ps.stem(i) for i in text.split()])

new_df['tags'] = new_df['tags'].apply(stem)
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())

# vectorization
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()
similarity = cosine_similarity(vectors)

# Save to artifacts/
if not os.path.exists('artifacts'):
    os.makedirs('artifacts')

new_df.to_pickle('artifacts/movie_list.pkl')
pickle.dump(similarity, open('artifacts/similarity.pkl', 'wb'))

print("✅ Artifacts saved in 'artifacts/' folder.")
