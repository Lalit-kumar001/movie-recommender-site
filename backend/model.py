import pandas as pd
import pickle

# Load the model files
movies = pd.read_pickle('artifacts/movie_list.pkl')
similarity = pickle.load(open('artifacts/similarity.pkl', 'rb'))

def recommend(movie):
    print("🔍 Received movie from frontend:", movie)
    movie = movie.lower()
    
    movie_index = None
    for idx, title in enumerate(movies['title']):
        if title.lower() == movie:
            movie_index = idx
            break

    if movie_index is None:
        print("❌ Movie not found in dataset!")
        raise Exception("Movie not found!")

    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended = []
    for i in movies_list:
        recommended.append(movies.iloc[i[0]]['title'])

    print("✅ Recommendations ready:", recommended)
    return recommended
