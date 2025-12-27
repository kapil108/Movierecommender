import pandas as pd
import numpy as np
import ast
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def convert(obj):
    L = []
    try:
        for i in ast.literal_eval(obj):
            L.append(i['name'])
    except:
        pass
    return L

def convert3(obj):
    L = []
    counter = 0
    try:
        for i in ast.literal_eval(obj):
            if counter != 3:
                L.append(i['name'])
                counter += 1
            else:
                break
    except:
        pass
    return L

def fetch_director(obj):
    L = []
    try:
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director':
                L.append(i['name'])
                break
    except:
        pass
    return L

def prepare_data():
    print("Loading data...")
    movies = pd.read_csv('tmdb_5000_movies.csv')
    credits = pd.read_csv('tmdb_5000_credits.csv')

    print("Merging data...")
    # Merge datasets
    movies = movies.merge(credits, on='title')

    # Select relevant columns Use 'id' from movies as movie_id matches usually but let's be safe
    # The notebook used 'movie_id' from credits or movies. Let's stick to what works.
    # checking notebook: movies = movies.merge(credits, left_on='title', right_on='title')
    # resulting columns had movie_id from credits.
    # In the provided notebook, it selected: ['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']
    # I will add: 'vote_average', 'release_date', 'runtime'
    
    # Note: after merge, if keys overlap they get suffixes. 
    # tmdb_5000_movies has 'id'. tmdb_5000_credits has 'movie_id'.
    # usually 'id' in movies equals 'movie_id' in credits.
    
    movies['movie_id'] = movies['id'] # Ensure we have a movie_id column
    
    columns_to_keep = [
        'movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew', 
        'vote_average', 'release_date', 'runtime'
    ]
    
    movies = movies[columns_to_keep]
    
    print("Processing features...")
    # Drop missing values for critical text fields to avoid errors
    movies.dropna(subset=['overview', 'genres', 'keywords', 'cast', 'crew'], inplace=True)

    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(convert3)
    movies['crew'] = movies['crew'].apply(fetch_director)

    # Handle Overview which is a string
    movies['overview'] = movies['overview'].apply(lambda x: x.split())

    # Create tags for similarity
    # We want to remove spaces in names to treat "Sam Worthington" as one entity "SamWorthington"
    # ONLY for the tags calculation, but for display we might want original.
    # The original notebook replaced spaces. Let's do that for the 'tags' column only.
    
    movies['genres_tag'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['keywords_tag'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['cast_tag'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['crew_tag'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])

    movies['tags'] = movies['overview'] + movies['genres_tag'] + movies['keywords_tag'] + movies['cast_tag'] + movies['crew_tag']
    
    new_df = movies[['movie_id', 'title', 'tags', 'vote_average', 'release_date', 'runtime', 'overview', 'genres', 'cast', 'crew']]
    
    # Convert tags back to string
    new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
    new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())
    
    # Convert overview back to string for display (it was split earlier)
    new_df['overview'] = new_df['overview'].apply(lambda x: " ".join(x))

    print("Vectorizing tags...")
    cv = TfidfVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(new_df['tags']).toarray()

    print("Calculating similarity...")
    similarity = cosine_similarity(vectors)

    print("Saving data to 'movie_data.pkl'...")
    pickle.dump((new_df.to_dict('records'), similarity), open('movie_data.pkl', 'wb'))
    print("Done!")

if __name__ == '__main__':
    prepare_data()
