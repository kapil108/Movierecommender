import streamlit as st
import pandas as pd
import requests
import pickle
from streamlit_agraph import agraph, Node, Edge, Config

import os
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
@st.cache_data
def load_data():
    if os.path.exists('movie_data.pkl'):
        with open('movie_data.pkl', 'rb') as file:
            movie_dict, cosine_sim = pickle.load(file)
        movies = pd.DataFrame(movie_dict)
        return movies, cosine_sim
    else:
        with st.spinner('Building recommendation model... (This runs once)'):
            # Logic from prepare_data.py
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

            movies_df = pd.read_csv('tmdb_5000_movies.csv')
            
            if os.path.exists('tmdb_5000_credits.csv'):
                credits_df = pd.read_csv('tmdb_5000_credits.csv')
            elif os.path.exists('tmdb_5000_credits.zip'):
                credits_df = pd.read_csv('tmdb_5000_credits.zip') 
            else:
                 # Fallback/Error if neither exists
                 st.error("Critical Error: 'tmdb_5000_credits.csv' or 'tmdb_5000_credits.zip' not found.")
                 st.stop()
            
            movies_df = movies_df.merge(credits_df, on='title')
            
            # Keep consistent with prepare_data.py columns
            # merging creates suffixes if columns overlap. tmdb_5000_movies has 'id', credits has 'movie_id'.
            # checking typical merge behavior:
            # If 'title' is unique enough.
            # We need to ensure we have 'movie_id'
            if 'movie_id' not in movies_df.columns:
                 if 'id' in movies_df.columns:
                     movies_df['movie_id'] = movies_df['id']
            
            columns_to_keep = [
                'movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew', 
                'vote_average', 'release_date', 'runtime'
            ]
            # Ensure columns exist before selecting
            existing_cols = [c for c in columns_to_keep if c in movies_df.columns]
            movies_df = movies_df[existing_cols]

            movies_df.dropna(subset=['overview', 'genres', 'keywords', 'cast', 'crew'], inplace=True)

            movies_df['genres'] = movies_df['genres'].apply(convert)
            movies_df['keywords'] = movies_df['keywords'].apply(convert)
            movies_df['cast'] = movies_df['cast'].apply(convert3)
            movies_df['crew'] = movies_df['crew'].apply(fetch_director)
            
            movies_df['overview'] = movies_df['overview'].apply(lambda x: x.split())
            
            movies_df['genres_tag'] = movies_df['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
            movies_df['keywords_tag'] = movies_df['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
            movies_df['cast_tag'] = movies_df['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
            movies_df['crew_tag'] = movies_df['crew'].apply(lambda x: [i.replace(" ", "") for i in x])

            movies_df['tags'] = movies_df['overview'] + movies_df['genres_tag'] + movies_df['keywords_tag'] + movies_df['cast_tag'] + movies_df['crew_tag']
            
            new_df = movies_df[['movie_id', 'title', 'tags', 'vote_average', 'release_date', 'runtime', 'overview', 'genres', 'cast', 'crew']].copy()
            
            new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x).lower())
            # Join overview back for display
            new_df['overview'] = new_df['overview'].apply(lambda x: " ".join(x))
            
            cv = TfidfVectorizer(max_features=5000, stop_words='english')
            vectors = cv.fit_transform(new_df['tags']).toarray()
            cosine_sim_generated = cosine_similarity(vectors)
            
            return new_df, cosine_sim_generated

movies, cosine_sim = load_data()

# Helper: Extract unique genres
all_genres = set()
for g_list in movies['genres']:
    all_genres.update(g_list)
all_genres = sorted(list(all_genres))

# Extract Years
movies['year'] = pd.to_datetime(movies['release_date'], errors='coerce').dt.year
min_year = int(movies['year'].min()) if not movies['year'].isnull().all() else 1980
max_year = int(movies['year'].max()) if not movies['year'].isnull().all() else 2024

# Sidebar configuration
st.sidebar.markdown("# 🍿 Movie Recommender")
st.sidebar.title("Filter Options")

selected_genres = st.sidebar.multiselect("Select Genres", all_genres)
min_rating = st.sidebar.slider("Minimum Rating", 0.0, 10.0, 5.0, 0.5)
year_range = st.sidebar.slider("Release Year", min_year, max_year, (1990, max_year))

# Fetch Poster
def fetch_poster(movie_id):
    try:
        api_key = '7b995d3c6fd91a2284b4ad8cb390c7b8' 
        url = f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}'
        response = requests.get(url, timeout=5)
        data = response.json()
        poster_path = data.get('poster_path')
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except Exception as e:
        return "https://via.placeholder.com/500x750?text=No+Poster"
    return "https://via.placeholder.com/500x750?text=No+Poster"

# Recommendation Function
def get_recommendations(title, cosine_sim=cosine_sim):
    try:
        idx = movies[movies['title'] == title].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        # Fetch top 30 to allow for filtering
        sim_scores = sim_scores[1:31]
        movie_indices = [i[0] for i in sim_scores]
        return movies.iloc[movie_indices]
    except IndexError:
        return pd.DataFrame()

# Main UI
st.title("🍿 Better Movie Recommendations")
st.markdown("Find your next favorite movie with **advanced filtering**!")

selected_movie = st.selectbox("Type or select a movie you like:", movies['title'].values)

if st.button('Show Recommendations', type='primary'):
    with st.spinner('Finding the best matches...'):
        recommendations = get_recommendations(selected_movie)
        
        # Apply Filters
        if not recommendations.empty:
            # Filter by Genre
            if selected_genres:
                recommendations = recommendations[recommendations['genres'].apply(
                    lambda x: any(genre in x for genre in selected_genres)
                )]
            
            # Filter by Rating
            recommendations = recommendations[recommendations['vote_average'] >= min_rating]
            
            # Filter by Year
            recommendations = recommendations[
                (recommendations['year'] >= year_range[0]) & 
                (recommendations['year'] <= year_range[1])
            ]

            st.divider()
            
            if recommendations.empty:
                st.warning(f"No movies found similar to **{selected_movie}** with these filters. Try adjusting the sliders in the sidebar.")
            else:
                # TABS for View Selection
                tab1, tab2 = st.tabs(["📝 List View", "🕸️ Network Graph"])
                
                with tab1:
                    st.subheader(f"Top matches for **{selected_movie}**:")
                    for i, row in recommendations.head(10).iterrows():
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.image(fetch_poster(row['movie_id']), width=150)
                        with col2:
                            st.markdown(f"### {row['title']}")
                            st.caption(f"📅 {int(row['year']) if pd.notna(row['year']) else 'N/A'} | ⭐ {row['vote_average']}/10 | ⏱ {row['runtime']} min")
                            st.markdown(f"**Genres**: {', '.join(row['genres'])}")
                            st.write(row['overview'])
                        st.divider()

                with tab2:
                    st.subheader("Movie Connections")
                    st.info("Explore how these movies are connected! Drag nodes to rearrange.")
                    
                    nodes = []
                    edges = []
                    
                    # Central Node (Selected Movie)
                    try:
                        center_img = fetch_poster(movies[movies['title'] == selected_movie].iloc[0]['movie_id'])
                    except:
                        center_img = "https://img.icons8.com/clouds/200/000000/movie-projector.png"

                    nodes.append(Node(id=selected_movie, 
                                      label=selected_movie, 
                                      size=40, 
                                      shape="circularImage",
                                      image=center_img))
                    
                    # Recommendation Nodes
                    for i, row in recommendations.head(15).iterrows():
                        try:
                            img_url = fetch_poster(row['movie_id'])
                        except:
                            img_url = "https://img.icons8.com/clouds/200/000000/film-reel.png"
                            
                        nodes.append(Node(id=row['title'], 
                                          label=row['title'], 
                                          size=25, 
                                          shape="circularImage", 
                                          image=img_url))
                        
                        # Edge from Center to Rec
                        edges.append(Edge(source=selected_movie, 
                                          target=row['title'], 
                                          type="STRAIGHT",
                                          strokeWidth=2,
                                          color="#ff5733"))
                        
                        # Optional: Inter-movie connections (if they share genre)
                        # Minimal implementation to avoid clutter:
                        # Logic: If they share at least 2 genres, link them.
                        main_genres = set(movies[movies['title'] == selected_movie].iloc[0]['genres'])
                        rec_genres = set(row['genres'])
                        common = main_genres.intersection(rec_genres)
                        if common:
                             edges.append(Edge(source=selected_movie, target=row['title'], label=list(common)[0], color="#BDC3C7"))

                    config = Config(width=700, 
                                    height=500, 
                                    directed=True, 
                                    physics=True, 
                                    hierarchical=False)
                    
                    return_value = agraph(nodes=nodes, 
                                          edges=edges, 
                                          config=config)
