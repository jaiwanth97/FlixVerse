import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_data():
    movies = pd.read_csv("data/movies.csv")
    ratings = pd.read_csv("data/ratings.csv")
    return movies, ratings

def train_model(movies, ratings):
    # Calculate average ratings and rating counts
    movie_stats = ratings.groupby('movieId').agg({
        'rating': ['mean', 'count']
    }).reset_index()
    movie_stats.columns = ['movieId', 'avg_rating', 'rating_count']
    
    # Filter movies with at least 10 ratings for better quality
    movie_stats = movie_stats[movie_stats['rating_count'] >= 10]
    
    # Merge with movies
    movies_with_stats = movies.merge(movie_stats, on='movieId', how='left')
    movies_with_stats['avg_rating'] = movies_with_stats['avg_rating'].fillna(0)
    movies_with_stats['rating_count'] = movies_with_stats['rating_count'].fillna(0)
    
    # Create TF-IDF matrix from genres
    tfidf = TfidfVectorizer(stop_words='english', token_pattern=r'[a-zA-Z0-9]+')
    tfidf_matrix = tfidf.fit_transform(movies_with_stats['genres'].fillna(''))
    
    # Use cosine similarity (better than linear_kernel for sparse matrices)
    sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    return sim_matrix, movies_with_stats

def recommend(movie_title, movies, sim_matrix, top_n=10):
    import re
    
    movie_title = movie_title.strip().lower()
    
    # Better matching: try exact match first, then contains
    exact_match = movies[movies['title'].str.lower() == movie_title]
    if not exact_match.empty:
        matches = exact_match
    else:
        # Try partial match
        matches = movies[movies['title'].str.lower().str.contains(movie_title, na=False, regex=False)]
    
    if matches.empty:
        return pd.DataFrame(columns=['title', 'genres'])

    # Use the first match (most relevant)
    idx = matches.index[0]
    sim_scores = list(enumerate(sim_matrix[idx]))
    
    def extract_year(title):
        match = re.search(r'\((\d{4})\)', str(title))
        return int(match.group(1)) if match else -1  # Use -1 for unknown so they sort last
    
    # Combine similarity with rating quality
    enriched = []
    for i, sim_score in sim_scores:
        if i == idx:
            continue
        
        # Filter: only consider movies with minimum similarity (0.1 = 10% similar genres)
        if sim_score < 0.1:
            continue
        
        avg_rating = movies.iloc[i]['avg_rating']
        rating_count = movies.iloc[i]['rating_count']
        year = extract_year(movies.iloc[i]['title'])
        
        # Filter out very low rated movies unless they're very similar
        if avg_rating < 2.5 and sim_score < 0.5:
            continue
        
        # Normalize rating (0-5 scale) to 0-1
        rating_score = (avg_rating / 5.0) if avg_rating > 0 else 0
        # Boost movies with more ratings (log scale)
        popularity_boost = min(np.log1p(rating_count) / 5.0, 0.2) if rating_count > 0 else 0
        
        enriched.append((i, sim_score, avg_rating, rating_count, year))
    
    # Sort by similarity FIRST (most relevant), then by year (newest), then by rating
    # This ensures similar movies come first, but newer ones are prioritized among similar ones
    enriched.sort(key=lambda x: (x[1], x[4], x[2]), reverse=True)
    
    filtered = enriched
    
    top = filtered[:top_n]
    movie_indices = [i for i, _, _, _, _ in top]
    
    recs = movies.iloc[movie_indices][['title', 'genres']].copy()
    recs['year'] = [extract_year(t) for t in recs['title']]
    return recs.reset_index(drop=True)