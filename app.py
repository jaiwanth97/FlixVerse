import streamlit as st
import pandas as pd
import time
import urllib.parse
from src.recommender import load_data, train_model, recommend
from src.tmdb_utils import get_movie_details, get_full_movie_details

GENRE_COLORS = {
    "Action": "#ff4b4b",
    "Adventure": "#f39c12",
    "Animation": "#9b59b6",
    "Comedy": "#f1c40f",
    "Crime": "#e67e22",
    "Documentary": "#16a085",
    "Drama": "#3498db",
    "Family": "#1abc9c",
    "Fantasy": "#8e44ad",
    "History": "#e74c3c",
    "Horror": "#c0392b",
    "Music": "#2ecc71",
    "Mystery": "#9b59b6",
    "Romance": "#fd79a8",
    "Sci-Fi": "#00cec9",
    "Thriller": "#e84393",
    "War": "#636e72",
    "Western": "#b2bec3",
    "IMAX": "#0984e3"
}

st.set_page_config(page_title="AI Movie Recommender", layout="wide")

# Get movie from query parameters
query_params = st.query_params
selected_movie_title = None
if query_params.get("movie"):
    # Handle both single value and list
    movie_param = query_params.get("movie")
    if isinstance(movie_param, list):
        selected_movie_title = movie_param[0] if movie_param else None
    else:
        selected_movie_title = movie_param
    
    # URL decode the title
    if selected_movie_title:
        selected_movie_title = urllib.parse.unquote(selected_movie_title)

@st.cache_data
def prepare():
    movies, ratings = load_data()
    sim_matrix, movies_with_stats = train_model(movies, ratings)
    return movies_with_stats, sim_matrix

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_cached_movie_details(title):
    """Cache movie details to avoid unnecessary fetching"""
    return get_movie_details(title)

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_cached_full_movie_details(title):
    """Cache full movie details to avoid unnecessary fetching"""
    return get_full_movie_details(title)

movies, sim_matrix = prepare()

# If movie is selected via query params, show detailed view
if selected_movie_title:
    st.title(f"🎬 {selected_movie_title}")
    
    # Back button
    if st.button("← Back to Recommendations"):
        st.query_params.clear()
        st.rerun()
    
    # Load full movie details
    full_details = get_cached_full_movie_details(selected_movie_title)
    
    # If no details found or wrong movie, show error
    if not full_details:
        st.error(f"❌ Could not find movie details for: {selected_movie_title}")
        st.info("This might be due to:")
        st.write("- The movie is not available in TMDB")
        st.write("- The title doesn't match any results")
        st.write("- There was an error fetching the data")
        if st.button("← Back to Recommendations"):
            st.query_params.clear()
            st.rerun()
    elif full_details:
        backdrop = full_details.get('backdrop') or full_details.get('poster') or ''
        
        # Hero section with backdrop
        if backdrop:
            st.markdown(f"""
            <div style="
                height: 500px;
                background-image: url('{backdrop}');
                background-size: cover;
                background-position: center;
                margin: -20px -20px 30px -20px;
                display: flex;
                align-items: flex-end;
                padding: 40px;
                position: relative;
            ">
                <div style="
                    position: absolute;
                    top: 0;
                    left: 0;
                    right: 0;
                    bottom: 0;
                    background: linear-gradient(to top, rgba(0,0,0,0.95) 0%, rgba(0,0,0,0.7) 50%, transparent 100%);
                "></div>
                <div style="position: relative; z-index: 1; width: 100%;">
                    <h1 style="color: white; font-size: 3.5em; margin-bottom: 15px; text-shadow: 3px 3px 6px rgba(0,0,0,0.9); font-weight: bold;">
                        {full_details.get('title', 'N/A')}
                    </h1>
                    <div style="display: flex; gap: 20px; color: white; margin-bottom: 20px; font-size: 1.1em;">
                        <span>⭐ <strong>{full_details.get('rating', 0):.1f}</strong></span>
                        <span>•</span>
                        <span>{full_details.get('year', 'N/A')}</span>
                        <span>•</span>
                        <span>{full_details.get('runtime', 'N/A')} min</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Main content
        col1, col2 = st.columns([1, 2], gap="large")
        
        with col1:
            if full_details.get('poster'):
                st.image(full_details.get('poster'), use_container_width=True)
        
        with col2:
            if full_details.get('tagline'):
                st.markdown(f"### *{full_details.get('tagline')}*")
                st.markdown("---")
            
            if full_details.get('overview'):
                st.markdown("### 📖 Overview")
                st.write(full_details.get('overview'))
            
            if full_details.get('genres'):
                st.markdown("### 🎭 Genres")
                genre_html = "<div style='display:flex; flex-wrap:wrap; gap:10px; margin-bottom: 20px;'>"
                for genre in full_details.get('genres', []):
                    color = GENRE_COLORS.get(genre, "#555")
                    genre_html += (
                        f"<span style='background:transparent; color:{color}; border:2px solid {color}; "
                        f"border-radius:9999px; padding:8px 16px; font-size:0.95em; font-weight:700;'>"
                        f"{genre}</span>"
                    )
                genre_html += "</div>"
                st.markdown(genre_html, unsafe_allow_html=True)
            
            details_cols = st.columns(2)
            with details_cols[0]:
                if full_details.get('director'):
                    st.markdown("### 🎬 Director")
                    st.write(f"**{full_details.get('director')}**")
            
            with details_cols[1]:
                if full_details.get('production_companies'):
                    st.markdown("### 🏢 Production")
                    st.write("**" + ", ".join(full_details.get('production_companies', [])) + "**")
            
            if full_details.get('cast'):
                st.markdown("### 👥 Top Cast")
                cast_list = full_details.get('cast', [])
                num_cols = min(5, len(cast_list))
                cast_cols = st.columns(num_cols)
                for i, actor in enumerate(cast_list[:num_cols]):
                    with cast_cols[i]:
                        if actor.get('profile_path'):
                            st.image(actor['profile_path'], width=120)
                        st.write(f"**{actor.get('name', 'N/A')}**")
                        st.caption(f"as {actor.get('character', 'N/A')}")
            
            if full_details.get('trailer_key'):
                st.markdown("### 🎥 Trailer")
                st.video(f"https://www.youtube.com/watch?v={full_details.get('trailer_key')}")
            
            with st.expander("📊 Additional Information"):
                if full_details.get('release_date'):
                    st.write(f"**Release Date:** {full_details.get('release_date')}")
                if full_details.get('status'):
                    st.write(f"**Status:** {full_details.get('status')}")
                if full_details.get('vote_count'):
                    st.write(f"**Total Votes:** {full_details.get('vote_count'):,}")
                if full_details.get('budget') and full_details.get('budget') > 0:
                    st.write(f"**Budget:** ${full_details.get('budget'):,}")
                if full_details.get('revenue') and full_details.get('revenue') > 0:
                    st.write(f"**Revenue:** ${full_details.get('revenue'):,}")
                if full_details.get('imdb_id'):
                    st.markdown(f"**[View on IMDB](https://www.imdb.com/title/{full_details.get('imdb_id')})**")
                if full_details.get('homepage'):
                    st.markdown(f"**[Official Website]({full_details.get('homepage')})**")
    else:
        st.error("❌ Could not load movie details. Please try again.")
        if st.button("← Back to Recommendations"):
            st.query_params.clear()
            st.rerun()
else:
    # Main recommendation page
    st.title("🎬 AI-Powered Movie Recommendation System")
    
    movie_name = st.text_input("Enter a movie name (e.g., Inception, Avatar, Titanic):")

    if movie_name:
        recs = recommend(movie_name, movies, sim_matrix)
        if recs.empty:
            st.error("❌ Movie not found. Try another title.")
        else:
            st.subheader("Recommended Movies:")
            
            # Create consistent grid layout - 5 columns with equal spacing
            num_cols = 5
            cols = st.columns(num_cols)
            
            for idx, (_, row) in enumerate(recs.iterrows()):
                col_idx = idx % num_cols
                with cols[col_idx]:
                    # Use cached details to avoid unnecessary fetching - no delay needed since cached
                    details = get_cached_movie_details(row['title'])
                    poster = details['poster'] or "https://via.placeholder.com/200x300?text=No+Poster"
                    year = details['year'] or "N/A"
                    rating = details['rating'] or "N/A"

                    # Use original title from dataset (includes year for better matching)
                    original_title = row['title']
                    title_clean = original_title.split('(')[0].strip()
                    genres = [g.strip() for g in row['genres'].split('|') if g.strip()]

                    # Build genres into rows of max 2 per line (outlined, subtle pills)
                    genre_rows_html = ""
                    for i in range(0, len(genres), 2):
                        chunk = genres[i:i+2]
                        row_html = "<div style='display:flex; justify-content:center; gap:8px; margin:6px 0;'>"
                        for g in chunk:
                            color = GENRE_COLORS.get(g, "#555")
                            row_html += (
                                f"<span style='background:transparent; color:{color}; border:1px solid {color}; "
                                f"border-radius:9999px; padding:4px 10px; font-size:0.78em; font-weight:700; "
                                f"letter-spacing:0.2px; display:inline-block; line-height:1;'>"
                                f"{g}</span>"
                            )
                        row_html += "</div>"
                        genre_rows_html += row_html

                    # Display card with consistent spacing and make it clickable
                    card_html = (
                        "<div style='background:transparent; border:none; border-radius:14px; "
                        "padding:0; margin:8px auto 20px auto; max-width:220px; text-align:center; "
                        "box-shadow:none; cursor:pointer;'>"
                        f"<img src='{poster}' alt='poster' style='width:150px; height:auto; display:block; margin:0 auto 10px auto; border-radius:10px;'/>"
                        f"<div style='font-weight:700; color:#e5e7eb; margin-bottom:4px;'>{title_clean} ({year})</div>"
                        f"<div style='color:#fbbf24; margin-bottom:2px;'>⭐ {rating}</div>"
                        f"{genre_rows_html}"
                        "</div>"
                    )

                    st.markdown(card_html, unsafe_allow_html=True)
                    
                    # Button to navigate to movie details page - use original title for accurate matching
                    if st.button("View Details", key=f"view_{idx}_{row['title']}", use_container_width=True):
                        st.query_params["movie"] = original_title
                        st.rerun()