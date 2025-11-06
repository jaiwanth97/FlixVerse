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

# Global Netflix-like theming
st.markdown(
    """
    <style>
      :root { --bg: #141414; --text:#e5e5e5; --muted:#b3b3b3; --red:#e50914; --red-dark:#b20710; --yellow:#f5c518; }
      html, body, [data-testid="stAppViewContainer"] { background: var(--bg); color: var(--text); }
      html { scroll-behavior: smooth; }
      body { margin: 0; }
      [data-testid="stAppViewContainer"] > .main { padding-top: 64px !important; }
      .block-container { padding-top: 50px !important; }
      [data-testid="stHeader"], footer, #MainMenu { display: none; }
      .container { padding: 0 24px; }
      h1, h2, h3, h4, h5, h6 { color: var(--text); letter-spacing: -0.02em; }
      h1 { font-weight: 900; }
      h2, h3 { font-weight: 800; }

      /* Navbar */
      .navbar { position: fixed; top: 0; left: 0; right: 0; z-index: 100; display: flex; align-items: center; justify-content: flex-start; gap: 24px; padding: 12px 24px; background: rgba(0,0,0,0.85); width: 100%; margin-left: 0; margin-right: 0; margin-top: 0; border-bottom: 2px solid var(--red); }
      .navbar .nav-left { display: flex; align-items: center; gap: 24px; }
      .navbar .brand { color: var(--red); font-size: 24px; font-weight: 900; text-transform: uppercase; letter-spacing: 0.5px; }
      .navbar .links { display: flex; gap: 30px; color: var(--muted); font-weight: 600; align-items: center; }
      .navbar .links a { 
        color: var(--muted); 
        text-decoration: none; 
        cursor: pointer; 
        padding: 8px 16px;
        border-radius: 6px;
        transition: all 0.3s ease;
        font-size: 14px;
        font-weight: 600;
        letter-spacing: 0.3px;
        border: 1px solid transparent;
        background: rgba(255, 255, 255, 0.05);
      }
      .navbar .links a:hover { 
        color: #fff; 
        background: rgba(229, 9, 20, 0.2);
        border-color: rgba(229, 9, 20, 0.4);
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(229, 9, 20, 0.2);
      }
      .navbar .links a:active {
        transform: translateY(0);
        box-shadow: 0 2px 4px rgba(229, 9, 20, 0.15);
      }
      /* Right-side navbar elements removed for now */

      /* Cards */
      .movie-card { 
        background: transparent; 
        border-radius: 12px; 
        transition: transform .25s ease, box-shadow .25s ease; 
        padding: 10px; 
        width: 220px; 
        height: 500px; 
        display: flex; 
        flex-direction: column; 
        align-items: center; 
        gap: 6px; 
        box-sizing: border-box;
      }
      .movie-card img { width: 150px; height: 225px; object-fit: cover; border-radius: 10px; display: block; margin: 0 auto; box-shadow: 0 12px 24px rgba(0,0,0,0.35); flex-shrink: 0; }
      .movie-card:hover { transform: scale(1.03); z-index: 2; }
      .poster-title { 
        font-weight: 800; 
        color: #f5f5f5; 
        margin: 8px 0 2px; 
        font-size: 0.98rem; 
        text-align: center; 
        line-height: 1.3; 
        height: calc(1.3em * 2); 
        overflow: hidden; 
        display: -webkit-box; 
        -webkit-line-clamp: 2; 
        -webkit-box-orient: vertical;
        flex-shrink: 0;
      }
      .poster-meta { 
        color: var(--yellow); 
        text-align: center; 
        font-weight: 700; 
        margin-bottom: 6px; 
        height: 22px; 
        flex-shrink: 0;
      }
      .genre-row { 
        display:flex; 
        justify-content:center; 
        gap:8px; 
        margin: 2px 0; 
        flex-wrap: wrap; 
      }
      .genre-pill { 
        border: 1px solid rgba(255,255,255,0.25); 
        color: var(--muted); 
        border-radius: 999px; 
        padding: 4px 10px; 
        font-size: 0.78em; 
        font-weight: 700; 
        letter-spacing: 0.2px; 
        line-height: 1; 
      }
      .genre-stack { 
        height: 90px; 
        max-height: 90px; 
        display: flex; 
        flex-direction: column; 
        justify-content: flex-start; 
        align-items: center;
        overflow: hidden;
        flex-shrink: 0;
        width: 100%;
      }
      .card-spacer { 
        flex: 1 1 auto; 
        min-height: 0;
      }
      .btn-link { display: inline-block; text-decoration: none; width: 100%; flex-shrink: 0; }
      .movie-card a, .btn-link, .btn-link:link, .btn-link:visited, .btn-link:hover, .btn-link:active { text-decoration: none !important; border-bottom: none !important; }
      .btn-primary { background: var(--red); color: #fff; border: none; border-radius: 6px; padding: 10px 12px; font-weight: 800; text-align:center; box-shadow: 0 6px 14px rgba(229,9,20,0.25); transition: background .2s ease; text-decoration: none !important; width: 100%; }
      .btn-primary:hover { background: var(--red-dark); }

      /* Buttons */
      .stButton>button { background: var(--red); color: #fff; border: none; border-radius: 4px; padding: 10px 12px; font-weight: 800; transition: background .2s ease; box-shadow: 0 6px 14px rgba(229,9,20,0.25); }
      .stButton>button:hover { background: var(--red-dark); }
      .stButton>button:focus { outline: 2px solid rgba(229,9,20,0.5); }

      /* Specific styling for the top movie search button */
      #movie-search .stButton>button {
        background: #d8242491 !important; /* your custom color */
        color: #fff !important;
        border: none !important;
      }   
      #movie-search .stButton>button:hover {
        filter: brightness(1.05);
      }

      /* Inputs */
      input, textarea { background: #0f0f0f !important; color: var(--text) !important; border: 1px solid rgba(255,255,255,0.15) !important; }

      /* Hero */
      .hero { height: 480px; position: relative; margin: -16px -16px 24px -16px; display: flex; align-items: flex-end; }
      .hero::before { content: ""; position: absolute; inset: 0; background: rgba(0,0,0,0.25); }
      .hero::after { content: ""; position: absolute; inset: 0; background: linear-gradient(180deg, rgba(0,0,0,0) 0%, rgba(20,20,20,0.85) 65%, rgba(20,20,20,1) 100%); }
      .hero-content { position: relative; z-index: 1; width: 100%; padding: 32px; }
      .hero-title { color: #fff; font-size: 3.2rem; line-height: 1.1; margin-bottom: 14px; text-shadow: 3px 3px 6px rgba(0,0,0,0.6); font-weight: 900; }
      .hero-meta { display: flex; gap: 20px; color: #fff; margin-bottom: 16px; font-size: 1.05rem; font-weight: 700; }

      /* Subtle dividers */
      hr { border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 12px 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Top Navbar
st.markdown(
    """
    <div class="navbar">
      <div class="nav-left">
        <div class="brand">FilmVerse</div>
        <div class="links">
          <a href="?" target="_self">Home</a>
          <a href="?view=recommendations#top" target="_self">Find your next movie</a>
          <a href="?view=surprise-me" target="_self">Surprise Me</a>
          <a href="?view=top-rated" target="_self">Top Rated Movies</a>
          <a href="?view=advanced-search" target="_self">Advanced Search</a>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

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

# Determine current view (home vs advanced search)
current_view = query_params.get("view")
if isinstance(current_view, list):
    current_view = current_view[0] if current_view else None

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

# Advanced Search renderer
def render_advanced_search():
    st.title("Advanced Search")

    def get_all_genres(df):
        if 'genres' not in df.columns:
            return []
        all_genres = set()
        for g in df['genres'].dropna().astype(str):
            for part in g.split('|'):
                part = part.strip()
                if part:
                    all_genres.add(part)
        return sorted(list(all_genres))

    def find_movie_index_by_title(df, title_query):
        if not title_query:
            return None
        title_query_lower = title_query.lower().strip()
        exact = df.index[df['title'].astype(str).str.lower() == title_query_lower]
        if len(exact) > 0:
            return int(exact[0])
        sw = df.index[df['title'].astype(str).str.lower().str.startswith(title_query_lower)]
        if len(sw) > 0:
            return int(sw[0])
        contains = df.index[df['title'].astype(str).str.lower().str.contains(title_query_lower, na=False)]
        if len(contains) > 0:
            return int(contains[0])
        return None

    def normalize_series(s):
        s = pd.to_numeric(s, errors='coerce').fillna(0)
        if s.max() == s.min():
            return pd.Series([0.0]*len(s), index=s.index)
        return (s - s.min()) / (s.max() - s.min())

    def extract_year_from_title(title):
        """Extract year from title like 'Movie Name (2005)'"""
        import re
        match = re.search(r'\((\d{4})\)', str(title))
        return int(match.group(1)) if match else None

    with st.form("adv_search_form"):
        c1, c2, c3 = st.columns([1,1,1])
        with c1:
            selected_genres = st.multiselect("Genre(s)", options=get_all_genres(movies))
            rating_min = st.slider("Min rating", 0.0, 10.0, 6.0, 0.1)
            language = st.text_input("Language (e.g., en, hi, fr)")
        with c2:
            # Extract years from titles to get min/max
            years = movies['title'].apply(extract_year_from_title).dropna()
            if len(years) > 0:
                min_year = int(years.min())
                max_year = int(years.max())
            else:
                min_year = 1990
                max_year = 2024
            year_range = st.slider("Release year range", min_year, max_year, (max(min_year, max_year-20), max_year))
            actor = st.text_input("Actor contains")
        with c3:
            director = st.text_input("Director contains")
            keywords = st.text_input("Keywords (mood/theme/plot)")
            similar_to = st.text_input("Similar to (movie)")

        submitted = st.form_submit_button("Search")

    if not submitted:
        return

    df = movies.copy()

    if selected_genres and 'genres' in df.columns:
        mask = df['genres'].fillna('').apply(lambda x: all(g in x for g in selected_genres))
        df = df[mask]

    # Extract year from title and filter by year range
    df['extracted_year'] = df['title'].apply(extract_year_from_title)
    df = df[df['extracted_year'].notna()]  # Remove movies without year
    df = df[(df['extracted_year'] >= year_range[0]) & (df['extracted_year'] <= year_range[1])]
    df = df.drop('extracted_year', axis=1)  # Remove temporary column

    rating_col = 'rating' if 'rating' in df.columns else ('vote_average' if 'vote_average' in df.columns else None)
    if rating_col:
        df = df[pd.to_numeric(df[rating_col], errors='coerce').fillna(0) >= rating_min]

    lang_col = 'language' if 'language' in df.columns else ('original_language' if 'original_language' in df.columns else None)
    if language and lang_col:
        df = df[df[lang_col].astype(str).str.contains(language.strip(), case=False, na=False)]

    if keywords:
        kw = keywords.strip()
        cols = [c for c in ['title','overview','tagline','genres'] if c in df.columns]
        if cols:
            kw_mask = False
            for c in cols:
                kw_mask = kw_mask | df[c].astype(str).str.contains(kw, case=False, na=False)
            df = df[kw_mask]

    # Filter by director and actor - need to fetch from TMDB since dataset doesn't have these
    # Do this BEFORE scoring to avoid unnecessary API calls
    if director or actor:
        # Filter movies by fetching their details from TMDB
        filtered_indices = []
        progress_bar = None
        status_text = None
        
        total_movies = len(df)
        if total_movies > 10:  # Only show progress if many movies
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        for enum_idx, (original_idx, row) in enumerate(df.iterrows()):
            if progress_bar and enum_idx % 5 == 0:
                progress_bar.progress((enum_idx + 1) / total_movies)
                status_text.text(f"Checking movie details... {enum_idx + 1}/{total_movies}")
            
            title_val = str(row.get('title', 'Unknown'))
            full_details = get_cached_full_movie_details(title_val)
            
            # Check director match
            director_match = True
            if director:
                director_match = False
                if full_details and full_details.get('director'):
                    movie_director = full_details.get('director', '').lower()
                    if director.strip().lower() in movie_director:
                        director_match = True
            
            # Check actor match
            actor_match = True
            if actor:
                actor_match = False
                if full_details and full_details.get('cast'):
                    cast_names = [c.get('name', '').lower() for c in full_details.get('cast', [])]
                    actor_query = actor.strip().lower()
                    for cast_name in cast_names:
                        if actor_query in cast_name:
                            actor_match = True
                            break
            
            # If both matches pass, include this movie
            if director_match and actor_match:
                filtered_indices.append(original_idx)
        
        if progress_bar:
            progress_bar.empty()
            status_text.empty()
        
        # Filter dataframe to only include matching indices
        if filtered_indices:
            df = df.loc[filtered_indices]
        else:
            df = df.iloc[[]]  # Empty dataframe if no matches

    score = pd.Series(0.0, index=df.index)
    if 'similar_to' in locals() and similar_to:
        idx = find_movie_index_by_title(movies, similar_to)
        if idx is not None:
            sim_vec = pd.Series(sim_matrix[idx], index=movies.index)
            score = score.add(normalize_series(sim_vec.reindex(df.index).fillna(0)) * 0.55, fill_value=0)

    rcol2 = 'rating' if 'rating' in df.columns else ('vote_average' if 'vote_average' in df.columns else ('avg_rating' if 'avg_rating' in df.columns else None))
    if rcol2:
        score = score.add(normalize_series(pd.to_numeric(df[rcol2], errors='coerce').fillna(0)) * 0.30, fill_value=0)

    # Use extracted year for scoring
    df['extracted_year'] = df['title'].apply(extract_year_from_title)
    score = score.add(normalize_series(pd.to_numeric(df['extracted_year'], errors='coerce').fillna(0)) * 0.15, fill_value=0)
    df = df.drop('extracted_year', axis=1)  # Remove temporary column

    df = df.loc[score.sort_values(ascending=False).head(10).index]

    st.markdown("### Results")
    if df.empty:
        st.info("No matches found. Try relaxing your filters.")
        return

    num_cols = 5
    cols2 = st.columns(num_cols)
    for jdx, (_, r) in enumerate(df.iterrows()):
        cidx = jdx % num_cols
        with cols2[cidx]:
            title_value = str(r.get('title', 'Unknown'))
            title_clean_adv = title_value.split('(')[0].strip()
            details = get_cached_movie_details(title_value)
            poster = (details.get('poster') if details else None) or "https://via.placeholder.com/200x300?text=No+Poster"
            year_val = r.get('year') or (details.get('year') if details else 'N/A')
            rcol = 'rating' if 'rating' in r.index else ('vote_average' if 'vote_average' in r.index else None)
            rating_display = r.get(rcol) if rcol else (details.get('rating') if details else 'N/A')
            genres2 = []
            if 'genres' in r and isinstance(r['genres'], str):
                genres2 = [g.strip() for g in r['genres'].split('|') if g.strip()]

            genre_rows_html2 = "<div class='genre-stack'>"
            for i2 in range(0, len(genres2), 2):
                chunk2 = genres2[i2:i2+2]
                row_html2 = "<div class='genre-row'>"
                for g2 in chunk2:
                    color2 = GENRE_COLORS.get(g2, "#555")
                    row_html2 += (
                        f"<span class='genre-pill' style='border-color:{color2}; color:{color2};'>"
                        f"{g2}</span>"
                    )
                row_html2 += "</div>"
                genre_rows_html2 += row_html2
            genre_rows_html2 += "</div>"

            encoded2 = urllib.parse.quote(title_value)
            card_html2 = (
                "<div class='movie-card'>"
                f"<img src='{poster}' alt='poster'/>"
                f"<div class='poster-title'>{title_clean_adv} ({year_val})</div>"
                f"<div class='poster-meta'>⭐ {rating_display}</div>"
                f"{genre_rows_html2}"
                "<div class='card-spacer'></div>"
                f"<a class='btn-link' href='?movie={encoded2}' target='_self'><div class='btn-primary'>View Details</div></a>"
                "</div>"
            )
            st.markdown(card_html2, unsafe_allow_html=True)

# Surprise Me renderer
def render_surprise_me():
    st.title("🎲 Surprise Me!")
    st.markdown("Discover a random movie from our collection!")
    
    # Initialize session state for surprise movie
    if 'surprise_movie' not in st.session_state:
        st.session_state.surprise_movie = None
        st.session_state.surprise_recs = None
    
    # Get a random movie
    if st.button("🎲 Get Surprised!", use_container_width=False):
        random_movie = movies.sample(n=1).iloc[0]
        random_title = random_movie['title']
        
        # Get recommendations based on the random movie
        recs = recommend(random_title, movies, sim_matrix)
        
        # Remove the random movie from recommendations
        if not recs.empty:
            recs = recs[recs['title'] != random_title]
        
        # Store in session state
        st.session_state.surprise_movie = random_movie
        st.session_state.surprise_recs = recs
    
    # Display the surprise movie if available
    if st.session_state.surprise_movie is not None:
        random_movie = st.session_state.surprise_movie
        random_title = random_movie['title']
        recs = st.session_state.surprise_recs
        
        st.subheader(f"Your Surprise Movie: {random_title.split('(')[0].strip()}")
        
        # Display the random movie
        details = get_cached_movie_details(random_title)
        poster = details['poster'] or "https://via.placeholder.com/200x300?text=No+Poster"
        year = details['year'] or "N/A"
        rating = details['rating'] or "N/A"
        title_clean = random_title.split('(')[0].strip()
        genres = [g.strip() for g in random_movie['genres'].split('|') if g.strip()] if isinstance(random_movie.get('genres'), str) else []

        genre_rows_html = "<div class='genre-stack'>"
        for i in range(0, len(genres), 2):
            chunk = genres[i:i+2]
            row_html = "<div class='genre-row'>"
            for g in chunk:
                color = GENRE_COLORS.get(g, "#555")
                row_html += (
                    f"<span class='genre-pill' style='border-color:{color}; color:{color};'>"
                    f"{g}</span>"
                )
            row_html += "</div>"
            genre_rows_html += row_html
        genre_rows_html += "</div>"

        encoded = urllib.parse.quote(random_title)
        card_html = (
            "<div class='movie-card'>"
            f"<img src='{poster}' alt='poster'/>"
            f"<div class='poster-title'>{title_clean} ({year})</div>"
            f"<div class='poster-meta'>⭐ {rating}</div>"
            f"{genre_rows_html}"
            "<div class='card-spacer'></div>"
            f"<a class='btn-link' href='?movie={encoded}' target='_self'><div class='btn-primary'>View Details</div></a>"
            "</div>"
        )
        st.markdown(card_html, unsafe_allow_html=True)
        
        # Show similar movies
        if recs is not None and not recs.empty:
            st.subheader("Similar Movies You Might Like:")
            num_cols = 5
            cols = st.columns(num_cols)
            
            for idx, (_, row) in enumerate(recs.iterrows()):
                col_idx = idx % num_cols
                with cols[col_idx]:
                    details = get_cached_movie_details(row['title'])
                    poster = details['poster'] or "https://via.placeholder.com/200x300?text=No+Poster"
                    year = details['year'] or "N/A"
                    rating = details['rating'] or "N/A"
                    original_title = row['title']
                    title_clean = original_title.split('(')[0].strip()
                    genres = [g.strip() for g in row['genres'].split('|') if g.strip()]

                    genre_rows_html = "<div class='genre-stack'>"
                    for i in range(0, len(genres), 2):
                        chunk = genres[i:i+2]
                        row_html = "<div class='genre-row'>"
                        for g in chunk:
                            color = GENRE_COLORS.get(g, "#555")
                            row_html += (
                                f"<span class='genre-pill' style='border-color:{color}; color:{color};'>"
                                f"{g}</span>"
                            )
                        row_html += "</div>"
                        genre_rows_html += row_html
                    genre_rows_html += "</div>"

                    encoded = urllib.parse.quote(original_title)
                    card_html = (
                        "<div class='movie-card'>"
                        f"<img src='{poster}' alt='poster'/>"
                        f"<div class='poster-title'>{title_clean} ({year})</div>"
                        f"<div class='poster-meta'>⭐ {rating}</div>"
                        f"{genre_rows_html}"
                        "<div class='card-spacer'></div>"
                        f"<a class='btn-link' href='?movie={encoded}' target='_self'><div class='btn-primary'>View Details</div></a>"
                        "</div>"
                    )
                    st.markdown(card_html, unsafe_allow_html=True)

# Top Rated Movies renderer
def render_top_rated():
    st.title("⭐ Top Rated Movies")
    st.markdown("Discover the highest-rated movies in our collection!")
    
    # Determine rating column - check for avg_rating first (from train_model), then fallback to rating/vote_average
    rating_col = None
    if 'avg_rating' in movies.columns:
        rating_col = 'avg_rating'
    elif 'rating' in movies.columns:
        rating_col = 'rating'
    elif 'vote_average' in movies.columns:
        rating_col = 'vote_average'
    
    if rating_col:
        # Get top rated movies (sorted by rating, then by vote count if available)
        top_movies = movies.copy()
        top_movies['rating_num'] = pd.to_numeric(top_movies[rating_col], errors='coerce').fillna(0)
        
        # Filter out movies with 0 rating (no ratings available)
        top_movies = top_movies[top_movies['rating_num'] > 0]
        
        # Sort by rating (descending) and limit to top 50
        top_movies = top_movies.sort_values('rating_num', ascending=False).head(50)
        
        if top_movies.empty:
            st.warning("⚠️ No rated movies found in the dataset.")
            return
        
        st.subheader(f"Top {len(top_movies)} Highest-Rated Movies")
        
        # Show progress indicator while fetching movie details
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Create consistent grid layout - 5 columns
        num_cols = 5
        cols = st.columns(num_cols)
        
        total_movies = len(top_movies)
        
        for idx, (_, row) in enumerate(top_movies.iterrows()):
            col_idx = idx % num_cols
            with cols[col_idx]:
                try:
                    # Update progress (only every 5 movies to reduce overhead)
                    if idx % 5 == 0 or idx == total_movies - 1:
                        progress = (idx + 1) / total_movies
                        progress_bar.progress(progress)
                        status_text.text(f"Loading movie details... {idx + 1}/{total_movies}")
                    
                    title_val = str(row.get('title', 'Unknown'))
                    details = get_cached_movie_details(title_val)
                    poster = (details.get('poster') if details else None) or "https://via.placeholder.com/200x300?text=No+Poster"
                    year_val = (details.get('year') if details else None) or row.get('year', 'N/A')
                    
                    # Use dataset rating if available, otherwise use TMDB rating
                    rating_display = row.get(rating_col, 'N/A')
                    if rating_display == 'N/A' or pd.isna(rating_display):
                        rating_display = details.get('rating') if details else 'N/A'
                    
                    # Format rating display
                    if isinstance(rating_display, (int, float)):
                        rating_display = f"{rating_display:.1f}"

                    title_clean = title_val.split('(')[0].strip()
                    genres = []
                    if isinstance(row.get('genres'), str):
                        genres = [g.strip() for g in row['genres'].split('|') if g.strip()]

                    genre_rows_html = "<div class='genre-stack'>"
                    for i in range(0, len(genres), 2):
                        chunk = genres[i:i+2]
                        row_html = "<div class='genre-row'>"
                        for g in chunk:
                            color = GENRE_COLORS.get(g, "#555")
                            row_html += (
                                f"<span class='genre-pill' style='border-color:{color}; color:{color};'>"
                                f"{g}</span>"
                            )
                        row_html += "</div>"
                        genre_rows_html += row_html
                    genre_rows_html += "</div>"

                    encoded = urllib.parse.quote(title_val)
                    card_html = (
                        "<div class='movie-card'>"
                        f"<img src='{poster}' alt='poster'/>"
                        f"<div class='poster-title'>{title_clean} ({year_val})</div>"
                        f"<div class='poster-meta'>⭐ {rating_display}</div>"
                        f"{genre_rows_html}"
                        "<div class='card-spacer'></div>"
                        f"<a class='btn-link' href='?movie={encoded}' target='_self'><div class='btn-primary'>View Details</div></a>"
                        "</div>"
                    )
                    st.markdown(card_html, unsafe_allow_html=True)
                except Exception as e:
                    # Handle errors gracefully - show movie card without details
                    title_val = str(row.get('title', 'Unknown'))
                    title_clean = title_val.split('(')[0].strip()
                    rating_display = row.get(rating_col, 'N/A')
                    if isinstance(rating_display, (int, float)):
                        rating_display = f"{rating_display:.1f}"
                    
                    encoded = urllib.parse.quote(title_val)
                    card_html = (
                        "<div class='movie-card'>"
                        f"<img src='https://via.placeholder.com/200x300?text=No+Poster' alt='poster'/>"
                        f"<div class='poster-title'>{title_clean}</div>"
                        f"<div class='poster-meta'>⭐ {rating_display}</div>"
                        "<div class='card-spacer'></div>"
                        f"<a class='btn-link' href='?movie={encoded}' target='_self'><div class='btn-primary'>View Details</div></a>"
                        "</div>"
                    )
                    st.markdown(card_html, unsafe_allow_html=True)
        
        # Clear progress indicator
        progress_bar.empty()
        status_text.empty()
    else:
        st.error("❌ Rating information not available in the dataset.")

# If movie is selected via query params, show detailed view
if current_view == "advanced-search":
    render_advanced_search()
elif current_view == "surprise-me":
    render_surprise_me()
elif current_view == "top-rated":
    render_top_rated()
elif selected_movie_title:
    st.title(f"🎬 {selected_movie_title}")
    
    # Back button
    if st.button("← Back to Recommendations", key="back_button_1"):
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
        if st.button("← Back to Recommendations", key="back_button_2"):
            st.query_params.clear()
            st.rerun()
    elif full_details:
        backdrop = full_details.get('backdrop') or full_details.get('poster') or ''
        
        # Hero section with backdrop
        if backdrop:
            st.markdown(
                f"""
                <div class=\"hero\" style=\"background-image:url('{backdrop}'); background-size:cover; background-position:center;\">
                  <div class=\"hero-content\">
                    <div class=\"hero-title\">{full_details.get('title', 'N/A')}</div>
                    <div class=\"hero-meta\">
                      <span>⭐ <strong>{full_details.get('rating', 0):.1f}</strong></span>
                      <span>•</span>
                      <span>{full_details.get('year', 'N/A')}</span>
                      <span>•</span>
                      <span>{full_details.get('runtime', 'N/A')} min</span>
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        
        # Main content
        col1, col2 = st.columns([1, 2], gap="large")
        
        with col1:
            if full_details.get('poster'):
                st.image(full_details.get('poster'), use_container_width=True)
        
        with col2:
            if full_details.get('tagline'):
                st.markdown(f"### *{full_details.get('tagline')}*")
                st.markdown("<hr />", unsafe_allow_html=True)
            
            if full_details.get('overview'):
                st.markdown("### 📖 Overview")
                st.write(full_details.get('overview'))
            
            if full_details.get('genres'):
                st.markdown("### 🎭 Genres")
                genre_html = "<div style='display:flex; flex-wrap:wrap; gap:10px; margin-bottom: 20px;'>"
                for genre in full_details.get('genres', []):
                    color = GENRE_COLORS.get(genre, "#555")
                    genre_html += (
                        f"<span class='genre-pill' style='border-color:{color}; color:{color};'>"
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
        if st.button("← Back to Recommendations", key="back_button_3"):
            st.query_params.clear()
            st.rerun()
else:
    # Main recommendation page
    st.markdown('<div id="top"></div>', unsafe_allow_html=True)
    
    # Show title only on "Find your next movie" page
    if current_view == "recommendations":
        st.title("🎬 AI-Powered Movie Recommendation System")
    
    # Search input with button on the right (aligned)
    submitted_search = False
    st.markdown("<div id=\"movie-search\">", unsafe_allow_html=True)
    if current_view == "recommendations":
        st.markdown("Enter a movie name (e.g., Inception, Avatar, Titanic):")
    else:
        st.markdown("Enter a movie name :")
    with st.form("movie_search_form"):
        left_col, right_col = st.columns([12, 2])
        with left_col:
            movie_name = st.text_input(
                "movie_input",
                placeholder="Type a movie title...",
                label_visibility="collapsed",
            )
        with right_col:
            submitted_search = st.form_submit_button("Search", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if submitted_search and movie_name:
            # Search for movies matching the search term
            search_term = movie_name.lower().strip()
            
            def extract_year_from_title(title):
                """Extract year from title like 'Movie Name (2005)'"""
                import re
                match = re.search(r'\((\d{4})\)', str(title))
                return int(match.group(1)) if match else 0
            
            # First try exact match (case-insensitive)
            exact_match = movies[movies['title'].astype(str).str.lower() == search_term]
            
            # If no exact match, try title starts with search term
            if exact_match.empty:
                exact_match = movies[movies['title'].astype(str).str.lower().str.startswith(search_term)]
            
            # If still no match, try contains search term
            if exact_match.empty:
                exact_match = movies[movies['title'].astype(str).str.lower().str.contains(search_term, na=False)]
            
            if exact_match.empty:
                st.error("❌ Movie not found. Try another title.")
            else:
                # Sort by year descending to get the latest movie first
                exact_match = exact_match.copy()
                exact_match['extracted_year'] = exact_match['title'].apply(extract_year_from_title)
                exact_match = exact_match.sort_values('extracted_year', ascending=False)
                
                # Get the latest matching movie title
                searched_movie_title = exact_match.iloc[0]['title']
                
                # Only get recommendations if we're on the "Find your next movie" page
                show_recommendations = (current_view == "recommendations")
                
                if show_recommendations:
                    # Get recommendations based on the searched movie
                    recs = recommend(searched_movie_title, movies, sim_matrix)
                    
                    # Remove the searched movie from recommendations to avoid duplication
                    if not recs.empty:
                        recs = recs[recs['title'] != searched_movie_title]
                    
                    # Display the searched movie first
                    st.subheader(f"You searched for: {searched_movie_title.split('(')[0].strip()}")
                    
                    # Display searched movie card
                    searched_row = exact_match.iloc[0]
                    cols_searched = st.columns(5)
                    with cols_searched[0]:
                        details = get_cached_movie_details(searched_movie_title)
                        poster = details['poster'] or "https://via.placeholder.com/200x300?text=No+Poster"
                        year = details['year'] or "N/A"
                        rating = details['rating'] or "N/A"
                        title_clean = searched_movie_title.split('(')[0].strip()
                        genres = [g.strip() for g in str(searched_row.get('genres','')).split('|') if g.strip()]

                        genre_rows_html = "<div class='genre-stack'>"
                        for i in range(0, len(genres), 2):
                            chunk = genres[i:i+2]
                            row_html = "<div class='genre-row'>"
                            for g in chunk:
                                color = GENRE_COLORS.get(g, "#555")
                                row_html += (
                                    f"<span class='genre-pill' style='border-color:{color}; color:{color};'>"
                                    f"{g}</span>"
                                )
                            row_html += "</div>"
                            genre_rows_html += row_html
                        genre_rows_html += "</div>"

                        encoded = urllib.parse.quote(searched_movie_title)
                        card_html = (
                            "<div class='movie-card'>"
                            f"<img src='{poster}' alt='poster'/>"
                            f"<div class='poster-title'>{title_clean} ({year})</div>"
                            f"<div class='poster-meta'>⭐ {rating}</div>"
                            f"{genre_rows_html}"
                            "<div class='card-spacer'></div>"
                            f"<a class='btn-link' href='?movie={encoded}' target='_self'><div class='btn-primary'>View Details</div></a>"
                            "</div>"
                        )
                        st.markdown(card_html, unsafe_allow_html=True)
                    
                    # Display recommendations
                    if not recs.empty:
                        st.subheader("Recommended Movies Based on Your Search:")
                        
                        num_cols = 5
                        cols = st.columns(num_cols)
                        
                        for idx, (_, row) in enumerate(recs.iterrows()):
                            col_idx = idx % num_cols
                            with cols[col_idx]:
                                details = get_cached_movie_details(row['title'])
                                poster = details['poster'] or "https://via.placeholder.com/200x300?text=No+Poster"
                                year = details['year'] or "N/A"
                                rating = details['rating'] or "N/A"
                                original_title = row['title']
                                title_clean = original_title.split('(')[0].strip()
                                genres = [g.strip() for g in str(row.get('genres','')).split('|') if g.strip()]

                                genre_rows_html = "<div class='genre-stack'>"
                                for i in range(0, len(genres), 2):
                                    chunk = genres[i:i+2]
                                    row_html = "<div class='genre-row'>"
                                    for g in chunk:
                                        color = GENRE_COLORS.get(g, "#555")
                                        row_html += (
                                            f"<span class='genre-pill' style='border-color:{color}; color:{color};'>"
                                            f"{g}</span>"
                                        )
                                    row_html += "</div>"
                                    genre_rows_html += row_html
                                genre_rows_html += "</div>"

                                encoded = urllib.parse.quote(original_title)
                                card_html = (
                                    "<div class='movie-card'>"
                                    f"<img src='{poster}' alt='poster'/>"
                                    f"<div class='poster-title'>{title_clean} ({year})</div>"
                                    f"<div class='poster-meta'>⭐ {rating}</div>"
                                    f"{genre_rows_html}"
                                    "<div class='card-spacer'></div>"
                                    f"<a class='btn-link' href='?movie={encoded}' target='_self'><div class='btn-primary'>View Details</div></a>"
                                    "</div>"
                                )
                                st.markdown(card_html, unsafe_allow_html=True)
                    else:
                        st.info("No similar movies found for this selection.")
                else:
                    # On home page, show all matching movies (latest first)
                    st.subheader("Search Results")

                    results = exact_match.head(50)
                    num_cols = 5
                    cols = st.columns(num_cols)

                    for idx, (_, searched_row) in enumerate(results.iterrows()):
                        col_idx = idx % num_cols
                        with cols[col_idx]:
                            details = get_cached_movie_details(searched_row['title'])
                            poster = details['poster'] or "https://via.placeholder.com/200x300?text=No+Poster"
                            year = details['year'] or "N/A"
                            rating = details['rating'] or "N/A"
                            original_title = searched_row['title']
                            title_clean = original_title.split('(')[0].strip()
                            genres = [g.strip() for g in str(searched_row.get('genres','')).split('|') if g.strip()]

                            genre_rows_html = "<div class='genre-stack'>"
                            for i in range(0, len(genres), 2):
                                chunk = genres[i:i+2]
                                row_html = "<div class='genre-row'>"
                                for g in chunk:
                                    color = GENRE_COLORS.get(g, "#555")
                                    row_html += (
                                        f"<span class='genre-pill' style='border-color:{color}; color:{color};'>"
                                        f"{g}</span>"
                                    )
                                row_html += "</div>"
                                genre_rows_html += row_html
                            genre_rows_html += "</div>"

                            encoded = urllib.parse.quote(original_title)
                            card_html = (
                                "<div class='movie-card'>"
                                f"<img src='{poster}' alt='poster'/>"
                                f"<div class='poster-title'>{title_clean} ({year})</div>"
                                f"<div class='poster-meta'>⭐ {rating}</div>"
                                f"{genre_rows_html}"
                                "<div class='card-spacer'></div>"
                                f"<a class='btn-link' href='?movie={encoded}' target='_self'><div class='btn-primary'>View Details</div></a>"
                                "</div>"
                            )
                            st.markdown(card_html, unsafe_allow_html=True)

    else:
        # Show all available movies
        st.subheader("All Movies")

        # Create consistent grid layout - 5 columns
        num_cols = 5
        cols = st.columns(num_cols)

        for idx, (_, row) in enumerate(movies.iterrows()):
            col_idx = idx % num_cols
            with cols[col_idx]:
                title_val = str(row.get('title', 'Unknown'))
                details = get_cached_movie_details(title_val)
                poster = (details.get('poster') if details else None) or "https://via.placeholder.com/200x300?text=No+Poster"
                year_val = (details.get('year') if details else None) or row.get('year', 'N/A')
                rating_display = (details.get('rating') if details else None) or row.get('rating') or row.get('vote_average', 'N/A')

                title_clean = title_val.split('(')[0].strip()
                genres = []
                if isinstance(row.get('genres'), str):
                    genres = [g.strip() for g in row['genres'].split('|') if g.strip()]

                genre_rows_html = "<div class='genre-stack'>"
                for i in range(0, len(genres), 2):
                    chunk = genres[i:i+2]
                    row_html = "<div class='genre-row'>"
                    for g in chunk:
                        color = GENRE_COLORS.get(g, "#555")
                        row_html += (
                            f"<span class='genre-pill' style='border-color:{color}; color:{color};'>"
                            f"{g}</span>"
                        )
                    row_html += "</div>"
                    genre_rows_html += row_html
                genre_rows_html += "</div>"

                encoded = urllib.parse.quote(title_val)
                card_html = (
                    "<div class='movie-card'>"
                    f"<img src='{poster}' alt='poster'/>"
                    f"<div class='poster-title'>{title_clean} ({year_val})</div>"
                    f"<div class='poster-meta'>⭐ {rating_display}</div>"
                    f"{genre_rows_html}"
                    "<div class='card-spacer'></div>"
                    f"<a class='btn-link' href='?movie={encoded}' target='_self'><div class='btn-primary'>View Details</div></a>"
                    "</div>"
                )

                st.markdown(card_html, unsafe_allow_html=True)