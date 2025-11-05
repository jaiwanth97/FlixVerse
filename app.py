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
      .block-container { padding-top: 0 !important; }
      [data-testid="stHeader"], footer, #MainMenu { display: none; }
      .container { padding: 0 24px; }
      h1, h2, h3, h4, h5, h6 { color: var(--text); letter-spacing: -0.02em; }
      h1 { font-weight: 900; }
      h2, h3 { font-weight: 800; }

      /* Navbar */
      .navbar { position: fixed; top: 0; left: 0; right: 0; z-index: 100; display: flex; align-items: center; justify-content: flex-start; gap: 24px; padding: 12px 24px; background: linear-gradient(180deg, rgba(0,0,0,0.85), rgba(0,0,0,0)); width: 100%; margin-left: 0; margin-right: 0; margin-top: 0; }
      .navbar .nav-left { display: flex; align-items: center; gap: 24px; }
      .navbar .brand { color: var(--red); font-size: 24px; font-weight: 900; text-transform: uppercase; letter-spacing: 0.5px; }
      .navbar .links { display: flex; gap: 16px; color: var(--muted); font-weight: 600; }
      .navbar .links a { color: var(--muted); text-decoration: none; cursor: pointer; }
      .navbar .links a:hover { color: #fff; }
      /* Right-side navbar elements removed for now */

      /* Cards */
      .movie-card { background: transparent; border-radius: 12px; transition: transform .25s ease, box-shadow .25s ease; padding: 10px; width: 220px; display: flex; flex-direction: column; align-items: center; gap: 6px; min-height: 460px; }
      .movie-card img { width: 150px; height: 225px; object-fit: cover; border-radius: 10px; display: block; margin: 0 auto; box-shadow: 0 12px 24px rgba(0,0,0,0.35); }
      .movie-card:hover { transform: scale(1.03); z-index: 2; }
      .poster-title { font-weight: 800; color: #f5f5f5; margin: 8px 0 2px; font-size: 0.98rem; text-align: center; line-height: 1.3; min-height: calc(1.3em * 2); max-height: calc(1.3em * 2); overflow: hidden; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; }
      .poster-meta { color: var(--yellow); text-align: center; font-weight: 700; margin-bottom: 6px; min-height: 22px; }
      .genre-row { display:flex; justify-content:center; gap:8px; margin:6px 0; flex-wrap: wrap; }
      .genre-pill { border: 1px solid rgba(255,255,255,0.25); color: var(--muted); border-radius: 999px; padding: 4px 10px; font-size: 0.78em; font-weight: 700; letter-spacing: 0.2px; line-height: 1; }
      .genre-stack { min-height: 80px; display: flex; flex-direction: column; justify-content: center; }
      .card-spacer { flex: 1 1 auto; }
      .btn-link { display: inline-block; text-decoration: none; width: 100%; }
      .movie-card a, .btn-link, .btn-link:link, .btn-link:visited, .btn-link:hover, .btn-link:active { text-decoration: none !important; border-bottom: none !important; }
      .btn-primary { background: var(--red); color: #fff; border: none; border-radius: 6px; padding: 10px 12px; font-weight: 800; text-align:center; box-shadow: 0 6px 14px rgba(229,9,20,0.25); transition: background .2s ease; text-decoration: none !important; }
      .btn-primary:hover { background: var(--red-dark); }

      /* Buttons */
      .stButton>button { background: var(--red); color: #fff; border: none; border-radius: 4px; padding: 10px 12px; font-weight: 800; transition: background .2s ease; box-shadow: 0 6px 14px rgba(229,9,20,0.25); }
      .stButton>button:hover { background: var(--red-dark); }
      .stButton>button:focus { outline: 2px solid rgba(229,9,20,0.5); }

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
        <div class="brand">CineRecommender</div>
        <div class="links">
          <a href="?">Home</a>
          <a href="?">Find your next movie</a>
          <a href="?view=advanced-search">Advanced Search</a>
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

    with st.form("adv_search_form"):
        c1, c2, c3 = st.columns([1,1,1])
        with c1:
            selected_genres = st.multiselect("Genre(s)", options=get_all_genres(movies))
            rating_min = st.slider("Min rating", 0.0, 10.0, 6.0, 0.1)
            language = st.text_input("Language (e.g., en, hi, fr)")
        with c2:
            min_year = int(pd.to_numeric(movies.get('year', pd.Series([1990])), errors='coerce').fillna(1990).min())
            max_year = int(pd.to_numeric(movies.get('year', pd.Series([2024])), errors='coerce').fillna(2024).max())
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

    if 'year' in df.columns:
        df = df[pd.to_numeric(df['year'], errors='coerce').fillna(0).between(year_range[0], year_range[1])]

    rating_col = 'rating' if 'rating' in df.columns else ('vote_average' if 'vote_average' in df.columns else None)
    if rating_col:
        df = df[pd.to_numeric(df[rating_col], errors='coerce').fillna(0) >= rating_min]

    lang_col = 'language' if 'language' in df.columns else ('original_language' if 'original_language' in df.columns else None)
    if language and lang_col:
        df = df[df[lang_col].astype(str).str.contains(language.strip(), case=False, na=False)]

    if 'cast' in df.columns and st.session_state.get('dummy', True):
        pass
    if 'director' in df.columns and st.session_state.get('dummy', True):
        pass
    if 'cast' in df.columns and 'actor' in locals() and actor:
        df = df[df['cast'].astype(str).str.contains(actor.strip(), case=False, na=False)]
    if 'director' in df.columns and 'director' in locals() and director:
        df = df[df['director'].astype(str).str.contains(director.strip(), case=False, na=False)]

    if 'keywords' in locals() and keywords:
        kw = keywords.strip()
        cols = [c for c in ['title','overview','tagline','genres'] if c in df.columns]
        if cols:
            kw_mask = False
            for c in cols:
                kw_mask = kw_mask | df[c].astype(str).str.contains(kw, case=False, na=False)
            df = df[kw_mask]

    score = pd.Series(0.0, index=df.index)
    if 'similar_to' in locals() and similar_to:
        idx = find_movie_index_by_title(movies, similar_to)
        if idx is not None:
            sim_vec = pd.Series(sim_matrix[idx], index=movies.index)
            score = score.add(normalize_series(sim_vec.reindex(df.index).fillna(0)) * 0.55, fill_value=0)

    rcol2 = 'rating' if 'rating' in df.columns else ('vote_average' if 'vote_average' in df.columns else None)
    if rcol2:
        score = score.add(normalize_series(pd.to_numeric(df[rcol2], errors='coerce').fillna(0)) * 0.30, fill_value=0)

    if 'year' in df.columns:
        score = score.add(normalize_series(pd.to_numeric(df['year'], errors='coerce').fillna(0)) * 0.15, fill_value=0)

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
                f"<a class='btn-link' href='?movie={encoded2}'><div class='btn-primary'>View Details</div></a>"
                "</div>"
            )
            st.markdown(card_html2, unsafe_allow_html=True)

# If movie is selected via query params, show detailed view
if current_view == "advanced-search":
    render_advanced_search()
elif selected_movie_title:
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
        if st.button("← Back to Recommendations"):
            st.query_params.clear()
            st.rerun()
else:
    # Main recommendation page
    st.markdown('<div id="top"></div>', unsafe_allow_html=True)
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

                    # Display card with built-in button; use query string navigation for details
                    encoded = urllib.parse.quote(original_title)
                    card_html = (
                        "<div class='movie-card'>"
                        f"<img src='{poster}' alt='poster'/>"
                        f"<div class='poster-title'>{title_clean} ({year})</div>"
                        f"<div class='poster-meta'>⭐ {rating}</div>"
                        f"{genre_rows_html}"
                        "<div class='card-spacer'></div>"
                        f"<a class='btn-link' href='?movie={encoded}'><div class='btn-primary'>View Details</div></a>"
                        "</div>"
                    )

                    st.markdown(card_html, unsafe_allow_html=True)

            # Advanced Search Section
            st.markdown('<div id="advanced-search"></div>', unsafe_allow_html=True)
            st.markdown("<hr />", unsafe_allow_html=True)
            st.subheader("Advanced Search")

            # Helper utilities
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

            with st.form("adv_search_form"):
                c1, c2, c3 = st.columns([1,1,1])
                with c1:
                    selected_genres = st.multiselect("Genre(s)", options=get_all_genres(movies))
                    rating_min = st.slider("Min rating", 0.0, 10.0, 6.0, 0.1)
                    language = st.text_input("Language (e.g., en, hi, fr)")
                with c2:
                    min_year = int(pd.to_numeric(movies.get('year', pd.Series([1990])), errors='coerce').fillna(1990).min())
                    max_year = int(pd.to_numeric(movies.get('year', pd.Series([2024])), errors='coerce').fillna(2024).max())
                    year_range = st.slider("Release year range", min_year, max_year, (max(min_year, max_year-20), max_year))
                    actor = st.text_input("Actor contains")
                with c3:
                    director = st.text_input("Director contains")
                    keywords = st.text_input("Keywords (mood/theme/plot)")
                    similar_to = st.text_input("Similar to (movie)")

                submitted = st.form_submit_button("Search")

            if submitted:
                df = movies.copy()
                # Filter by genres
                if selected_genres and 'genres' in df.columns:
                    mask = df['genres'].fillna('').apply(lambda x: all(g in x for g in selected_genres))
                    df = df[mask]

                # Filter by year
                if 'year' in df.columns:
                    df = df[pd.to_numeric(df['year'], errors='coerce').fillna(0).between(year_range[0], year_range[1])]

                # Filter by rating
                rating_col = 'rating' if 'rating' in df.columns else ('vote_average' if 'vote_average' in df.columns else None)
                if rating_col:
                    df = df[pd.to_numeric(df[rating_col], errors='coerce').fillna(0) >= rating_min]

                # Filter by language
                lang_col = 'language' if 'language' in df.columns else ('original_language' if 'original_language' in df.columns else None)
                if language and lang_col:
                    df = df[df[lang_col].astype(str).str.contains(language.strip(), case=False, na=False)]

                # Actor / Director keyword filters (if present)
                if actor and 'cast' in df.columns:
                    df = df[df['cast'].astype(str).str.contains(actor.strip(), case=False, na=False)]
                if director and 'director' in df.columns:
                    df = df[df['director'].astype(str).str.contains(director.strip(), case=False, na=False)]

                # Keywords search across available text columns
                if keywords:
                    kw = keywords.strip()
                    cols = [c for c in ['title','overview','tagline','genres'] if c in df.columns]
                    if cols:
                        kw_mask = False
                        for c in cols:
                            kw_mask = kw_mask | df[c].astype(str).str.contains(kw, case=False, na=False)
                        df = df[kw_mask]

                # Scoring: similarity + rating + recency
                score = pd.Series(0.0, index=df.index)
                if similar_to:
                    idx = find_movie_index_by_title(movies, similar_to)
                    if idx is not None:
                        sim_vec = pd.Series(sim_matrix[idx], index=movies.index)
                        score = score.add(normalize_series(sim_vec.reindex(df.index).fillna(0)) * 0.55, fill_value=0)

                if 'rating' in df.columns or 'vote_average' in df.columns:
                    rcol = 'rating' if 'rating' in df.columns else 'vote_average'
                    score = score.add(normalize_series(pd.to_numeric(df[rcol], errors='coerce').fillna(0)) * 0.30, fill_value=0)

                if 'year' in df.columns:
                    score = score.add(normalize_series(pd.to_numeric(df['year'], errors='coerce').fillna(0)) * 0.15, fill_value=0)

                top_idx = score.sort_values(ascending=False).head(10).index
                df = df.loc[top_idx]

                st.markdown("### Results")
                if df.empty:
                    st.info("No matches found. Try relaxing your filters.")
                else:
                    num_cols = 5
                    cols2 = st.columns(num_cols)
                    for jdx, (_, r) in enumerate(df.iterrows()):
                        cidx = jdx % num_cols
                        with cols2[cidx]:
                            title_value = str(r.get('title', 'Unknown'))
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
                                f"<div class='poster-title'>{title_value} ({year_val})</div>"
                                f"<div class='poster-meta'>⭐ {rating_display}</div>"
                                f"{genre_rows_html2}"
                                "<div class='card-spacer'></div>"
                                f"<a class='btn-link' href='?movie={encoded2}'><div class='btn-primary'>View Details</div></a>"
                                "</div>"
                            )
                            st.markdown(card_html2, unsafe_allow_html=True)