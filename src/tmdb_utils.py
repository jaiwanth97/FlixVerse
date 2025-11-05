import requests
import os
import re
import time
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

load_dotenv()
API_KEY = os.getenv("TMDB_API_KEY")
BASE_URL = "https://api.themoviedb.org/3"

# Create a session with retry strategy
session = requests.Session()

# Configure retry strategy
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"]
)

adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("http://", adapter)
session.mount("https://", adapter)

# Set default headers
session.headers.update({
    "User-Agent": "MovieRecommender/1.0",
    "Accept": "application/json"
})

def clean_title(title):
    """Normalize MovieLens-style titles like 'Phantom, The (1996)' → 'The Phantom'."""
    title_no_year = re.sub(r"\s*\(\d{4}\)", "", title)
    m = re.match(r"^(.*),\s*(The|An|A)$", title_no_year)
    if m:
        title_no_year = f"{m.group(2)} {m.group(1)}"
    return title_no_year.strip()

def title_similarity(title1, title2):
    """Check if two titles are similar (strict word-based check)."""
    # Remove special characters and normalize
    import string
    title1_clean = ''.join(c for c in title1.lower() if c.isalnum() or c.isspace())
    title2_clean = ''.join(c for c in title2.lower() if c.isalnum() or c.isspace())
    
    words1 = set(word for word in title1_clean.split() if len(word) > 1)  # Ignore single char words
    words2 = set(word for word in title2_clean.split() if len(word) > 1)
    
    if not words1 or not words2:
        return False
    
    # Check for significant word overlap - need at least 60% of words to match
    common_words = words1.intersection(words2)
    if not common_words:
        return False  # No words in common = definitely not similar
    
    # For short titles (1-2 words), require all words to match
    min_words = min(len(words1), len(words2))
    if min_words <= 2:
        return len(common_words) == min_words
    
    # For longer titles, require at least 60% overlap
    overlap_ratio = len(common_words) / min_words
    return overlap_ratio >= 0.6

def get_movie_details(title, retry_count=0):
    """Search TMDb and return poster, release year, and rating."""
    clean = clean_title(title)
    
    # Extract year from original title if available
    year_match = re.search(r'\((\d{4})\)', title)
    year = int(year_match.group(1)) if year_match else None
    
    search_url = f"{BASE_URL}/search/movie"
    params = {"api_key": API_KEY, "query": clean}
    if year:
        params["year"] = year  # Add year to search for better matching

    try:
        r = session.get(
            search_url, 
            params=params,
            timeout=(5, 10)  # (connect timeout, read timeout)
        )
        r.raise_for_status()  # Raise an exception for bad status codes
        
        data = r.json().get("results", [])
        if not data:
            return {"poster": None, "year": None, "rating": None}

        # Find the best match: prioritize title similarity AND year match
        movie = None
        clean_lower = clean.lower()
        
        # First, try to find exact match with both year and title similarity
        for m in data:
            release_year = m.get("release_date", "")
            movie_title = m.get("title", "").lower()
            
            # Check year match
            year_match = release_year and release_year.startswith(str(year)) if year else True
            
            # Check title similarity
            title_match = title_similarity(clean_lower, movie_title)
            
            # If both match, this is our best match
            if year_match and title_match:
                movie = m
                break
        
        # If no perfect match, try year-only match (if year provided)
        if not movie and year:
            for m in data:
                release_year = m.get("release_date", "")
                if release_year and release_year.startswith(str(year)):
                    movie_title = m.get("title", "").lower()
                    # Still check basic title similarity
                    if title_similarity(clean_lower, movie_title):
                        movie = m
                        break
        
        # If still no match, try title-only match
        if not movie:
            for m in data:
                movie_title = m.get("title", "").lower()
                if title_similarity(clean_lower, movie_title):
                    movie = m
                    break
        
        # Last resort: use first result only if it has some similarity
        if not movie and data:
            first_movie = data[0]
            first_title = first_movie.get("title", "").lower()
            if title_similarity(clean_lower, first_title):
                movie = first_movie
            else:
                # If first result is completely different, return defaults
                return {"poster": None, "year": None, "rating": None}
        
        poster = f"https://image.tmdb.org/t/p/w500{movie['poster_path']}" if movie.get("poster_path") else None
        year = movie.get("release_date", "")[:4] if movie.get("release_date") else None
        rating = movie.get("vote_average", None)
        return {"poster": poster, "year": year, "rating": rating}
    
    except requests.exceptions.RequestException as e:
        # Retry logic for connection errors
        if retry_count < 2:
            time.sleep(2 ** retry_count)  # Exponential backoff
            return get_movie_details(title, retry_count + 1)
        else:
            # Return default values after max retries
            return {"poster": None, "year": None, "rating": None}

def get_full_movie_details(title, retry_count=0):
    """Get full movie details from TMDB including description, cast, director, etc."""
    clean = clean_title(title)
    
    # Extract year from original title if available (e.g., "Last Knight (2017)")
    year_match = re.search(r'\((\d{4})\)', title)
    year = int(year_match.group(1)) if year_match else None
    
    search_url = f"{BASE_URL}/search/movie"
    params = {"api_key": API_KEY, "query": clean}
    if year:
        params["year"] = year  # Add year to search for better matching

    try:
        # First, search for the movie
        r = session.get(search_url, params=params, timeout=(5, 10))
        r.raise_for_status()
        
        data = r.json().get("results", [])
        if not data:
            return None
        
        # Find the best match: prioritize title similarity AND year match
        movie_id = None
        best_match = None
        
        # First, try to find exact match with both year and title similarity
        for movie in data:
            release_year = movie.get("release_date", "")
            movie_title = movie.get("title", "").lower()
            clean_lower = clean.lower()
            
            # Check year match
            year_match = release_year and release_year.startswith(str(year)) if year else True
            
            # Check title similarity - must have significant word overlap
            title_match = title_similarity(clean_lower, movie_title)
            
            # If both match, this is our best match
            if year_match and title_match:
                best_match = movie
                movie_id = movie.get("id")
                break
        
        # If no perfect match, try year-only match (if year provided)
        if not movie_id and year:
            for movie in data:
                release_year = movie.get("release_date", "")
                if release_year and release_year.startswith(str(year)):
                    movie_title = movie.get("title", "").lower()
                    # Still check basic title similarity
                    if title_similarity(clean_lower, movie_title):
                        best_match = movie
                        movie_id = movie.get("id")
                        break
        
        # If still no match, try title-only match
        if not movie_id:
            for movie in data:
                movie_title = movie.get("title", "").lower()
                if title_similarity(clean_lower, movie_title):
                    best_match = movie
                    movie_id = movie.get("id")
                    break
        
        # Last resort: use first result ONLY if it has strong similarity
        if not movie_id and data:
            first_movie = data[0]
            first_title = first_movie.get("title", "").lower()
            # Must have strong similarity to use first result
            if title_similarity(clean_lower, first_title):
                movie_id = first_movie.get("id")
            else:
                # If first result is completely different, return None - don't show wrong movie
                return None
        
        if not movie_id:
            return None
        
        # Get full movie details
        details_url = f"{BASE_URL}/movie/{movie_id}"
        params_details = {
            "api_key": API_KEY,
            "append_to_response": "credits,videos"  # Get cast, crew, and videos
        }
        
        r_details = session.get(details_url, params=params_details, timeout=(5, 10))
        r_details.raise_for_status()
        movie_data = r_details.json()
        
        # Extract all relevant information
        poster = f"https://image.tmdb.org/t/p/w500{movie_data.get('poster_path', '')}" if movie_data.get("poster_path") else None
        backdrop = f"https://image.tmdb.org/t/p/w1280{movie_data.get('backdrop_path', '')}" if movie_data.get("backdrop_path") else None
        
        # Get director from crew
        director = None
        crew = movie_data.get("credits", {}).get("crew", [])
        for person in crew:
            if person.get("job") == "Director":
                director = person.get("name")
                break
        
        # Get top cast (first 5)
        cast = []
        cast_list = movie_data.get("credits", {}).get("cast", [])[:5]
        for actor in cast_list:
            cast.append({
                "name": actor.get("name"),
                "character": actor.get("character"),
                "profile_path": f"https://image.tmdb.org/t/p/w185{actor.get('profile_path', '')}" if actor.get("profile_path") else None
            })
        
        # Get genres
        genres = [g.get("name") for g in movie_data.get("genres", [])]
        
        # Get production companies
        companies = [c.get("name") for c in movie_data.get("production_companies", [])[:3]]
        
        # Get trailer
        trailer_key = None
        videos = movie_data.get("videos", {}).get("results", [])
        for video in videos:
            if video.get("type") == "Trailer" and video.get("site") == "YouTube":
                trailer_key = video.get("key")
                break
        
        # Final validation: verify the returned movie actually matches what we searched for
        returned_title = movie_data.get("title", "").lower()
        if not title_similarity(clean.lower(), returned_title):
            # The returned movie doesn't match our search - return None instead of wrong movie
            return None
        
        return {
            "title": movie_data.get("title"),
            "original_title": movie_data.get("original_title"),
            "overview": movie_data.get("overview"),
            "poster": poster,
            "backdrop": backdrop,
            "release_date": movie_data.get("release_date"),
            "year": movie_data.get("release_date", "")[:4] if movie_data.get("release_date") else None,
            "rating": movie_data.get("vote_average"),
            "vote_count": movie_data.get("vote_count"),
            "runtime": movie_data.get("runtime"),
            "genres": genres,
            "director": director,
            "cast": cast,
            "production_companies": companies,
            "budget": movie_data.get("budget"),
            "revenue": movie_data.get("revenue"),
            "tagline": movie_data.get("tagline"),
            "status": movie_data.get("status"),
            "trailer_key": trailer_key,
            "imdb_id": movie_data.get("imdb_id"),
            "homepage": movie_data.get("homepage")
        }
    
    except requests.exceptions.RequestException as e:
        # Retry logic for connection errors
        if retry_count < 2:
            time.sleep(2 ** retry_count)
            return get_full_movie_details(title, retry_count + 1)
        else:
            return None
