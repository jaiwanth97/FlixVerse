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
    
    # First check: exact match after cleaning
    if title1_clean == title2_clean:
        return True
    
    words1 = set(word for word in title1_clean.split() if len(word) > 1)  # Ignore single char words
    words2 = set(word for word in title2_clean.split() if len(word) > 1)
    
    if not words1 or not words2:
        return False
    
    # Check for significant word overlap - need at least 60% of words to match
    common_words = words1.intersection(words2)
    if not common_words:
        return False  # No words in common = definitely not similar
    
    # For short titles (1-2 words), require ALL words to match
    # This prevents "Black Panther" matching "Schwarzer Panther" (only "panther" matches)
    min_words = min(len(words1), len(words2))
    if min_words <= 2:
        return len(common_words) == min_words and len(common_words) == len(words1) and len(common_words) == len(words2)
    
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
        
        def extract_year_from_date(date_str):
            """Extract year from date string like '2018-05-15'"""
            if not date_str:
                return None
            try:
                return int(date_str[:4])
            except (ValueError, TypeError):
                return None
        
        # First, try to find exact match with both year and title similarity
        # When year is provided, be strict about year matching (exact match or ±1 year for remakes)
        if year:
            for m in data:
                release_year = extract_year_from_date(m.get("release_date", ""))
                movie_title = m.get("title", "").lower()
                
                # Check year match - must be exact or within ±1 year
                year_match = (release_year is not None and abs(release_year - year) <= 1) if year else True
                
                # Check title similarity
                title_match = title_similarity(clean_lower, movie_title)
                
                # If both match, this is our best match
                if year_match and title_match:
                    movie = m
                    break
        
        # If no perfect match with year, try exact year match only (if year provided)
        if not movie and year:
            for m in data:
                release_year = extract_year_from_date(m.get("release_date", ""))
                if release_year == year:  # Exact year match required
                    movie_title = m.get("title", "").lower()
                    # Still check basic title similarity
                    if title_similarity(clean_lower, movie_title):
                        movie = m
                        break
        
        # If still no match and year was provided, try title-only but prefer movies with similar year
        # Also prefer English titles and newer movies
        if not movie:
            best_match = None
            best_year_diff = float('inf')
            best_score = -1
            
            for m in data:
                movie_title = m.get("title", "").lower()
                if title_similarity(clean_lower, movie_title):
                    release_year = extract_year_from_date(m.get("release_date", ""))
                    original_language = m.get("original_language", "").lower()
                    
                    # Score: prefer English titles, prefer newer movies, prefer closer year match
                    score = 0
                    if original_language == "en":
                        score += 100  # Strong preference for English
                    if release_year:
                        if year:
                            year_diff = abs(release_year - year)
                            score += (100 - year_diff * 10)  # Prefer closer year match
                        else:
                            # If no year specified, prefer newer movies
                            score += release_year - 1900  # Newer movies get higher score
                    
                    if score > best_score or (score == best_score and release_year and best_match and extract_year_from_date(best_match.get("release_date", "")) and release_year > extract_year_from_date(best_match.get("release_date", ""))):
                        best_match = m
                        best_score = score
                        if year and release_year:
                            best_year_diff = abs(release_year - year)
            
            if not movie and best_match:
                movie = best_match
        
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
        movie_year = movie.get("release_date", "")[:4] if movie.get("release_date") else None
        rating = movie.get("vote_average", None)
        return {"poster": poster, "year": movie_year, "rating": rating}
    
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
        clean_lower = clean.lower()
        
        def extract_year_from_date(date_str):
            """Extract year from date string like '2018-05-15'"""
            if not date_str:
                return None
            try:
                return int(date_str[:4])
            except (ValueError, TypeError):
                return None
        
        # First, try to find exact match with both year and title similarity
        # When year is provided, be strict about year matching (exact match or ±1 year for remakes)
        if year:
            for movie in data:
                release_year = extract_year_from_date(movie.get("release_date", ""))
                movie_title = movie.get("title", "").lower()
                
                # Check year match - must be exact or within ±1 year
                year_match = (release_year is not None and abs(release_year - year) <= 1) if year else True
                
                # Check title similarity - must have significant word overlap
                title_match = title_similarity(clean_lower, movie_title)
                
                # If both match, this is our best match
                if year_match and title_match:
                    best_match = movie
                    movie_id = movie.get("id")
                    break
        
        # If no perfect match with year, try exact year match only (if year provided)
        if not movie_id and year:
            for movie in data:
                release_year = extract_year_from_date(movie.get("release_date", ""))
                if release_year == year:  # Exact year match required
                    movie_title = movie.get("title", "").lower()
                    # Still check basic title similarity
                    if title_similarity(clean_lower, movie_title):
                        best_match = movie
                        movie_id = movie.get("id")
                        break
        
        # If still no match and year was provided, try title-only but prefer movies with similar year
        # Also prefer English titles and newer movies
        if not movie_id:
            best_match_candidate = None
            best_year_diff = float('inf')
            best_score = -1
            
            for movie in data:
                movie_title = movie.get("title", "").lower()
                if title_similarity(clean_lower, movie_title):
                    release_year = extract_year_from_date(movie.get("release_date", ""))
                    original_language = movie.get("original_language", "").lower()
                    
                    # Score: prefer English titles, prefer newer movies, prefer closer year match
                    score = 0
                    if original_language == "en":
                        score += 100  # Strong preference for English
                    if release_year:
                        if year:
                            year_diff = abs(release_year - year)
                            score += (100 - year_diff * 10)  # Prefer closer year match
                        else:
                            # If no year specified, prefer newer movies
                            score += release_year - 1900  # Newer movies get higher score
                    
                    if score > best_score or (score == best_score and release_year and best_match_candidate and extract_year_from_date(best_match_candidate.get("release_date", "")) and release_year > extract_year_from_date(best_match_candidate.get("release_date", ""))):
                        best_match_candidate = movie
                        best_score = score
                        if year and release_year:
                            best_year_diff = abs(release_year - year)
                    elif not year and not best_match_candidate:
                        # If no year specified and no candidate yet, use first good match
                        best_match = movie
                        movie_id = movie.get("id")
                        break
            
            if not movie_id and best_match_candidate:
                best_match = best_match_candidate
                movie_id = best_match.get("id")
        
        # Last resort: use first result ONLY if it has strong similarity
        if not movie_id and data:
            first_movie = data[0]
            first_title = first_movie.get("title", "").lower()
            # Must have strong similarity to use first result
            if title_similarity(clean_lower, first_title):
                movie_id = first_movie.get("id")
                best_match = first_movie
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
