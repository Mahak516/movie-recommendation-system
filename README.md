# AuraRec AI: Movie Recommendation Web App

A Flask web app that recommends movies from an 11,810-title catalog. It ranks titles by embedding similarity (built with TensorFlow) and then re-ranks them using industry, mood and cast matches. It also searches by movie title or actor, and uses the TMDB and OMDb APIs for actor search and poster artwork.

## Features

- **Home page** with a banner for the most popular title and rows for Trending, Bollywood, Hollywood, Chinese, Korean and Anime.
- **Search** by movie title or cast member. Local catalog results are merged with live TMDB results, and duplicates are removed.
- **Recommendations** from a title typed into the form, returning the 12 closest matches.
- **Movie page** with details, 12 related movies and a YouTube trailer search link.
- **Graceful fallbacks:** TMDB and OMDb calls use short timeouts, and a placeholder poster or an empty result is used when an API is unreachable, so an API outage does not break a page.

## Tech stack

Python, Flask, TensorFlow (Keras), pandas, NumPy, requests, HTML/Jinja2 templates. TMDB and OMDb are used as external APIs.

## How the recommendations work

Each movie is turned into a 96-dimensional vector by concatenating four embeddings:

| Feature | Embedding size | Notes |
|---|---|---|
| Title | 32 | one vector per title |
| Cast | 16 | mean of the cast members' vectors |
| Keywords | 16 | mean of the keywords' vectors |
| Mood | 32 | one vector per mood label |

For a chosen movie, the app computes the dot product between its vector and the vectors of all 11,810 movies, then multiplies each score by boosts for matching attributes:

| Route | Same industry | Same mood | Shared cast member | Query movie |
|---|---|---|---|---|
| `/movie/<id>` | ×2.5 | ×3.0 | ×1.2 | score set to 0 |
| `/recommend` | ×1.2 | ×1.3 | ×2.0 | first-ranked result dropped |

The 12 highest scores are returned.

The model loads `enhanced_movie_weights.weights.h5` from the project root if that file exists. No weights file is included in this repository, so without one the app runs with freshly initialised embeddings.

## Project structure

```
app.py                      Flask app, model definition, routes, API helpers
movies_with_popularity.csv  Catalog: movieId, title, genres, cast, keywords,
                            popularity, rating, poster, mood, Industry
templates/                  Jinja2 pages (index, search, movie_page, ...)
tests/test_app.py           Automated tests
```

## Setup

```bash
git clone https://github.com/Mahak516/movie-recommendation-system.git
cd movie-recommendation-system
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install flask pandas numpy requests tensorflow tf-keras
```

`app.py` sets `TF_USE_LEGACY_KERAS=1`, which is why `tf-keras` is installed.

### API keys (optional)

Poster enrichment and actor search use TMDB and OMDb. Set your own keys as environment variables:

```bash
export TMDB_API_KEY="your_tmdb_key"
export OMDB_API_KEY="your_omdb_key"
```

Without keys the app still runs: those calls fail quietly and fall back to catalog data and placeholder posters.

### Run

```bash
python app.py
```

Then open http://127.0.0.1:5000. Recommendation requests score every title, so they are slower than the other pages (see Known issues).

## Routes

| Method | Path | What it does |
|---|---|---|
| GET | `/` | Home page with the banner and industry rows |
| GET | `/search?query=...` | Search by title or cast; an empty query redirects home |
| POST | `/recommend` | Form field `movie_name`; returns 12 recommendations |
| GET | `/movie/<movie_id>` | Movie page with 12 related titles and a trailer link |

## Tests

```bash
pip install pytest
pytest -v
```

The suite has 39 tests covering all 4 routes, the TMDB and OMDb helpers and the poster logic. Network calls are patched out, and by default the TensorFlow model is replaced with a deterministic fake so results are repeatable. Expected result: **35 passed, 3 xfailed, 1 skipped**.

To also run the integration test against the real TensorFlow model:

```bash
REAL_TF=1 pytest -v
```

The 3 expected failures are known bugs, recorded as tests so they are not forgotten (see below).

## Known issues and limitations

- **Regex characters in search:** the query is used as a regular expression, so a query like `(` raises an error and returns HTTP 500.
- **Missing form field:** `POST /recommend` without `movie_name` crashes instead of returning a clean error.
- **Query movie in its own results:** `/recommend` drops the top-ranked item by position, not by id. If another title outscores the query movie, the query movie can appear in its own recommendations.
- **Speed:** each request scores all 11,810 rows with a Python loop. With the model stubbed out and network calls disabled, that scoring alone took roughly 1.4 s (`/recommend`) to 2 s (`/movie/<id>`) on a cloud test machine. Vectorising the loop with NumPy would remove most of that time.
- **No trained weights are included** (see "How the recommendations work").

## Author

Mahak Bansal, [GitHub](https://github.com/Mahak516)
