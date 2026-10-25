import logging
import os
import re
import time

import pandas as pd
import requests
from flask import Flask, g, jsonify, redirect, render_template, request, url_for

import feedback as feedback_store
from recommender import Recommender

# ---------------------------------------------------------------- setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("aurarec")

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-only-change-me")
app.config["FEEDBACK_DB"] = os.getenv("FEEDBACK_DB", os.path.join(BASE_DIR, "feedback.db"))

# Keys come from the environment (see .env.example). Never commit real keys.
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")
OMDB_API_KEY = os.getenv("OMDB_API_KEY", "")
PLACEHOLDER_POSTER = "https://via.placeholder.com/300x450"

# Re-ranking presets applied on top of the base similarity score.
RECOMMEND_BOOSTS = {"industry": 1.2, "mood": 1.3, "cast": 2.0}
MOVIE_PAGE_BOOSTS = {"industry": 2.5, "mood": 3.0, "cast": 1.2}


# ---------------------------------------------------------------- data + model
def extract_year(title):
    match = re.search(r"\((\d{4})\)", str(title))
    return match.group(1) if match else "N/A"


df = pd.read_csv(os.getenv("CSV_PATH", os.path.join(BASE_DIR, "movies_with_popularity.csv"))).fillna("")
df["year"] = df["title"].apply(extract_year)
id_to_idx = {int(m): i for i, m in enumerate(df["movieId"])}


def build_engine(frame):
    """scikit-learn engine by default; RECOMMENDER=tf selects the TensorFlow model."""
    if os.getenv("RECOMMENDER", "tfidf").lower() == "tf":
        try:
            from tf_engine import TFEngine
            return TFEngine(frame)
        except Exception as exc:  # TensorFlow missing or model failed to build
            log.warning("TensorFlow engine unavailable (%s); using TF-IDF engine.", exc)
    return Recommender(frame)


engine = build_engine(df)


# ---------------------------------------------------------------- external APIs
def get_tmdb_results(query):
    """Search TMDB for an actor (their movies) or, failing that, for a movie."""
    if not TMDB_API_KEY:
        return []
    base = "https://api.themoviedb.org/3"
    try:
        person = requests.get(f"{base}/search/person",
                              params={"api_key": TMDB_API_KEY, "query": query}, timeout=3).json()
        if person.get("results"):
            person_id = person["results"][0]["id"]
            credits = requests.get(f"{base}/person/{person_id}/movie_credits",
                                   params={"api_key": TMDB_API_KEY}, timeout=3).json()
            items, mood = credits.get("cast", [])[:20], "Actor Hit"
        else:
            found = requests.get(f"{base}/search/movie",
                                 params={"api_key": TMDB_API_KEY, "query": query}, timeout=3).json()
            items, mood = found.get("results", []), "New Discovery"
        results = []
        for item in items:
            path = item.get("poster_path")
            results.append({
                "movieId": item["id"],
                "title": item.get("title"),
                "poster": f"https://image.tmdb.org/t/p/w500{path}" if path else PLACEHOLDER_POSTER,
                "year": (item.get("release_date") or "N/A")[:4],
                "rating": item.get("vote_average", "N/A"),
                "mood": mood,
            })
        return results
    except (requests.RequestException, ValueError, KeyError) as exc:
        log.warning("TMDB lookup failed for %r: %s", query, exc)
        return []


def get_omdb(title):
    """Poster/plot/rating from OMDb; falls back to placeholders on any failure."""
    fallback = {"poster": PLACEHOLDER_POSTER, "plot": "No plot available.", "rating": "N/A"}
    if not OMDB_API_KEY:
        return fallback
    clean = re.sub(r"\s*\(\d{4}\)", "", title).strip()
    try:
        r = requests.get("https://www.omdbapi.com/",
                         params={"t": clean, "apikey": OMDB_API_KEY}, timeout=2).json()
        if r.get("Response") == "True":
            return {"poster": r.get("Poster"), "plot": r.get("Plot"), "rating": r.get("imdbRating")}
    except (requests.RequestException, ValueError) as exc:
        log.warning("OMDb lookup failed for %r: %s", title, exc)
    return fallback


def get_enriched_list(dataframe):
    enriched = []
    for _, row in dataframe.iterrows():
        try:
            poster = str(row.get("poster", "")).strip()
            if poster.startswith("/") and not poster.startswith("http"):
                final_poster = f"https://image.tmdb.org/t/p/w500{poster}"  # TMDB partial path
            elif not poster or poster in ("nan", "N/A"):
                meta = get_omdb(row["title"])
                final_poster = meta["poster"] if meta["poster"] != "N/A" else PLACEHOLDER_POSTER + "?text=No+Poster"
            else:
                final_poster = poster

            enriched.append({
                "movieId": row["movieId"],
                "title": row["title"],
                "year": row["year"],
                "mood": row["mood"],
                "Industry": row.get("Industry", "Global"),
                "cast": row["cast"],
                "poster": final_poster,
                "rating": row["rating"] if row["rating"] != "" else "7.5",
            })
        except Exception as exc:  # one bad row must not break the whole page
            log.error("Error enriching %r: %s", row.get("title"), exc)
    return enriched


def _feedback_for(movie_id):
    """Net like/dislike votes for this movie's recommendations, keyed by row index."""
    try:
        votes = feedback_store.net_votes(app.config["FEEDBACK_DB"], movie_id)
    except Exception as exc:
        log.warning("Could not read feedback: %s", exc)
        return {}
    return {id_to_idx[rec]: net for rec, net in votes.items() if rec in id_to_idx}


# ---------------------------------------------------------------- request logging
@app.before_request
def _start_timer():
    g.start = time.perf_counter()


@app.after_request
def _log_request(response):
    elapsed_ms = (time.perf_counter() - g.get("start", time.perf_counter())) * 1000
    log.info("%s %s -> %s in %.0f ms", request.method, request.path, response.status_code, elapsed_ms)
    return response


# ---------------------------------------------------------------- routes
@app.route("/health")
def health():
    return jsonify(status="ok", engine=type(engine).__name__, movies=len(df))


@app.route("/")
def home():
    trending_all = df.sort_values(by="popularity", ascending=False)
    banner_data = get_enriched_list(trending_all.head(1))
    banner_movie = banner_data[0] if banner_data else None

    def industry(name):
        return df[df["Industry"].str.upper() == name]

    return render_template(
        "index.html",
        banner=banner_movie,
        bollywood=get_enriched_list(industry("BOLLYWOOD").sort_values(by="year", ascending=False).head(12)),
        hollywood=get_enriched_list(industry("HOLLYWOOD").head(12)),
        chinese=get_enriched_list(industry("CHINESE").head(12)),
        korean=get_enriched_list(industry("KOREAN").head(12)),
        animation=get_enriched_list(industry("ANIME").head(12)),
        trending=get_enriched_list(trending_all.head(6)),
    )


@app.route("/search")
def search():
    query = request.args.get("query", "").strip()
    if not query:
        return redirect(url_for("home"))

    final_results = get_tmdb_results(query)  # best for actor searches
    existing_titles = {str(r["title"]).lower() for r in final_results}

    local = df[df["title"].str.contains(query, case=False, na=False, regex=False)
               | df["cast"].str.contains(query, case=False, na=False, regex=False)]
    for movie in get_enriched_list(local):
        if movie["title"].lower() not in existing_titles:
            final_results.append(movie)

    return render_template("search.html", query=query, results=final_results)


@app.route("/recommend", methods=["POST"])
def recommend():
    query = (request.form.get("movie_name") or "").strip()
    idx = engine.find(query)
    if idx is None:
        return render_template("index.html", error="Movie not found!")

    movie_id = int(df.iloc[idx]["movieId"])
    indices = engine.recommend(idx, k=12, boosts=RECOMMEND_BOOSTS, feedback=_feedback_for(movie_id))
    return render_template("index.html", results=get_enriched_list(df.iloc[indices]), query=query)


@app.route("/movie/<int:movie_id>")
def movie_details(movie_id):
    idx = id_to_idx.get(movie_id)
    if idx is None:
        return redirect(url_for("home"))

    indices = engine.recommend(idx, k=12, boosts=MOVIE_PAGE_BOOSTS, feedback=_feedback_for(movie_id))
    related_movies = get_enriched_list(df.iloc[indices])
    movie_info = get_enriched_list(df.iloc[[idx]])[0]

    clean_title = re.sub(r"\s*\(\d{4}\)", "", movie_info["title"]).strip()
    movie_info["trailer_url"] = ("https://www.youtube.com/results?search_query="
                                 + f"{clean_title} official trailer".replace(" ", "+"))
    return render_template("movie_page.html", movie=movie_info, related=related_movies)


@app.route("/feedback", methods=["POST"])
def feedback():
    """Record a like/dislike: JSON or form body {movie_id, rec_id, liked}."""
    data = request.get_json(silent=True) or request.form
    try:
        movie_id, rec_id = int(data["movie_id"]), int(data["rec_id"])
        liked_raw = str(data["liked"]).strip().lower()
    except (KeyError, TypeError, ValueError):
        return jsonify(error="movie_id, rec_id (integers) and liked are required"), 400
    if liked_raw not in {"true", "1", "like", "false", "0", "dislike"}:
        return jsonify(error="liked must be true/false"), 400
    if movie_id not in id_to_idx or rec_id not in id_to_idx:
        return jsonify(error="unknown movie id"), 404

    feedback_store.add_feedback(app.config["FEEDBACK_DB"], movie_id, rec_id,
                                liked_raw in {"true", "1", "like"})
    return jsonify(status="recorded"), 201


if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_DEBUG") == "1", port=int(os.getenv("PORT", "5000")))
