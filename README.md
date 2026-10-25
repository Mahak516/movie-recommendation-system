# AuraRec AI: Movie Recommendation Web App

Flask web app that recommends movies from an 11,810-title catalog (Hollywood, Bollywood, Korean, Chinese, Anime). Recommendations are ranked by TF-IDF cosine similarity over keywords and cast, with optional re-ranking by industry, mood and shared cast, and a like/dislike feedback loop. The original TensorFlow embedding model is kept as an optional engine.

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env        # add TMDB_API_KEY / OMDB_API_KEY (optional: app runs without them, with placeholder posters)
export $(grep -v '^#' .env | xargs)
python app.py               # http://127.0.0.1:5000
```

Docker: `docker build -t aurarec . && docker run -p 8080:8080 aurarec`

## Endpoints

| Route | Purpose |
|---|---|
| `GET /` | Home page (trending + per-industry rows) |
| `GET /search?query=` | Search titles and cast (TMDB for actors when a key is set) |
| `POST /recommend` | Recommendations for a movie title |
| `GET /movie/<id>` | Movie page with related movies |
| `POST /feedback` | `{movie_id, rec_id, liked}` stores a vote; net votes nudge future ranking (bounded to +/-50%) |
| `GET /health` | Liveness/readiness check (engine and catalog size) |

Every request is logged with method, path, status and latency.

## Evaluation (offline, reproducible)

`python evaluate.py` writes `reports/evaluation.md`. The dataset has no user ratings, so relevance is a proxy: a recommendation counts as relevant when its genres overlap the query movie's genres (Jaccard >= 0.5). Genres are not model features. k = 10, 1,000 random query movies, seed 42.

| Strategy | Precision@10 | HitRate@10 | NDCG@10 |
|---|---|---|---|
| Random | 0.111 | 0.593 | 0.165 |
| Most popular | 0.095 | 0.545 | 0.140 |
| **TF-IDF cosine (no boosts)** | **0.288** | **0.861** | **0.317** |
| TF-IDF + `/recommend` boosts | 0.345 | 0.885 | 0.365 |
| TF-IDF + movie-page boosts | 0.461 | 0.925 | 0.449 |
| TF-IDF + movie-page boosts, **no mood** | 0.286 | 0.857 | 0.316 |

**How to read this.** TF-IDF is about 2.6x better than random on precision. The boosts look better, but the mood labels in this dataset are tied to genres (Feel-Good is 40% Comedy, Futuristic is 75% Science Fiction), so mood boosts are partly rewarded for matching the answer key. With the mood boost removed, the industry and cast boosts add essentially nothing on this metric (0.286 vs 0.288). Real user feedback (the `/feedback` endpoint) is what would settle whether the boosts help.

### Bias check

- Hollywood is 84.5% of the catalog and about 83-84% of recommendations, so the recommender mirrors the catalog rather than amplifying it.
- Without boosts, minority industries drift to Hollywood titles: only 40% of recommendations for a Korean query, 45% for Chinese and 55% for Bollywood stay in the same industry (Hollywood: 91%). The industry boost raises these to 64%, 77% and 77%.
- Precision is similar across industries (0.27-0.43 without boosts), so quality is not obviously worse for minority industries. The Bollywood estimate rests on only 66 titles.

## Handling imbalanced data

The `mood` label is heavily imbalanced (Adrenaline 2,925 titles; Neutral 411; Futuristic 11). `python mood_classifier.py` trains a logistic regression on a stratified split with and without `class_weight="balanced"` and reports `reports/mood_classifier.md`.

| Config | Accuracy | Macro-F1 | Neutral recall |
|---|---|---|---|
| Unweighted | 0.535 | 0.395 | 0.04 |
| `class_weight="balanced"` | 0.541 | 0.427 | 0.18 |

Balancing helps the minority class (Neutral) and macro-F1. The 11-title Futuristic class is still not learned (recall 0.00, only 3 test examples), so more data would be needed.

## Performance

The original per-row boost loop (3 `df.iloc` lookups for each of 11,810 rows) took about 1.7 s per request on its own. Scoring and boosts are now vectorized: a full recommendation takes about 2 ms (measured locally, single process).

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

27 tests cover the recommender (self-exclusion, boosts, feedback bounds, empty keywords), the metrics, the imbalance comparison, the Flask routes and the feedback store. GitHub Actions runs the tests, a quick evaluation and a Docker build on every push.

## Optional TensorFlow engine

`RECOMMENDER=tf` (after `pip install -r requirements-tf.txt`) uses the original embedding model in `tf_engine.py`. The repo does not contain `enhanced_movie_weights.weights.h5`, so without it the embeddings are untrained and the engine logs a warning. Train the model and add the weights before relying on this engine.

## Deploying to a cloud (not done yet)

The Dockerfile works with any container host. Example for Google Cloud Run:

```bash
gcloud run deploy aurarec --source . --region asia-south1 --allow-unauthenticated \
  --set-env-vars RECOMMENDER=tfidf --set-secrets TMDB_API_KEY=tmdb:latest,OMDB_API_KEY=omdb:latest
```

Note: `feedback.db` is a local SQLite file, so votes reset when a serverless container restarts. Use a managed database for production.

## Project layout

```
app.py                Flask routes, API clients, request logging
recommender.py        TF-IDF recommender with boosts and feedback nudge
tf_engine.py          Optional TensorFlow embedding engine
evaluate.py           Offline metrics, baselines, bias check
mood_classifier.py    Imbalanced-data demo (macro-F1, class weights)
feedback.py           SQLite like/dislike store
tests/                27 unit tests
reports/              Generated evaluation reports
```

## Known limitations

- Relevance is evaluated against a genre-overlap proxy, not real user behavior.
- Poster data depends on the TMDB/OMDb keys; without them the app shows placeholders.
- The TF-IDF engine uses keywords and cast only; the 15.5% of titles with no keywords are ranked by cast alone.
