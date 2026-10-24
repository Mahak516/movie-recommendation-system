"""Tests for the movie recommendation Flask app (app.py).

Run from the repository root:

    pip install pytest
    pytest -v                      # fast unit tests (TensorFlow model is replaced by a fake)
    REAL_TF=1 pytest -v            # also runs the integration test with the real TensorFlow model

They are written as unittest.TestCase classes, so `python -m unittest -v tests.test_app`
works too. No test touches the network: requests.get is patched in setUp.

What is real and what is faked
- Real: the Flask routes, the CSV data, pandas filtering, the boosting logic, the poster logic.
- Faked (unless REAL_TF=1): the TensorFlow MovieModel. The fake returns deterministic vectors,
  so ranking tests are repeatable and do not depend on random weights.
- Faked: TMDB / OMDb HTTP calls.
"""
import os
import sys
import types
import unittest
import zlib
from unittest import mock

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USE_REAL_TF = os.environ.get("REAL_TF") == "1"


# --------------------------------------------------------------------------- #
# Loading app.py
# --------------------------------------------------------------------------- #
class _Scores:
    """Mimics a TensorFlow tensor just enough for `.numpy()`."""

    def __init__(self, arr):
        self._arr = arr

    def numpy(self):
        return self._arr


def _install_fake_tensorflow():
    """Minimal stand-in for `tensorflow` so app.py imports without TensorFlow installed."""

    class _ModelBase:
        def __call__(self, inputs):
            return self.call(inputs)

        def load_weights(self, *args, **kwargs):
            return None

    passthrough = lambda *args, **kwargs: (lambda x: x)  # noqa: E731
    fake_tf = types.ModuleType("tensorflow")
    fake_tf.keras = types.SimpleNamespace(
        Model=_ModelBase,
        layers=types.SimpleNamespace(StringLookup=passthrough, Embedding=passthrough),
    )
    fake_tf.constant = lambda value: np.array(value)
    fake_tf.strings = types.SimpleNamespace(split=lambda x, sep: x)
    fake_tf.reduce_mean = lambda x, axis=None: x
    fake_tf.concat = lambda xs, axis=-1: xs[0]
    fake_tf.matmul = lambda a, b, transpose_b=False: _Scores(
        np.asarray(a) @ (np.asarray(b).T if transpose_b else np.asarray(b))
    )
    sys.modules["tensorflow"] = fake_tf


def _load_app():
    os.chdir(ROOT)  # app.py reads the CSV through a relative path
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    if not USE_REAL_TF:
        _install_fake_tensorflow()
    import app as app_module

    return app_module


app_module = _load_app()
DF = app_module.df


# --------------------------------------------------------------------------- #
# Test helpers
# --------------------------------------------------------------------------- #
def _titles_from(value):
    arr = value.numpy() if hasattr(value, "numpy") else value
    return [v.decode() if isinstance(v, bytes) else str(v) for v in np.asarray(arr).reshape(-1)]


class FakeEmbedder:
    """Deterministic replacement for MovieModel.

    mode="random":   every title gets a fixed pseudo-random positive vector.
    mode="constant": every title gets the same vector, so similarity is a tie and
                     only the boosting rules in app.py decide the ranking.
    overrides:       {title: vector} to give specific titles a different vector.
    """

    _cache = {}

    def __init__(self, mode="random", overrides=None):
        self.mode = mode
        self.overrides = overrides or {}

    def _vector(self, title):
        if title in self.overrides:
            return np.asarray(self.overrides[title], dtype="float32")
        if self.mode == "constant":
            return np.ones(8, dtype="float32")
        if title not in self._cache:
            rng = np.random.default_rng(zlib.crc32(title.encode()))
            self._cache[title] = (np.abs(rng.normal(size=8)) + 0.1).astype("float32")
        return self._cache[title]

    def __call__(self, inputs):
        return np.stack([self._vector(t) for t in _titles_from(inputs["title"])])


def _json_response(payload):
    response = mock.Mock()
    response.json.return_value = payload
    return response


def _tmdb_fake_get(person_results, credits=None, movies=None):
    def fake_get(url, timeout=None):
        if "search/person" in url:
            return _json_response({"results": person_results})
        if "movie_credits" in url:
            return _json_response({"cast": credits or []})
        if "search/movie" in url:
            return _json_response({"results": movies or []})
        raise AssertionError(f"unexpected URL: {url}")

    return fake_get


def _row_frame(**overrides):
    row = {
        "movieId": 1,
        "title": "Test Movie (2001)",
        "year": "2001",
        "mood": "Epic",
        "Industry": "HOLLYWOOD",
        "cast": "A|B",
        "poster": "https://example.com/p.jpg",
        "rating": 8.0,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def _largest_industry_mood_group():
    keys = DF["Industry"].astype(str).str.upper() + "|" + DF["mood"].astype(str).str.lower()
    key = keys.value_counts().index[0]
    movie_id = int(DF[keys == key].iloc[0]["movieId"])
    return movie_id, key, keys


class AppTestCase(unittest.TestCase):
    def setUp(self):
        app_module.app.config["TESTING"] = True
        self.client = app_module.app.test_client()

        # Unit tests never use the network: every HTTP call fails unless a test overrides it.
        patcher = mock.patch.object(
            app_module.requests,
            "get",
            side_effect=app_module.requests.ConnectionError("network disabled in tests"),
        )
        self.requests_get = patcher.start()
        self.addCleanup(patcher.stop)

    def use_model(self, model):
        patcher = mock.patch.object(app_module, "model", model)
        patcher.start()
        self.addCleanup(patcher.stop)

    def capture_render(self):
        """Replace render_template so tests can inspect what each route passes to it."""
        patcher = mock.patch.object(app_module, "render_template", return_value="rendered")
        mocked = patcher.start()
        self.addCleanup(patcher.stop)
        return mocked


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
class TestExtractYear(AppTestCase):
    def test_reads_year_from_title(self):
        self.assertEqual(app_module.extract_year("Gladiator (2000)"), "2000")

    def test_returns_na_when_title_has_no_year(self):
        self.assertEqual(app_module.extract_year("Gladiator"), "N/A")

    def test_every_csv_row_gets_a_year_column(self):
        self.assertIn("year", DF.columns)
        self.assertEqual(len(DF), len(DF["year"]))


# --------------------------------------------------------------------------- #
# OMDb helper
# --------------------------------------------------------------------------- #
class TestOmdbHelper(AppTestCase):
    def test_returns_poster_plot_and_rating_when_movie_found(self):
        self.requests_get.side_effect = None
        self.requests_get.return_value = _json_response(
            {"Response": "True", "Poster": "http://p.jpg", "Plot": "A plot.", "imdbRating": "8.5"}
        )
        meta = app_module.get_omdb("Gladiator (2000)")
        self.assertEqual(meta, {"poster": "http://p.jpg", "plot": "A plot.", "rating": "8.5"})

    def test_year_is_removed_from_the_search_title(self):
        self.requests_get.side_effect = None
        self.requests_get.return_value = _json_response({"Response": "False"})
        app_module.get_omdb("Gladiator (2000)")
        url = self.requests_get.call_args.args[0]
        self.assertIn("t=Gladiator&", url)
        self.assertNotIn("(2000)", url)

    def test_request_has_a_timeout_so_a_slow_api_cannot_hang_the_page(self):
        self.requests_get.side_effect = None
        self.requests_get.return_value = _json_response({"Response": "False"})
        app_module.get_omdb("Gladiator (2000)")
        self.assertIsNotNone(self.requests_get.call_args.kwargs.get("timeout"))

    def test_falls_back_to_placeholder_when_movie_not_found(self):
        self.requests_get.side_effect = None
        self.requests_get.return_value = _json_response({"Response": "False"})
        meta = app_module.get_omdb("Unknown Film (1999)")
        self.assertEqual(meta["plot"], "No plot available.")
        self.assertEqual(meta["rating"], "N/A")
        self.assertIn("placeholder", meta["poster"])

    def test_falls_back_to_placeholder_on_network_error(self):
        meta = app_module.get_omdb("Gladiator (2000)")  # setUp makes every request fail
        self.assertEqual(meta["plot"], "No plot available.")
        self.assertEqual(meta["rating"], "N/A")


# --------------------------------------------------------------------------- #
# TMDB search helper
# --------------------------------------------------------------------------- #
class TestTmdbSearch(AppTestCase):
    def test_actor_query_returns_at_most_20_actor_movies(self):
        credits = [
            {"id": i, "title": f"Movie {i}", "poster_path": "/x.jpg",
             "release_date": "2001-05-05", "vote_average": 7.1}
            for i in range(30)
        ]
        self.requests_get.side_effect = _tmdb_fake_get([{"id": 7}], credits=credits)
        results = app_module.get_tmdb_results("some actor")
        self.assertEqual(len(results), 20)
        self.assertTrue(all(r["mood"] == "Actor Hit" for r in results))
        self.assertEqual(results[0]["poster"], "https://image.tmdb.org/t/p/w500/x.jpg")
        self.assertEqual(results[0]["year"], "2001")

    def test_missing_poster_gets_placeholder(self):
        credits = [{"id": 1, "title": "No Poster", "poster_path": None,
                    "release_date": "1999-01-01", "vote_average": 6.0}]
        self.requests_get.side_effect = _tmdb_fake_get([{"id": 7}], credits=credits)
        results = app_module.get_tmdb_results("some actor")
        self.assertIn("placeholder", results[0]["poster"])

    def test_movie_query_used_when_no_actor_matches(self):
        movies = [{"id": 5, "title": "Found Film", "poster_path": "/f.jpg",
                   "release_date": "2010-02-02", "vote_average": 7.7}]
        self.requests_get.side_effect = _tmdb_fake_get([], movies=movies)
        results = app_module.get_tmdb_results("found film")
        self.assertEqual([r["title"] for r in results], ["Found Film"])
        self.assertEqual(results[0]["mood"], "New Discovery")

    def test_returns_empty_list_when_tmdb_is_unreachable(self):
        self.assertEqual(app_module.get_tmdb_results("anything"), [])


# --------------------------------------------------------------------------- #
# Poster / card building
# --------------------------------------------------------------------------- #
class TestEnrichedList(AppTestCase):
    def test_relative_tmdb_poster_path_becomes_full_url(self):
        out = app_module.get_enriched_list(_row_frame(poster="/abc.jpg"))
        self.assertEqual(out[0]["poster"], "https://image.tmdb.org/t/p/w500/abc.jpg")

    def test_full_poster_url_is_kept(self):
        out = app_module.get_enriched_list(_row_frame(poster="https://example.com/full.jpg"))
        self.assertEqual(out[0]["poster"], "https://example.com/full.jpg")

    def test_missing_poster_is_fetched_from_omdb(self):
        with mock.patch.object(app_module, "get_omdb", return_value={"poster": "http://omdb/p.jpg"}):
            out = app_module.get_enriched_list(_row_frame(poster=""))
        self.assertEqual(out[0]["poster"], "http://omdb/p.jpg")

    def test_omdb_na_poster_becomes_no_poster_placeholder(self):
        with mock.patch.object(app_module, "get_omdb", return_value={"poster": "N/A"}):
            out = app_module.get_enriched_list(_row_frame(poster="N/A"))
        self.assertIn("No+Poster", out[0]["poster"])

    def test_empty_rating_defaults_to_7_5(self):
        out = app_module.get_enriched_list(_row_frame(rating=""))
        self.assertEqual(out[0]["rating"], "7.5")

    def test_card_carries_title_year_mood_and_industry(self):
        out = app_module.get_enriched_list(_row_frame())[0]
        self.assertEqual(
            (out["title"], out["year"], out["mood"], out["Industry"]),
            ("Test Movie (2001)", "2001", "Epic", "HOLLYWOOD"),
        )


# --------------------------------------------------------------------------- #
# Home and search routes
# --------------------------------------------------------------------------- #
class TestHomeRoute(AppTestCase):
    def test_home_page_renders(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

    def test_banner_is_the_most_popular_movie(self):
        render = self.capture_render()
        self.client.get("/")
        banner = render.call_args.kwargs["banner"]
        popularity = DF.loc[DF["movieId"] == banner["movieId"], "popularity"].iloc[0]
        self.assertEqual(popularity, DF["popularity"].max())

    def test_each_industry_row_has_at_most_12_movies(self):
        render = self.capture_render()
        self.client.get("/")
        for section in ("bollywood", "hollywood", "chinese", "korean", "animation"):
            self.assertLessEqual(len(render.call_args.kwargs[section]), 12, section)


class TestSearchRoute(AppTestCase):
    def setUp(self):
        super().setUp()
        patcher = mock.patch.object(app_module, "get_tmdb_results", return_value=[])
        self.tmdb = patcher.start()
        self.addCleanup(patcher.stop)

    def titles_for(self, query):
        render = self.capture_render() if not hasattr(self, "_render") else self._render
        self._render = render
        self.client.get("/search", query_string={"query": query})
        return [r["title"] for r in render.call_args.kwargs["results"]]

    def test_missing_query_redirects_to_home(self):
        response = self.client.get("/search")
        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.headers["Location"].endswith("/"))

    def test_blank_query_redirects_to_home(self):
        response = self.client.get("/search", query_string={"query": "   "})
        self.assertEqual(response.status_code, 302)

    def test_finds_movie_by_title(self):
        self.assertIn("Gladiator (2000)", self.titles_for("Gladiator"))

    def test_search_is_case_insensitive(self):
        self.assertIn("Gladiator (2000)", self.titles_for("gLaDiAtOr"))

    def test_finds_movie_by_cast_member(self):
        self.assertIn("Gladiator (2000)", self.titles_for("Russell Crowe"))

    def test_query_with_no_match_returns_empty_results(self):
        self.assertEqual(self.titles_for("zzzzqqqqxxxx"), [])

    def test_duplicates_between_tmdb_and_csv_are_removed(self):
        self.tmdb.return_value = [{"movieId": 1, "title": "Gladiator (2000)", "poster": "p",
                                   "year": "2000", "rating": 8, "mood": "Actor Hit"}]
        titles = [t.lower() for t in self.titles_for("Gladiator")]
        self.assertEqual(titles.count("gladiator (2000)"), 1)

    @unittest.expectedFailure  # KNOWN BUG: query is used as a regex, so "(" raises re.error -> HTTP 500
    def test_query_with_regex_characters_does_not_crash(self):
        response = self.client.get("/search", query_string={"query": "("})
        self.assertLess(response.status_code, 500)


# --------------------------------------------------------------------------- #
# Movie details route
# --------------------------------------------------------------------------- #
class TestMovieDetailsRoute(AppTestCase):
    GLADIATOR_ID = 98

    def test_unknown_movie_id_redirects_home(self):
        response = self.client.get("/movie/99999999")
        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.headers["Location"].endswith("/"))

    def test_returns_12_related_movies_excluding_the_movie_itself(self):
        self.use_model(FakeEmbedder("random"))
        render = self.capture_render()
        self.client.get(f"/movie/{self.GLADIATOR_ID}")
        kwargs = render.call_args.kwargs
        related_ids = [m["movieId"] for m in kwargs["related"]]
        self.assertEqual(len(related_ids), 12)
        self.assertNotIn(self.GLADIATOR_ID, related_ids)
        self.assertEqual(kwargs["movie"]["movieId"], self.GLADIATOR_ID)

    def test_trailer_link_searches_youtube_with_clean_title(self):
        self.use_model(FakeEmbedder("random"))
        render = self.capture_render()
        self.client.get(f"/movie/{self.GLADIATOR_ID}")
        url = render.call_args.kwargs["movie"]["trailer_url"]
        self.assertTrue(url.startswith("https://www.youtube.com/results?search_query="))
        self.assertIn("Gladiator+official+trailer", url)
        self.assertNotIn("2000", url)

    def test_same_industry_and_mood_rank_first_when_similarity_is_tied(self):
        """With identical embeddings, only the 2.5x industry and 3.0x mood boosts decide the order."""
        movie_id, key, keys = _largest_industry_mood_group()
        self.assertGreaterEqual((keys == key).sum(), 13, "need a group big enough to fill 12 slots")
        self.use_model(FakeEmbedder("constant"))
        render = self.capture_render()
        self.client.get(f"/movie/{movie_id}")
        related_ids = [m["movieId"] for m in render.call_args.kwargs["related"]]
        related_keys = set(keys[DF["movieId"].isin(related_ids)])
        self.assertEqual(related_keys, {key})


# --------------------------------------------------------------------------- #
# Recommend route
# --------------------------------------------------------------------------- #
class TestRecommendRoute(AppTestCase):
    def post(self, name):
        return self.client.post("/recommend", data={"movie_name": name})

    def test_unknown_movie_shows_not_found_error(self):
        render = self.capture_render()
        self.post("zzzz-not-a-movie-zzzz")
        self.assertEqual(render.call_args.args[0], "index.html")
        self.assertEqual(render.call_args.kwargs["error"], "Movie not found!")

    def test_exact_title_match_is_case_insensitive_and_returns_12(self):
        self.use_model(FakeEmbedder("random"))
        render = self.capture_render()
        self.post("gladiator (2000)")
        self.assertEqual(len(render.call_args.kwargs["results"]), 12)
        self.assertEqual(render.call_args.kwargs["query"], "gladiator (2000)")

    def test_partial_title_falls_back_to_contains_match(self):
        self.use_model(FakeEmbedder("random"))
        render = self.capture_render()
        self.post("Gladiator")
        self.assertEqual(len(render.call_args.kwargs["results"]), 12)

    @unittest.expectedFailure  # KNOWN BUG: request.form.get() returns None -> None.lower() -> HTTP 500
    def test_missing_movie_name_field_does_not_crash(self):
        response = self.client.post("/recommend", data={})
        self.assertLess(response.status_code, 500)

    @unittest.expectedFailure  # KNOWN BUG: the query movie is dropped by position ([1:13]), not by id
    def test_searched_movie_never_appears_in_its_own_recommendations(self):
        """If any other movie scores higher than the query movie, the query movie leaks into the list."""
        other_title = next(t for t in DF["title"] if t != "Gladiator (2000)")
        self.use_model(FakeEmbedder("constant", overrides={other_title: np.full(8, 100.0)}))
        render = self.capture_render()
        self.post("Gladiator (2000)")
        titles = [m["title"] for m in render.call_args.kwargs["results"]]
        self.assertNotIn("Gladiator (2000)", titles)

    @unittest.skipUnless(USE_REAL_TF, "integration test: run with REAL_TF=1 and TensorFlow installed")
    def test_real_tensorflow_model_returns_12_recommendations(self):
        render = self.capture_render()
        self.post("Gladiator (2000)")
        self.assertEqual(len(render.call_args.kwargs["results"]), 12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
