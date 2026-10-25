import os
import tempfile
import unittest
from unittest import mock

import app as app_module
import feedback as feedback_store


class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        app_module.app.config["FEEDBACK_DB"] = os.path.join(cls.tmp.name, "fb.db")
        app_module.app.config["TESTING"] = True
        cls.client = app_module.app.test_client()
        # Never hit the network from tests.
        cls.patches = [
            mock.patch.object(app_module, "get_tmdb_results", return_value=[]),
            mock.patch.object(app_module, "get_omdb", return_value={"poster": "p.jpg", "plot": "", "rating": "7"}),
        ]
        for p in cls.patches:
            p.start()

    @classmethod
    def tearDownClass(cls):
        for p in cls.patches:
            p.stop()
        cls.tmp.cleanup()

    def test_health(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        body = r.get_json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["movies"], len(app_module.df))

    def test_home_renders(self):
        self.assertEqual(self.client.get("/").status_code, 200)

    def test_movie_page_renders_and_unknown_id_redirects(self):
        movie_id = int(app_module.df.iloc[0]["movieId"])
        self.assertEqual(self.client.get(f"/movie/{movie_id}").status_code, 200)
        self.assertEqual(self.client.get("/movie/999999999").status_code, 302)

    def test_recommend_known_and_unknown_title(self):
        title = app_module.df.iloc[0]["title"]
        self.assertEqual(self.client.post("/recommend", data={"movie_name": title}).status_code, 200)
        r = self.client.post("/recommend", data={"movie_name": "zzzz no such film"})
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"Movie not found", r.data)

    def test_recommend_without_form_field_does_not_crash(self):
        self.assertEqual(self.client.post("/recommend", data={}).status_code, 200)

    def test_search_empty_query_redirects_home(self):
        self.assertEqual(self.client.get("/search?query=").status_code, 302)

    def test_search_handles_regex_characters(self):
        self.assertEqual(self.client.get("/search?query=(").status_code, 200)

    def test_feedback_validation_and_storage(self):
        ids = [int(x) for x in app_module.df["movieId"].iloc[:2]]
        ok = self.client.post("/feedback", json={"movie_id": ids[0], "rec_id": ids[1], "liked": True})
        self.assertEqual(ok.status_code, 201)
        self.assertEqual(self.client.post("/feedback", json={"movie_id": ids[0]}).status_code, 400)
        self.assertEqual(self.client.post("/feedback", json={"movie_id": ids[0], "rec_id": ids[1],
                                                              "liked": "maybe"}).status_code, 400)
        self.assertEqual(self.client.post("/feedback", json={"movie_id": ids[0], "rec_id": 999999999,
                                                              "liked": True}).status_code, 404)
        votes = feedback_store.net_votes(app_module.app.config["FEEDBACK_DB"], ids[0])
        self.assertEqual(votes, {ids[1]: 1})


class FeedbackStoreTests(unittest.TestCase):
    def test_net_votes_sums_likes_and_dislikes(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "fb.db")
            for liked in (True, True, False):
                feedback_store.add_feedback(path, 1, 2, liked)
            feedback_store.add_feedback(path, 1, 3, False)
            self.assertEqual(feedback_store.net_votes(path, 1), {2: 1, 3: -1})
            self.assertEqual(feedback_store.net_votes(path, 99), {})


if __name__ == "__main__":
    unittest.main()
