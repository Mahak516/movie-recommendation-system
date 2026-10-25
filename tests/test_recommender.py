import unittest

import numpy as np
import pandas as pd

from recommender import Recommender, split_pipe


def make_df():
    rows = [
        # id, title, cast, keywords, mood, industry
        (1, "Space War (2001)", "Ann Lee|Bob Ray", "space|war|robots", "Adrenaline", "HOLLYWOOD"),
        (2, "Star Battle (2002)", "Cy Dunn", "space|war|aliens", "Adrenaline", "HOLLYWOOD"),
        (3, "Quiet Love (2003)", "Di Fox", "romance|paris|letters", "Romantic", "HOLLYWOOD"),
        (4, "Paris Nights (2004)", "Di Fox|Ed Poe", "romance|paris|dance", "Romantic", "Bollywood"),
        (5, "Robot Uprising (2005)", "Ann Lee", "robots|war|space", "Adrenaline", "KOREAN"),
        (6, "No Keywords (2006)", "Gus Hall", "", "Neutral", "HOLLYWOOD"),
        (7, "Another Drama (2007)", "Hal Ice", "family|loss", "Dark", "ANIME"),
    ]
    return pd.DataFrame(rows, columns=["movieId", "title", "cast", "keywords", "mood", "Industry"])


class SplitPipeTests(unittest.TestCase):
    def test_splits_and_cleans(self):
        self.assertEqual(split_pipe(" A | b|| C "), ["a", "b", "c"])

    def test_missing_values_give_empty_list(self):
        self.assertEqual(split_pipe(None), [])
        self.assertEqual(split_pipe(float("nan")), [])
        self.assertEqual(split_pipe(""), [])


class RecommenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rec = Recommender(make_df())

    def test_excludes_itself_and_returns_unique_k(self):
        out = self.rec.recommend(0, k=4)
        self.assertEqual(len(out), 4)
        self.assertNotIn(0, out)
        self.assertEqual(len(set(out.tolist())), 4)

    def test_k_is_capped_by_catalog_size(self):
        self.assertEqual(len(self.rec.recommend(0, k=100)), self.rec.n - 1)

    def test_most_similar_ranks_first(self):
        # "Star Battle" and "Robot Uprising" share keywords with "Space War".
        self.assertIn(self.rec.recommend(0, k=1)[0], (1, 4))

    def test_results_are_deterministic(self):
        a = self.rec.recommend(2, k=5)
        b = self.rec.recommend(2, k=5)
        np.testing.assert_array_equal(a, b)

    def test_movie_with_empty_keywords_does_not_crash(self):
        out = self.rec.recommend(5, k=3)  # row 5 has no keywords
        self.assertEqual(len(out), 3)
        self.assertNotIn(5, out)

    def test_industry_boost_lifts_same_industry(self):
        plain = self.rec.scores(0)
        boosted = self.rec.scores(0, boosts={"industry": 3.0})
        same = self.rec.industry == self.rec.industry[0]
        np.testing.assert_allclose(boosted[same], plain[same] * 3.0)
        np.testing.assert_allclose(boosted[~same], plain[~same])

    def test_shared_cast_boost_applies_only_to_shared_cast(self):
        shared = self.rec.shared_cast(0)  # "Ann Lee" also stars in row 4
        self.assertTrue(shared[4])
        self.assertFalse(shared[2])
        plain = self.rec.scores(0)
        boosted = self.rec.scores(0, boosts={"cast": 2.0})
        self.assertAlmostEqual(boosted[4], plain[4] * 2.0)
        self.assertAlmostEqual(boosted[2], plain[2])

    def test_feedback_reorders_and_is_bounded(self):
        base = self.rec.scores(0)
        liked = self.rec.scores(0, feedback={4: 100})      # huge net votes
        disliked = self.rec.scores(0, feedback={1: -100})
        self.assertAlmostEqual(liked[4], base[4] * 1.5)      # capped at +50%
        self.assertAlmostEqual(disliked[1], base[1] * 0.5)   # capped at -50%

    def test_find_exact_partial_and_miss(self):
        self.assertEqual(self.rec.find("space war (2001)"), 0)
        self.assertEqual(self.rec.find("paris"), 3)
        self.assertIsNone(self.rec.find("does not exist"))
        self.assertIsNone(self.rec.find("  "))


if __name__ == "__main__":
    unittest.main()
