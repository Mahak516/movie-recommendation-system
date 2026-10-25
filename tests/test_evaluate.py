import unittest

import numpy as np
import pandas as pd

import evaluate
from mood_classifier import compare


class MetricTests(unittest.TestCase):
    def test_jaccard_to_all(self):
        df = pd.DataFrame({"genres": ["Action|Drama", "Action", "Comedy", ""]})
        G = evaluate.genre_matrix(df)
        jac = evaluate.jaccard_to_all(G, 0)
        np.testing.assert_allclose(jac, [1.0, 0.5, 0.0, 0.0])

    def test_ndcg_perfect_and_worst(self):
        ideal = np.array([1.0, 0.8, 0.5, 0.2, 0.0])
        self.assertAlmostEqual(evaluate.ndcg_at_k(np.array([1.0, 0.8, 0.5]), ideal), 1.0)
        self.assertEqual(evaluate.ndcg_at_k(np.zeros(3), ideal), 0.0)

    def test_ndcg_handles_all_zero_ideal(self):
        self.assertEqual(evaluate.ndcg_at_k(np.zeros(3), np.zeros(5)), 0.0)


class EvaluationRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = evaluate.run(quick=True)

    def test_tfidf_beats_random_and_popularity(self):
        by_name = {r["strategy"]: r for r in self.result["overall"]}
        key = f"precision@{evaluate.K}"
        tfidf = by_name["TF-IDF cosine (no boosts)"][key]
        self.assertGreater(tfidf, by_name["random"][key])
        self.assertGreater(tfidf, by_name["most popular"][key])

    def test_metrics_are_valid_fractions(self):
        for r in self.result["overall"]:
            for key in (f"precision@{evaluate.K}", f"hit_rate@{evaluate.K}", f"ndcg@{evaluate.K}"):
                self.assertTrue(0.0 <= r[key] <= 1.0, (r["strategy"], key, r[key]))

    def test_every_industry_is_reported(self):
        self.assertEqual(set(self.result["by_industry"]), {"HOLLYWOOD", "ANIME", "KOREAN", "CHINESE", "BOLLYWOOD"})


class ImbalanceTests(unittest.TestCase):
    def test_balanced_weighting_improves_minority_recall(self):
        rng = np.random.default_rng(0)
        rows = []
        for _ in range(400):  # majority class: words a/b
            rows.append(("a|b|common", "x", "big"))
        for _ in range(30):   # minority class: overlaps with the majority via 'common'
            rows.append(("z|common", "y", "small"))
        rng.shuffle(rows)
        df = pd.DataFrame(rows, columns=["keywords", "cast", "mood"])
        res = compare(df, target="mood")
        recall = {k: v["per_class"]["small"]["recall"] for k, v in res["configs"].items()}
        self.assertGreaterEqual(recall["class_weight=balanced"], recall["unweighted"])
        self.assertIn("macro_f1", res["configs"]["unweighted"])


if __name__ == "__main__":
    unittest.main()
