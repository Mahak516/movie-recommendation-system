"""Offline evaluation of the recommenders.

There are no user ratings in the dataset, so relevance is a *proxy*: a
recommended movie is "relevant" to a query movie when their genre sets have
Jaccard similarity >= 0.5. Genres are never used as model features, so this is
a held-out signal, but it is still a proxy and not real user satisfaction.

Metrics (k = 10): Precision@k, HitRate@k, NDCG@k (graded by Jaccard), catalog
coverage, and the industry mix of what gets recommended (bias check).

Usage:  python evaluate.py [--quick]
Writes reports/evaluation.md and reports/evaluation.json.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from recommender import Recommender

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "movies_with_popularity.csv")
K = 10
RELEVANT_JACCARD = 0.5

# The hand-tuned boost presets used by the Flask routes in app.py.
RECOMMEND_PRESET = {"industry": 1.2, "mood": 1.3, "cast": 2.0}
MOVIE_PAGE_PRESET = {"industry": 2.5, "mood": 3.0, "cast": 1.2}
# Same preset without the mood boost. Mood labels in this dataset are tied to
# genres (e.g. Feel-Good is 40% Comedy), so mood boosts partly "see" the answer
# key; this variant gives a non-circular read on the other two boosts.
MOVIE_PAGE_NO_MOOD = {"industry": 2.5, "cast": 1.2}


def genre_matrix(df: pd.DataFrame) -> np.ndarray:
    """Binary movie x genre matrix built from the pipe-separated genres column."""
    sets = [set(g for g in str(v).split("|") if g) for v in df["genres"]]
    vocab = sorted(set().union(*sets))
    pos = {g: i for i, g in enumerate(vocab)}
    mat = np.zeros((len(df), len(vocab)), dtype=np.float32)
    for row, s in enumerate(sets):
        for g in s:
            mat[row, pos[g]] = 1.0
    return mat


def jaccard_to_all(G: np.ndarray, q: int) -> np.ndarray:
    inter = G @ G[q]
    union = G.sum(axis=1) + G[q].sum() - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def ndcg_at_k(gains: np.ndarray, ideal_gains: np.ndarray) -> float:
    k = len(gains)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float((gains * discounts).sum())
    idcg = float((np.sort(ideal_gains)[::-1][:k] * discounts[:k]).sum())
    return dcg / idcg if idcg > 0 else 0.0


def score_strategy(name, recs_by_query, queries, G, industry, catalog_hollywood_share):
    precisions, hits, ndcgs, same_ind, hollywood = [], [], [], [], []
    seen = set()
    for q in queries:
        recs = np.asarray(recs_by_query[q])[:K]
        jac = jaccard_to_all(G, q)
        gains = jac[recs]
        ideal = jac.copy()
        ideal[q] = -1  # a movie is not a recommendation for itself
        relevant = gains >= RELEVANT_JACCARD
        precisions.append(relevant.mean())
        hits.append(float(relevant.any()))
        ndcgs.append(ndcg_at_k(gains, ideal))
        same_ind.append(float((industry[recs] == industry[q]).mean()))
        hollywood.append(float((industry[recs] == "HOLLYWOOD").mean()))
        seen.update(int(r) for r in recs)
    return {
        "strategy": name,
        "n_queries": len(queries),
        f"precision@{K}": round(float(np.mean(precisions)), 4),
        f"hit_rate@{K}": round(float(np.mean(hits)), 4),
        f"ndcg@{K}": round(float(np.mean(ndcgs)), 4),
        "catalog_coverage": round(len(seen) / len(industry), 4),
        "same_industry_share": round(float(np.mean(same_ind)), 4),
        "hollywood_share_of_recs": round(float(np.mean(hollywood)), 4),
        "hollywood_share_of_catalog": round(catalog_hollywood_share, 4),
    }


def build_strategies(rec: Recommender, df: pd.DataFrame, rng: np.random.Generator):
    popularity_order = np.argsort(-pd.to_numeric(df["popularity"], errors="coerce").fillna(0).to_numpy(),
                                  kind="stable")

    def random_fn(q):
        pool = np.delete(np.arange(rec.n), q)
        return rng.choice(pool, size=K, replace=False)

    def popularity_fn(q):
        return [i for i in popularity_order[: K + 1] if i != q][:K]

    return {
        "random": random_fn,
        "most popular": popularity_fn,
        "TF-IDF cosine (no boosts)": lambda q: rec.recommend(q, K),
        "TF-IDF + /recommend boosts": lambda q: rec.recommend(q, K, boosts=RECOMMEND_PRESET),
        "TF-IDF + movie-page boosts": lambda q: rec.recommend(q, K, boosts=MOVIE_PAGE_PRESET),
        "TF-IDF + movie-page boosts, no mood": lambda q: rec.recommend(q, K, boosts=MOVIE_PAGE_NO_MOOD),
    }


def run(quick: bool = False, seed: int = 42) -> dict:
    df = pd.read_csv(CSV_PATH).fillna("")
    rec = Recommender(df)
    G = genre_matrix(df)
    industry = rec.industry
    hollywood_share = float((industry == "HOLLYWOOD").mean())
    rng = np.random.default_rng(seed)
    has_genre = np.flatnonzero(G.sum(axis=1) > 0)

    n_overall = 200 if quick else 1000
    n_per_industry = 30 if quick else 150

    overall_q = rng.choice(has_genre, size=min(n_overall, len(has_genre)), replace=False).tolist()
    strategies = build_strategies(rec, df, rng)

    t0 = time.perf_counter()
    overall = []
    for name, fn in strategies.items():
        recs = {q: fn(q) for q in overall_q}
        overall.append(score_strategy(name, recs, overall_q, G, industry, hollywood_share))
    elapsed = time.perf_counter() - t0

    # Bias check: precision for each industry's own query movies.
    by_industry = {}
    for ind in sorted(set(industry)):
        pool = np.intersect1d(np.flatnonzero(industry == ind), has_genre)
        if pool.size == 0:
            continue
        qs = rng.choice(pool, size=min(n_per_industry, pool.size), replace=False).tolist()
        rows = {}
        for name in ("TF-IDF cosine (no boosts)", "TF-IDF + movie-page boosts"):
            fn = strategies[name]
            res = score_strategy(name, {q: fn(q) for q in qs}, qs, G, industry, hollywood_share)
            rows[name] = {"n_queries": len(qs), f"precision@{K}": res[f"precision@{K}"],
                          "same_industry_share": res["same_industry_share"]}
        by_industry[ind] = {"catalog_titles": int((industry == ind).sum()), **rows}

    return {"seed": seed, "k": K, "relevance_rule": f"genre Jaccard >= {RELEVANT_JACCARD}",
            "catalog_size": int(len(df)), "quick": quick, "overall": overall,
            "by_industry": by_industry, "eval_seconds": round(elapsed, 1)}


def to_markdown(res: dict) -> str:
    p, h, n = f"precision@{K}", f"hit_rate@{K}", f"ndcg@{K}"
    out = ["# Offline evaluation", "",
           f"Catalog: {res['catalog_size']} titles. Relevance proxy: {res['relevance_rule']} "
           f"(genres are not model features). k = {res['k']}, seed = {res['seed']}"
           f"{' (quick run, small sample)' if res['quick'] else ''}.", "",
           "## Overall (random sample of query movies)", "",
           f"| Strategy | Queries | Precision@{K} | HitRate@{K} | NDCG@{K} | Coverage | Same-industry | Hollywood share of recs |",
           "|---|---|---|---|---|---|---|---|"]
    for r in res["overall"]:
        out.append(f"| {r['strategy']} | {r['n_queries']} | {r[p]:.3f} | {r[h]:.3f} | {r[n]:.3f} | "
                   f"{r['catalog_coverage']:.3f} | {r['same_industry_share']:.3f} | "
                   f"{r['hollywood_share_of_recs']:.3f} |")
    out += ["", f"Hollywood is {res['overall'][0]['hollywood_share_of_catalog']:.1%} of the catalog.", "",
            "Caveat: mood labels in the dataset are closely tied to genres, so boosts that use mood are "
            "partly rewarded for matching the answer key. The 'no mood' row is the non-circular comparison.", "",
            "## Bias check: quality per query industry", "",
            f"| Industry | Titles | Queries | P@{K} no boosts | P@{K} movie-page boosts | "
            "Same-industry share (no boosts) | Same-industry share (boosts) |",
            "|---|---|---|---|---|---|---|"]
    a, b = "TF-IDF cosine (no boosts)", "TF-IDF + movie-page boosts"
    for ind, row in res["by_industry"].items():
        out.append(f"| {ind} | {row['catalog_titles']} | {row[a]['n_queries']} | {row[a][p]:.3f} | "
                   f"{row[b][p]:.3f} | {row[a]['same_industry_share']:.3f} | {row[b]['same_industry_share']:.3f} |")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small sample for CI smoke runs")
    args = ap.parse_args()
    result = run(quick=args.quick)
    os.makedirs(os.path.join(BASE_DIR, "reports"), exist_ok=True)
    suffix = "_quick" if args.quick else ""
    with open(os.path.join(BASE_DIR, "reports", f"evaluation{suffix}.json"), "w") as f:
        json.dump(result, f, indent=2)
    md = to_markdown(result)
    with open(os.path.join(BASE_DIR, "reports", f"evaluation{suffix}.md"), "w") as f:
        f.write(md)
    print(md)
