"""Content-based movie recommender built on scikit-learn.

Features come from the pipe-separated ``keywords`` and ``cast`` columns and are
weighted with TF-IDF. The ``genres`` column is deliberately NOT used as a
feature, so it can serve as held-out ground truth in ``evaluate.py``.

Optional business-rule boosts (same industry / same mood / shared cast) and a
feedback-based nudge are applied on top of the base cosine similarity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize
import scipy.sparse as sp

NEUTRAL_BOOSTS = {"industry": 1.0, "mood": 1.0, "cast": 1.0}


def split_pipe(value) -> list[str]:
    """Split a 'a|b|c' cell into clean lower-case tokens ([] for missing)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [t.strip().lower() for t in str(value).split("|") if t.strip()]


class Recommender:
    """TF-IDF + cosine similarity recommender with optional re-ranking."""

    def __init__(self, df: pd.DataFrame, keyword_weight: float = 1.0, cast_weight: float = 1.0):
        self.df = df.reset_index(drop=True)
        self.n = len(self.df)

        kw_vec = TfidfVectorizer(analyzer=split_pipe, sublinear_tf=True)
        cast_vec = TfidfVectorizer(analyzer=split_pipe, sublinear_tf=True)
        kw = kw_vec.fit_transform(self.df["keywords"])
        cast = cast_vec.fit_transform(self.df["cast"])
        # Rows are L2-normalised, so a dot product equals cosine similarity.
        self.features = normalize(sp.hstack([kw * keyword_weight, cast * cast_weight]).tocsr())

        cast_bin = CountVectorizer(analyzer=split_pipe, binary=True)
        self._cast_bin = cast_bin.fit_transform(self.df["cast"]).tocsr()

        self.industry = self.df["Industry"].astype(str).str.upper().to_numpy()
        self.mood = self.df["mood"].astype(str).str.lower().to_numpy()
        self._titles_lower = self.df["title"].astype(str).str.lower().to_numpy()

    # ----------------------------------------------------------------- lookup
    def find(self, query: str) -> int | None:
        """Exact (case-insensitive) title match first, then substring match."""
        q = (query or "").strip().lower()
        if not q:
            return None
        exact = np.flatnonzero(self._titles_lower == q)
        if exact.size:
            return int(exact[0])
        partial = [i for i, t in enumerate(self._titles_lower) if q in t]
        return int(partial[0]) if partial else None

    # ---------------------------------------------------------------- scoring
    def base_scores(self, idx: int) -> np.ndarray:
        """Cosine similarity of movie ``idx`` to every movie."""
        return (self.features @ self.features[idx].T).toarray().ravel()

    def shared_cast(self, idx: int) -> np.ndarray:
        overlap = (self._cast_bin @ self._cast_bin[idx].T).toarray().ravel()
        return overlap > 0

    def scores(self, idx: int, boosts: dict | None = None, feedback: dict | None = None) -> np.ndarray:
        s = self.base_scores(idx).astype(float)
        b = {**NEUTRAL_BOOSTS, **(boosts or {})}
        if b["industry"] != 1.0:
            s = np.where(self.industry == self.industry[idx], s * b["industry"], s)
        if b["mood"] != 1.0:
            s = np.where(self.mood == self.mood[idx], s * b["mood"], s)
        if b["cast"] != 1.0:
            s = np.where(self.shared_cast(idx), s * b["cast"], s)
        if feedback:
            # feedback maps item index -> net votes (likes minus dislikes).
            factor = np.ones(self.n)
            for item, net in feedback.items():
                factor[item] = np.clip(1.0 + 0.1 * net, 0.5, 1.5)
            s = s * factor
        return s

    def recommend(self, idx: int, k: int = 12, boosts: dict | None = None,
                  feedback: dict | None = None) -> np.ndarray:
        """Indices of the top-``k`` movies, excluding ``idx`` itself."""
        s = self.scores(idx, boosts, feedback)
        s[idx] = -np.inf
        k = max(0, min(k, self.n - 1))
        return np.argsort(-s, kind="stable")[:k]
