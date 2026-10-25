"""Mood classifier that demonstrates imbalanced-data handling.

The ``mood`` label is heavily imbalanced (e.g. 'Futuristic' has 11 titles, while
'Adrenaline' has ~2,900). Accuracy hides that, so this script reports
macro-F1 and per-class precision/recall, and compares an unweighted logistic
regression with one trained using ``class_weight="balanced"``.

Usage:  python mood_classifier.py
Writes reports/mood_classifier.md.
"""
from __future__ import annotations

import os

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from recommender import split_pipe

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "movies_with_popularity.csv")


def make_model(class_weight=None) -> Pipeline:
    features = ColumnTransformer([
        ("keywords", TfidfVectorizer(analyzer=split_pipe, sublinear_tf=True), "keywords"),
        ("cast", TfidfVectorizer(analyzer=split_pipe, sublinear_tf=True), "cast"),
    ])
    clf = LogisticRegression(max_iter=1000, class_weight=class_weight)
    return Pipeline([("features", features), ("clf", clf)])


def compare(df: pd.DataFrame, target: str = "mood", seed: int = 42) -> dict:
    """Train with and without class balancing on a stratified split."""
    X, y = df[["keywords", "cast"]], df[target]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
    out = {"train_size": len(X_tr), "test_size": len(X_te), "configs": {}}
    for name, cw in (("unweighted", None), ("class_weight=balanced", "balanced")):
        model = make_model(cw).fit(X_tr, y_tr)
        pred = model.predict(X_te)
        out["configs"][name] = {
            "accuracy": round(float(accuracy_score(y_te, pred)), 4),
            "macro_f1": round(float(f1_score(y_te, pred, average="macro")), 4),
            "per_class": classification_report(y_te, pred, output_dict=True, zero_division=0),
        }
    out["class_counts"] = y.value_counts().to_dict()
    return out


def to_markdown(res: dict) -> str:
    out = ["# Mood classifier: handling class imbalance", "",
           f"Stratified split: {res['train_size']} train / {res['test_size']} test. "
           "Features: TF-IDF over keywords and cast.", "",
           "## Class counts (full dataset)", "", "| Mood | Titles |", "|---|---|"]
    out += [f"| {k} | {v} |" for k, v in res["class_counts"].items()]
    out += ["", "## Overall", "", "| Config | Accuracy | Macro-F1 |", "|---|---|---|"]
    for name, r in res["configs"].items():
        out.append(f"| {name} | {r['accuracy']:.3f} | {r['macro_f1']:.3f} |")
    out += ["", "## Per-class recall (the minority classes are the point)", "",
            "| Mood | Test support | Recall unweighted | Recall balanced | F1 unweighted | F1 balanced |",
            "|---|---|---|---|---|---|"]
    u, b = res["configs"]["unweighted"]["per_class"], res["configs"]["class_weight=balanced"]["per_class"]
    for mood in res["class_counts"]:
        out.append(f"| {mood} | {int(u[mood]['support'])} | {u[mood]['recall']:.2f} | {b[mood]['recall']:.2f} | "
                   f"{u[mood]['f1-score']:.2f} | {b[mood]['f1-score']:.2f} |")
    out += ["", "Note: classes with only a handful of test examples (e.g. 'Futuristic') give very noisy "
            "estimates; treat their rows as indicative, not conclusive."]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    data = pd.read_csv(CSV_PATH).fillna("")
    result = compare(data)
    os.makedirs(os.path.join(BASE_DIR, "reports"), exist_ok=True)
    md = to_markdown(result)
    with open(os.path.join(BASE_DIR, "reports", "mood_classifier.md"), "w") as f:
        f.write(md)
    print(md)
