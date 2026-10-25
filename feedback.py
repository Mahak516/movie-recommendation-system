"""Tiny SQLite store for like/dislike feedback on recommendations.

Votes feed back into ranking: ``Recommender.recommend(..., feedback=...)`` nudges
a recommended movie's score up or down by net votes (bounded to +/-50%).
"""
from __future__ import annotations

import sqlite3
import time
from contextlib import closing

SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    movie_id INTEGER NOT NULL,
    rec_id INTEGER NOT NULL,
    liked INTEGER NOT NULL CHECK (liked IN (0, 1)),
    created_at REAL NOT NULL
)
"""


def _connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute(SCHEMA)
    return conn


def add_feedback(path: str, movie_id: int, rec_id: int, liked: bool) -> None:
    with closing(_connect(path)) as conn, conn:
        conn.execute(
            "INSERT INTO feedback (movie_id, rec_id, liked, created_at) VALUES (?, ?, ?, ?)",
            (int(movie_id), int(rec_id), 1 if liked else 0, time.time()),
        )


def net_votes(path: str, movie_id: int) -> dict[int, int]:
    """Net votes (likes - dislikes) per recommended movie id for ``movie_id``."""
    with closing(_connect(path)) as conn:
        rows = conn.execute(
            "SELECT rec_id, SUM(CASE WHEN liked = 1 THEN 1 ELSE -1 END) "
            "FROM feedback WHERE movie_id = ? GROUP BY rec_id",
            (int(movie_id),),
        ).fetchall()
    return {int(rec_id): int(net) for rec_id, net in rows}
