"""Optional TensorFlow embedding engine (your original model).

Enable with ``RECOMMENDER=tf`` after ``pip install -r requirements-tf.txt``.

IMPORTANT: this model only produces meaningful similarities after it has been
trained. The original repo does not include ``enhanced_movie_weights.weights.h5``;
without that file the embeddings are random and recommendations are driven by
the re-ranking boosts alone. The engine logs a warning in that case.
This module was written against the original app.py logic but could not be
executed in the environment where it was authored (TensorFlow was not
installable there), so run it once locally before relying on it.
"""
from __future__ import annotations

import logging
import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from recommender import Recommender  # noqa: E402

log = logging.getLogger("aurarec.tf")
WEIGHTS_FILE = "enhanced_movie_weights.weights.h5"


class MovieModel(tf.keras.Model):
    """Title/cast/keyword/mood embeddings concatenated into one vector."""

    def __init__(self, titles, cast, keywords, moods):
        super().__init__()
        self.title_lookup = tf.keras.layers.StringLookup(vocabulary=titles, mask_token=None)
        self.title_emb = tf.keras.layers.Embedding(len(titles) + 1, 32)
        self.cast_lookup = tf.keras.layers.StringLookup(vocabulary=cast, mask_token=None)
        self.cast_emb = tf.keras.layers.Embedding(len(cast) + 1, 16)
        self.key_lookup = tf.keras.layers.StringLookup(vocabulary=keywords, mask_token=None)
        self.key_emb = tf.keras.layers.Embedding(len(keywords) + 1, 16)
        self.mood_lookup = tf.keras.layers.StringLookup(vocabulary=moods, mask_token=None)
        self.mood_emb = tf.keras.layers.Embedding(len(moods) + 1, 32)

    def call(self, inputs):
        t_v = self.title_emb(self.title_lookup(inputs["title"]))
        c_v = tf.reduce_mean(self.cast_emb(self.cast_lookup(tf.strings.split(inputs["cast"], "|"))), axis=1)
        k_v = tf.reduce_mean(self.key_emb(self.key_lookup(tf.strings.split(inputs["keywords"], "|"))), axis=1)
        m_v = self.mood_emb(self.mood_lookup(inputs["mood"]))
        return tf.concat([t_v, c_v, k_v, m_v], axis=-1)


def _inputs(frame):
    return {
        "title": tf.constant(frame["title"].values.astype(str)),
        "cast": tf.constant(frame["cast"].values.astype(str)),
        "keywords": tf.constant(frame["keywords"].values.astype(str)),
        "mood": tf.constant(frame["mood"].values.astype(str)),
    }


class TFEngine(Recommender):
    """Same ranking/boost API as ``Recommender`` but scores with the TF model."""

    def __init__(self, df, weights_path: str = WEIGHTS_FILE):
        super().__init__(df)
        frame = self.df
        self.model = MovieModel(
            frame["title"].unique(),
            np.unique("|".join(frame["cast"].astype(str)).split("|")),
            np.unique("|".join(frame["keywords"].astype(str)).split("|")),
            frame["mood"].unique(),
        )
        self.model(_inputs(frame.iloc[:1]))  # build the layers
        if os.path.exists(weights_path):
            self.model.load_weights(weights_path, by_name=True, skip_mismatch=True)
        else:
            log.warning("%s not found: embeddings are untrained (random).", weights_path)
        # Computed once at start-up instead of on every request.
        self._all_vecs = self.model(_inputs(frame)).numpy()

    def base_scores(self, idx: int) -> np.ndarray:
        return self._all_vecs @ self._all_vecs[idx]
