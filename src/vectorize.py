"""
vectorize.py — Text vectorization utilities.

Defaults to character TF-IDF (n-grams 1..4), but can optionally build a
hybrid representation (char + word TF-IDF concatenated).

Usage:
    vec = CharTfidf(ngram_range=(1,4), max_features=5000, min_df=2)
    X_train = vec.fit_transform(texts_train)
    X_val   = vec.transform(texts_val)

    # Hybrid:
    vec = HybridTfidf(
        char_params=dict(ngram_range=(1,4), max_features=5000, min_df=2),
        word_params=dict(ngram_range=(1,2), max_features=20000, min_df=2),
    )
    X_train = vec.fit_transform(texts_train)
    X_val   = vec.transform(texts_val)
"""

from __future__ import annotations
from typing import Iterable, Optional, Tuple, Dict, Any
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, hstack


class CharTfidf:
    """
    Character-level TF-IDF wrapper returning dense float32 arrays.
    """

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (1, 4),
        max_features: Optional[int] = 5000,
        min_df: int = 2,
        lowercase: bool = True,
    ):
        self.vec = TfidfVectorizer(
            analyzer="char",
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=min_df,
            lowercase=lowercase,
        )

    def fit(self, texts: Iterable[str]) -> "CharTfidf":
        self.vec.fit(texts)
        return self

    def transform(self, texts: Iterable[str]) -> np.ndarray:
        return self.vec.transform(texts).toarray().astype("float32")

    def fit_transform(self, texts: Iterable[str]) -> np.ndarray:
        return self.vec.fit_transform(texts).toarray().astype("float32")

    # Utilities
    def get_feature_names(self) -> list[str]:
        return list(self.vec.get_feature_names_out())

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.vec, f)

    @staticmethod
    def load(path: str) -> "CharTfidf":
        with open(path, "rb") as f:
            vec = pickle.load(f)
        obj = CharTfidf()
        obj.vec = vec
        return obj


class HybridTfidf:
    """
    Concatenate char- and word-level TF-IDF features (sparse under the hood,
    returned as dense float32 arrays).

    Parameters:
        char_params: dict passed to TfidfVectorizer(analyzer="char", ...)
        word_params: dict passed to TfidfVectorizer(analyzer="word", ...)
        to_dense: return dense arrays (True) or keep sparse (False)
    """

    def __init__(
        self,
        char_params: Optional[Dict[str, Any]] = None,
        word_params: Optional[Dict[str, Any]] = None,
        to_dense: bool = True,
    ):
        char_defaults = dict(analyzer="char", ngram_range=(1, 4), max_features=5000, min_df=2, lowercase=True)
        word_defaults = dict(analyzer="word", ngram_range=(1, 2), max_features=20000, min_df=2, lowercase=True)

        self.char_vec = TfidfVectorizer(**{**char_defaults, **(char_params or {})})
        self.word_vec = TfidfVectorizer(**{**word_defaults, **(word_params or {})})
        self.to_dense = to_dense

    def fit(self, texts: Iterable[str]) -> "HybridTfidf":
        self.char_vec.fit(texts)
        self.word_vec.fit(texts)
        return self

    def transform(self, texts: Iterable[str]) -> np.ndarray | csr_matrix:
        Xc = self.char_vec.transform(texts)
        Xw = self.word_vec.transform(texts)
        X = hstack([Xc, Xw], format="csr")
        if self.to_dense:
            return X.toarray().astype("float32")
        return X

    def fit_transform(self, texts: Iterable[str]) -> np.ndarray | csr_matrix:
        Xc = self.char_vec.fit_transform(texts)
        Xw = self.word_vec.fit_transform(texts)
        X = hstack([Xc, Xw], format="csr")
        if self.to_dense:
            return X.toarray().astype("float32")
        return X

    # Utilities
    def get_feature_names(self) -> list[str]:
        # Prefix to avoid collisions/ambiguity
        names_c = [f"c:{n}" for n in self.char_vec.get_feature_names_out()]
        names_w = [f"w:{n}" for n in self.word_vec.get_feature_names_out()]
        return names_c + names_w

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "char": self.char_vec,
                    "word": self.word_vec,
                    "to_dense": self.to_dense,
                },
                f,
            )

    @staticmethod
    def load(path: str) -> "HybridTfidf":
        with open(path, "rb") as f:
            obj = HybridTfidf()
            state = pickle.load(f)
        obj.char_vec = state["char"]
        obj.word_vec = state["word"]
        obj.to_dense = state.get("to_dense", True)
        return obj
