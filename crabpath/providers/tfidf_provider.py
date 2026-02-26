"""Pure-stdlib TF-IDF embedding provider."""

from __future__ import annotations

import math
import re
from collections import Counter

from .base import EmbeddingProvider


class TfidfEmbeddingProvider(EmbeddingProvider):
    name = "tfidf-local"

    def __init__(self, dim: int = 1024) -> None:
        self._dim = dim
        self._idf: dict[str, float] | None = None
        self._corpus_size: int = 0

    def dimensions(self) -> int:
        return self._dim

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) >= 2]

    @classmethod
    def _features(cls, text: str) -> list[str]:
        tokens = cls._tokenize(text)
        bigrams = [f"{left}_{right}" for left, right in zip(tokens, tokens[1:])]
        return tokens + bigrams

    def fit(self, texts: list[str]) -> None:
        """Build IDF from a corpus. Call this once with all node contents."""
        document_frequency: dict[str, int] = {}
        self._corpus_size = len(texts)

        for text in texts:
            features = set(self._features(text))
            for feature in features:
                document_frequency[feature] = document_frequency.get(feature, 0) + 1

        idf: dict[str, float] = {}
        n = self._corpus_size or 0
        for feature, df in document_frequency.items():
            idf[feature] = math.log((n + 1) / (df + 1)) + 1
        self._idf = idf

    def _vectorize(self, text: str) -> list[float]:
        feature_counts = Counter(self._features(text))
        vector: list[float] = [0.0] * self._dim

        use_idf = self._idf is not None
        for feature, tf in feature_counts.items():
            if tf <= 0:
                continue

            term_weight = 1.0 + math.log(tf)
            if use_idf:
                term_weight *= self._idf.get(feature, 0.0)
            bucket = hash(feature) % self._dim
            sign = -1.0 if hash(feature) < 0 else 1.0
            vector[bucket] += sign * term_weight

        magnitude = math.sqrt(sum(value * value for value in vector))
        if magnitude == 0.0:
            return vector
        return [value / magnitude for value in vector]

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using TF-IDF with feature hashing.
        If fit() hasn't been called, uses raw TF (no IDF weighting)."""
        return [self._vectorize(text) for text in texts]
