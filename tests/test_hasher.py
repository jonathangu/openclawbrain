from __future__ import annotations

import math

from crabpath.hasher import HashEmbedder


def test_hash_embedder_deterministic() -> None:
    embedder = HashEmbedder()
    first = embedder.embed("deterministic text")
    second = embedder.embed("deterministic text")
    assert first == second


def test_hash_embedder_different_texts() -> None:
    embedder = HashEmbedder()
    first = embedder.embed("first text")
    second = embedder.embed("completely different text")
    assert first != second


def test_hash_embedder_dimension() -> None:
    dim = 128
    embedder = HashEmbedder(dim=dim)
    assert len(embedder.embed("some text")) == dim


def test_hash_embedder_normalized() -> None:
    embedder = HashEmbedder()
    vector = embedder.embed("norm test")
    norm = math.sqrt(sum(value * value for value in vector))
    assert norm == 1.0


def test_hash_embedder_camel_case() -> None:
    embedder = HashEmbedder()
    assert embedder._tokenize("camelCase") == ["camel", "case"]


def test_hash_embedder_batch() -> None:
    embedder = HashEmbedder()
    vectors = embedder.embed_batch([("a", "first"), ("b", "second")])
    assert list(vectors.keys()) == ["a", "b"]
