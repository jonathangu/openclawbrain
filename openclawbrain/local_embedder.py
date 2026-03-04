"""Local ONNX embedder wrapper with normalized vectors."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np

DEFAULT_LOCAL_MODEL = "BAAI/bge-large-en-v1.5"
DEFAULT_LOCAL_MODEL_TAG = "bge-large-en-v1.5"

_LOCAL_MODEL_TAG_MAP = {
    "bge-large-en-v1.5": "BAAI/bge-large-en-v1.5",
    "bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
}

FASTEMBED_CACHE_ENV = "FASTEMBED_CACHE_PATH"
DEFAULT_FASTEMBED_CACHE_DIR = Path.home() / ".cache" / "fastembed"

_MODEL_CACHE: dict[tuple[str, str | None], object] = {}


def resolve_fastembed_cache_dir(cache_dir: str | None = None) -> str | None:
    """Resolve fastembed cache dir with env override."""
    if cache_dir and str(cache_dir).strip():
        return str(Path(cache_dir).expanduser())
    env_value = os.environ.get(FASTEMBED_CACHE_ENV)
    if env_value and env_value.strip():
        return str(Path(env_value).expanduser())
    return str(DEFAULT_FASTEMBED_CACHE_DIR)


def _stub_dim_for_model(model_name: str) -> int:
    lower = model_name.lower()
    if "small" in lower:
        return 384
    if "large" in lower:
        return 1024
    return 768


class _StubTextEmbedding:
    def __init__(self, model_name: str) -> None:
        from .hasher import HashEmbedder

        self.model_name = model_name
        self._embedder = HashEmbedder(dim=_stub_dim_for_model(model_name))

    def embed(self, texts: list[str]):
        for text in texts:
            yield self._embedder.embed(text)


def _normalize(vec: object) -> list[float]:
    arr = np.asarray(vec, dtype=float)
    if arr.ndim != 1:
        raise ValueError("embedding output must be a 1D vector")
    norm = float(np.linalg.norm(arr))
    if norm > 0:
        arr = arr / norm
    return [float(v) for v in arr.tolist()]


def local_tag_from_model(model_name: str | None) -> str:
    """Derive the local tag name used in metadata from a model name."""
    if not model_name:
        return DEFAULT_LOCAL_MODEL_TAG
    return model_name.rsplit("/", 1)[-1]


def local_embedder_name(model_name: str | None) -> str:
    """Create the stable embedder_name string for local embeddings."""
    return f"local:{local_tag_from_model(model_name)}"


def resolve_local_model(
    meta: dict[str, object] | None = None,
    *,
    embed_model: str | None = None,
    default_model: str = DEFAULT_LOCAL_MODEL,
) -> str:
    """Resolve the full local model name from metadata and overrides."""
    if embed_model:
        cleaned = str(embed_model).strip()
        if cleaned:
            return cleaned

    if meta:
        embedder_model = meta.get("embedder_model")
        if isinstance(embedder_model, str) and embedder_model.strip():
            return embedder_model.strip()
        embedder_name = meta.get("embedder_name") or meta.get("embedder")
        if isinstance(embedder_name, str) and embedder_name.startswith("local:"):
            local_tag = embedder_name.split(":", 1)[1].strip()
            if not local_tag:
                return default_model
            if "/" in local_tag:
                return local_tag
            mapped = _LOCAL_MODEL_TAG_MAP.get(local_tag)
            if mapped:
                return mapped
            return local_tag

    return default_model


@dataclass
class LocalEmbedder:
    """Fast local embedder powered by `fastembed`."""

    model_name: str = DEFAULT_LOCAL_MODEL
    cache_dir: str | None = None
    _dim: int | None = None

    @property
    def name(self) -> str:
        return local_embedder_name(self.model_name)

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = len(self.embed(""))
        return self._dim

    def _model(self):
        cache_dir = resolve_fastembed_cache_dir(self.cache_dir)
        cache_key = (self.model_name, cache_dir)
        cached = _MODEL_CACHE.get(cache_key)
        if cached is not None:
            return cached
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            if os.environ.get("OPENCLAWBRAIN_FASTEMBED_STUB"):
                model = _StubTextEmbedding(self.model_name)
                _MODEL_CACHE[cache_key] = model
                return model
            raise ImportError("fastembed is required for local embeddings") from exc
        model = TextEmbedding(model_name=self.model_name, cache_dir=cache_dir)
        _MODEL_CACHE[cache_key] = model
        return model

    def embed(self, text: str) -> list[float]:
        vectors = list(self._model().embed([text]))
        if not vectors:
            raise RuntimeError("local embedder returned no vectors")
        normalized = _normalize(vectors[0])
        self._dim = len(normalized)
        return normalized

    def embed_batch(self, texts: list[tuple[str, str]]) -> dict[str, list[float]]:
        if not texts:
            return {}
        ids = [node_id for node_id, _ in texts]
        contents = [content for _, content in texts]
        vectors = list(self._model().embed(contents))
        if len(vectors) != len(ids):
            raise RuntimeError("local embedder batch size mismatch")
        normalized_vectors = [_normalize(vector) for vector in vectors]
        if normalized_vectors:
            self._dim = len(normalized_vectors[0])
        return dict(zip(ids, normalized_vectors))
