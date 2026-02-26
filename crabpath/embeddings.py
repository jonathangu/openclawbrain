"""
CrabPath Embeddings — Semantic seeding via vector similarity.

Embeds node content and finds the best seeds for a query
using cosine similarity. Supports any embedding function.

Ships with an OpenAI adapter. Bring your own for local models.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from .graph import Graph

OPENAI_EMBEDDING_MAX_BATCH_SIZE = 2048


@dataclass
class EmbeddingIndex:
    """A vector index over graph nodes for semantic seeding.

    Build once, save to disk, reload on startup.
    Reindex when nodes change.
    """

    vectors: dict[str, list[float]] = field(default_factory=dict)  # node_id -> embedding
    dim: int = 0

    @staticmethod
    def _validate_vector_batch(
        vectors: list[list[float]],
        expected_batch_size: int,
        *,
        context: str,
        expected_dim: int | None = None,
    ) -> list[list[float]]:
        if not isinstance(vectors, list):
            raise TypeError(
                f"embedding function must return a list for {context}; got {type(vectors).__name__}"
            )
        if len(vectors) != expected_batch_size:
            raise ValueError(
                f"embedding function returned {len(vectors)} vectors for {context}, "
                f"expected {expected_batch_size}"
            )
        observed_dim = expected_dim
        for idx, vector in enumerate(vectors):
            if not isinstance(vector, list):
                raise TypeError(
                    f"embedding vector at position {idx} for {context} must be a list, "
                    f"got {type(vector).__name__}"
                )
            if observed_dim is None:
                observed_dim = len(vector)
            elif len(vector) != observed_dim:
                raise ValueError(
                    f"embedding vector at position {idx} for {context} has "
                    f"dimension {len(vector)} but expected {observed_dim}"
                )
            for value in vector:
                if not isinstance(value, (int, float)):
                    raise TypeError(
                        f"embedding vector element at position {idx} for {context} must be numeric, "
                        f"got {type(value).__name__}"
                    )
        return vectors

    @staticmethod
    def _single_vector(
        vectors: list[list[float]],
        *,
        context: str,
        expected_dim: int | None = None,
    ) -> list[float]:
        if not vectors:
            raise ValueError(f"embedding function returned no vectors for {context}")
        validated = EmbeddingIndex._validate_vector_batch(
            vectors,
            1,
            context=context,
            expected_dim=expected_dim,
        )
        return validated[0]

    def build(
        self,
        graph: Graph,
        embed_fn: Callable[[list[str]], list[list[float]]],
        batch_size: int = 100,
    ) -> None:
        """Embed all node content in the graph.

        Args:
            graph: The CrabPath graph.
            embed_fn: Function that takes a list of strings and returns
                a list of embedding vectors. Signature: (texts) -> vectors.
            batch_size: How many texts to embed per API call.
        """
        nodes = graph.nodes()
        if not nodes:
            return

        # Prepare texts: id + content + error_class for richer matching
        texts = []
        ids = []
        for node in nodes:
            node_text = node.summary if node.summary else node.content
            parts = [node_text]
            ec = node.metadata.get("error_class", "")
            if ec:
                parts.append(ec)
            texts.append(" ".join(parts))
            ids.append(node.id)

        # Batch embed
        all_vectors = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            vectors = self._validate_vector_batch(
                embed_fn(batch),
                expected_batch_size=len(batch),
                context=f"build batch {i}-{i + len(batch)}",
                expected_dim=self.dim or None,
            )
            all_vectors.extend(vectors)

        # Store
        self.vectors = dict(zip(ids, all_vectors))
        if all_vectors:
            self.dim = len(all_vectors[0])

    def seed(
        self,
        query: str,
        embed_fn: Callable[[list[str]], list[list[float]]],
        top_k: int = 10,
        min_score: float = 0.25,
        energy: float = 2.0,
    ) -> dict[str, float]:
        """Find the best seed nodes for a query.

        Returns {node_id: energy} for the top-k most similar nodes
        above min_score. Energy is scaled so the best match gets
        the full `energy` value and others are proportional.

        Args:
            query: The task or question text.
            embed_fn: Embedding function.
            top_k: Maximum seeds to return.
            min_score: Minimum cosine similarity to include.
            energy: Energy for the top-scoring seed. Others scale proportionally.
        """
        if not self.vectors:
            return {}

        q_vec = self._single_vector(
            embed_fn([query]),
            context=f"query seed: {query[:80]!r}",
            expected_dim=self.dim or None,
        )

        # Cosine similarity against all nodes
        scores = {}
        for node_id, node_vec in self.vectors.items():
            if self.dim and len(node_vec) != self.dim:
                raise ValueError(
                    f"stored embedding dimension mismatch for node {node_id}: "
                    f"expected {self.dim}, got {len(node_vec)}"
                )
            sim = _cosine(q_vec, node_vec)
            if sim >= min_score:
                scores[node_id] = sim

        if not scores:
            return {}

        # Top-k, scaled to energy
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        max_sim = ranked[0][1] if ranked else 1.0
        return {nid: (sim / max_sim) * energy for nid, sim in ranked}

    def upsert(
        self,
        node_id: str,
        content_or_vector: list[float] | str,
        embed_fn: Optional[Callable[[list[str]], list[list[float]]]] = None,
        metadata: Any | None = None,
    ) -> None:
        """Add or replace one vector in the index."""
        _ = metadata
        if isinstance(content_or_vector, list):
            vector = self._validate_vector_batch(
                [content_or_vector],
                expected_batch_size=1,
                context=f"upsert node_id={node_id}",
                expected_dim=self.dim or None,
            )[0]
        elif isinstance(content_or_vector, tuple):
            vector = self._validate_vector_batch(
                [list(content_or_vector)],
                expected_batch_size=1,
                context=f"upsert node_id={node_id}",
                expected_dim=self.dim or None,
            )[0]
        else:
            if embed_fn is None:
                raise TypeError("embed_fn is required when upserting from content")
            vector = self._single_vector(
                embed_fn([str(content_or_vector)]),
                context=f"upsert node_id={node_id}",
                expected_dim=self.dim or None,
            )
        self.vectors[node_id] = vector
        if vector:
            self.dim = len(vector)

    def remove(self, node_id: str) -> None:
        """Remove one vector from the index."""
        self.vectors.pop(node_id, None)

    def raw_scores(
        self,
        query: str,
        embed_fn: Callable[[list[str]], list[list[float]]],
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Return raw cosine scores for all nodes, sorted descending."""
        if not self.vectors:
            return []

        q_vec = self._single_vector(
            embed_fn([query]),
            context=f"query raw_scores: {query[:80]!r}",
            expected_dim=self.dim or None,
        )
        scores: list[tuple[str, float]] = []
        for node_id, node_vec in self.vectors.items():
            if self.dim and len(node_vec) != self.dim:
                raise ValueError(
                    f"stored embedding dimension mismatch for node {node_id}: "
                    f"expected {self.dim}, got {len(node_vec)}"
                )
            scores.append((node_id, _cosine(q_vec, node_vec)))

        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:top_k]

    def save(self, path: str) -> None:
        """Save index to JSON."""
        target = Path(path)
        data = {"dim": self.dim, "vectors": self.vectors}

        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", delete=False, dir=str(target.parent), suffix=".tmp"
        ) as f:
            json.dump(data, f)
            temp_path = Path(f.name)

        os.replace(temp_path, target)

    @classmethod
    def load(cls, path: str) -> EmbeddingIndex:
        """Load index from JSON."""
        with open(path) as f:
            data = json.load(f)
        idx = cls()
        idx.dim = data.get("dim", 0)
        idx.vectors = data.get("vectors", {})
        if idx.dim <= 0 and isinstance(idx.vectors, dict):
            inferred_dims = {
                len(value) for value in idx.vectors.values() if isinstance(value, list)
            }
            if len(inferred_dims) == 1:
                idx.dim = next(iter(inferred_dims))
        return idx


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    if len(a) != len(b):
        raise ValueError("cosine vectors must have matching dimensions")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---- OpenAI adapter ----


def openai_embed(
    model: str = "text-embedding-3-small",
) -> Callable[[list[str]], list[list[float]]]:
    """Returns an embed_fn that uses the OpenAI API.

    Usage:
        embed = openai_embed()
        index.build(graph, embed)
        seeds = index.seed("my query", embed)
    """
    from openai import OpenAI

    client = OpenAI()

    def _status_code(exc: Exception) -> int | None:
        status = getattr(exc, "status_code", None)
        if isinstance(status, int):
            return status
        if isinstance(status, str) and status.isdigit():
            return int(status)

        response = getattr(exc, "response", None)
        response_status = getattr(response, "status_code", None)
        if isinstance(response_status, int):
            return response_status
        return None

    def _embed_batch(batch: list[str]) -> list[list[float]]:
        try:
            response = client.embeddings.create(model=model, input=batch)
        except Exception as exc:
            status = _status_code(exc)
            if status == 429:
                raise ValueError(
                    "OpenAI embeddings rate-limited (HTTP 429); retry to recover."
                ) from exc
            if status is not None and status >= 500:
                raise ValueError(f"OpenAI embeddings HTTP {status}") from exc
            raise ValueError(f"OpenAI embeddings request failed: {exc}") from exc

        data = getattr(response, "data", None)
        if not isinstance(data, list):
            raise ValueError("OpenAI embeddings response missing data payload.")
        return [list(map(float, item.embedding)) for item in data]


    def embed_fn(texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        if not isinstance(texts, list):
            raise TypeError("texts must be a list of strings")

        for i in range(0, len(texts), OPENAI_EMBEDDING_MAX_BATCH_SIZE):
            batch = texts[i : i + OPENAI_EMBEDDING_MAX_BATCH_SIZE]
            vectors.extend(_embed_batch(batch))

        return vectors

    return embed_fn


def gemini_embed(
    model: str = "text-embedding-004",
) -> Callable[[list[str]], list[list[float]]]:
    """Returns an embed_fn that uses Google Gemini embeddings."""
    from google import generativeai as genai

    api_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("Gemini API key missing. Set GEMINI_API_KEY or GOOGLE_API_KEY.")

    genai.configure(api_key=api_key)

    model_name = model if model.startswith("models/") else f"models/{model}"

    def embed_fn(texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            try:
                result = genai.embed_content(
                    model=model_name,
                    content=text,
                    task_type="retrieval_document",
                )
            except Exception as exc:
                status = getattr(exc, "status", None)
                if str(status) in {"500", "503"}:
                    raise ValueError(
                        f"Gemini embeddings service returned HTTP {status}"
                    ) from exc
                raise ValueError(f"Gemini embeddings request failed: {exc}") from exc

            if not isinstance(result, dict):
                raise ValueError(f"Gemini embed_content returned non-dict result: {result!r}")

            embedding = result.get("embedding")
            if not isinstance(embedding, (tuple, list)):
                raise ValueError(f"Gemini response missing embedding: {result!r}")
            vectors.append([float(x) for x in embedding])
        return vectors

    return embed_fn


def cohere_embed(
    model: str = "embed-v4",
) -> Callable[[list[str]], list[list[float]]]:
    """Returns an embed_fn that uses Cohere embeddings."""
    import cohere

    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise RuntimeError("Cohere API key missing. Set COHERE_API_KEY.")

    client = cohere.Client(api_key=api_key)

    def embed_fn(texts: list[str]) -> list[list[float]]:
        response = client.embed(texts=texts, model=model)
        return [[float(x) for x in row] for row in response.embeddings]

    return embed_fn


def ollama_embed(
    model: str = "nomic-embed-text",
    base_url: str = "http://localhost:11434",
) -> Callable[[list[str]], list[list[float]]]:
    """Returns an embed_fn that uses a local Ollama endpoint."""
    import requests

    endpoint = f"{base_url.rstrip('/')}/api/embeddings"

    # Verify service and model existence up front.
    try:
        resp = requests.post(
            endpoint,
            json={"model": model, "prompt": "probe"},
            timeout=1.5,
        )
    except Exception as exc:
        raise ValueError(
            f"Ollama unavailable while verifying model '{model}' at {base_url}: {exc}"
        ) from exc
    if resp.status_code != 200:
        raise ValueError(
            f"Ollama returned HTTP {resp.status_code} for model '{model}' at {base_url}"
        )

    def embed_fn(texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            payload = {"model": model, "prompt": text}
            try:
                response = requests.post(endpoint, json=payload, timeout=30)
                response.raise_for_status()
            except requests.RequestException as exc:
                raise ValueError(f"Ollama request failed while embedding: {exc}") from exc
            except Exception as exc:
                raise ValueError(f"Ollama error while embedding: {exc}") from exc

            try:
                result = response.json()
            except ValueError as exc:
                raise ValueError(f"Ollama response was not valid JSON: {exc}") from exc

            embedding = result.get("embedding")
            if not isinstance(embedding, (tuple, list)):
                raise ValueError(f"Ollama response missing embedding: {result}")
            vectors.append([float(x) for x in embedding])
        return vectors

    return embed_fn


def auto_embed(
    openai_model: str = "text-embedding-3-small",
    gemini_model: str = "text-embedding-004",
    ollama_model: str = "nomic-embed-text",
    ollama_base_url: str = "http://localhost:11434",
) -> Callable[[list[str]], list[list[float]]]:
    """Return the first available embedding provider adapter."""
    from .providers import get_embedding_provider

    provider = get_embedding_provider()
    if provider is None:
        raise RuntimeError(
            "No embedding provider found. Tried: OPENAI, GEMINI, and Ollama. "
            "Set OPENAI_API_KEY (pip install crabpath[openai]) or GEMINI/GOOGLE_API_KEY or run Ollama locally."
        )

    if provider.__class__.__name__ == "OpenAIEmbeddingProvider" and openai_model:
        current_model = getattr(provider, "model", None)
        if current_model != openai_model:
            from .providers.openai_provider import OpenAIEmbeddingProvider

            return OpenAIEmbeddingProvider(model=openai_model).embed

    if provider.__class__.__name__ == "GeminiEmbeddingProvider" and gemini_model:
        current_model = getattr(provider, "model", None)
        if current_model != gemini_model:
            from .providers.gemini_provider import GeminiEmbeddingProvider

            return GeminiEmbeddingProvider(model=gemini_model).embed

    if provider.__class__.__name__ == "OllamaEmbeddingProvider":
        current_model = getattr(provider, "model", None)
        current_base_url = getattr(provider, "base_url", None)
        if current_model != ollama_model or current_base_url != ollama_base_url:
            from .providers.ollama_provider import OllamaEmbeddingProvider

            return OllamaEmbeddingProvider(
                model=ollama_model,
                base_url=ollama_base_url,
            ).embed

    return provider.embed
