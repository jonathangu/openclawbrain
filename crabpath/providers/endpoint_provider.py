"""OpenAI-compatible endpoint provider implementations."""

from __future__ import annotations

import json
from urllib.parse import urlparse
from typing import Any

from .base import EmbeddingProvider, RouterProvider


def _validate_endpoint_url(url: str, component: str) -> str:
    """Validate user-supplied endpoint URLs.

    We only allow HTTP(S) endpoints with explicit hosts. This prevents local and
    file-path based injection through env vars.
    """
    parsed = urlparse(url)
    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError(f"{component} URL must use http or https: {url!r}")
    if not parsed.netloc:
        raise ValueError(f"{component} URL must include a host: {url!r}")
    return url


class EndpointEmbeddingProvider(EmbeddingProvider):
    """Embedding provider that calls any OpenAI-compatible /v1/embeddings endpoint."""

    name = "endpoint"

    def __init__(self, url: str, token: str | None = None, model: str = "text-embedding-3-small") -> None:
        self._url = _validate_endpoint_url(url, "Embedding endpoint")
        self._token = token
        self._model = model
        self._dim: int | None = None

    def dimensions(self) -> int:
        return self._dim or 1536

    def embed(self, texts: list[str]) -> list[list[float]]:
        import requests

        headers = {"Content-Type": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        resp = requests.post(
            self._url,
            headers=headers,
            json={
                "model": self._model,
                "input": texts,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        vectors = [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]
        if vectors and self._dim is None:
            self._dim = len(vectors[0])
        return vectors


class EndpointRouterProvider(RouterProvider):
    """Router provider that calls any OpenAI-compatible /v1/chat/completions endpoint."""

    name = "endpoint"

    def __init__(self, url: str, token: str | None = None, model: str = "gpt-5-mini") -> None:
        self._url = _validate_endpoint_url(url, "Router endpoint")
        self._token = token
        self._model = model

    def route(self, query: str, candidates: list[dict], schema: dict | None = None) -> dict[str, Any]:
        import requests

        # Build the routing prompt (same as OpenAI provider)
        candidate_text = "\n".join(
            f'- {c.get("node_id", c.get("id", ""))}: {c.get("summary", c.get("content", "")[:100])}'
            for c in candidates
        )

        system = (
            'You are a memory router. Given a query and candidate document pointers, '
            'choose which to follow. Output JSON: '
            '{"target": "node_id", "confidence": 0.0-1.0, "rationale": "brief"}'
        )
        user = f"Query: {query}\n\nCandidates:\n{candidate_text}"

        headers = {"Content-Type": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        resp = requests.post(
            self._url,
            headers=headers,
            json={
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_tokens": 100,
                "temperature": 0.0,
            },
            timeout=15,
        )
        resp.raise_for_status()

        data = resp.json()
        content = data["choices"][0]["message"]["content"]

        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            import re

            match = re.search(r"\{[^}]+\}", content)
            if match:
                result = json.loads(match.group())
            else:
                return {"target": "", "confidence": 0.0, "rationale": "Failed to parse LLM response", "provider": self.name}

        result["provider"] = self.name
        return result
