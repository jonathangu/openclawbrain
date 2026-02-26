"""Gemini embedding and routing provider implementations."""

from __future__ import annotations

import os
from typing import Any

from ..embeddings import gemini_embed
from ..router import Router, RouterConfig
from .base import EmbeddingProvider, RouterProvider


def _ensure_gemini_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is required for Gemini provider.")
    return api_key


def _candidate_pairs(
    candidates: list[Any],
) -> tuple[list[tuple[str, float]], str, str | None]:
    candidate_pairs: list[tuple[str, float]] = []
    current_node_id = ""
    context_summary = None

    for candidate in candidates:
        if isinstance(candidate, dict):
            candidate_id = str(candidate.get("node_id") or candidate.get("id") or "")
            weight = candidate.get("weight", 0.0)
            context_summary = context_summary or candidate.get("summary")
            current_node_id = (
                current_node_id
                or str(candidate.get("current_node_id") or candidate.get("current_node", ""))
            )
        elif isinstance(candidate, (tuple, list)) and candidate:
            candidate_id = str(candidate[0])
            weight = candidate[1] if len(candidate) > 1 else 0.0
        else:
            candidate_id = str(candidate)
            weight = 0.0

        try:
            candidate_weight = float(weight)
        except (TypeError, ValueError):
            candidate_weight = 0.0
        candidate_pairs.append((candidate_id, candidate_weight))

    return candidate_pairs, current_node_id, str(context_summary or "")


class GeminiEmbeddingProvider(EmbeddingProvider):
    name = "gemini"

    def __init__(self, model: str = "text-embedding-004") -> None:
        _ensure_gemini_key()
        self.model = model
        self._embed_fn = gemini_embed(model=model)
        self._dimensions = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._embed_fn(texts)
        if self._dimensions == 0 and vectors:
            self._dimensions = len(vectors[0])
        return vectors

    def dimensions(self) -> int:
        if self._dimensions > 0:
            return self._dimensions

        vectors = self.embed(["__crabpath_probe__"])
        self._dimensions = len(vectors[0])
        return self._dimensions


class GeminiRouterProvider(RouterProvider):
    name = "gemini"

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        timeout_s: float = 8.0,
        temperature: float | None = None,
        max_retries: int = 2,
    ) -> None:
        api_key = _ensure_gemini_key()

        from google import generativeai as genai

        genai.configure(api_key=api_key)
        self.model = model
        self.client = genai.GenerativeModel(model)
        self.router = Router(
            config=RouterConfig(
                model=model,
                timeout_s=timeout_s,
                temperature=temperature,
                max_retries=max_retries,
                fallback_behavior="heuristic",
            ),
            client=self._call_chat,
        )

    def _call_chat(self, messages: list[dict[str, str]]) -> str:
        prompt = "\n".join(
            str(message.get("content", "")) for message in messages if message.get("content")
        ).strip()

        if not prompt:
            return "{}"

        response = self.client.generate_content(prompt)
        content = response.text if hasattr(response, "text") else None
        if not content:
            raise RuntimeError("Gemini returned empty chat output.")
        return str(content)

    def route(
        self,
        query: str,
        candidates: list[dict],
        schema: dict | None = None,
    ) -> dict[str, Any]:
        schema = schema or {}
        normalized, current_node_id, context_summary = _candidate_pairs(candidates)
        decision = self.router.decide_next(
            query=query,
            current_node_id=str(schema.get("current_node_id", current_node_id)),
            candidate_nodes=normalized[:max(1, int(schema.get("branch_beam", 5)))],
            context={
                "node_summary": schema.get("node_summary", context_summary),
                "current_node_summary": schema.get("current_node_summary", context_summary),
            },
            tier=str(schema.get("tier", "habitual")),
        )

        return {
            "target": decision.chosen_target,
            "confidence": decision.confidence,
            "rationale": decision.rationale,
            "tier": decision.tier,
            "alternatives": decision.alternatives,
            "provider": self.name,
        }
