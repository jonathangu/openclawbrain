"""OpenAI embedding and routing provider implementations."""

from __future__ import annotations

import os
from typing import Any

from ..embeddings import openai_embed
from ..router import Router, RouterConfig
from .base import EmbeddingProvider, RouterProvider


def _ensure_openai_key() -> None:
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for OpenAI provider.")


def _candidate_pairs(
    candidates: list[Any],
) -> tuple[list[tuple[str, float]], str, str | None]:
    candidate_pairs: list[tuple[str, float]] = []
    context_summary = None
    current_node_id = ""

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


def _safe_branch_beam(value: Any) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return 5


class OpenAIEmbeddingProvider(EmbeddingProvider):
    name = "openai"

    def __init__(self, model: str = "text-embedding-3-small") -> None:
        _ensure_openai_key()
        self.model = model
        self._embed_fn = openai_embed(model=model)
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


class OpenAIRouterProvider(RouterProvider):
    name = "openai"

    def __init__(
        self,
        model: str = "gpt-5-mini",
        timeout_s: float = 8.0,
        temperature: float | None = None,
        max_retries: int = 2,
    ) -> None:
        _ensure_openai_key()
        from openai import OpenAI

        self.client = OpenAI()
        self.config = RouterConfig(
            model=model,
            timeout_s=timeout_s,
            temperature=temperature,
            max_retries=max_retries,
            fallback_behavior="heuristic",
        )
        self.router = Router(config=self.config, client=self._call_chat)

    def _call_chat(self, messages: list[dict[str, str]]) -> str:
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=list(messages),
            timeout=self.config.timeout_s,
            reasoning_effort="minimal",
            **({} if self.config.temperature is None else {"temperature": self.config.temperature}),
        )

        if not response.choices:
            raise RuntimeError("No choices returned by OpenAI chat API.")
        choice = response.choices[0]
        content = choice.message.content if hasattr(choice, "message") else None
        if not content:
            raise RuntimeError("Empty content from OpenAI chat API.")
        return str(content)

    def route(
        self,
        query: str,
        candidates: list[dict],
        schema: dict | None = None,
    ) -> dict[str, Any]:
        schema = schema or {}
        normalized, current_node_id, context_summary = _candidate_pairs(candidates)
        branch_beam = _safe_branch_beam(schema.get("branch_beam", 5))
        decision = self.router.decide_next(
            query=query,
            current_node_id=str(schema.get("current_node_id", current_node_id)),
            candidate_nodes=normalized[:branch_beam],
            context={
                "node_summary": schema.get("node_summary", context_summary),
                "current_node_summary": schema.get("current_node_summary", context_summary),
            },
            tier=str(schema.get("tier", "habitual")),
        )
        raw = dict(decision.raw)
        raw["provider"] = self.name
        return {
            "target": decision.chosen_target,
            "confidence": decision.confidence,
            "rationale": decision.rationale,
            "tier": decision.tier,
            "alternatives": decision.alternatives,
            "provider": self.name,
            "raw": raw,
        }
