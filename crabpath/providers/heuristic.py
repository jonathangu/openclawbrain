"""Heuristic, no-LLM router provider."""

from __future__ import annotations

import math
import re
from typing import Any

from .base import RouterProvider


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9']+", text.lower()) if token}


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class HeuristicRouter(RouterProvider):
    name = "heuristic"

    def __init__(self, top_k: int = 5) -> None:
        if top_k <= 0:
            top_k = 1
        self.top_k = top_k

    def _normalize_candidate(self, candidate: Any) -> tuple[str, float, str | None]:
        if isinstance(candidate, dict):
            candidate_id = candidate.get("node_id") or candidate.get("id") or ""
            if not candidate_id:
                candidate_id = str(candidate.get("node", ""))
            weight = _to_float(candidate.get("weight"))
            summary = candidate.get("summary")
            return str(candidate_id), float(weight) if weight is not None else 0.0, (
                None if summary is None else str(summary)
            )
        if isinstance(candidate, (tuple, list)):
            if len(candidate) >= 2:
                return str(candidate[0]), _to_float(candidate[1]) or 0.0, (
                    str(candidate[2]) if len(candidate) >= 3 and candidate[2] is not None else None
                )
            return str(candidate[0]), 0.0, None
        return str(candidate), 0.0, None

    def route(self, query: str, candidates: list[dict], schema: dict | None = None) -> dict[str, Any]:
        if not query:
            query_tokens = set()
        else:
            query_tokens = _tokenize(query)

        schema = schema or {}
        edge_routing = bool(schema.get("edge_routing"))
        ranked: list[tuple[str, float]] = []

        for candidate in candidates:
            node_id, weight, summary = self._normalize_candidate(candidate)
            if not node_id:
                continue

            if edge_routing and isinstance(candidate, dict):
                raw_embedding_similarity = candidate.get("embedding_similarity")
                embedding_similarity = _to_float(raw_embedding_similarity)
            else:
                embedding_similarity = None

            if embedding_similarity is None:
                node_text = " ".join(
                    part
                    for part in (str(node_id), str(summary) if summary else "", str(candidate.get("content", "") if isinstance(candidate, dict) else ""))
                    if part
                )
                candidate_tokens = _tokenize(node_text)
                overlap = len(query_tokens.intersection(candidate_tokens))
                denominator = len(query_tokens) if query_tokens else 1
                score = overlap / denominator if denominator else 0.0
            else:
                score = 0.6 * embedding_similarity + 0.4 * max(0.0, weight)

            if embedding_similarity is None and edge_routing:
                score = 0.0 + max(0.0, weight)

            ranked.append((node_id, score))

        if not ranked:
            return {
                "target": "",
                "confidence": 0.0,
                "rationale": "No candidates were available for heuristic routing.",
                "tier": schema.get("tier", "heuristic"),
                "provider": self.name,
                "alternatives": [],
                "raw": {
                    "provider": self.name,
                    "target": "",
                    "confidence": 0.0,
                    "rationale": "No candidates were available for heuristic routing.",
                    "tier": schema.get("tier", "heuristic"),
                    "alternatives": [],
                },
            }

        ranked.sort(key=lambda item: item[1], reverse=True)
        max_k = max(1, self.top_k)
        top_candidates = ranked[:max_k]
        target, top_score = top_candidates[0]
        max_score = ranked[0][1] if ranked else 0.0
        confidence = max(0.0, min(1.0, float(max_score)))

        return {
            "target": str(target),
            "confidence": confidence,
            "rationale": (
                "Heuristic routing selected top weighted/token-overlap match."
            ),
            "tier": schema.get("tier", "heuristic"),
            "provider": self.name,
            "alternatives": [[node_id, score] for node_id, score in top_candidates],
            "candidates": ranked,
            "raw": {
                "provider": self.name,
                "target": str(target),
                "confidence": confidence,
                "rationale": "Heuristic routing selected top weighted/token-overlap match.",
                "tier": schema.get("tier", "heuristic"),
                "alternatives": [[node_id, score] for node_id, score in top_candidates],
            },
        }
