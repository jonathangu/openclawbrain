"""Industry-standard evaluation baselines for OpenClawBrain."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
import time
from typing import Any, Callable, Protocol

from openclawbrain.graph import Graph
from openclawbrain.prompt_context import build_prompt_context_ranked_with_stats

_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


class Reranker(Protocol):
    """Minimal reranker interface."""

    name: str

    def rerank(self, query: str, candidates: list[tuple[str, str, float]]) -> list[tuple[str, float]]:
        """Return [(node_id, score)] sorted by descending score."""


class _BM25Reranker:
    """BM25 reranker using rank_bm25."""

    name = "bm25"

    def __init__(self) -> None:
        from rank_bm25 import BM25Okapi

        self._bm25_cls = BM25Okapi

    def rerank(self, query: str, candidates: list[tuple[str, str, float]]) -> list[tuple[str, float]]:
        if not candidates:
            return []
        tokenized_docs = [_tokenize_text(content) for _node_id, content, _score in candidates]
        bm25 = self._bm25_cls(tokenized_docs)
        query_tokens = _tokenize_text(query)
        scores = bm25.get_scores(query_tokens)
        reranked = [(node_id, float(score)) for (node_id, _content, _score), score in zip(candidates, scores)]
        reranked.sort(key=lambda item: item[1], reverse=True)
        return reranked


def try_load_bm25_reranker() -> tuple[Reranker | None, str | None]:
    """Return (reranker, reason_if_missing)."""
    try:
        reranker = _BM25Reranker()
    except ImportError:
        return None, "rank_bm25 not installed; install openclawbrain[reranker]"
    return reranker, None


def _tokenize_text(text: str) -> list[str]:
    return [match.group(0).lower() for match in _TOKEN_RE.finditer(text or "")]


def _approx_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, int(len(text) / 4))


def _normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    low = min(scores)
    high = max(scores)
    if math.isclose(low, high):
        return [1.0 for _ in scores]
    return [(score - low) / (high - low) for score in scores]


def run_vector_topk(
    *,
    graph: Graph,
    index: Any,
    embed_fn: Callable[[str], list[float]],
    query_text: str,
    top_k: int,
    max_prompt_context_chars: int,
    prompt_context_include_node_ids: bool,
) -> dict[str, object]:
    q_vec = embed_fn(query_text)
    seeds = index.search(q_vec, top_k=top_k)
    fired_nodes = [node_id for node_id, _score in seeds]
    fired_scores = {node_id: float(score) for node_id, score in seeds}
    prompt_context, prompt_stats = build_prompt_context_ranked_with_stats(
        graph=graph,
        node_ids=fired_nodes,
        node_scores=fired_scores,
        max_chars=max_prompt_context_chars,
        include_node_ids=prompt_context_include_node_ids,
    )
    return {
        "fired_nodes": fired_nodes,
        "prompt_context": prompt_context,
        **prompt_stats,
        "route_router_conf_mean": 0.0,
        "route_relevance_conf_mean": 0.0,
        "route_policy_disagreement_mean": 0.0,
        "route_decision_count": 0,
    }


def run_vector_topk_rerank(
    *,
    graph: Graph,
    index: Any,
    embed_fn: Callable[[str], list[float]],
    query_text: str,
    top_k: int,
    max_prompt_context_chars: int,
    prompt_context_include_node_ids: bool,
    reranker: Reranker | None,
) -> dict[str, object]:
    if reranker is None:
        return {"skipped": True, "skipped_reason": "reranker unavailable"}

    q_vec = embed_fn(query_text)
    seeds = index.search(q_vec, top_k=top_k)
    candidates: list[tuple[str, str, float]] = []
    for node_id, score in seeds:
        node = graph.get_node(node_id)
        if node is None:
            continue
        candidates.append((node_id, node.content or "", float(score)))

    reranked = reranker.rerank(query_text, candidates)
    fired_nodes = [node_id for node_id, _score in reranked]
    scores = _normalize_scores([score for _node_id, score in reranked])
    fired_scores = {node_id: score for node_id, score in zip(fired_nodes, scores)}
    prompt_context, prompt_stats = build_prompt_context_ranked_with_stats(
        graph=graph,
        node_ids=fired_nodes,
        node_scores=fired_scores,
        max_chars=max_prompt_context_chars,
        include_node_ids=prompt_context_include_node_ids,
    )
    return {
        "fired_nodes": fired_nodes,
        "prompt_context": prompt_context,
        **prompt_stats,
        "route_router_conf_mean": 0.0,
        "route_relevance_conf_mean": 0.0,
        "route_policy_disagreement_mean": 0.0,
        "route_decision_count": 0,
        "reranker_name": reranker.name,
    }


@dataclass(frozen=True)
class PointerChaseConfig:
    top_k: int = 4
    max_turns: int = 6
    max_tool_calls: int = 8
    max_candidates: int = 6
    edge_weight_factor: float = 0.55
    similarity_factor: float = 0.45
    min_next_score: float = 0.05
    tool_latency_ms: float = 30.0
    llm_latency_ms: float = 200.0
    completion_tokens_per_turn: int = 24
    cost_per_1k_tokens: float = 0.0
    candidate_snippet_chars: int = 180
    max_prompt_context_chars: int = 20000
    prompt_context_include_node_ids: bool = True


class PointerChaseSimulator:
    """Deterministic pointer-chasing simulator for baseline comparisons."""

    def __init__(
        self,
        *,
        graph: Graph,
        index: Any,
        embed_fn: Callable[[str], list[float]],
        config: PointerChaseConfig | None = None,
    ) -> None:
        self._graph = graph
        self._index = index
        self._embed_fn = embed_fn
        self._config = config or PointerChaseConfig()

    def run(self, query_text: str) -> dict[str, object]:
        cfg = self._config
        start = time.perf_counter()
        query_vec = self._embed_fn(query_text)
        seeds = self._index.search(query_vec, top_k=cfg.top_k)
        tool_calls = 1 if seeds else 0
        tool_latency = cfg.tool_latency_ms if seeds else 0.0
        if not seeds:
            return {
                "fired_nodes": [],
                "prompt_context": "",
                "prompt_context_len": 0,
                "pointer_turns": 0,
                "pointer_tool_calls": tool_calls,
                "pointer_prompt_tokens": 0,
                "pointer_completion_tokens": 0,
                "pointer_total_tokens": 0,
                "pointer_cost": 0.0,
                "pointer_latency_ms": 0.0,
            }

        current_node_id, current_score = seeds[0]
        visited = [current_node_id]
        visited_scores = {current_node_id: float(current_score)}
        llm_turns = 0
        prompt_tokens = 0
        completion_tokens = 0

        for _turn in range(cfg.max_turns):
            if tool_calls >= cfg.max_tool_calls:
                break
            outgoing = self._graph.outgoing(current_node_id)
            tool_calls += 1
            tool_latency += cfg.tool_latency_ms
            if not outgoing:
                break

            candidates = []
            for node, edge in outgoing[: cfg.max_candidates]:
                sim = _safe_cosine(query_vec, self._embed_fn(node.content or ""))
                combined = (cfg.edge_weight_factor * float(edge.weight)) + (
                    cfg.similarity_factor * sim
                )
                candidates.append((node.id, node.content or "", float(edge.weight), sim, combined))

            if not candidates:
                break

            candidates.sort(key=lambda item: item[4], reverse=True)
            next_id, _content, _edge_weight, _sim, score = candidates[0]
            llm_turns += 1

            prompt_tokens += _approx_tokens(
                _render_pointer_prompt(query_text, current_node_id, candidates, cfg.candidate_snippet_chars)
            )
            completion_tokens += cfg.completion_tokens_per_turn

            if score < cfg.min_next_score:
                break
            if next_id in visited:
                break

            visited.append(next_id)
            visited_scores[next_id] = score
            current_node_id = next_id

        prompt_context, prompt_stats = build_prompt_context_ranked_with_stats(
            graph=self._graph,
            node_ids=visited,
            node_scores=visited_scores,
            max_chars=cfg.max_prompt_context_chars,
            include_node_ids=cfg.prompt_context_include_node_ids,
        )
        total_tokens = prompt_tokens + completion_tokens
        cost = (total_tokens / 1000.0) * cfg.cost_per_1k_tokens
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        simulated_latency = tool_latency + (llm_turns * cfg.llm_latency_ms)

        return {
            "fired_nodes": visited,
            "prompt_context": prompt_context,
            **prompt_stats,
            "pointer_turns": llm_turns,
            "pointer_tool_calls": tool_calls,
            "pointer_prompt_tokens": prompt_tokens,
            "pointer_completion_tokens": completion_tokens,
            "pointer_total_tokens": total_tokens,
            "pointer_cost": cost,
            "pointer_latency_ms": simulated_latency,
            "latency_ms": elapsed_ms,
        }


def run_pointer_chase(
    *,
    graph: Graph,
    index: Any,
    embed_fn: Callable[[str], list[float]],
    query_text: str,
    config: PointerChaseConfig | None = None,
) -> dict[str, object]:
    simulator = PointerChaseSimulator(graph=graph, index=index, embed_fn=embed_fn, config=config)
    return simulator.run(query_text)


def _safe_cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def _render_pointer_prompt(
    query_text: str,
    current_node_id: str,
    candidates: list[tuple[str, str, float, float, float]],
    snippet_chars: int,
) -> str:
    lines = [f"Query: {query_text}", f"Current node: {current_node_id}", "Candidates:"]
    for node_id, content, edge_weight, sim, score in candidates:
        snippet = (content or "").replace("\n", " ")[:snippet_chars]
        lines.append(f"- {node_id} w={edge_weight:.3f} sim={sim:.3f} score={score:.3f} :: {snippet}")
    return "\n".join(lines)
