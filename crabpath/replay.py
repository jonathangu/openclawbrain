from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable

from .graph import Graph
from .learn import LearningConfig, apply_outcome
from .traverse import TraversalConfig, traverse


_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def _tokenize_text(text: str) -> set[str]:
    return {match.group(0).lower() for match in _WORD_RE.finditer(text)}


def _extract_user_query_content(content: object) -> str | None:
    if isinstance(content, str):
        value = content.strip()
        return value if value else None

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    parts.append(text)
                continue

            if isinstance(item, dict):
                item_text = _extract_user_query_content(item.get("text"))
                if item_text:
                    parts.append(item_text)
                    continue
                item_content = _extract_user_query_content(item.get("content"))
                if item_content:
                    parts.append(item_content)
        return " ".join(parts) if parts else None

    return None


def _extract_openclaw_query(payload: dict) -> str | None:
    if payload.get("type") != "message":
        return None

    message = payload.get("message")
    if not isinstance(message, dict) or message.get("role") != "user":
        return None

    return _extract_user_query_content(message.get("content"))


def _extract_flat_query(payload: dict) -> str | None:
    if payload.get("role") != "user":
        return None
    return _extract_user_query_content(payload.get("content"))


def extract_queries(session_path: str | Path) -> list[str]:
    """Extract user queries from an OpenClaw session log.

    OpenClaw format: JSONL with records like:
    {"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": "..."}]}}

    Also handles flat format: {"role": "user", "content": "..."}
    Also handles plain text lines.

    Returns list of query strings.
    """
    path = Path(session_path).expanduser()
    if not path.exists():
        raise SystemExit(f"missing sessions file: {path}")

    queries: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue

            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                queries.append(raw)
                continue

            if not isinstance(payload, dict):
                continue

            query = _extract_openclaw_query(payload)
            if query is None:
                query = _extract_flat_query(payload)

            if query:
                queries.append(query)

    return queries


def extract_queries_from_dir(sessions_dir: str | Path) -> list[str]:
    """Extract queries from all .jsonl files in a directory."""
    path = Path(sessions_dir).expanduser()
    if not path.exists():
        raise SystemExit(f"missing sessions directory: {path}")
    if not path.is_dir():
        raise SystemExit(f"not a directory: {path}")

    queries: list[str] = []
    for session_file in sorted(path.glob("*.jsonl")):
        queries.extend(extract_queries(session_file))
    return queries


def default_keyword_seed_fn(graph: Graph, query_text: str) -> list[tuple[str, float]]:
    query_tokens = _tokenize_text(query_text)
    if not query_tokens:
        return []

    scores: list[tuple[str, float]] = []
    for node in graph.nodes():
        node_tokens = _tokenize_text(node.content)
        overlap = len(query_tokens & node_tokens)
        scores.append((node.id, overlap / len(query_tokens)))

    scores.sort(key=lambda item: (item[1], item[0]), reverse=True)
    return scores[:10]


def _snapshot_edges(graph: Graph) -> dict[tuple[str, str], float]:
    weights: dict[tuple[str, str], float] = {}
    for source_id, edges in graph._edges.items():
        for target_id, edge in edges.items():
            weights[(source_id, target_id)] = edge.weight
    return weights


def _cross_file_edges(graph: Graph) -> set[tuple[str, str]]:
    edges: set[tuple[str, str]] = set()
    for source_id, source_edges in graph._edges.items():
        source_node = graph.get_node(source_id)
        source_file = source_node.metadata.get("file") if source_node else None
        for target_id in source_edges:
            target_node = graph.get_node(target_id)
            target_file = target_node.metadata.get("file") if target_node else None
            if source_file is not None and target_file is not None and source_file != target_file:
                edges.add((source_id, target_id))
    return edges


def replay_queries(
    graph: Graph,
    queries: list[str],
    config: TraversalConfig | None = None,
    keyword_seed_fn: Callable[[Graph, str], list[tuple[str, float]]] | None = None,
    verbose: bool = False,
) -> dict:
    """Replay historical queries to warm up graph edges.

    For each query:
    1. Seed from keyword matching (or provided seed_fn)
    2. Traverse the graph
    3. Apply positive outcome (+1) — assumes historical queries were useful
    4. Apply Hebbian co-firing for co-selected nodes
    """
    cfg = config or TraversalConfig()
    seed_fn = keyword_seed_fn or default_keyword_seed_fn

    stats = {
        "queries_replayed": 0,
        "edges_reinforced": 0,
        "cross_file_edges_created": 0,
    }
    total_queries = len(queries)

    for query in queries:
        stats["queries_replayed"] += 1

        seeds = seed_fn(graph, query)
        result = traverse(graph=graph, seeds=seeds, config=cfg)
        if not result.fired:
            if verbose:
                print(
                    f"Replayed {stats['queries_replayed']}/{total_queries} queries, "
                    f"{stats['cross_file_edges_created']} cross-file edges created"
                )
            continue

        before_weights = _snapshot_edges(graph)
        before_cross_edges = _cross_file_edges(graph)

        apply_outcome(graph=graph, fired_nodes=result.fired, outcome=1.0, config=LearningConfig())

        after_weights = _snapshot_edges(graph)
        after_cross_edges = _cross_file_edges(graph)

        for key, weight in after_weights.items():
            if before_weights.get(key) != weight:
                stats["edges_reinforced"] += 1

        new_cross_edges = after_cross_edges - before_cross_edges
        stats["cross_file_edges_created"] += len(new_cross_edges)

        if verbose:
            print(
                f"Replayed {stats['queries_replayed']}/{total_queries} queries, "
                f"{stats['cross_file_edges_created']} cross-file edges created"
            )

    return stats
