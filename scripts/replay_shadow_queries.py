"""Replay shadow queries against an existing CrabPath graph."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crabpath._structural_utils import count_cross_file_edges, node_file_id  # noqa: E402
from crabpath.decay import DecayConfig, apply_decay  # noqa: E402
from crabpath.graph import Graph  # noqa: E402
from crabpath.lifecycle_sim import SimConfig, make_mock_llm_all, make_mock_router  # noqa: E402
from crabpath.mitosis import MitosisConfig, MitosisState, mitosis_maintenance  # noqa: E402
from crabpath.synaptogenesis import (  # noqa: E402
    SynaptogenesisConfig,
    SynaptogenesisState,
    decay_proto_edges,
    edge_tier_stats,
    record_cofiring,
    record_skips,
)

DEFAULT_SEED = 2026
REAL_QUERY_COUNT = 35
REPHRASE_VARIANTS_PER_QUERY = 4
CROSS_FILE_QUERIES = 60


@dataclass
class ReplayQuery:
    text: str
    source: str
    origin: str


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _read_jsonl_queries(path: Path, limit: int) -> list[str]:
    queries: list[str] = []

    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue

            text = None
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                text = line
            else:
                if isinstance(payload, str):
                    text = payload
                elif isinstance(payload, dict):
                    text = (
                        payload.get("query")
                        or payload.get("text")
                        or payload.get("question")
                        or payload.get("prompt")
                        or payload.get("query_text")
                    )

            if not isinstance(text, str):
                continue
            text = text.strip()
            if text:
                queries.append(text)

            if len(queries) >= limit:
                break

    if len(queries) < limit:
        raise ValueError(
            f"expected at least {limit} queries in {path}, found {len(queries)}"
        )

    return queries


SUBSTITUTIONS: dict[str, tuple[str, ...]] = {
    "show": ("display", "reveal", "outline"),
    "list": ("enumerate", "show", "outline"),
    "check": ("verify", "confirm", "inspect"),
    "create": ("build", "set up", "define"),
    "reset": ("reinitialize", "restart", "rebuild"),
    "worktree": ("tree", "workspace", "branch"),
    "codex": ("agent", "assistant", "tool"),
    "cookie": ("token", "session", "credential"),
    "query": ("lookup", "request", "question"),
    "rules": ("policies", "guidelines", "constraints"),
    "browser": ("web client", "chrome", "headless session"),
    "trading": ("market", "deal flow", "finance"),
}

REWRITE_TEMPLATES = [
    "Can you {query}?",
    "How can I {query}?",
    "I need to know how to {query}.",
    "What is the best way to {query}?",
    "When should I {query}?",
]

CROSS_FILE_TEMPLATES = [
    "{left} AND {right}: how should this work?",
    "How do {left} and {right} work together?",
    "What are the tradeoffs between {left} and {right}?",
    "I need a workflow that combines {left} with {right}",
    "How to align {left} with {right}?",
]

STOPWORDS = {
    "the",
    "and",
    "or",
    "to",
    "for",
    "a",
    "an",
    "of",
    "in",
    "on",
    "at",
    "with",
    "without",
    "from",
    "by",
    "if",
    "then",
    "it",
    "this",
    "that",
    "i",
    "my",
    "me",
    "you",
    "your",
    "we",
    "our",
    "can",
    "should",
    "would",
    "is",
    "are",
    "be",
}


def _rewrite_once(query: str, rng: random.Random) -> str:
    words = query.split()
    rewritten = []

    for word in words:
        key = re.sub(r"[^a-z0-9]", "", word.lower())
        replacements = SUBSTITUTIONS.get(key)
        if replacements and rng.random() < 0.55:
            rewritten.append(rng.choice(replacements))
        else:
            rewritten.append(word)

    base = " ".join(rewritten).strip()
    template = rng.choice(REWRITE_TEMPLATES)
    return template.format(query=base)


def _build_query_variants(query: str, rng: random.Random, target: int = 4) -> list[str]:
    variants: list[str] = []
    seen: set[str] = set()

    while len(variants) < target:
        rewritten = _rewrite_once(query, rng)
        if rewritten and rewritten not in seen:
            variants.append(rewritten)
            seen.add(rewritten)

    return variants


def _extract_file_id(node_id: str, metadata: dict[str, Any]) -> str:
    file_path = metadata.get("file") if isinstance(metadata, dict) else None
    if isinstance(file_path, str) and file_path.strip():
        cleaned = file_path.replace("\\", "/").split("/")[-1]
        return cleaned.rsplit(".", 1)[0] if "." in cleaned else cleaned
    return node_file_id(node_id)


def _extract_file_topics(graph: Graph) -> dict[str, list[str]]:
    counters: dict[str, Counter[str]] = {}
    for node in graph.nodes():
        file_id = _extract_file_id(node.id, node.metadata)
        counter = counters.setdefault(file_id, Counter())
        counter.update(token for token in _tokenize(node.content) if token not in STOPWORDS)

    topics: dict[str, list[str]] = {}
    for file_id, counter in counters.items():
        top_words = [word for word, _ in counter.most_common(3)]
        if not top_words:
            top_words = [file_id]
        topics[file_id] = [file_id, *top_words]

    return topics


def _pick_file_pairs(topic_map: dict[str, list[str]], rng: random.Random) -> list[tuple[str, str]]:
    file_ids = list(topic_map)
    if len(file_ids) >= 2:
        return file_ids

    fallback = ["identity", "tools", "safety", "workspace", "workflow", "trading"]
    return fallback


def _build_rephrased_queries(real_queries: list[str], rng: random.Random) -> list[str]:
    variants: list[str] = []
    for query in real_queries:
        variants.extend(_build_query_variants(query, rng, target=REPHRASE_VARIANTS_PER_QUERY))
    return variants


def _build_cross_file_queries(graph: Graph, rng: random.Random, target: int) -> list[str]:
    topic_map = _extract_file_topics(graph)
    file_ids = list(topic_map)

    if len(file_ids) < 2:
        file_ids = _pick_file_pairs(topic_map, rng)

    queries: list[str] = []
    seen: set[str] = set()
    while len(queries) < target:
        left_file, right_file = rng.sample(file_ids, 2)
        left_candidates = topic_map.get(left_file, [left_file])
        right_candidates = topic_map.get(right_file, [right_file])

        left_topic = rng.choice(left_candidates)
        right_topic = rng.choice(right_candidates)

        query = rng.choice(CROSS_FILE_TEMPLATES).format(
            left=left_topic,
            right=right_topic,
        )

        if query in seen:
            continue
        queries.append(query)
        seen.add(query)

    return queries


def _build_replay_queries(
    real_queries: list[str],
    graph: Graph,
    rng: random.Random,
) -> list[ReplayQuery]:
    base = [ReplayQuery(text=q, source=q, origin="real") for q in real_queries]

    rephrases = [
        ReplayQuery(text=variant, source=query, origin="rephrase")
        for query in real_queries
        for variant in _build_query_variants(
            query,
            rng,
            target=REPHRASE_VARIANTS_PER_QUERY,
        )
    ]

    cross_file = [
        ReplayQuery(text=query, source="cross_file", origin="cross_file")
        for query in _build_cross_file_queries(
            graph=graph,
            rng=rng,
            target=CROSS_FILE_QUERIES,
        )
    ]

    return base + rephrases + cross_file


def _query_reward(selected_nodes: list[str], context_chars: int) -> float:
    selected_count = len(selected_nodes)
    if selected_count == 0:
        return 0.0
    if context_chars >= 320:
        return 1.0
    return min(1.0, (selected_count / 3.0) + (context_chars / 400.0))


def _run_simulation(graph: Graph, queries: list[ReplayQuery], config: SimConfig) -> tuple[
    list[dict[str, Any]], dict[str, int], int
]:
    mitosis_state = MitosisState()
    synapse_state = SynaptogenesisState()
    mitosis_config = MitosisConfig()
    synapse_config = SynaptogenesisConfig()
    decay_config = DecayConfig(half_life_turns=config.decay_half_life)

    llm = make_mock_llm_all()
    router = make_mock_router()

    records: list[dict[str, Any]] = []
    for qi, replay_query in enumerate(queries, 1):
        query_text = replay_query.text
        query_words = set(query_text.lower().split())

        candidates: list[tuple[str, float, str]] = []
        for node in graph.nodes():
            node_words = set((node.content or "").lower().split())
            overlap = len(query_words & node_words)
            score = min(overlap / max(len(query_words), 1), 1.0)
            if score > 0.1:
                candidates.append((node.id, score, node.summary or node.content[:80]))

        candidates.sort(key=lambda item: item[1], reverse=True)
        candidates = candidates[:10]

        selected = router(query_text, candidates)

        edges_before = graph.edge_count
        proto_before = len(synapse_state.proto_edges)

        cofire_result = record_cofiring(graph, selected, synapse_state, synapse_config)

        skips = 0
        if selected:
            candidate_ids = [node_id for node_id, _, _ in candidates]
            skips = record_skips(graph, selected[0], candidate_ids, selected, synapse_config)

        if qi % config.decay_interval == 0:
            apply_decay(graph, turns_elapsed=config.decay_interval, config=decay_config)
            decay_proto_edges(synapse_state, synapse_config)

        if qi % config.maintenance_interval == 0:
            mitosis_maintenance(graph, llm, mitosis_state, mitosis_config)

        tiers = edge_tier_stats(graph, synapse_config)

        edges_after = graph.edge_count
        proto_after = len(synapse_state.proto_edges)

        context_chars = 0
        for node_id in selected:
            node = graph.get_node(node_id)
            if node is not None:
                context_chars += len(node.content)

        reward = _query_reward(selected, context_chars)

        records.append(
            {
                "query_num": qi,
                "query_type": replay_query.origin,
                "query": query_text,
                "source_query": replay_query.source,
                "selected_nodes": list(selected),
                "selected_nodes_count": len(selected),
                "reward": reward,
                "context_chars": context_chars,
                "edge_updates": {
                    "edges_before": edges_before,
                    "edges_after": edges_after,
                    "edges_delta": edges_after - edges_before,
                    "proto_edges_before": proto_before,
                    "proto_edges_after": proto_after,
                    "proto_edges_delta": proto_after - proto_before,
                    "promotions": cofire_result["promoted"],
                    "reinforcements": cofire_result["reinforced"],
                    "skips_penalized": skips,
                },
                "tiers": tiers,
            }
        )

    final_tiers = edge_tier_stats(graph, synapse_config)
    cross_file_edges = count_cross_file_edges(graph)
    return records, final_tiers, cross_file_edges


def _format_summary(
    total_queries: int,
    records: list[dict[str, Any]],
    final_tiers: dict[str, int],
    cross_file_edges: int,
) -> str:
    rewards = [record["reward"] for record in records]
    avg_reward = (sum(rewards) / total_queries) if total_queries else 0.0
    return (
        "\n=== Replay summary ===\n"
        f"total_queries: {total_queries}\n"
        f"avg_reward: {avg_reward:.4f}\n"
        f"reflex_edges_formed: {final_tiers.get('reflex', 0)}\n"
        f"cross_file_edges: {cross_file_edges}\n"
        f"tier_distribution: {final_tiers}\n"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay 35 real queries + 200 synthetic variants through a graph"
    )
    parser.add_argument(
        "--queries",
        required=True,
        help="JSONL file containing shadow queries",
    )
    parser.add_argument(
        "--graph",
        required=True,
        help="CrabPath graph JSON file",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL path for per-query replay records",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Deterministic seed for synthetic query generation",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    query_path = Path(args.queries)
    if not query_path.exists():
        raise FileNotFoundError(f"queries file not found: {query_path}")

    graph_path = Path(args.graph)
    if not graph_path.exists():
        raise FileNotFoundError(f"graph file not found: {graph_path}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    graph = Graph.load(str(graph_path))
    real_queries = _read_jsonl_queries(query_path, REAL_QUERY_COUNT)
    rng = random.Random(args.seed)

    replay_queries = _build_replay_queries(real_queries, graph, rng)
    print(
        "Total queries prepared:"
        f"{len(real_queries)} real + "
        f"{len(replay_queries)-len(real_queries)} synthetic ({len(replay_queries)} total)"
    )

    records, final_tiers, cross_file_edges = _run_simulation(
        graph,
        replay_queries,
        SimConfig(),
    )

    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record))
            handle.write("\n")

    print(_format_summary(len(records), records, final_tiers, cross_file_edges))


if __name__ == "__main__":
    main()
