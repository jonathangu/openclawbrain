#!/usr/bin/env python3
"""Sparsity and scale experiment with CrabPath-vs-BM25 crossover analysis."""

from __future__ import annotations

import copy
import json
import math
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crabpath.graph import Edge, Graph, Node
from crabpath.legacy.activation import activate, learn
from scripts import ablation_study

SEED = 2026
TOP_K = 3
WARMUP_QUERIES = 50
EVAL_QUERIES = 50
TOTAL_QUERIES = 100
NIAH_QUERIES = 20
NIAH_WARMUP = 10
NIAH_EVAL = 10
RESULTS_PATH = ROOT / "scripts" / "sparsity_scale_results.json"

GRAPH_SIZES = [20, 50, 100, 200, 500]
SPARSITY_LEVELS = [
    ("Dense", 1.0),
    ("Medium", 0.3),
    ("Sparse", 0.1),
    ("V.Sparse", 0.05),
]

CLUSTERS = {
    "deploy": {
        "topic": "deployment operations",
        "anchor_terms": [
            "artifact",
            "canary",
            "rollout",
            "release",
            "pipeline",
            "rollback",
        ],
    },
    "incident": {
        "topic": "incident response",
        "anchor_terms": [
            "alert",
            "outage",
            "severity",
            "playbook",
            "mitigation",
            "postmortem",
        ],
    },
    "security": {
        "topic": "security controls",
        "anchor_terms": [
            "hardening",
            "audit",
            "credential",
            "policy",
            "threat",
            "compliance",
        ],
    },
    "monitoring": {
        "topic": "system monitoring",
        "anchor_terms": [
            "latency",
            "telemetry",
            "threshold",
            "dashboards",
            "anomaly",
            "incident",
        ],
    },
    "architecture": {
        "topic": "system architecture",
        "anchor_terms": [
            "boundary",
            "service",
            "contract",
            "dependency",
            "storage",
            "throughput",
        ],
    },
}


@dataclass
class DocSpec:
    node_id: str
    cluster: str
    keywords: list[str]
    content: str


@dataclass
class QueryCase:
    text: str
    expected_node_ids: list[str]
    hops: int
    path: list[str]


def _safe_divide(num: float, den: float) -> float:
    if den <= 0:
        return 0.0
    return num / den


def _token_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9_]+", text.lower()))


def _build_cluster_distribution(size: int) -> dict[str, int]:
    base = size // len(CLUSTERS)
    extra = size % len(CLUSTERS)
    counts = {}
    for i, cluster in enumerate(CLUSTERS):
        counts[cluster] = base + (1 if i < extra else 0)
    return counts


def _make_documents(size: int, rng: random.Random) -> tuple[dict[str, DocSpec], list[DocSpec]]:
    counts = _build_cluster_distribution(size)
    node_specs: dict[str, DocSpec] = {}
    ordered_docs: list[DocSpec] = []

    for cluster, count in counts.items():
        terms = CLUSTERS[cluster]["anchor_terms"]
        for i in range(count):
            node_index = f"{cluster}-{i + 1:03d}"
            keywords = [
                f"{cluster}_topic_{i + 1:03d}_1",
                f"{cluster}_signal_{i + 1:03d}_{rng.randint(10, 99)}",
            ]
            theme = CLUSTERS[cluster]["topic"]

            sentences = [
                (
                    f"{keywords[0]} documents {theme} guidance in sentence one."
                    f" It mentions {terms[i % len(terms)]} and {terms[(i + 1) % len(terms)]}."
                ),
                (
                    f"This note links {keywords[1]} to local evidence around"
                    f" {cluster} edge behavior and repeatable operations."
                ),
            ]

            if rng.random() < 0.75:
                neighbor_term = terms[(i + 2) % len(terms)]
                sentences.append(
                    f"It references {neighbor_term} in a downstream handoff and"
                    f" captures a practical follow-on action."
                )

            if rng.random() < 0.60:
                sentences.append(
                    f"Cross-functional review expects {keywords[0]} to align with"
                    f" {keywords[1]} under operational pressure."
                )

            content = " ".join(sentences)
            node_id = f"{cluster}_{node_index}"
            spec = DocSpec(
                node_id=node_id,
                cluster=cluster,
                keywords=keywords,
                content=content,
            )
            node_specs[node_id] = spec
            ordered_docs.append(spec)

    return node_specs, ordered_docs


def _add_edge(graph: Graph, source: str, target: str, *, weight_hint: float, rng: random.Random) -> None:
    if source == target:
        return
    existing = graph.get_edge(source, target)
    weight = weight_hint + rng.uniform(-0.08, 0.08)
    if existing is None:
        graph.add_edge(
            Edge(
                source=source,
                target=target,
                weight=max(0.05, min(0.95, weight)),
            )
        )


def _build_graph(size: int, edge_ratio: float, seed: int) -> tuple[Graph, dict[str, DocSpec], list[DocSpec]]:
    rng = random.Random(seed)
    node_specs, ordered_docs = _make_documents(size, rng)
    graph = Graph()

    for spec in ordered_docs:
        graph.add_node(
            Node(
                id=spec.node_id,
                content=spec.content,
                cluster_id=spec.cluster,
            )
        )

    node_ids = [spec.node_id for spec in ordered_docs]
    by_cluster: dict[str, list[str]] = {}
    for cluster, count in _build_cluster_distribution(size).items():
        by_cluster[cluster] = [
            spec.node_id for spec in ordered_docs if spec.cluster == cluster
        ]

    # Ensure baseline connectivity using a directed spanning structure (both directions).
    ordered = node_ids.copy()
    rng.shuffle(ordered)
    for i in range(len(ordered) - 1):
        _add_edge(
            graph,
            ordered[i],
            ordered[i + 1],
            weight_hint=0.60,
            rng=rng,
        )
        _add_edge(
            graph,
            ordered[i + 1],
            ordered[i],
            weight_hint=0.60,
            rng=rng,
        )

    # Guarantee at least one inter-cluster bridge per adjacent pair in the cluster list.
    cluster_names = list(CLUSTERS.keys())
    for i in range(len(cluster_names) - 1):
        source_cluster = by_cluster[cluster_names[i]]
        target_cluster = by_cluster[cluster_names[i + 1]]
        if source_cluster and target_cluster:
            s = rng.choice(source_cluster)
            t = rng.choice(target_cluster)
            _add_edge(graph, s, t, weight_hint=0.50, rng=rng)
            _add_edge(graph, t, s, weight_hint=0.50, rng=rng)

    # Add sparse-dense edges with intra-cluster preference and exact node ratios.
    if edge_ratio < 1.0:
        desired_out = max(1, max(1, int(round(edge_ratio * (size - 1)))))
    else:
        desired_out = size - 1

    for source in node_ids:
        source_cluster = next(
            spec.cluster
            for spec in ordered_docs
            if spec.node_id == source
        )

        existing = {edge_target.id for edge_target, _ in graph.outgoing(source)}
        needed = max(0, desired_out - len(existing))
        if needed <= 0:
            continue

        same_cluster = [nid for nid in node_ids if nid != source and nid not in existing and node_specs[nid].cluster == source_cluster]
        other_cluster = [nid for nid in node_ids if nid != source and nid not in existing and node_specs[nid].cluster != source_cluster]

        # Bias intra-cluster edges while keeping a few inter-cluster bridges.
        if edge_ratio == 1.0:
            random_targets = [nid for nid in node_ids if nid != source and nid not in existing]
        else:
            weighted_candidates = same_cluster * 3 + other_cluster
            if not weighted_candidates:
                random_targets = []
            else:
                k = min(len(weighted_candidates), needed)
                sampled = random.sample(weighted_candidates, k=k)
                random_targets = []
                seen: set[str] = set(existing)
                for nid in sampled:
                    if nid not in seen:
                        random_targets.append(nid)
                        seen.add(nid)
                        if len(random_targets) >= needed:
                            break

                if len(random_targets) < needed:
                    remaining = [
                        nid
                        for nid in node_ids
                        if nid != source and nid not in seen
                    ]
                    if remaining:
                        extra = random.sample(
                            remaining,
                            k=min(needed - len(random_targets), len(remaining)),
                        )
                        random_targets.extend(extra)

        for target in random_targets[:needed]:
            weight_hint = 0.72 if node_specs[target].cluster == source_cluster else 0.48
            _add_edge(
                graph,
                source,
                target,
                weight_hint=weight_hint,
                rng=rng,
            )

    if edge_ratio < 1.0 and len(node_ids) > 2:
        # Ensure exact ratio-like density via sample-based supplementation.
        for source in node_ids:
            existing_count = len(graph._outgoing.get(source, []))
            target_count = max(1, int(round(edge_ratio * (size - 1))))
            if existing_count >= target_count:
                continue
            missing = target_count - existing_count
            candidates = [nid for nid in node_ids if nid != source and nid not in graph._outgoing.get(source, [])]
            if not candidates:
                continue
            picks = random.sample(candidates, k=min(len(candidates), missing))
            for target in picks:
                _add_edge(
                    graph,
                    source,
                    target,
                    weight_hint=0.45,
                    rng=rng,
                )

    return graph, node_specs, ordered_docs


def _metric_at_3(ranking: list[str], expected: list[str]) -> dict[str, float]:
    expected_set = set(expected)
    if not expected_set:
        return {"recall@3": 0.0, "ndcg@3": 0.0, "mrr@3": 0.0}

    top3 = ranking[:3]
    expected_count = len(expected_set)
    hits = sum(1 for node_id in top3 if node_id in expected_set)
    recall3 = _safe_divide(hits, expected_count)

    ideal_hits = min(expected_count, 3)
    ideal_dcg = sum(1.0 / math.log2(i + 2.0) for i in range(ideal_hits))
    dcg = 0.0
    for idx, node_id in enumerate(top3):
        if node_id in expected_set:
            dcg += 1.0 / math.log2(idx + 2.0)
    ndcg3 = _safe_divide(dcg, ideal_dcg)

    mrr3 = 0.0
    for idx, node_id in enumerate(top3, start=1):
        if node_id in expected_set:
            mrr3 = 1.0 / idx
            break

    return {"recall@3": recall3, "ndcg@3": ndcg3, "mrr@3": mrr3}


def _select_by_bm25(graph: Graph, query: str) -> list[str]:
    scores = ablation_study._bm25_score(query, list(graph.nodes()))
    return [node_id for node_id, _ in scores[:TOP_K]]


def _seed_from_bm25(ranking: list[str], graph: Graph) -> dict[str, float]:
    seeds: dict[str, float] = {}
    for idx, node_id in enumerate(ranking):
        if graph.get_node(node_id):
            seeds[node_id] = 1.0 - 0.2 * idx
    return seeds


def _draw_path(
    graph: Graph,
    rng: random.Random,
    start: str,
    length: int,
) -> list[str]:
    if length == 0:
        return [start]
    path = [start]
    for _ in range(length):
        outgoing = [edge_target.id for edge_target, _ in graph.outgoing(path[-1])]
        if not outgoing:
            return []
        next_hops = [node_id for node_id in outgoing if node_id not in path]
        if not next_hops:
            return []
        path.append(rng.choice(next_hops))
    return path


def _query_keywords(specs: dict[str, DocSpec], node_id: str, rng: random.Random) -> str:
    return rng.choice(specs[node_id].keywords)


def _make_query(case_nodes: list[str], specs: dict[str, DocSpec], hops: int, idx: int, rng: random.Random) -> QueryCase:
    if hops == 1:
        anchor_start = _query_keywords(specs, case_nodes[0], rng)
        text = (
            f"Locate the runbook note that continues from {anchor_start}"
            f" in the same cluster flow. Query ID {idx + 1}."
        )
        expected = [case_nodes[1]]
    elif hops == 2:
        anchor_start = _query_keywords(specs, case_nodes[0], rng)
        anchor_mid = _query_keywords(specs, case_nodes[1], rng)
        text = (
            f"Starting from {anchor_start}, follow the linked path through"
            f" {anchor_mid} and identify the next handoff document."
            f" Query ID {idx + 1}."
        )
        expected = [case_nodes[2]]
    else:
        anchor_a = _query_keywords(specs, case_nodes[0], rng)
        anchor_b = _query_keywords(specs, case_nodes[1], rng)
        anchor_c = _query_keywords(specs, case_nodes[2], rng)
        text = (
            f"Given {anchor_a}, trace through {anchor_b} and {anchor_c} to confirm"
            f" the final chain target document."
            f" Query ID {idx + 1}."
        )
        expected = [case_nodes[3]]

    return QueryCase(text=text, expected_node_ids=expected, hops=hops, path=case_nodes)


def _build_queries(graph: Graph, specs: dict[str, DocSpec], rng: random.Random) -> list[QueryCase]:
    node_ids = list(specs.keys())
    queries: list[QueryCase] = []

    targets = {1: 60, 2: 30, 3: 10}
    for hops, count in targets.items():
        produced = 0
        attempts = 0
        while produced < count and attempts < 10_000:
            attempts += 1
            start = rng.choice(node_ids)
            path = _draw_path(graph, rng, start, hops)
            if not path or len(path) != hops + 1:
                continue
            # Avoid identical paths for readability in trace output.
            if path in [q.path for q in queries]:
                continue

            case = _make_query(path, specs, hops, produced, rng)
            if not case.expected_node_ids:
                continue
            queries.append(case)
            produced += 1
    return queries[:TOTAL_QUERIES]


def _run_query_suite(
    graph: Graph,
    queries: list[QueryCase],
    learn_graph: bool,
    warmup: int,
) -> dict[str, Any]:
    initial_inhibitory = sum(1 for edge in graph.edges() if edge.weight < 0.0)
    initial_edge_count = graph.edge_count

    per_query: list[dict[str, Any]] = []

    total_tokens_bm25 = 0
    total_tokens_cp = 0
    total_tokens_bm25_eval = 0
    total_tokens_cp_phase2 = 0

    for i, query in enumerate(queries, start=1):
        bm25_ranking = _select_by_bm25(graph, query.text)
        bm25_top = bm25_ranking[:TOP_K]
        bm25_metrics = _metric_at_3(bm25_top, query.expected_node_ids)
        bm25_tokens = sum(len(graph.get_node(nid).content) for nid in bm25_top if graph.get_node(nid))
        total_tokens_bm25 += bm25_tokens

        selected: list[str]
        activation = None
        if learn_graph:
            seeds = _seed_from_bm25(bm25_top, graph)
            if not seeds:
                fallback = [node.id for node in graph.nodes() if node is not None]
                if fallback:
                    seeds = {fallback[0]: 1.0}
            if seeds:
                activation = activate(
                    graph=graph,
                    seeds=seeds,
                    max_steps=max(3, query.hops),
                    top_k=TOP_K,
                )
                selected = [node.id for node, _ in activation.fired]
            else:
                selected = bm25_top.copy()
        else:
            selected = bm25_top.copy()

        if not selected:
            selected = bm25_top

        cp_metrics = _metric_at_3(selected, query.expected_node_ids)
        cp_tokens = sum(
            len(graph.get_node(nid).content)
            for nid in selected
            if graph.get_node(nid) is not None
        )
        total_tokens_cp += cp_tokens

        if i > warmup and learn_graph:
            total_tokens_cp_phase2 += cp_tokens
            total_tokens_bm25_eval += bm25_tokens

        if learn_graph:
            expected = set(query.expected_node_ids)
            reward = 1.0 if set(selected[:TOP_K]) & expected else -1.0
            if activation is not None:
                learn(graph=graph, result=activation, outcome=reward, rate=0.12)

        per_query.append(
            {
                "index": i,
                "query": query.text,
                "hops": query.hops,
                "path": query.path,
                "expected": list(query.expected_node_ids),
                "bm25_ranking": bm25_top,
                "crabpath_ranking": selected[:TOP_K],
                "bm25_metrics": bm25_metrics,
                "crabpath_metrics": cp_metrics,
                "tokens_loaded": {"bm25": bm25_tokens, "crabpath": cp_tokens},
            }
        )

    if learn_graph:
        eval_count = max(1, len(per_query) - warmup)
        eval_window = per_query[warmup:]
        phase2_scores = {
            "recall@3": sum(q["crabpath_metrics"]["recall@3"] for q in eval_window) / eval_count,
            "ndcg@3": sum(q["crabpath_metrics"]["ndcg@3"] for q in eval_window) / eval_count,
            "mrr@3": sum(q["crabpath_metrics"]["mrr@3"] for q in eval_window) / eval_count,
        }
        avg_tokens_cp = total_tokens_cp_phase2 / float(eval_count)
        avg_tokens_bm25 = total_tokens_bm25_eval / float(eval_count)
        metric_key = "crabpath_phase2"
    else:
        phase2_scores = {
            "recall@3": sum(q["bm25_metrics"]["recall@3"] for q in per_query) / len(per_query),
            "ndcg@3": sum(q["bm25_metrics"]["ndcg@3"] for q in per_query) / len(per_query),
            "mrr@3": sum(q["bm25_metrics"]["mrr@3"] for q in per_query) / len(per_query),
        }
        avg_tokens_cp = 0.0
        avg_tokens_bm25 = total_tokens_bm25 / float(len(per_query))
        metric_key = "bm25"

    final_inhibitory = sum(1 for edge in graph.edges() if edge.weight < 0.0)
    inhibitory_formed = max(0, final_inhibitory - initial_inhibitory)

    return {
        "metric_key": metric_key,
        "metrics": phase2_scores,
        "avg_tokens_loaded": {"crabpath": avg_tokens_cp, "bm25": avg_tokens_bm25},
        "tokens_total": {"crabpath": total_tokens_cp, "bm25": total_tokens_bm25},
        "inhibitory_edges": {
            "initial": initial_inhibitory,
            "final": final_inhibitory,
            "formed": inhibitory_formed,
        },
        "edge_count_final": graph.edge_count,
        "edge_count_initial": initial_edge_count,
        "query_results": per_query,
        "queries_eval": eval_count if learn_graph else len(per_query),
    }


def _run_scale_configuration(size: int, label: str, edge_ratio: float, seed: int) -> dict[str, Any]:
    graph, docs, ordered_docs = _build_graph(size, edge_ratio, seed=seed)
    query_rng = random.Random(seed + 1)
    queries = _build_queries(graph=graph, specs=docs, rng=query_rng)

    crab = _run_query_suite(
        graph=graph,
        queries=queries,
        learn_graph=True,
        warmup=WARMUP_QUERIES,
    )
    bm25 = _run_query_suite(
        graph=copy.deepcopy(graph),
        queries=queries,
        learn_graph=False,
        warmup=0,
    )

    return {
        "size": size,
        "sparsity_label": label,
        "edge_ratio": edge_ratio,
        "seed": seed,
        "num_nodes": size,
        "num_docs": len(ordered_docs),
        "crabpath": {
            "metrics": crab["metrics"],
            "avg_tokens_loaded": crab["avg_tokens_loaded"],
            "query_results": crab["query_results"],
            "inhibitory_edges": crab["inhibitory_edges"],
            "edge_count_final": crab["edge_count_final"],
        },
        "bm25": {
            "metrics": bm25["metrics"],
            "avg_tokens_loaded": bm25["avg_tokens_loaded"],
            "query_results": bm25["query_results"],
            "inhibitory_edges": bm25["inhibitory_edges"],
            "edge_count_final": bm25["edge_count_final"],
        },
    }


def _run_niah_variant(size: int, rng: random.Random) -> dict[str, Any]:
    size_seed = rng.randint(0, 2**31 - 1)
    base_graph, base_docs, _ = _build_graph(size, 0.1, seed=size_seed)
    node_ids = list(base_docs.keys())
    results: list[dict[str, Any]] = []
    winner_by_depth: list[tuple[int, str, float]] = []

    for depth in range(1, 5):
        graph = copy.deepcopy(base_graph)
        specs = copy.deepcopy(base_docs)

        start_candidates = node_ids.copy()
        path_to_predecessor: list[str] = []
        for _ in range(500):
            start = rng.choice(start_candidates)
            path_to_predecessor = _draw_path(graph, rng, start, depth - 1)
            if path_to_predecessor and len(path_to_predecessor) == depth:
                break

        if not path_to_predecessor:
            start = rng.choice(start_candidates)
            path_to_predecessor = [start]
            if depth > 1 and len(node_ids) >= depth:
                extra = [n for n in node_ids if n != start][: depth - 1]
                if len(extra) == depth - 1:
                    path_to_predecessor = [start] + extra

        needle_id = f"needle_{size}_{depth}_{rng.randint(100, 999)}"
        needle_keywords = [f"needle_{size}_{depth}_{rng.randint(10, 999)}", f"needle_secret_{depth}"]
        needle_content = (
            "Needle evidence is a synthetic hidden artifact used for retrieval stress."
            f" The needle topic is {needle_keywords[0]} with fallback marker {needle_keywords[1]}."
        )
        graph.add_node(
            Node(
                id=needle_id,
                content=needle_content,
                cluster_id="needle",
                metadata={"is_needle": True},
            )
        )
        specs[needle_id] = DocSpec(
            node_id=needle_id,
            cluster="needle",
            keywords=needle_keywords,
            content=needle_content,
        )

        predecessor = path_to_predecessor[-1]
        _add_edge(graph, predecessor, needle_id, weight_hint=0.55, rng=rng)
        chain = path_to_predecessor + [needle_id]

        niah_queries: list[QueryCase] = []
        for q_idx in range(NIAH_QUERIES):
            base_terms = []
            for node_id in chain[:-1]:
                base_terms.append(_query_keywords(specs, node_id, rng))
            query_text = (
                "Trace the chain from "
                + " -> ".join(base_terms)
                + f" to resolve the final follow-up action for case {q_idx + 1}."
            )
            niah_queries.append(
                QueryCase(
                    text=query_text,
                    expected_node_ids=[needle_id],
                    hops=depth,
                    path=chain,
                )
            )

        crab = _run_query_suite(
            graph=graph,
            queries=niah_queries,
            learn_graph=True,
            warmup=NIAH_WARMUP,
        )
        bm25 = _run_query_suite(
            graph=copy.deepcopy(graph),
            queries=niah_queries,
            learn_graph=False,
            warmup=0,
        )

        cp_recall = crab["metrics"]["recall@3"]
        bm_recall = bm25["metrics"]["recall@3"]
        margin = abs(cp_recall - bm_recall)
        winner = "CRAB" if cp_recall >= bm_recall else "BM25"
        winner_by_depth.append((depth, winner, margin))

        results.append(
            {
                "depth": depth,
                "chain": chain,
                "crabpath": {
                    "metrics": crab["metrics"],
                    "avg_tokens_loaded": crab["avg_tokens_loaded"],
                    "inhibitory_edges": crab["inhibitory_edges"],
                },
                "bm25": {
                    "metrics": bm25["metrics"],
                    "avg_tokens_loaded": bm25["avg_tokens_loaded"],
                    "inhibitory_edges": bm25["inhibitory_edges"],
                },
                "query_results": crab["query_results"],
            }
        )

    return {"size": size, "depth_results": results, "winner_by_depth": winner_by_depth}


def _winner_cell(cp_recall: float, bm25_recall: float) -> str:
    if cp_recall >= bm25_recall:
        return f"CRAB+{cp_recall - bm25_recall:.3f}"
    return f"BM25+{bm25_recall - cp_recall:.3f}"


def _build_crossover_matrix(results: list[dict[str, Any]]) -> dict[str, dict[int, str]]:
    by_sparsity: dict[str, dict[int, dict[str, float]]] = {}
    for item in results:
        by_sparsity.setdefault(item["sparsity_label"], {})[item["size"]] = {
            "crab": item["crabpath"]["metrics"]["recall@3"],
            "bm25": item["bm25"]["metrics"]["recall@3"],
        }

    matrix: dict[str, dict[int, str]] = {}
    for label, by_size in by_sparsity.items():
        row: dict[int, str] = {}
        for size in GRAPH_SIZES:
            size_data = by_size[size]
            row[size] = _winner_cell(size_data["crab"], size_data["bm25"])
        matrix[label] = row
    return matrix


def _find_crossover(size_results: dict[str, dict[str, Any]]) -> int | None:
    for size in GRAPH_SIZES:
        metrics = size_results.get(size, {})
        if not metrics:
            continue
        if metrics["crab"] > metrics["bm25"]:
            return size
    return None


def _print_crossover_table(matrix: dict[str, dict[int, str]]) -> None:
    col_widths = [11] + [10 for _ in GRAPH_SIZES]
    header = ["Sparsity"] + [str(size) for size in GRAPH_SIZES]
    print("\n" + " | ".join(item.ljust(width) for item, width in zip(header, col_widths)))
    print("-" * sum(col_widths) + "-" * (3 * (len(col_widths) - 1)))
    for label in ["Dense", "Medium", "Sparse", "V.Sparse"]:
        row = [label] + [matrix[label][size] for size in GRAPH_SIZES]
        print(" | ".join(item.ljust(width) for item, width in zip(row, col_widths)))


def _print_niah_table(niah_results: list[dict[str, Any]]) -> None:
    print("\nNIAH (Needle in a Haystack) depth comparison @ edge_ratio=0.1")
    header = ["Size"] + [f"Depth {d}" for d in range(1, 5)]
    print(" | ".join(header))
    print("-" * 50)
    for result in niah_results:
        line = [str(result["size"])]
        for item in result["depth_results"]:
            depth = item["depth"]
            cp = item["crabpath"]["metrics"]["recall@3"]
            bm = item["bm25"]["metrics"]["recall@3"]
            line.append(_winner_cell(cp, bm))
        print(" | ".join(line))


def run_experiment() -> dict[str, Any]:
    rng = random.Random(SEED)
    all_results: list[dict[str, Any]] = []
    niah_inputs: list[dict[str, Any]] = []

    for size in GRAPH_SIZES:
        for label, edge_ratio in SPARSITY_LEVELS:
            cfg_seed = rng.randint(0, 2**31 - 1)
            config_result = _run_scale_configuration(
                size=size,
                label=label,
                edge_ratio=edge_ratio,
                seed=cfg_seed,
            )
            all_results.append(config_result)

        niah_inputs.append(_run_niah_variant(size=size, rng=random.Random(rng.randint(0, 2**31 - 1))))

    matrix = _build_crossover_matrix(all_results)
    _print_crossover_table(matrix)

    # Print crossover point by sparsity for phase-2 CrabPath Recall@3.
    print("\nCrossover point (CRAB recall > BM25 recall):")
    for label, _ in SPARSITY_LEVELS:
        size_to_compare = {
            size: {
                "crab": item["crabpath"]["metrics"]["recall@3"],
                "bm25": item["bm25"]["metrics"]["recall@3"],
            }
            for item in all_results
            if item["size"] in GRAPH_SIZES and item["sparsity_label"] == label
            for size in [item["size"]]
        }
        cross_size = _find_crossover(size_to_compare)
        print(f"{label}: {cross_size if cross_size is not None else 'none by 500'}")

    _print_niah_table(niah_inputs)

    return {
        "seed": SEED,
        "top_k": TOP_K,
        "sizes": GRAPH_SIZES,
        "sparsity_levels": [{"name": label, "edge_ratio": edge_ratio} for label, edge_ratio in SPARSITY_LEVELS],
        "configurations": all_results,
        "crossover_table": matrix,
        "niah": niah_inputs,
    }


def main() -> None:
    payload = run_experiment()
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"\nWrote detailed results to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
