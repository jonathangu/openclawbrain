#!/usr/bin/env python3
"""Evaluation + ablation harness for OpenClawBrain retrieval modes.

Outputs per-mode JSON metrics suitable for paper tables.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openclawbrain.daemon import _handle_query
from openclawbrain.hasher import HashEmbedder
from openclawbrain.local_embedder import DEFAULT_LOCAL_MODEL, LocalEmbedder
from openclawbrain.prompt_context import build_prompt_context_ranked_with_stats
from openclawbrain.route_model import RouteModel
from openclawbrain.store import load_state

VALID_CATEGORIES = {"decision-history", "project-boundary", "pointer", "ops"}
VALID_MODES = {
    "vector_only",
    "graph_prior_only",
    "qtsim_only",
    "learned",
    "edge_sim_legacy",
}


class _NullEventStore:
    def append(self, event: dict[str, object]) -> None:  # noqa: D401
        """Discard events so eval does not mutate journal files."""
        _ = event


@dataclass(frozen=True)
class EvalQuery:
    id: str
    query: str
    category: str
    expected_keywords: tuple[str, ...] = ()
    acceptable_node_ids: tuple[str, ...] = ()
    required_node_ids: tuple[str, ...] = ()


def _parse_string_list(payload: object, *, line_no: int, field_name: str) -> tuple[str, ...]:
    if payload is None:
        return ()
    if not isinstance(payload, list) or any(
        not isinstance(item, str) or not item.strip() for item in payload
    ):
        raise SystemExit(f"line {line_no}: {field_name} must be a list of non-empty strings")
    return tuple(item.strip() for item in payload)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if pct <= 0:
        return float(min(values))
    if pct >= 100:
        return float(max(values))
    ordered = sorted(float(v) for v in values)
    rank = (len(ordered) - 1) * (pct / 100.0)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _distribution(values: list[float]) -> dict[str, int]:
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0000001]
    labels = ["[0.0,0.2)", "[0.2,0.4)", "[0.4,0.6)", "[0.6,0.8)", "[0.8,1.0]"]
    counts = {label: 0 for label in labels}
    for raw in values:
        value = max(0.0, min(1.0, float(raw)))
        for idx in range(len(bins) - 1):
            if bins[idx] <= value < bins[idx + 1]:
                counts[labels[idx]] += 1
                break
    return counts


def _load_queries(path: Path) -> list[EvalQuery]:
    if not path.exists():
        raise SystemExit(f"queries file not found: {path}")
    loaded: list[EvalQuery] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        raw = line.strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"invalid JSON on line {line_no}: {exc}") from exc
        if not isinstance(payload, dict):
            raise SystemExit(f"line {line_no}: expected JSON object")
        query_id = payload.get("id")
        query_text = payload.get("query")
        category = payload.get("category")
        if not isinstance(query_id, str) or not query_id.strip():
            raise SystemExit(f"line {line_no}: id must be a non-empty string")
        if not isinstance(query_text, str) or not query_text.strip():
            raise SystemExit(f"line {line_no}: query must be a non-empty string")
        if not isinstance(category, str) or category not in VALID_CATEGORIES:
            raise SystemExit(
                f"line {line_no}: category must be one of {sorted(VALID_CATEGORIES)}"
            )
        expected_keywords = payload.get("expected_keywords", [])
        acceptable_node_ids = payload.get("acceptable_node_ids", [])
        required_node_ids = payload.get("required_node_ids", [])
        loaded.append(
            EvalQuery(
                id=query_id.strip(),
                query=query_text.strip(),
                category=category,
                expected_keywords=_parse_string_list(
                    expected_keywords,
                    line_no=line_no,
                    field_name="expected_keywords",
                ),
                acceptable_node_ids=_parse_string_list(
                    acceptable_node_ids,
                    line_no=line_no,
                    field_name="acceptable_node_ids",
                ),
                required_node_ids=_parse_string_list(
                    required_node_ids,
                    line_no=line_no,
                    field_name="required_node_ids",
                ),
            )
        )
    if not loaded:
        raise SystemExit("queries file has no valid rows")
    seen_ids: set[str] = set()
    for item in loaded:
        if item.id in seen_ids:
            raise SystemExit(f"duplicate query id: {item.id}")
        seen_ids.add(item.id)
    return loaded


def _resolve_embed(meta: dict[str, object], embed_model: str) -> Any:
    model = embed_model.strip().lower()
    configured_dim = meta.get("embedder_dim")
    hash_dim = configured_dim if isinstance(configured_dim, int) and configured_dim > 0 else 1024
    if model == "hash":
        return HashEmbedder(dim=hash_dim).embed
    if model.startswith("hash:"):
        try:
            requested_dim = int(model.split(":", 1)[1].strip())
        except ValueError as exc:
            raise SystemExit("hash embed-model must be hash or hash:<positive-dim>") from exc
        if requested_dim <= 0:
            raise SystemExit("hash embed-model dimension must be > 0")
        return HashEmbedder(dim=requested_dim).embed
    if model == "local":
        return LocalEmbedder(DEFAULT_LOCAL_MODEL).embed
    if model.startswith("local:"):
        return LocalEmbedder(embed_model.split(":", 1)[1].strip() or DEFAULT_LOCAL_MODEL).embed

    embedder_name = str(meta.get("embedder_name", "hash-v1")).strip().lower()
    if embedder_name.startswith("local:"):
        return LocalEmbedder(DEFAULT_LOCAL_MODEL).embed
    return HashEmbedder(dim=hash_dim).embed


def _run_vector_only(
    *,
    graph: Any,
    index: Any,
    embed_fn: Any,
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


def _mode_params(mode: str, *, top_k: int, max_prompt_context_chars: int, max_fired_nodes: int) -> dict[str, object]:
    params: dict[str, object] = {
        "top_k": top_k,
        "max_prompt_context_chars": max_prompt_context_chars,
        "max_context_chars": max_prompt_context_chars,
        "max_fired_nodes": max_fired_nodes,
        "prompt_context_include_node_ids": True,
    }
    if mode == "edge_sim_legacy":
        params.update({"route_mode": "edge+sim"})
    elif mode in {"learned", "graph_prior_only", "qtsim_only"}:
        params.update({"route_mode": "learned"})
        if mode == "graph_prior_only":
            params.update(
                {
                    "debug_allow_confidence_override": True,
                    "router_conf_override": 0.0,
                }
            )
        elif mode == "qtsim_only":
            params.update(
                {
                    "debug_allow_confidence_override": True,
                    "router_conf_override": 1.0,
                    "relevance_conf_override": 1.0,
                }
            )
    return params


def _keyword_hit_ratio(prompt_context: str, expected_keywords: tuple[str, ...]) -> float | None:
    if not expected_keywords:
        return None
    text = prompt_context.lower()
    hits = sum(1 for key in expected_keywords if key.lower() in text)
    return hits / max(1, len(expected_keywords))


def _prompt_node_ids(payload: dict[str, object]) -> list[str]:
    raw_ids = payload.get("prompt_context_included_node_ids")
    if isinstance(raw_ids, list):
        return [str(node_id) for node_id in raw_ids if isinstance(node_id, str) and node_id]
    fired_nodes = payload.get("fired_nodes")
    if isinstance(fired_nodes, list):
        return [str(node_id) for node_id in fired_nodes if isinstance(node_id, str) and node_id]
    return []


def _required_node_coverage(
    prompt_node_ids: list[str],
    required_node_ids: tuple[str, ...],
) -> float | None:
    if not required_node_ids:
        return None
    present = sum(1 for node_id in required_node_ids if node_id in prompt_node_ids)
    return present / max(1, len(required_node_ids))


def _acceptable_node_hit(
    prompt_node_ids: list[str],
    acceptable_node_ids: tuple[str, ...],
) -> float | None:
    if not acceptable_node_ids:
        return None
    return 1.0 if any(node_id in prompt_node_ids for node_id in acceptable_node_ids) else 0.0


def _target_success(
    *,
    required_node_coverage: float | None,
    acceptable_node_hit: float | None,
) -> float | None:
    if required_node_coverage is None and acceptable_node_hit is None:
        return None
    required_ok = required_node_coverage is None or required_node_coverage >= 0.999999
    acceptable_ok = acceptable_node_hit is None or acceptable_node_hit >= 0.999999
    return 1.0 if required_ok and acceptable_ok else 0.0


def _summarize_mode(rows: list[dict[str, object]]) -> dict[str, object]:
    latencies = [float(row["latency_ms"]) for row in rows]
    context_lens = [float(row["prompt_context_len"]) for row in rows]
    fired_counts = [float(row["fired_count"]) for row in rows]
    router_conf = [float(row["route_router_conf_mean"]) for row in rows]
    relevance_conf = [float(row["route_relevance_conf_mean"]) for row in rows]
    policy_disagreement = [float(row["route_policy_disagreement_mean"]) for row in rows]

    keyword_scores = [float(row["keyword_hit_ratio"]) for row in rows if row["keyword_hit_ratio"] is not None]
    required_coverage = [
        float(row["required_node_coverage"])
        for row in rows
        if row.get("required_node_coverage") is not None
    ]
    acceptable_hits = [
        float(row["acceptable_node_hit"])
        for row in rows
        if row.get("acceptable_node_hit") is not None
    ]
    target_success = [
        float(row["target_success"])
        for row in rows
        if row.get("target_success") is not None
    ]
    return {
        "query_count": len(rows),
        "latency": {
            "p50_ms": _percentile(latencies, 50),
            "p95_ms": _percentile(latencies, 95),
            "mean_ms": _mean(latencies),
        },
        "context": {
            "prompt_context_len_mean": _mean(context_lens),
            "prompt_context_len_median": _median(context_lens),
            "fired_count_mean": _mean(fired_counts),
            "fired_count_median": _median(fired_counts),
        },
        "routing_diagnostics": {
            "route_router_conf_mean": _mean(router_conf),
            "route_relevance_conf_mean": _mean(relevance_conf),
            "route_policy_disagreement_mean": _mean(policy_disagreement),
        },
        "qtsim_proxies": {
            "route_router_conf_distribution": _distribution(router_conf),
            "fraction_router_conf_gt_0_7": (
                sum(1 for value in router_conf if value > 0.7) / len(router_conf) if router_conf else 0.0
            ),
        },
        "keyword_hit_ratio_mean": _mean(keyword_scores) if keyword_scores else None,
        "ground_truth": {
            "queries_with_required_node_ids": len(required_coverage),
            "queries_with_acceptable_node_ids": len(acceptable_hits),
            "queries_with_targets": len(target_success),
            "required_node_coverage_mean": _mean(required_coverage) if required_coverage else None,
            "acceptable_node_hit_rate": _mean(acceptable_hits) if acceptable_hits else None,
            "target_success_rate": _mean(target_success) if target_success else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluation and ablation matrix for OpenClawBrain.")
    parser.add_argument("--state", required=True, help="Path to state.json")
    parser.add_argument(
        "--queries",
        default=str(Path(__file__).resolve().parent / "queries.jsonl"),
        help="Path to query JSONL dataset",
    )
    parser.add_argument(
        "--modes",
        default="vector_only,graph_prior_only,qtsim_only,learned,edge_sim_legacy",
        help="Comma-separated modes",
    )
    parser.add_argument("--route-model", help="Optional path to route_model.npz")
    parser.add_argument("--embed-model", default="auto", help="auto|hash|local|local:<model>")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--max-fired-nodes", type=int, default=30)
    parser.add_argument("--max-prompt-context-chars", type=int, default=30000)
    parser.add_argument("--output", help="Write summary JSON to this file")
    parser.add_argument("--print-per-query", action="store_true", help="Include per-query rows in output JSON")
    args = parser.parse_args()

    if args.top_k <= 0:
        raise SystemExit("--top-k must be > 0")
    if args.max_fired_nodes <= 0:
        raise SystemExit("--max-fired-nodes must be > 0")
    if args.max_prompt_context_chars <= 0:
        raise SystemExit("--max-prompt-context-chars must be > 0")

    selected_modes = [part.strip() for part in args.modes.split(",") if part.strip()]
    if not selected_modes:
        raise SystemExit("--modes must include at least one mode")
    unknown = [mode for mode in selected_modes if mode not in VALID_MODES]
    if unknown:
        raise SystemExit(f"unknown mode(s): {unknown}; valid: {sorted(VALID_MODES)}")

    state_path = Path(args.state).expanduser()
    graph, index, meta = load_state(str(state_path))
    embed_fn = _resolve_embed(meta, args.embed_model)
    queries = _load_queries(Path(args.queries).expanduser())

    route_model_path = Path(args.route_model).expanduser() if args.route_model else state_path.parent / "route_model.npz"
    learned_model = RouteModel.load_npz(route_model_path) if route_model_path.exists() else None
    if any(mode in {"learned", "graph_prior_only", "qtsim_only"} for mode in selected_modes) and learned_model is None:
        raise SystemExit(
            "selected modes require a learned route model but none was found; set --route-model or create route_model.npz"
        )
    target_projections = learned_model.precompute_target_projections(index) if learned_model is not None else {}

    per_mode_rows: dict[str, list[dict[str, object]]] = {mode: [] for mode in selected_modes}
    null_store = _NullEventStore()

    for mode in selected_modes:
        params_template = _mode_params(
            mode,
            top_k=args.top_k,
            max_prompt_context_chars=args.max_prompt_context_chars,
            max_fired_nodes=args.max_fired_nodes,
        )
        for item in queries:
            start = time.perf_counter()
            if mode == "vector_only":
                payload = _run_vector_only(
                    graph=graph,
                    index=index,
                    embed_fn=embed_fn,
                    query_text=item.query,
                    top_k=args.top_k,
                    max_prompt_context_chars=args.max_prompt_context_chars,
                    prompt_context_include_node_ids=True,
                )
            else:
                params = {"query": item.query, **params_template}
                payload = _handle_query(
                    graph=graph,
                    index=index,
                    meta=meta,
                    embed_fn=embed_fn,
                    params=params,
                    event_store=null_store,
                    learned_model=learned_model,
                    target_projections=target_projections,
                )
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            fired_nodes = payload.get("fired_nodes", [])
            prompt_context = str(payload.get("prompt_context", ""))
            prompt_node_ids = _prompt_node_ids(payload)
            acceptable_node_hit = _acceptable_node_hit(prompt_node_ids, item.acceptable_node_ids)
            required_node_coverage = _required_node_coverage(prompt_node_ids, item.required_node_ids)
            row = {
                "query_id": item.id,
                "category": item.category,
                "mode": mode,
                "latency_ms": round(elapsed_ms, 6),
                "prompt_context_len": int(payload.get("prompt_context_len", len(prompt_context))),
                "fired_count": len(fired_nodes) if isinstance(fired_nodes, list) else 0,
                "prompt_context_included_node_ids": prompt_node_ids,
                "acceptable_node_ids": list(item.acceptable_node_ids),
                "required_node_ids": list(item.required_node_ids),
                "route_router_conf_mean": float(payload.get("route_router_conf_mean", 0.0)),
                "route_relevance_conf_mean": float(payload.get("route_relevance_conf_mean", 0.0)),
                "route_policy_disagreement_mean": float(payload.get("route_policy_disagreement_mean", 0.0)),
                "keyword_hit_ratio": _keyword_hit_ratio(prompt_context, item.expected_keywords),
                "acceptable_node_hit": acceptable_node_hit,
                "required_node_coverage": required_node_coverage,
                "target_success": _target_success(
                    required_node_coverage=required_node_coverage,
                    acceptable_node_hit=acceptable_node_hit,
                ),
            }
            per_mode_rows[mode].append(row)

    summary = {
        "state": str(state_path),
        "queries_path": str(Path(args.queries).expanduser()),
        "route_model_path": str(route_model_path),
        "query_count": len(queries),
        "modes": selected_modes,
        "mode_summaries": {mode: _summarize_mode(rows) for mode, rows in per_mode_rows.items()},
    }
    if args.print_per_query:
        summary["per_query"] = per_mode_rows

    rendered = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
