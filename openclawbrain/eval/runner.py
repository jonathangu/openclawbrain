"""Baseline evaluation runner for OpenClawBrain."""

from __future__ import annotations

import csv
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openclawbrain.daemon import _handle_query
from openclawbrain.eval.baselines import (
    PointerChaseConfig,
    run_pointer_chase,
    run_vector_topk,
    run_vector_topk_rerank,
    try_load_bm25_reranker,
)
from openclawbrain.hasher import HashEmbedder
from openclawbrain.local_embedder import DEFAULT_LOCAL_MODEL, LocalEmbedder
from openclawbrain.route_model import RouteModel
from openclawbrain.store import load_state

VALID_CATEGORIES = {"decision-history", "project-boundary", "pointer", "ops"}


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
        if expected_keywords is None:
            expected_keywords = []
        if not isinstance(expected_keywords, list) or any(
            not isinstance(item, str) or not item.strip() for item in expected_keywords
        ):
            raise SystemExit(f"line {line_no}: expected_keywords must be a list of non-empty strings")
        loaded.append(
            EvalQuery(
                id=query_id.strip(),
                query=query_text.strip(),
                category=category,
                expected_keywords=tuple(item.strip() for item in expected_keywords),
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
    if model == "hash":
        return HashEmbedder().embed
    if model == "local":
        return LocalEmbedder(DEFAULT_LOCAL_MODEL).embed
    if model.startswith("local:"):
        return LocalEmbedder(embed_model.split(":", 1)[1].strip() or DEFAULT_LOCAL_MODEL).embed

    embedder_name = str(meta.get("embedder_name", "hash-v1")).strip().lower()
    if embedder_name.startswith("local:"):
        return LocalEmbedder(DEFAULT_LOCAL_MODEL).embed
    return HashEmbedder().embed


def _keyword_hit_ratio(prompt_context: str, expected_keywords: tuple[str, ...]) -> float | None:
    if not expected_keywords:
        return None
    text = prompt_context.lower()
    hits = sum(1 for key in expected_keywords if key.lower() in text)
    return hits / max(1, len(expected_keywords))


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


def _summarize_mode(rows: list[dict[str, object]]) -> dict[str, object]:
    latencies = [float(row["latency_ms"]) for row in rows if row.get("latency_ms") is not None]
    context_lens = [float(row["prompt_context_len"]) for row in rows]
    fired_counts = [float(row["fired_count"]) for row in rows]
    router_conf = [float(row.get("route_router_conf_mean", 0.0)) for row in rows]
    relevance_conf = [float(row.get("route_relevance_conf_mean", 0.0)) for row in rows]
    policy_disagreement = [float(row.get("route_policy_disagreement_mean", 0.0)) for row in rows]

    keyword_scores = [float(row["keyword_hit_ratio"]) for row in rows if row.get("keyword_hit_ratio") is not None]

    pointer_turns = [float(row["pointer_turns"]) for row in rows if row.get("pointer_turns") is not None]
    pointer_tool_calls = [float(row["pointer_tool_calls"]) for row in rows if row.get("pointer_tool_calls") is not None]
    pointer_tokens = [float(row["pointer_total_tokens"]) for row in rows if row.get("pointer_total_tokens") is not None]
    pointer_cost = [float(row["pointer_cost"]) for row in rows if row.get("pointer_cost") is not None]
    pointer_latency = [float(row["pointer_latency_ms"]) for row in rows if row.get("pointer_latency_ms") is not None]

    summary = {
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
    }

    if pointer_turns:
        summary["pointer_chase"] = {
            "turns_mean": _mean(pointer_turns),
            "tool_calls_mean": _mean(pointer_tool_calls),
            "total_tokens_mean": _mean(pointer_tokens),
            "cost_mean": _mean(pointer_cost),
            "latency_mean_ms": _mean(pointer_latency),
        }
    return summary


def _render_report(summary: dict[str, object], modes: list[str]) -> str:
    lines = [
        "# Baseline Evaluation Report",
        "",
        "## Summary",
        "",
        "| mode | status | queries | p50 latency (ms) | p95 latency (ms) | ctx mean | fired mean |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    mode_status = summary.get("mode_status", {}) if isinstance(summary.get("mode_status"), dict) else {}
    mode_summaries = summary.get("mode_summaries", {}) if isinstance(summary.get("mode_summaries"), dict) else {}
    for mode in modes:
        status = mode_status.get(mode, {}).get("status", "ok") if isinstance(mode_status.get(mode), dict) else "ok"
        mode_summary = mode_summaries.get(mode, {}) if isinstance(mode_summaries.get(mode), dict) else {}
        latency = mode_summary.get("latency", {}) if isinstance(mode_summary.get("latency"), dict) else {}
        context = mode_summary.get("context", {}) if isinstance(mode_summary.get("context"), dict) else {}
        lines.append(
            f"| {mode} | {status} | {mode_summary.get('query_count', 0)} | {latency.get('p50_ms', 0):.2f} | {latency.get('p95_ms', 0):.2f} | {context.get('prompt_context_len_mean', 0):.2f} | {context.get('fired_count_mean', 0):.2f} |"
        )

    lines.append("")
    lines.append("## Pointer-Chase Diagnostics")
    lines.append("")
    lines.append("| mode | turns mean | tool calls mean | total tokens mean | cost mean | latency mean (ms) |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for mode in modes:
        mode_summary = mode_summaries.get(mode, {}) if isinstance(mode_summaries.get(mode), dict) else {}
        pointer = mode_summary.get("pointer_chase", {}) if isinstance(mode_summary.get("pointer_chase"), dict) else {}
        if pointer:
            lines.append(
                f"| {mode} | {pointer.get('turns_mean', 0):.2f} | {pointer.get('tool_calls_mean', 0):.2f} | {pointer.get('total_tokens_mean', 0):.2f} | {pointer.get('cost_mean', 0):.4f} | {pointer.get('latency_mean_ms', 0):.2f} |"
            )
        else:
            lines.append(f"| {mode} | - | - | - | - | - |")

    return "\n".join(lines)


def run_baseline_suite(
    *,
    state_path: Path,
    queries_path: Path,
    modes: list[str],
    embed_model: str,
    route_model_path: Path | None,
    top_k: int,
    max_fired_nodes: int,
    max_prompt_context_chars: int,
    output_dir: Path,
    include_per_query: bool,
) -> dict[str, object]:
    graph, index, meta = load_state(str(state_path))
    embed_fn = _resolve_embed(meta, embed_model)
    queries = _load_queries(queries_path)

    route_model = None
    if any(mode == "learned" for mode in modes):
        model_path = route_model_path or (state_path.parent / "route_model.npz")
        if model_path.exists():
            route_model = RouteModel.load_npz(model_path)
        else:
            raise SystemExit(
                "selected modes require a learned route model but none was found; set --route-model or create route_model.npz"
            )

    target_projections = route_model.precompute_target_projections(index) if route_model is not None else {}
    reranker, reranker_reason = try_load_bm25_reranker()

    per_mode_rows: dict[str, list[dict[str, object]]] = {mode: [] for mode in modes}
    mode_status: dict[str, dict[str, object]] = {mode: {"status": "ok"} for mode in modes}
    null_store = _NullEventStore()

    for mode in modes:
        if mode == "vector_topk_rerank" and reranker is None:
            mode_status[mode] = {"status": "skipped", "reason": reranker_reason}
            continue

        for item in queries:
            start = time.perf_counter()
            if mode == "vector_topk":
                payload = run_vector_topk(
                    graph=graph,
                    index=index,
                    embed_fn=embed_fn,
                    query_text=item.query,
                    top_k=top_k,
                    max_prompt_context_chars=max_prompt_context_chars,
                    prompt_context_include_node_ids=True,
                )
            elif mode == "vector_topk_rerank":
                payload = run_vector_topk_rerank(
                    graph=graph,
                    index=index,
                    embed_fn=embed_fn,
                    query_text=item.query,
                    top_k=top_k,
                    max_prompt_context_chars=max_prompt_context_chars,
                    prompt_context_include_node_ids=True,
                    reranker=reranker,
                )
            elif mode == "pointer_chase":
                cfg = PointerChaseConfig(
                    top_k=top_k,
                    max_prompt_context_chars=max_prompt_context_chars,
                )
                payload = run_pointer_chase(
                    graph=graph,
                    index=index,
                    embed_fn=embed_fn,
                    query_text=item.query,
                    config=cfg,
                )
            elif mode in {"learned", "edge_sim_legacy"}:
                params = {
                    "query": item.query,
                    "top_k": top_k,
                    "max_prompt_context_chars": max_prompt_context_chars,
                    "max_context_chars": max_prompt_context_chars,
                    "max_fired_nodes": max_fired_nodes,
                    "prompt_context_include_node_ids": True,
                }
                if mode == "edge_sim_legacy":
                    params["route_mode"] = "edge+sim"
                else:
                    params["route_mode"] = "learned"
                payload = _handle_query(
                    graph=graph,
                    index=index,
                    meta=meta,
                    embed_fn=embed_fn,
                    params=params,
                    event_store=null_store,
                    learned_model=route_model,
                    target_projections=target_projections,
                )
            else:
                raise SystemExit(f"unknown mode: {mode}")

            elapsed_ms = (time.perf_counter() - start) * 1000.0
            fired_nodes = payload.get("fired_nodes", [])
            prompt_context = str(payload.get("prompt_context", ""))
            row = {
                "query_id": item.id,
                "category": item.category,
                "mode": mode,
                "latency_ms": round(elapsed_ms, 6),
                "prompt_context_len": int(payload.get("prompt_context_len", len(prompt_context))),
                "fired_count": len(fired_nodes) if isinstance(fired_nodes, list) else 0,
                "route_router_conf_mean": float(payload.get("route_router_conf_mean", 0.0)),
                "route_relevance_conf_mean": float(payload.get("route_relevance_conf_mean", 0.0)),
                "route_policy_disagreement_mean": float(payload.get("route_policy_disagreement_mean", 0.0)),
                "keyword_hit_ratio": _keyword_hit_ratio(prompt_context, item.expected_keywords),
                "pointer_turns": payload.get("pointer_turns"),
                "pointer_tool_calls": payload.get("pointer_tool_calls"),
                "pointer_total_tokens": payload.get("pointer_total_tokens"),
                "pointer_cost": payload.get("pointer_cost"),
                "pointer_latency_ms": payload.get("pointer_latency_ms"),
                "reranker_name": payload.get("reranker_name"),
            }
            per_mode_rows[mode].append(row)

    summary = {
        "state": str(state_path),
        "queries_path": str(queries_path),
        "modes": modes,
        "mode_status": mode_status,
        "query_count": len(queries),
        "mode_summaries": {
            mode: _summarize_mode(rows) for mode, rows in per_mode_rows.items() if rows
        },
    }
    if include_per_query:
        summary["per_query"] = per_mode_rows

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    csv_path = output_dir / "summary.csv"
    _write_csv(csv_path, modes, per_mode_rows)

    report_path = output_dir / "report.md"
    report_path.write_text(_render_report(summary, modes) + "\n", encoding="utf-8")

    summary["output"] = {
        "summary_json": str(summary_path),
        "summary_csv": str(csv_path),
        "report_md": str(report_path),
    }
    return summary


def _write_csv(path: Path, modes: list[str], rows: dict[str, list[dict[str, object]]]) -> None:
    fieldnames = [
        "query_id",
        "category",
        "mode",
        "latency_ms",
        "prompt_context_len",
        "fired_count",
        "keyword_hit_ratio",
        "route_router_conf_mean",
        "route_relevance_conf_mean",
        "route_policy_disagreement_mean",
        "pointer_turns",
        "pointer_tool_calls",
        "pointer_total_tokens",
        "pointer_cost",
        "pointer_latency_ms",
        "reranker_name",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for mode in modes:
            for row in rows.get(mode, []):
                writer.writerow({name: row.get(name) for name in fieldnames})
