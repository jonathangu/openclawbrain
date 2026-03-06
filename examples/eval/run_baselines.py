#!/usr/bin/env python3
"""Industry-standard baseline suite runner for OpenClawBrain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from openclawbrain.eval.runner import run_baseline_suite

VALID_MODES = {
    "vector_topk",
    "vector_only",
    "vector_topk_rerank",
    "pointer_chase",
    "graph_prior_only",
    "qtsim_only",
    "learned",
    "edge_sim_legacy",
}
MODE_ALIASES = {"vector_only": "vector_topk"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run industry-standard baseline suite for OpenClawBrain.")
    parser.add_argument("--state", required=True, help="Path to state.json")
    parser.add_argument(
        "--queries",
        default=str(Path(__file__).resolve().parent / "queries.jsonl"),
        help="Path to query JSONL dataset",
    )
    parser.add_argument(
        "--modes",
        default="vector_topk,vector_topk_rerank,pointer_chase,learned",
        help="Comma-separated modes",
    )
    parser.add_argument("--route-model", help="Optional path to route_model.npz")
    parser.add_argument("--embed-model", default="auto", help="auto|hash|local|local:<model>")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--route-top-k", type=int, default=5)
    parser.add_argument("--max-fired-nodes", type=int, default=30)
    parser.add_argument("--max-prompt-context-chars", type=int, default=30000)
    parser.add_argument(
        "--output-dir",
        default=str(Path("scratch") / "industry-baselines" / "latest"),
        help="Directory for summary.json/summary.csv/report.md",
    )
    parser.add_argument("--print-per-query", action="store_true", help="Include per-query rows in output JSON")
    args = parser.parse_args()

    if args.top_k <= 0:
        raise SystemExit("--top-k must be > 0")
    if args.route_top_k <= 0:
        raise SystemExit("--route-top-k must be > 0")
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

    normalized_modes: list[str] = []
    for mode in selected_modes:
        resolved = MODE_ALIASES.get(mode, mode)
        if resolved not in normalized_modes:
            normalized_modes.append(resolved)

    summary = run_baseline_suite(
        state_path=Path(args.state).expanduser(),
        queries_path=Path(args.queries).expanduser(),
        modes=normalized_modes,
        embed_model=args.embed_model,
        route_model_path=Path(args.route_model).expanduser() if args.route_model else None,
        top_k=args.top_k,
        route_top_k=args.route_top_k,
        max_fired_nodes=args.max_fired_nodes,
        max_prompt_context_chars=args.max_prompt_context_chars,
        output_dir=Path(args.output_dir).expanduser(),
        include_per_query=args.print_per_query,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
