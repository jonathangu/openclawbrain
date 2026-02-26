"""Hero simulations for the CrabPath paper.

Two scenarios:
1) Brain death followed by self-tuning recovery.
2) Twin brains with identical starting genome but different query diets.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crabpath._structural_utils import count_cross_file_edges, node_file_id  # noqa: E402
from crabpath.autotune import (  # noqa: E402
    HEALTH_TARGETS,
    GraphHealth,
    SafetyBounds,
    measure_health,
    self_tune,
)
from crabpath.decay import DecayConfig, apply_decay  # noqa: E402
from crabpath.graph import Graph  # noqa: E402
from crabpath.lifecycle_sim import (  # noqa: E402
    Query,
    make_mock_llm_all,
    make_mock_router,
    workspace_scenario,
)
from crabpath.mitosis import (  # noqa: E402
    MitosisConfig,
    MitosisState,
    bootstrap_workspace,
    mitosis_maintenance,
)
from crabpath.synaptogenesis import (  # noqa: E402
    SynaptogenesisConfig,
    SynaptogenesisState,
    classify_tier,
    decay_proto_edges,
    record_cofiring,
    record_skips,
)


def _metric_within_range(value: float, target: tuple[float | None, float | None]) -> bool:
    min_v, max_v = target
    if min_v is not None and value < min_v:
        return False
    if max_v is not None and value > max_v:
        return False
    return True


def _health_in_range_count(health: GraphHealth) -> int:
    return sum(
        1
        for metric, target in HEALTH_TARGETS.items()
        if _metric_within_range(float(getattr(health, metric)), target)
    )


def _health_to_dict(health: GraphHealth) -> dict[str, float | int]:
    return {
        "avg_nodes_fired_per_query": health.avg_nodes_fired_per_query,
        "cross_file_edge_pct": health.cross_file_edge_pct,
        "dormant_pct": health.dormant_pct,
        "reflex_pct": health.reflex_pct,
        "context_compression": health.context_compression,
        "proto_promotion_rate": health.proto_promotion_rate,
        "reconvergence_rate": health.reconvergence_rate,
        "orphan_nodes": health.orphan_nodes,
    }


def _query_stats_from_run(
    *,
    fired_counts: list[int],
    context_chars: list[int],
    promotions: int,
    proto_created: int,
    reconverged_families: int,
    queries_seen: int,
) -> dict[str, object]:
    return {
        "fired_counts": list(fired_counts),
        "context_chars": list(context_chars),
        "chars": list(context_chars),
        "avg_nodes_fired_per_query": (sum(fired_counts) / queries_seen) if queries_seen else 0.0,
        "promotions": promotions,
        "proto_created": proto_created,
        "reconverged_families": reconverged_families,
        "queries": queries_seen,
    }


def _simulate_queries(
    *,
    workspace_files: dict[str, str],
    queries: list[Query],
    decay_half_life: int,
    self_tune_start: int,
    checkpoints: set[int],
    sim_label: str,
) -> dict[str, Any]:
    graph = Graph()
    mitosis_state = MitosisState()
    synapse_state = SynaptogenesisState()
    mitosis_config = MitosisConfig()
    synapse_config = SynaptogenesisConfig()
    decay_config = DecayConfig(half_life_turns=decay_half_life)

    llm = make_mock_llm_all()
    router = make_mock_router()
    bootstrap_workspace(graph, workspace_files, llm, mitosis_state, mitosis_config)

    fired_counts: list[int] = []
    context_chars: list[int] = []
    promotions = 0
    proto_created = 0

    last_adjusted: dict[str, int] = {}
    tune_history = []
    events: list[dict[str, Any]] = []
    tune_records: list[dict[str, Any]] = []

    for qi, query in enumerate(queries, 1):
        candidates: list[tuple[str, float, str]] = []
        query_words = set(query.text.lower().split())

        for node in graph.nodes():
            node_words = set((node.content or "").lower().split())
            overlap = len(query_words & node_words)
            score = min(overlap / max(len(query_words), 1), 1.0)
            if score > 0.1:
                candidates.append((node.id, score, node.summary or node.content[:80]))

        candidates.sort(key=lambda c: c[1], reverse=True)
        candidates = candidates[:10]

        selected = router(query.text, candidates)
        cofire_result = record_cofiring(graph, selected, synapse_state, synapse_config)
        promotions += cofire_result["promoted"]
        proto_created += cofire_result["proto_created"]

        if selected:
            candidate_ids = [c[0] for c in candidates]
            record_skips(graph, selected[0], candidate_ids, selected, synapse_config)

        fired_count = len(selected)
        fired_counts.append(fired_count)
        selected_chars = sum(
            len(graph.get_node(node_id).content) if graph.get_node(node_id) else 0
            for node_id in selected
        )
        context_chars.append(selected_chars)

        if qi % 5 == 0:
            apply_decay(graph, turns_elapsed=5, config=decay_config)
            decay_proto_edges(synapse_state, synapse_config)

        if qi % 25 == 0:
            mitosis_maintenance(graph, llm, mitosis_state, mitosis_config)

        query_stats = _query_stats_from_run(
            fired_counts=fired_counts,
            context_chars=context_chars,
            promotions=promotions,
            proto_created=proto_created,
            reconverged_families=0,
            queries_seen=qi,
        )
        health = measure_health(graph, mitosis_state, query_stats)

        if qi >= self_tune_start:
            tuned_health, adjustments, changes = self_tune(
                graph=graph,
                state=mitosis_state,
                query_stats=query_stats,
                syn_config=synapse_config,
                decay_config=decay_config,
                mitosis_config=mitosis_config,
                cycle_number=qi,
                last_adjusted=last_adjusted,
                safety_bounds=SafetyBounds(),
            )
            health = tuned_health
            if adjustments or changes:
                tune_records.append(
                    {
                        "query": qi,
                        "query_text": query.text,
                        "health_before": _health_to_dict(health),
                        "adjustments": [
                            {
                                "metric": item.metric,
                                "reason": item.reason,
                                "suggested": item.suggested_change,
                            }
                            for item in adjustments
                        ],
                        "applied_changes": changes,
                    }
                )

            if adjustments:
                events.append({"query": qi, "type": "tune", "changes": changes})

        if qi in checkpoints:
            events.append(
                {
                    "query": qi,
                    "type": "checkpoint",
                    "query_text": query.text,
                    "health": _health_to_dict(health),
                    "in_range": _health_in_range_count(health),
                    "reflex_pct": health.reflex_pct,
                    "dormant_pct": health.dormant_pct,
                    "cross_file_edge_pct": health.cross_file_edge_pct,
                    "decay_half_life": decay_config.half_life_turns,
                    "promotion_threshold": synapse_config.promotion_threshold,
                    "hebbian_increment": synapse_config.hebbian_increment,
                    "skip_factor": synapse_config.skip_factor,
                    "reflex_threshold": synapse_config.reflex_threshold,
                    "total_edges": graph.edge_count,
                    "reflex_edges": len(
                        [
                            e
                            for e in graph.edges()
                            if classify_tier(e.weight, synapse_config) == "reflex"
                        ]
                    ),
                    "dormant_edges": len(
                        [
                            e
                            for e in graph.edges()
                            if classify_tier(e.weight, synapse_config) == "dormant"
                        ]
                    ),
                }
            )

        tune_history.append(health)

    final_health = (
        tune_history[-1]
        if tune_history
        else _health_to_dict(
            measure_health(
                graph,
                mitosis_state,
                _query_stats_from_run(
                    fired_counts=fired_counts,
                    context_chars=context_chars,
                    promotions=promotions,
                    proto_created=proto_created,
                    reconverged_families=0,
                    queries_seen=len(queries),
                ),
            )
        )
    )
    if hasattr(final_health, "__dict__"):
        final_health_dict: dict[str, float | int] = _health_to_dict(final_health)
    else:
        final_health_dict = final_health

    return {
        "label": sim_label,
        "final_health": final_health_dict,
        "final_config": {
            "decay_half_life": decay_config.half_life_turns,
            "promotion_threshold": synapse_config.promotion_threshold,
            "hebbian_increment": synapse_config.hebbian_increment,
            "skip_factor": synapse_config.skip_factor,
            "reflex_threshold": synapse_config.reflex_threshold,
        },
        "events": events,
        "tune_records": tune_records,
        "metrics_at_checkpoints": [
            event
            for event in events
            if event.get("type") == "checkpoint"
        ],
        "bootstrap": {
            "files": len(workspace_files),
            "nodes": graph.node_count,
            "edges": graph.edge_count,
            "families": len(mitosis_state.families),
        },
        "summary": {
            "nodes": graph.node_count,
            "edges": graph.edge_count,
            "cross_file_edges": count_cross_file_edges(graph),
            "proto_edges": len(synapse_state.proto_edges),
            "reflex_count": len(
                [e for e in graph.edges() if classify_tier(e.weight, synapse_config) == "reflex"]
            ),
            "dormant_count": len(
                [e for e in graph.edges() if classify_tier(e.weight, synapse_config) == "dormant"]
            ),
        },
    }


def _extract_reflex_paths(graph: Graph, synapse_config: SynaptogenesisConfig) -> list[str]:
    return sorted(
        f"{edge.source} -> {edge.target} ({edge.weight:.3f})"
        for edge in graph.edges()
        if classify_tier(edge.weight, synapse_config) == "reflex"
    )


def _extract_cross_file_edges(graph: Graph) -> list[dict[str, str | float]]:
    return [
        {
            "source": edge.source,
            "target": edge.target,
            "source_file": node_file_id(edge.source),
            "target_file": node_file_id(edge.target),
            "weight": round(edge.weight, 4),
        }
        for edge in graph.edges()
        if node_file_id(edge.source) != node_file_id(edge.target)
    ]


def _extract_top_edges(graph: Graph, limit: int = 5) -> list[dict[str, str | float]]:
    top_edges = sorted(graph.edges(), key=lambda edge: edge.weight, reverse=True)[:limit]
    return [
        {
            "source": edge.source,
            "target": edge.target,
            "weight": round(edge.weight, 4),
            "source_file": node_file_id(edge.source),
            "target_file": node_file_id(edge.target),
        }
        for edge in top_edges
    ]


def _build_twin_queries_focus(
    *,
    focus: str,
) -> list[Query]:
    queries: list[Query] = []
    if focus == "coding":
        templates = [
            "how do I use codex",
            "run codex with --yolo",
            "reset worktree after codex",
            "git worktree hygiene",
            "codex worktree cleanup",
            "automate debugging with codex",
            "codex coding task",
        ]
        for i in range(100):
            base = templates[i % len(templates)]
            queries.append(Query(f"{base} {i+1}", ["codex", "tools", "coding"]))
    else:
        templates = [
            "can I show API keys",
            "how to handle credentials safely",
            "who are you",
            "what are your values",
            "destructive action policy",
            "never expose tokens",
            "privilege hierarchy explained",
        ]
        for i in range(100):
            base = templates[i % len(templates)]
            queries.append(Query(f"{base} {i+1}", ["safety", "identity", "credentials"]))

    return queries


def _run_and_collect_graph(
    *,
    workspace_files: dict[str, str],
    queries: list[Query],
) -> dict[str, Any]:
    graph = Graph()
    mitosis_state = MitosisState()
    synapse_state = SynaptogenesisState()
    mitosis_config = MitosisConfig()
    synapse_config = SynaptogenesisConfig()
    decay_config = DecayConfig(half_life_turns=80)

    llm = make_mock_llm_all()
    router = make_mock_router()
    bootstrap_workspace(graph, workspace_files, llm, mitosis_state, mitosis_config)

    fired_counts: list[int] = []
    context_chars: list[int] = []
    promotions = 0
    proto_created = 0

    for qi, query in enumerate(queries, 1):
        candidates: list[tuple[str, float, str]] = []
        query_words = set(query.text.lower().split())

        for node in graph.nodes():
            node_words = set((node.content or "").lower().split())
            overlap = len(query_words & node_words)
            score = min(overlap / max(len(query_words), 1), 1.0)
            if score > 0.1:
                candidates.append((node.id, score, node.summary or node.content[:80]))

        candidates.sort(key=lambda c: c[1], reverse=True)
        candidates = candidates[:10]

        selected = router(query.text, candidates)
        cofire_result = record_cofiring(graph, selected, synapse_state, synapse_config)
        promotions += cofire_result["promoted"]
        proto_created += cofire_result["proto_created"]

        if selected:
            candidate_ids = [c[0] for c in candidates]
            record_skips(graph, selected[0], candidate_ids, selected, synapse_config)

        fired_counts.append(len(selected))
        selected_chars = sum(
            len(graph.get_node(node_id).content) if graph.get_node(node_id) else 0
            for node_id in selected
        )
        context_chars.append(selected_chars)

        if qi % 5 == 0:
            apply_decay(graph, turns_elapsed=5, config=decay_config)
            decay_proto_edges(synapse_state, synapse_config)

        if qi % 25 == 0:
            mitosis_maintenance(graph, llm, mitosis_state, mitosis_config)

    query_stats = _query_stats_from_run(
        fired_counts=fired_counts,
        context_chars=context_chars,
        promotions=promotions,
        proto_created=proto_created,
        reconverged_families=0,
        queries_seen=len(queries),
    )
    health = measure_health(graph, mitosis_state, query_stats)

    return {
        "graph": graph,
        "state": mitosis_state,
        "synapse_config": synapse_config,
        "decay_config": decay_config,
        "health": health,
        "cross_file_edges": _extract_cross_file_edges(graph),
        "reflex_paths": _extract_reflex_paths(graph, synapse_config),
        "top_edges": _extract_top_edges(graph),
    }


def _print_separator() -> None:
    print("-" * 92)


def _print_scenario1_report(result: dict[str, Any]) -> None:
    print("\nSCENARIO 1: Brain Death + Self-Healing Recovery")
    print("""\
  - Bootstrap: workspace_scenario files
  - Intentionally set decay_half_life=20 (overly aggressive)
  - 50 queries without self-tune (collapse phase), then 50 with self_tune active
""")
    _print_separator()

    checkpoints = result["metrics_at_checkpoints"]
    print("TIMELINE")
    for event in checkpoints:
        stage = "RECOVERY" if event["query"] >= 75 else "DECLINE"
        if event["query"] == 50:
            stage = "DEATH CHECK"
        if event["query"] == 75:
            stage = "EARLY RECOVER"
        if event["query"] == 100:
            stage = "RECOVERED"

        print(
            f"Q{event['query']:>3} [{stage}]  "
            f"health={_health_in_range_count(
                GraphHealth(**event['health'])
            )}/8 in-range  "
            f"dormant={event['dormant_pct']:.2f}% reflex={event['reflex_pct']:.2f}% "
            f"cross-file={event['cross_file_edge_pct']:.2f}%  "
            f"decay_half_life={event['decay_half_life']}  "
            f"config={event['promotion_threshold']}/{event['hebbian_increment']:.3f}/{event['skip_factor']:.2f}/{event['reflex_threshold']:.2f}"
        )

    _print_separator()
    final = result["final_health"]
    print(
        "Final health: "
        f"avg_nodes={final['avg_nodes_fired_per_query']:.2f}, "
        f"cross_file={final['cross_file_edge_pct']:.2f}%, "
        f"dormant={final['dormant_pct']:.2f}%, "
        f"reflex={final['reflex_pct']:.2f}%, "
        f"orphan_nodes={final['orphan_nodes']}"
    )

    print(f"Final config: {result['final_config']}")

    print("\nAUTOTUNER ADJUSTMENTS")
    if not result["tune_records"]:
        print("No configuration changes were applied")
    else:
        for record in result["tune_records"]:
            if not record["applied_changes"]:
                continue
            print(f"Q{record['query']:>3}:")
            for knob, change in record["applied_changes"].items():
                bounded = " (bounded)" if change.get("bounded") else ""
                print(
                    f"  {knob}: {change['before']} -> {change['after']}"
                    f" (delta {change['delta']:+.4f}){bounded}"
                )


def _print_scenario2_report(
    title: str,
    brain_a: dict[str, Any],
    brain_b: dict[str, Any],
) -> None:
    print("\nSCENARIO 2: Twin Brains — Same Files, Different Experiences")
    _print_separator()

    print(
        "Brain A: coding/tools-heavy (100 queries)\n"
        "Brain B: safety/identity-heavy (100 queries)"
    )

    a_health = _health_to_dict(brain_a["health"])
    b_health = _health_to_dict(brain_b["health"])

    print("\nMETRICS COMPARISON")
    print(f"  Dormant %: A={a_health['dormant_pct']:.2f}%  B={b_health['dormant_pct']:.2f}%")
    print(
        f"  Cross-file edges: A={len(brain_a['cross_file_edges'])}  "
        f"B={len(brain_b['cross_file_edges'])}"
    )

    # Side-by-side table style output
    print("\nSIDEBY-SIDE COMPARISON")
    left_w = 70
    right_w = 70

    def _fmt_rows(values: list[str], width: int) -> list[str]:
        out: list[str] = []
        current = ""
        for value in values:
            if len(current) + len(value) + 2 > width and current:
                out.append(current.rstrip(", "))
                current = ""
            current += value + ", "
        if current:
            out.append(current.rstrip(", "))
        if not out:
            out.append("<none>")
        return out

    a_reflex = _fmt_rows(brain_a["reflex_paths"], left_w)
    b_reflex = _fmt_rows(brain_b["reflex_paths"], right_w)

    a_cf = [
        f"{edge['source_file']}->{edge['target_file']} ({edge['weight']})"
        for edge in sorted(brain_a["cross_file_edges"], key=lambda e: e["weight"], reverse=True)
    ]
    b_cf = [
        f"{edge['source_file']}->{edge['target_file']} ({edge['weight']})"
        for edge in sorted(brain_b["cross_file_edges"], key=lambda e: e["weight"], reverse=True)
    ]
    a_cf_rows = _fmt_rows(a_cf, left_w)
    b_cf_rows = _fmt_rows(b_cf, right_w)

    a_top = [
        (
            f"{edge['source_file']}->{edge['target_file']} "
            f"{edge['source']}->{edge['target']} ({edge['weight']})"
        )
        for edge in brain_a["top_edges"]
    ]
    b_top = [
        (
            f"{edge['source_file']}->{edge['target_file']} "
            f"{edge['source']}->{edge['target']} ({edge['weight']})"
        )
        for edge in brain_b["top_edges"]
    ]
    a_top_rows = _fmt_rows(a_top, left_w)
    b_top_rows = _fmt_rows(b_top, right_w)

    header = (
        f"{'Metric / item':<34}{'Brain A'.ljust(left_w)} {'Brain B'.ljust(right_w)}"
    )
    print(f"\n{header}")
    print("-" * len(header))

    def _print_blocked_rows(label: str, left_rows: list[str], right_rows: list[str]) -> None:
        max_rows = max(len(left_rows), len(right_rows))
        for i in range(max_rows):
            left = left_rows[i] if i < len(left_rows) else ""
            right = right_rows[i] if i < len(right_rows) else ""
            prefix = label if i == 0 else ""
            print(f"{prefix:<34}{left:<{left_w}} {right:<{right_w}}")

    _print_blocked_rows("Reflex paths", a_reflex, b_reflex)
    _print_blocked_rows("Cross-file edges", a_cf_rows, b_cf_rows)
    _print_blocked_rows("Top-5 strongest", a_top_rows, b_top_rows)


def _serialize_graph_state(graph: Graph, checkpoints: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "nodes": [
            {
                "id": node.id,
                "content": node.content,
                "summary": node.summary,
                "metadata": node.metadata,
            }
            for node in graph.nodes()
        ],
        "edges": [
            {
                "source": edge.source,
                "target": edge.target,
                "weight": edge.weight,
                "created_by": edge.created_by,
                "follow_count": edge.follow_count,
                "skip_count": edge.skip_count,
            }
            for edge in graph.edges()
        ],
        "checkpoints": checkpoints,
    }


def main() -> None:
    workspace_files, base_queries = workspace_scenario()
    scenario1_queries = (base_queries * 2)[:100]

    scenario1 = _simulate_queries(
        workspace_files=workspace_files,
        queries=scenario1_queries,
        decay_half_life=20,
        self_tune_start=51,
        checkpoints={25, 50, 75, 100},
        sim_label="scenario_1",
    )

    coding_queries = _build_twin_queries_focus(focus="coding")
    safety_queries = _build_twin_queries_focus(focus="safety")

    brain_a = _run_and_collect_graph(workspace_files=workspace_files, queries=coding_queries)
    brain_b = _run_and_collect_graph(workspace_files=workspace_files, queries=safety_queries)

    _print_scenario1_report(scenario1)
    _print_scenario2_report("Twin brains", brain_a, brain_b)

    out_payload = {
        "scenario_1": scenario1,
        "scenario_2": {
            "brain_a": {
                "label": "coding",
                "health": _health_to_dict(brain_a["health"]),
                "cross_file_edges": brain_a["cross_file_edges"],
                "reflex_paths": brain_a["reflex_paths"],
                "top_edges": brain_a["top_edges"],
                "summary": {
                    "nodes": brain_a["graph"].node_count,
                    "edges": brain_a["graph"].edge_count,
                    "cross_file_edge_count": len(brain_a["cross_file_edges"]),
                },
                "graph": _serialize_graph_state(
                    brain_a["graph"], [{"query": 100, "health": _health_to_dict(brain_a["health"])}]
                ),
            },
            "brain_b": {
                "label": "safety_identity",
                "health": _health_to_dict(brain_b["health"]),
                "cross_file_edges": brain_b["cross_file_edges"],
                "reflex_paths": brain_b["reflex_paths"],
                "top_edges": brain_b["top_edges"],
                "summary": {
                    "nodes": brain_b["graph"].node_count,
                    "edges": brain_b["graph"].edge_count,
                    "cross_file_edge_count": len(brain_b["cross_file_edges"]),
                },
                "graph": _serialize_graph_state(
                    brain_b["graph"], [{"query": 100, "health": _health_to_dict(brain_b["health"])}]
                ),
            },
            "comparison": {
                "dormant_pct": {
                    "a": _health_to_dict(brain_a["health"])["dormant_pct"],
                    "b": _health_to_dict(brain_b["health"])["dormant_pct"],
                    "delta": _health_to_dict(brain_a["health"])["dormant_pct"]
                    - _health_to_dict(brain_b["health"])["dormant_pct"],
                },
                "cross_file_count": {
                    "a": len(brain_a["cross_file_edges"]),
                    "b": len(brain_b["cross_file_edges"]),
                    "delta": len(brain_a["cross_file_edges"]) - len(brain_b["cross_file_edges"]),
                },
            },
        },
    }

    output_path = ROOT / "scripts" / "hero_sim_results.json"
    output_path.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")
    print(f"\nDetailed results written to {output_path}")


if __name__ == "__main__":
    main()
