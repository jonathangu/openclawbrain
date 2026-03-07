#!/usr/bin/env python3
"""Workflow-shaped proof harness for OpenClawBrain.

This simulation approximates recurring OpenClaw workflows:
- incident history lookups
- deploy-after-incident checks
- customer status updates
- on-call/dashboard recall

It intentionally separates:
- cold-start graph priors from ongoing/history ingestion
- background teacher labels that train the runtime route_fn

Outputs:
- workflow_state.json
- train_traces.jsonl / train_labels.jsonl
- eval_queries.jsonl
- route_model_epoch_*.npz
- learning_curve.csv
- per_query_matrix.csv / per_query_matrix.md
- summary.json
- report.md
- worked_example.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

from openclawbrain import Edge, Graph, Node, VectorIndex, save_state
from openclawbrain.daemon import _handle_query
from openclawbrain.eval.runner import run_baseline_suite
from openclawbrain.eval.baselines import PointerChaseConfig, run_pointer_chase, run_vector_topk
from openclawbrain.hasher import HashEmbedder
from openclawbrain.labels import from_teacher_output, write_labels_jsonl
from openclawbrain.route_model import RouteModel
from openclawbrain.store import load_state
from openclawbrain.trace import RouteCandidate, RouteDecisionPoint, RouteTrace, route_trace_to_json
from openclawbrain.train_route_model import train_route_model


@dataclass(frozen=True)
class Scenario:
    key: str
    category: str
    prototype_text: str
    hub_id: str
    chosen_target_id: str
    required_node_ids: tuple[str, ...]
    expected_keywords: tuple[str, ...]
    train_queries: tuple[str, ...]
    eval_query_id: str
    eval_query: str


@dataclass(frozen=True)
class NodeSpec:
    node_id: str
    content: str
    file_path: str
    start_line: int
    end_line: int


@dataclass(frozen=True)
class EdgeSpec:
    target_id: str
    weight: float
    relevance: float


class _NullEventStore:
    def append(self, event: dict[str, object]) -> None:
        _ = event


MODE_ORDER = ("vector_topk", "pointer_chase", "graph_prior_only", "learned")

NOISE_TEXT = (
    "finance budget hiring vacation policy compliance legal invoice recruiting travel roadmap"
)

NODE_SPECS = {
    "hub::release": NodeSpec(
        node_id="hub::release",
        content=(
            "Release hub.\n"
            "- routes deploy, rollback, and canary decisions\n"
            "- used when the operator asks how to re-run or stop a rollout"
        ),
        file_path="workspace/ops/release_hub.md",
        start_line=1,
        end_line=3,
    ),
    "hub::incident": NodeSpec(
        node_id="hub::incident",
        content=(
            "Incident hub.\n"
            "- routes incident history, dashboards, and pager ownership\n"
            "- used when the operator asks what happened during a prior failure"
        ),
        file_path="workspace/ops/incident_hub.md",
        start_line=1,
        end_line=3,
    ),
    "hub::customer": NodeSpec(
        node_id="hub::customer",
        content=(
            "Customer hub.\n"
            "- routes external status wording and incident-specific customer notes\n"
            "- used during auth or customer-facing incidents"
        ),
        file_path="workspace/ops/customer_hub.md",
        start_line=1,
        end_line=3,
    ),
    "doc::generic_hotfix_runbook": NodeSpec(
        node_id="doc::generic_hotfix_runbook",
        content=(
            "Generic hotfix runbook.\n"
            "1. Confirm CI green.\n"
            "2. Deploy to staging.\n"
            "3. Run smoke tests.\n"
            "4. Promote and monitor."
        ),
        file_path="workspace/runbooks/hotfix.md",
        start_line=1,
        end_line=5,
    ),
    "doc::rollback_gate": NodeSpec(
        node_id="doc::rollback_gate",
        content=(
            "Rollback gate.\n"
            "Do not rerun deploy until the feature flag is off, the canary error rate is below 1%, "
            "and staging smoke tests pass."
        ),
        file_path="workspace/runbooks/rollback_gate.md",
        start_line=1,
        end_line=2,
    ),
    "doc::payments_incident_2026_02_14": NodeSpec(
        node_id="doc::payments_incident_2026_02_14",
        content=(
            "Payments canary incident on 2026-02-14.\n"
            "Recovery used three actions in order: disable the `payments_v3` flag, freeze deploys, "
            "then apply the rollback gate before re-running checkout."
        ),
        file_path="workspace/incidents/payments-2026-02-14.md",
        start_line=1,
        end_line=2,
    ),
    "doc::auth_incident_2026_02_02": NodeSpec(
        node_id="doc::auth_incident_2026_02_02",
        content=(
            "Auth outage on 2026-02-02.\n"
            "Customer updates referenced login saturation, the mitigation already in progress, "
            "and the next update time."
        ),
        file_path="workspace/incidents/auth-2026-02-02.md",
        start_line=1,
        end_line=2,
    ),
    "doc::customer_status_template": NodeSpec(
        node_id="doc::customer_status_template",
        content=(
            "Customer status template.\n"
            "Acknowledge impact, cite the current mitigation, give the next update time, "
            "and do not promise a resolution ETA."
        ),
        file_path="workspace/customer/status_template.md",
        start_line=1,
        end_line=2,
    ),
    "doc::monitoring_dashboards": NodeSpec(
        node_id="doc::monitoring_dashboards",
        content=(
            "Dashboards for checkout incidents.\n"
            "Use `grafana checkout-availability`, `sentry checkout errors`, and `kibana api-gateway logs`."
        ),
        file_path="workspace/monitoring/dashboards.md",
        start_line=1,
        end_line=2,
    ),
    "doc::oncall_schedule": NodeSpec(
        node_id="doc::oncall_schedule",
        content=(
            "On-call schedule snapshot.\n"
            "Mia owned the checkout/payments pager this week. Devon backed up auth."
        ),
        file_path="workspace/ops/oncall.md",
        start_line=1,
        end_line=2,
    ),
    "doc::deploy_after_incident": NodeSpec(
        node_id="doc::deploy_after_incident",
        content=(
            "Deploy-after-incident checklist.\n"
            "Keep rollback ready, verify the fix in staging, rerun smoke tests, and confirm pager coverage "
            "before re-running deploy."
        ),
        file_path="workspace/runbooks/deploy_after_incident.md",
        start_line=1,
        end_line=2,
    ),
    "doc::canary_rules": NodeSpec(
        node_id="doc::canary_rules",
        content=(
            "Canary rules.\n"
            "Hold at 5%, watch Grafana and Sentry, and flip the feature flag off before re-running deploy "
            "if errors climb."
        ),
        file_path="workspace/runbooks/canary_rules.md",
        start_line=1,
        end_line=2,
    ),
}

HUB_EDGES = {
    "hub::release": (
        EdgeSpec("doc::generic_hotfix_runbook", weight=0.58, relevance=0.54),
        EdgeSpec("doc::canary_rules", weight=0.46, relevance=0.43),
        EdgeSpec("doc::rollback_gate", weight=0.42, relevance=0.40),
        EdgeSpec("doc::deploy_after_incident", weight=0.36, relevance=0.35),
        EdgeSpec("doc::payments_incident_2026_02_14", weight=0.31, relevance=0.33),
    ),
    "hub::incident": (
        EdgeSpec("doc::oncall_schedule", weight=0.56, relevance=0.52),
        EdgeSpec("doc::monitoring_dashboards", weight=0.53, relevance=0.50),
        EdgeSpec("doc::payments_incident_2026_02_14", weight=0.34, relevance=0.36),
        EdgeSpec("doc::auth_incident_2026_02_02", weight=0.30, relevance=0.31),
    ),
    "hub::customer": (
        EdgeSpec("doc::customer_status_template", weight=0.55, relevance=0.53),
        EdgeSpec("doc::payments_incident_2026_02_14", weight=0.40, relevance=0.35),
        EdgeSpec("doc::auth_incident_2026_02_02", weight=0.28, relevance=0.30),
    ),
}

SCENARIOS = (
    Scenario(
        key="payments_recovery",
        category="decision-history",
        prototype_text=(
            "what exact rollback gate and feature flag action did we use when the payments canary failed yesterday"
        ),
        hub_id="hub::incident",
        chosen_target_id="doc::payments_incident_2026_02_14",
        required_node_ids=("doc::payments_incident_2026_02_14", "doc::rollback_gate"),
        expected_keywords=("payments_v3", "rollback gate", "feature flag"),
        train_queries=(
            "last payments canary failure what did we flip before rollback",
            "checkout payments incident repeat what flag and rollback gate worked last time",
            "when payments_v3 canary failed which gate and flag shutoff fixed it",
            "payments canary broke again which rollback gate and feature flag action mattered",
            "remind me how we recovered the checkout payments canary yesterday",
            "payments incident history which flag shutdown preceded the rollback gate",
        ),
        eval_query_id="payments_recovery_eval",
        eval_query=(
            "the payments canary failed again. which exact flag action and rollback gate did we use last time"
        ),
    ),
    Scenario(
        key="deploy_after_incident",
        category="ops",
        prototype_text="before rerunning checkout deploy after an incident what steps are mandatory",
        hub_id="hub::release",
        chosen_target_id="doc::deploy_after_incident",
        required_node_ids=("doc::deploy_after_incident", "doc::rollback_gate"),
        expected_keywords=("smoke tests", "rollback", "pager coverage"),
        train_queries=(
            "after stabilizing checkout incident what deploy guardrails must we do before rerun",
            "what checklist do we follow before retrying deploy after outage mitigation",
            "rerun deploy after incident which rollback readiness and smoke steps are required",
            "after the outage how do we safely rerun deploy",
            "what preflight steps are required before redeploying after incident recovery",
            "which release checklist applies before re-running deploy after an incident",
        ),
        eval_query_id="deploy_after_incident_eval",
        eval_query=(
            "checkout is stable again. what must we verify before we rerun deploy after the incident"
        ),
    ),
    Scenario(
        key="customer_auth_outage",
        category="decision-history",
        prototype_text=(
            "customer asks for a status update during the auth outage which template and incident notes matter"
        ),
        hub_id="hub::customer",
        chosen_target_id="doc::customer_status_template",
        required_node_ids=("doc::customer_status_template", "doc::auth_incident_2026_02_02"),
        expected_keywords=("next update time", "mitigation", "login"),
        train_queries=(
            "during login outage which customer wording and auth incident notes should we send",
            "auth outage response what status template and incident summary do we use for customers",
            "how should we message customers during auth incident and what notes matter",
            "what customer update template do we pair with auth outage notes",
            "which status-page wording and auth incident notes do we send externally",
            "for the auth outage what customer template and incident notes should support quote",
        ),
        eval_query_id="customer_auth_outage_eval",
        eval_query=(
            "support needs the auth outage customer update. which template and incident notes should we use"
        ),
    ),
    Scenario(
        key="oncall_dashboard_recall",
        category="ops",
        prototype_text="who was on call during the payments incident and which dashboard showed the spike",
        hub_id="hub::incident",
        chosen_target_id="doc::oncall_schedule",
        required_node_ids=("doc::oncall_schedule", "doc::monitoring_dashboards"),
        expected_keywords=("Mia", "Grafana", "Sentry"),
        train_queries=(
            "payments outage who owned the pager and what dashboard proved the error spike",
            "during checkout payments incident which on call person and dashboard did we use",
            "what pager owner and dashboard were used for the payments spike incident",
            "which on call engineer and graph did we reference in the payments incident",
            "who carried pager for the payments outage and what dashboard backed the diagnosis",
            "for the payments incident who had pager and what dashboard showed the spike first",
        ),
        eval_query_id="oncall_dashboard_recall_eval",
        eval_query=(
            "for the payments spike incident, who had the pager and which dashboard did the team rely on"
        ),
    ),
)


def _unit(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vec))
    if norm <= 0.0:
        return vec
    return [value / norm for value in vec]


def _mix(*items: tuple[float, list[float]]) -> list[float]:
    dim = len(items[0][1])
    mixed = [0.0] * dim
    for weight, vec in items:
        for idx, value in enumerate(vec):
            mixed[idx] += weight * value
    return _unit(mixed)


def _parse_string_list(payload: object) -> tuple[str, ...]:
    if not isinstance(payload, list):
        return ()
    return tuple(str(item) for item in payload if isinstance(item, str) and item)


def _required_node_coverage(prompt_node_ids: list[str], required_node_ids: tuple[str, ...]) -> float:
    if not required_node_ids:
        return 0.0
    present = sum(1 for node_id in required_node_ids if node_id in prompt_node_ids)
    return present / max(1, len(required_node_ids))


def _target_success(prompt_node_ids: list[str], required_node_ids: tuple[str, ...]) -> float:
    return 1.0 if set(required_node_ids).issubset(set(prompt_node_ids)) else 0.0


def _softmax(values: list[float]) -> list[float]:
    if not values:
        return []
    offset = max(values)
    exp_values = [math.exp(value - offset) for value in values]
    denom = sum(exp_values)
    if denom <= 0.0:
        return [1.0 / len(values)] * len(values)
    return [value / denom for value in exp_values]


def _normalized_entropy(probs: list[float]) -> float:
    if len(probs) <= 1:
        return 0.0
    entropy = -sum(prob * math.log(max(prob, 1e-12)) for prob in probs)
    return max(0.0, min(1.0, entropy / math.log(float(len(probs)))))


def _confidence(values: list[float]) -> tuple[float, float, float]:
    probs = _softmax(values)
    if not probs:
        return 0.0, 0.0, 0.0
    ordered = sorted(probs, reverse=True)
    margin = 1.0 if len(ordered) == 1 else max(0.0, min(1.0, ordered[0] - ordered[1]))
    entropy = _normalized_entropy(probs)
    conf = margin if len(values) <= 3 else (1.0 - entropy)
    return entropy, max(0.0, min(1.0, conf)), margin


def _build_vectors(embed: HashEmbedder) -> dict[str, list[float]]:
    prototypes = {scenario.key: embed.embed(scenario.prototype_text) for scenario in SCENARIOS}
    noise = embed.embed(NOISE_TEXT)
    return {
        "hub::release": _mix(
            (1.0, prototypes["payments_recovery"]),
            (1.2, prototypes["deploy_after_incident"]),
            (0.2, noise),
        ),
        "hub::incident": _mix(
            (1.0, prototypes["payments_recovery"]),
            (0.5, prototypes["deploy_after_incident"]),
            (0.7, prototypes["customer_auth_outage"]),
            (1.1, prototypes["oncall_dashboard_recall"]),
            (0.2, noise),
        ),
        "hub::customer": _mix(
            (1.2, prototypes["customer_auth_outage"]),
            (0.2, prototypes["oncall_dashboard_recall"]),
            (0.15, noise),
        ),
        "doc::generic_hotfix_runbook": _mix(
            (0.35, prototypes["deploy_after_incident"]),
            (0.2, prototypes["payments_recovery"]),
            (0.7, noise),
        ),
        "doc::rollback_gate": _mix(
            (0.45, prototypes["payments_recovery"]),
            (0.28, prototypes["deploy_after_incident"]),
            (0.65, noise),
        ),
        "doc::payments_incident_2026_02_14": _mix(
            (0.42, prototypes["payments_recovery"]),
            (0.18, prototypes["oncall_dashboard_recall"]),
            (0.7, noise),
        ),
        "doc::auth_incident_2026_02_02": _mix(
            (0.45, prototypes["customer_auth_outage"]),
            (0.1, prototypes["oncall_dashboard_recall"]),
            (0.68, noise),
        ),
        "doc::customer_status_template": _mix(
            (0.46, prototypes["customer_auth_outage"]),
            (0.05, prototypes["deploy_after_incident"]),
            (0.66, noise),
        ),
        "doc::monitoring_dashboards": _mix(
            (0.42, prototypes["oncall_dashboard_recall"]),
            (0.12, prototypes["deploy_after_incident"]),
            (0.68, noise),
        ),
        "doc::oncall_schedule": _mix(
            (0.43, prototypes["oncall_dashboard_recall"]),
            (0.08, prototypes["customer_auth_outage"]),
            (0.68, noise),
        ),
        "doc::deploy_after_incident": _mix(
            (0.5, prototypes["deploy_after_incident"]),
            (0.1, prototypes["payments_recovery"]),
            (0.66, noise),
        ),
        "doc::canary_rules": _mix(
            (0.25, prototypes["payments_recovery"]),
            (0.25, prototypes["deploy_after_incident"]),
            (0.12, prototypes["oncall_dashboard_recall"]),
            (0.75, noise),
        ),
    }


def _build_state(output_dir: Path, embed: HashEmbedder) -> Path:
    graph = Graph()
    index = VectorIndex()
    node_vectors = _build_vectors(embed)

    for node_id, spec in NODE_SPECS.items():
        graph.add_node(
            Node(
                node_id,
                spec.content,
                metadata={
                    "file": spec.file_path,
                    "start_line": spec.start_line,
                    "end_line": spec.end_line,
                    "authority": "canonical" if node_id.startswith("doc::") else "overlay",
                },
            )
        )
        index.upsert(node_id, node_vectors[node_id])

    for source_id, edges in HUB_EDGES.items():
        for edge in edges:
            graph.add_edge(
                Edge(
                    source_id,
                    edge.target_id,
                    edge.weight,
                    metadata={"relevance": edge.relevance},
                )
            )

    graph.add_edge(
        Edge(
            "doc::payments_incident_2026_02_14",
            "doc::rollback_gate",
            0.62,
            metadata={"relevance": 0.60},
        )
    )
    graph.add_edge(
        Edge(
            "doc::deploy_after_incident",
            "doc::rollback_gate",
            0.62,
            metadata={"relevance": 0.60},
        )
    )
    graph.add_edge(
        Edge(
            "doc::customer_status_template",
            "doc::auth_incident_2026_02_02",
            0.62,
            metadata={"relevance": 0.60},
        )
    )
    graph.add_edge(
        Edge(
            "doc::oncall_schedule",
            "doc::monitoring_dashboards",
            0.62,
            metadata={"relevance": 0.60},
        )
    )

    state_path = output_dir / "workflow_state.json"
    save_state(
        graph=graph,
        index=index,
        path=str(state_path),
        meta={
            "embedder_name": "hash-v1",
            "embedder_dim": embed.dim,
            "synthetic": "openclaw-workflows",
        },
    )
    return state_path


def _candidate_specs(source_id: str) -> tuple[EdgeSpec, ...]:
    edges = HUB_EDGES.get(source_id)
    if edges is None:
        raise KeyError(f"missing hub edge table for {source_id}")
    return edges


def _write_training_data(
    *,
    output_dir: Path,
    embed: HashEmbedder,
) -> tuple[Path, Path]:
    traces_path = output_dir / "train_traces.jsonl"
    labels_path = output_dir / "train_labels.jsonl"
    traces: list[RouteTrace] = []
    labels = []

    trace_idx = 0
    for scenario in SCENARIOS:
        candidates = _candidate_specs(scenario.hub_id)
        candidate_rows = [
            RouteCandidate(
                target_id=edge.target_id,
                edge_weight=edge.weight,
                edge_relevance=edge.relevance,
                target_preview=(NODE_SPECS[edge.target_id].content.splitlines()[0]),
                target_file=NODE_SPECS[edge.target_id].file_path,
                target_authority="canonical",
            )
            for edge in candidates
        ]
        teacher_scores = {
            edge.target_id: 3.0 if edge.target_id == scenario.chosen_target_id else -2.0
            for edge in candidates
        }
        for query_text in scenario.train_queries:
            query_id = f"{scenario.key}_train_{trace_idx:03d}"
            point = RouteDecisionPoint(
                query_text=query_text,
                source_id=scenario.hub_id,
                source_preview=NODE_SPECS[scenario.hub_id].content.splitlines()[0],
                chosen_target_id=scenario.chosen_target_id,
                candidates=candidate_rows,
                teacher_choose=[scenario.chosen_target_id],
                teacher_scores=teacher_scores,
                ts=1000.0 + float(trace_idx),
            )
            traces.append(
                RouteTrace(
                    query_id=query_id,
                    ts=1000.0 + float(trace_idx),
                    query_text=query_text,
                    seeds=[[scenario.hub_id, 1.0]],
                    fired_nodes=[scenario.hub_id, scenario.chosen_target_id],
                    traversal_config={"max_hops": 4, "max_fired_nodes": 4},
                    route_policy={"route_mode": "learned"},
                    query_vector=embed.embed(query_text),
                    decision_points=[point],
                )
            )
            labels.append(
                from_teacher_output(
                    query_id=query_id,
                    decision_point_idx=0,
                    teacher_scores=teacher_scores,
                    ts=1000.0 + float(trace_idx),
                    weight=1.0,
                    metadata={"synthetic": "openclaw-workflows", "scenario": scenario.key},
                )
            )
            trace_idx += 1

    traces_path.write_text(
        "\n".join(route_trace_to_json(trace) for trace in traces) + "\n",
        encoding="utf-8",
    )
    write_labels_jsonl(labels_path, labels)
    return traces_path, labels_path


def _write_eval_queries(path: Path) -> None:
    rows = []
    for scenario in SCENARIOS:
        rows.append(
            {
                "id": scenario.eval_query_id,
                "query": scenario.eval_query,
                "category": scenario.category,
                "expected_keywords": list(scenario.expected_keywords),
                "required_node_ids": list(scenario.required_node_ids),
            }
        )
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _evaluate_mode(
    *,
    graph: Graph,
    index: VectorIndex,
    meta: dict[str, object],
    embed: HashEmbedder,
    scenario_queries: tuple[Scenario, ...],
    mode: str,
    learned_model: RouteModel | None,
) -> dict[str, object]:
    null_store = _NullEventStore()
    target_projections = learned_model.precompute_target_projections(index) if learned_model is not None else {}
    rows: list[dict[str, object]] = []

    for scenario in scenario_queries:
        if mode == "vector_topk":
            payload = run_vector_topk(
                graph=graph,
                index=index,
                embed_fn=embed.embed,
                query_text=scenario.eval_query,
                top_k=1,
                max_prompt_context_chars=8000,
                prompt_context_include_node_ids=True,
            )
        elif mode == "pointer_chase":
            payload = run_pointer_chase(
                graph=graph,
                index=index,
                embed_fn=embed.embed,
                query_text=scenario.eval_query,
                config=PointerChaseConfig(
                    top_k=1,
                    max_turns=4,
                    max_prompt_context_chars=8000,
                ),
            )
        else:
            params: dict[str, object] = {
                "query": scenario.eval_query,
                "top_k": 1,
                "max_prompt_context_chars": 8000,
                "max_context_chars": 8000,
                "max_fired_nodes": 4,
                "prompt_context_include_node_ids": True,
                "route_top_k": 1,
            }
            if mode == "graph_prior_only":
                params.update(
                    {
                        "route_mode": "learned",
                        "debug_allow_confidence_override": True,
                        "router_conf_override": 0.0,
                    }
                )
            elif mode == "learned":
                params.update({"route_mode": "learned"})
            else:
                raise ValueError(f"unsupported mode: {mode}")

            payload = _handle_query(
                graph=graph,
                index=index,
                meta=meta,
                embed_fn=embed.embed,
                params=params,
                event_store=null_store,
                learned_model=learned_model,
                target_projections=target_projections,
            )

        prompt_node_ids = [
            str(node_id)
            for node_id in payload.get("prompt_context_included_node_ids", [])
            if isinstance(node_id, str)
        ]
        required_coverage = _required_node_coverage(prompt_node_ids, scenario.required_node_ids)
        rows.append(
            {
                "query_id": scenario.eval_query_id,
                "mode": mode,
                "prompt_node_ids": prompt_node_ids,
                "required_node_coverage": required_coverage,
                "target_success": _target_success(prompt_node_ids, scenario.required_node_ids),
                "route_router_conf_mean": float(payload.get("route_router_conf_mean", 0.0)),
                "pointer_turns": payload.get("pointer_turns"),
            }
        )

    target_success = [float(row["target_success"]) for row in rows]
    required_coverage = [float(row["required_node_coverage"]) for row in rows]
    router_conf = [float(row["route_router_conf_mean"]) for row in rows]
    pointer_turns = [float(row["pointer_turns"]) for row in rows if row.get("pointer_turns") is not None]
    return {
        "rows": rows,
        "target_success_rate": sum(target_success) / len(target_success),
        "required_node_coverage_mean": sum(required_coverage) / len(required_coverage),
        "route_router_conf_mean": sum(router_conf) / len(router_conf),
        "pointer_turns_mean": (sum(pointer_turns) / len(pointer_turns)) if pointer_turns else None,
    }


def _write_learning_curve(
    *,
    output_dir: Path,
    graph: Graph,
    index: VectorIndex,
    meta: dict[str, object],
    embed: HashEmbedder,
    state_path: Path,
    traces_path: Path,
    labels_path: Path,
    epochs: int,
    rank: int,
    lr: float,
    label_temp: float,
) -> tuple[Path, list[dict[str, float]], dict[str, object]]:
    curve_path = output_dir / "learning_curve.csv"
    graph_prior_metrics = _evaluate_mode(
        graph=graph,
        index=index,
        meta=meta,
        embed=embed,
        scenario_queries=SCENARIOS,
        mode="graph_prior_only",
        learned_model=RouteModel.init_identity(d=embed.dim, df=1),
    )
    rows: list[dict[str, float]] = []
    first_full_success_epoch: int | None = None

    for epoch in range(1, max(1, int(epochs)) + 1):
        model_path = output_dir / f"route_model_epoch_{epoch:02d}.npz"
        train_route_model(
            state_path=str(state_path),
            traces_in=str(traces_path),
            labels_in=str(labels_path),
            out_path=str(model_path),
            rank=rank,
            epochs=epoch,
            lr=lr,
            label_temp=label_temp,
        )
        model = RouteModel.load_npz(model_path)
        learned_metrics = _evaluate_mode(
            graph=graph,
            index=index,
            meta=meta,
            embed=embed,
            scenario_queries=SCENARIOS,
            mode="learned",
            learned_model=model,
        )
        if first_full_success_epoch is None and learned_metrics["target_success_rate"] >= 0.999999:
            first_full_success_epoch = epoch
        rows.append(
            {
                "epoch": float(epoch),
                "graph_prior_target_success_rate": float(graph_prior_metrics["target_success_rate"]),
                "graph_prior_required_node_coverage_mean": float(
                    graph_prior_metrics["required_node_coverage_mean"]
                ),
                "learned_target_success_rate": float(learned_metrics["target_success_rate"]),
                "learned_required_node_coverage_mean": float(
                    learned_metrics["required_node_coverage_mean"]
                ),
                "learned_route_router_conf_mean": float(learned_metrics["route_router_conf_mean"]),
            }
        )

    with curve_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "graph_prior_target_success_rate",
                "graph_prior_required_node_coverage_mean",
                "learned_target_success_rate",
                "learned_required_node_coverage_mean",
                "learned_route_router_conf_mean",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return curve_path, rows, {
        "graph_prior_target_success_rate": graph_prior_metrics["target_success_rate"],
        "graph_prior_required_node_coverage_mean": graph_prior_metrics["required_node_coverage_mean"],
        "first_full_success_epoch": first_full_success_epoch,
    }


def _mode_summary(summary: dict[str, object], mode: str) -> dict[str, object]:
    mode_summaries = summary.get("mode_summaries")
    if not isinstance(mode_summaries, dict):
        return {}
    entry = mode_summaries.get(mode)
    return entry if isinstance(entry, dict) else {}


def _ground_truth(summary: dict[str, object], mode: str) -> dict[str, object]:
    mode_summary = _mode_summary(summary, mode)
    ground_truth = mode_summary.get("ground_truth")
    return ground_truth if isinstance(ground_truth, dict) else {}


def _pointer_summary(summary: dict[str, object], mode: str) -> dict[str, object]:
    mode_summary = _mode_summary(summary, mode)
    pointer = mode_summary.get("pointer_chase")
    return pointer if isinstance(pointer, dict) else {}


def _fmt_metric(value: object) -> str:
    return f"{float(value):.2f}" if isinstance(value, (int, float)) else "-"


def _fmt_count(successes: int, total: int) -> str:
    return f"{successes}/{total}" if total > 0 else "-"


def _fmt_pointer_turns(value: object) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    turns = float(value)
    return str(int(turns)) if turns.is_integer() else f"{turns:.2f}"


def _row_metric(row: dict[str, object], field: str) -> float | None:
    value = row.get(field)
    return float(value) if isinstance(value, (int, float)) else None


def _mode_query_rows(final_summary: dict[str, object]) -> dict[str, dict[str, dict[str, object]]]:
    per_query = final_summary.get("per_query")
    if not isinstance(per_query, dict):
        return {}

    rows_by_mode: dict[str, dict[str, dict[str, object]]] = {}
    for mode, rows in per_query.items():
        if not isinstance(mode, str) or not isinstance(rows, list):
            continue
        rows_by_mode[mode] = {
            str(row.get("query_id")): row
            for row in rows
            if isinstance(row, dict) and isinstance(row.get("query_id"), str)
        }
    return rows_by_mode


def _build_per_query_matrix_rows(final_summary: dict[str, object]) -> list[dict[str, object]]:
    rows_by_mode = _mode_query_rows(final_summary)
    matrix_rows: list[dict[str, object]] = []

    for scenario in SCENARIOS:
        for mode in MODE_ORDER:
            mode_rows = rows_by_mode.get(mode)
            if mode_rows is None:
                raise KeyError(f"missing per-query rows for mode {mode}")
            row = mode_rows.get(scenario.eval_query_id)
            if row is None:
                raise KeyError(f"missing per-query row for {scenario.eval_query_id} in mode {mode}")

            matrix_rows.append(
                {
                    "query_id": scenario.eval_query_id,
                    "scenario": scenario.key,
                    "category": scenario.category,
                    "mode": mode,
                    "required_node_ids": "|".join(scenario.required_node_ids),
                    "prompt_context_included_node_ids": str(
                        row.get("prompt_context_included_node_ids", "")
                    ),
                    "target_success": _row_metric(row, "target_success"),
                    "required_node_coverage": _row_metric(row, "required_node_coverage"),
                    "pointer_turns": _row_metric(row, "pointer_turns"),
                }
            )

    return matrix_rows


def _write_per_query_matrix_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    fieldnames = [
        "query_id",
        "scenario",
        "category",
        "mode",
        "required_node_ids",
        "prompt_context_included_node_ids",
        "target_success",
        "required_node_coverage",
        "pointer_turns",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})
    return path


def _write_per_query_matrix_md(path: Path, rows: list[dict[str, object]]) -> Path:
    lines = [
        "# Per-Query Workflow Matrix",
        "",
        "Each row is one held-out workflow query under one retrieval mode. This is a deterministic evidence slice for the routing mechanism proof: it records which node IDs reached prompt context, not downstream production answer quality.",
        "",
        "Pointer turns are populated only for `pointer_chase`.",
        "",
        "| query_id | scenario | category | mode | required_node_ids | prompt_context_included_node_ids | target_success | required_node_coverage | pointer_turns |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["query_id"]),
                    str(row["scenario"]),
                    str(row["category"]),
                    str(row["mode"]),
                    str(row["required_node_ids"]),
                    str(row["prompt_context_included_node_ids"]),
                    _fmt_metric(row.get("target_success")),
                    _fmt_metric(row.get("required_node_coverage")),
                    _fmt_pointer_turns(row.get("pointer_turns")),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _render_report(
    *,
    final_summary: dict[str, object],
    curve_rows: list[dict[str, float]],
    curve_meta: dict[str, object],
    per_query_matrix_rows: list[dict[str, object]],
) -> str:
    rows_by_mode: dict[str, list[dict[str, object]]] = {mode: [] for mode in MODE_ORDER}
    row_lookup: dict[tuple[str, str], dict[str, object]] = {}
    for row in per_query_matrix_rows:
        mode = row.get("mode")
        scenario = row.get("scenario")
        if isinstance(mode, str) and mode in rows_by_mode:
            rows_by_mode[mode].append(row)
        if isinstance(mode, str) and isinstance(scenario, str):
            row_lookup[(scenario, mode)] = row

    lines = [
        "# OpenClaw Workflow Proof",
        "",
        "This harness models a realistic OpenClaw pattern: OpenClawBrain starts from local graph priors, then async teacher labels train a runtime route_fn that pulls the right historical note or runbook immediately. The sims are evidence for the routing mechanism, not a claim of production superiority.",
        "",
        "## Final metrics",
        "",
        "| mode | exact target success | required coverage mean | pointer turns mean |",
        "| --- | --- | --- | --- |",
    ]
    for mode in MODE_ORDER:
        ground_truth = _ground_truth(final_summary, mode)
        pointer = _pointer_summary(final_summary, mode)
        mode_rows = rows_by_mode.get(mode, [])
        success_count = sum(
            1 for row in mode_rows if (_row_metric(row, "target_success") or 0.0) >= 0.999999
        )
        total = len(mode_rows)
        lines.append(
            f"| {mode} | {_fmt_count(success_count, total)} ({_fmt_metric(ground_truth.get('target_success_rate'))}) | {_fmt_metric(ground_truth.get('required_node_coverage_mean'))} | {_fmt_metric(pointer.get('turns_mean'))} |"
        )

    lines.extend(
        [
            "",
            "See `per_query_matrix.csv` or `per_query_matrix.md` for the per-scenario node IDs behind these totals.",
            "",
            "## Scenario by mode",
            "",
            "Cells show `target_success / required_coverage`; `pointer_chase` also includes turns.",
            "",
            "| scenario | vector_topk | pointer_chase | graph_prior_only | learned |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for scenario in SCENARIOS:
        cell_text: list[str] = []
        for mode in MODE_ORDER:
            row = row_lookup[(scenario.key, mode)]
            cell = f"{int(_row_metric(row, 'target_success') or 0.0)} / {_fmt_metric(row.get('required_node_coverage'))}"
            if mode == "pointer_chase" and row.get("pointer_turns") is not None:
                cell = f"{cell}; t={_fmt_pointer_turns(row.get('pointer_turns'))}"
            cell_text.append(f"`{cell}`")
        lines.append(
            f"| {scenario.key} ({scenario.category}) | {cell_text[0]} | {cell_text[1]} | {cell_text[2]} | {cell_text[3]} |"
        )

    lines.extend(
        [
            "",
            "## Learning curve",
            "",
            f"- Graph-prior target success stays at `{_fmt_metric(curve_meta.get('graph_prior_target_success_rate'))}`.",
            f"- Learned routing first reaches full target success at epoch `{curve_meta.get('first_full_success_epoch')}`.",
            f"- Final learned target success is `{_fmt_metric(curve_rows[-1]['learned_target_success_rate'])}` with required coverage `{_fmt_metric(curve_rows[-1]['learned_required_node_coverage_mean'])}`.",
            "",
            "## What this proves now",
            "",
            "- Cold-start graph priors are already useful on some workflow queries (`customer_auth_outage`, `oncall_dashboard_recall`).",
            "- Background teacher labels can move runtime retrieval from generic priors to the exact history/runbook node needed (`payments_recovery`, `deploy_after_incident`).",
            "- Learned routing can beat vector-only retrieval without paying pointer-chase turns on the hot path.",
            "",
            "## What it does not prove yet",
            "",
            "- It does not prove live OpenClaw end-task success in production sessions.",
            "- It does not cover scanner/harvester label quality directly; the harness fixes graph priors and isolates teacher-driven runtime routing.",
            "- It uses deterministic synthetic vectors for CI reproducibility; production defaults remain local BGE-large embeddings plus a local async teacher such as Ollama `qwen3.5:9b-q4_K_M`.",
        ]
    )
    return "\n".join(lines)


def _write_worked_example(
    *,
    output_dir: Path,
    final_summary: dict[str, object],
    embed: HashEmbedder,
    index: VectorIndex,
    learned_model: RouteModel,
) -> Path:
    worked_example_path = output_dir / "worked_example.md"
    target_scenario = next(scenario for scenario in SCENARIOS if scenario.key == "payments_recovery")
    candidate_specs = _candidate_specs(target_scenario.hub_id)
    query_vec = embed.embed(target_scenario.eval_query)
    feat_vec = [0.0] * learned_model.df
    feat_vec[-1] = 1.0

    relevances = [edge.relevance for edge in candidate_specs]
    router_scores = []
    for edge in candidate_specs:
        target_vec = index._vectors[edge.target_id]
        router_scores.append(learned_model.score(query_vec, target_vec, feat_vec))
    _relevance_entropy, relevance_conf, _relevance_margin = _confidence(relevances)
    _router_entropy, router_conf, _router_margin = _confidence(router_scores)

    candidate_rows = []
    for edge, router_score in zip(candidate_specs, router_scores):
        graph_prior_score = (relevance_conf * edge.relevance) + ((1.0 - relevance_conf) * edge.weight)
        final_score = (router_conf * router_score) + ((1.0 - router_conf) * graph_prior_score)
        candidate_rows.append(
            {
                "target_id": edge.target_id,
                "graph_prior_score": graph_prior_score,
                "router_score": router_score,
                "final_score": final_score,
                "required": edge.target_id == target_scenario.chosen_target_id,
            }
        )

    candidate_rows.sort(key=lambda item: item["final_score"], reverse=True)
    per_query = final_summary.get("per_query")
    per_query_rows = per_query if isinstance(per_query, dict) else {}

    lines = [
        "# Worked Example: Payments Canary Recovery",
        "",
        f"Query: `{target_scenario.eval_query}`",
        "",
        "Required nodes in prompt context:",
        f"- `{target_scenario.required_node_ids[0]}`",
        f"- `{target_scenario.required_node_ids[1]}`",
        "",
        "## Candidate scores from `hub::incident`",
        "",
        "| target_id | graph prior | learned router | learned final | teacher target |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in candidate_rows:
        lines.append(
            f"| {row['target_id']} | {row['graph_prior_score']:.3f} | {row['router_score']:.3f} | {row['final_score']:.3f} | {'yes' if row['required'] else 'no'} |"
        )

    lines.extend(["", "## Per-mode retrieval", ""])
    for mode in ("vector_topk", "pointer_chase", "graph_prior_only", "learned"):
        rows = per_query_rows.get(mode, []) if isinstance(per_query_rows, dict) else []
        match = next(
            (
                row
                for row in rows
                if isinstance(row, dict) and row.get("query_id") == target_scenario.eval_query_id
            ),
            None,
        )
        if not isinstance(match, dict):
            continue
        lines.append(f"- `{mode}` -> `{match.get('prompt_context_included_node_ids', '')}`")
        lines.append(
            f"  target_success={_fmt_metric(match.get('target_success'))} required_coverage={_fmt_metric(match.get('required_node_coverage'))}"
        )

    worked_example_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return worked_example_path


def run_openclaw_workflow_simulation(
    *,
    output_dir: Path,
    embed_dim: int = 64,
    epochs: int = 16,
    rank: int = 16,
    lr: float = 0.25,
    label_temp: float = 0.3,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    embed = HashEmbedder(dim=embed_dim)
    state_path = _build_state(output_dir, embed)
    traces_path, labels_path = _write_training_data(output_dir=output_dir, embed=embed)

    eval_queries_path = output_dir / "eval_queries.jsonl"
    _write_eval_queries(eval_queries_path)

    graph, index, meta = load_state(str(state_path))
    curve_path, curve_rows, curve_meta = _write_learning_curve(
        output_dir=output_dir,
        graph=graph,
        index=index,
        meta=meta,
        embed=embed,
        state_path=state_path,
        traces_path=traces_path,
        labels_path=labels_path,
        epochs=epochs,
        rank=rank,
        lr=lr,
        label_temp=label_temp,
    )

    final_model_path = output_dir / f"route_model_epoch_{max(1, int(epochs)):02d}.npz"
    final_summary = run_baseline_suite(
        state_path=state_path,
        queries_path=eval_queries_path,
        modes=list(MODE_ORDER),
        embed_model="auto",
        route_model_path=final_model_path,
        top_k=1,
        route_top_k=1,
        max_fired_nodes=4,
        max_prompt_context_chars=8000,
        output_dir=output_dir / "baseline_eval",
        include_per_query=True,
    )
    per_query_matrix_rows = _build_per_query_matrix_rows(final_summary)
    per_query_matrix_csv_path = _write_per_query_matrix_csv(
        output_dir / "per_query_matrix.csv",
        per_query_matrix_rows,
    )
    per_query_matrix_md_path = _write_per_query_matrix_md(
        output_dir / "per_query_matrix.md",
        per_query_matrix_rows,
    )
    report_path = output_dir / "report.md"
    report_path.write_text(
        _render_report(
            final_summary=final_summary,
            curve_rows=curve_rows,
            curve_meta=curve_meta,
            per_query_matrix_rows=per_query_matrix_rows,
        )
        + "\n",
        encoding="utf-8",
    )

    learned_model = RouteModel.load_npz(final_model_path)
    worked_example_path = _write_worked_example(
        output_dir=output_dir,
        final_summary=final_summary,
        embed=embed,
        index=index,
        learned_model=learned_model,
    )

    final_vector = _ground_truth(final_summary, "vector_topk")
    final_pointer = _ground_truth(final_summary, "pointer_chase")
    final_graph_prior = _ground_truth(final_summary, "graph_prior_only")
    final_learned = _ground_truth(final_summary, "learned")

    summary = {
        "state_path": str(state_path),
        "traces_path": str(traces_path),
        "labels_path": str(labels_path),
        "eval_queries_path": str(eval_queries_path),
        "curve_path": str(curve_path),
        "per_query_matrix_csv_path": str(per_query_matrix_csv_path),
        "per_query_matrix_md_path": str(per_query_matrix_md_path),
        "report_path": str(report_path),
        "worked_example_path": str(worked_example_path),
        "baseline_summary_path": str(Path(final_summary["output"]["summary_json"])),
        "baseline_report_path": str(Path(final_summary["output"]["report_md"])),
        "epochs": int(epochs),
        "vector_topk_target_success_rate": final_vector.get("target_success_rate"),
        "pointer_chase_target_success_rate": final_pointer.get("target_success_rate"),
        "graph_prior_target_success_rate": final_graph_prior.get("target_success_rate"),
        "graph_prior_required_coverage_mean": final_graph_prior.get("required_node_coverage_mean"),
        "learned_target_success_rate": final_learned.get("target_success_rate"),
        "learned_required_coverage_mean": final_learned.get("required_node_coverage_mean"),
        "learned_minus_graph_prior_target_success": (
            float(final_learned.get("target_success_rate", 0.0))
            - float(final_graph_prior.get("target_success_rate", 0.0))
        ),
        "learned_minus_vector_topk_target_success": (
            float(final_learned.get("target_success_rate", 0.0))
            - float(final_vector.get("target_success_rate", 0.0))
        ),
        "pointer_turns_mean": _pointer_summary(final_summary, "pointer_chase").get("turns_mean"),
        "first_full_success_epoch": curve_meta.get("first_full_success_epoch"),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the workflow-shaped OpenClaw proof harness.")
    parser.add_argument(
        "--output-dir",
        default=str(Path("scratch") / "workflow-proof" / "latest"),
        help="Directory for generated proof artifacts",
    )
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.25)
    parser.add_argument("--label-temp", type=float, default=0.3)
    args = parser.parse_args()

    if args.embed_dim <= 0:
        raise SystemExit("--embed-dim must be > 0")
    if args.epochs <= 0:
        raise SystemExit("--epochs must be > 0")
    if args.rank <= 0:
        raise SystemExit("--rank must be > 0")
    if args.lr <= 0.0:
        raise SystemExit("--lr must be > 0")
    if args.label_temp <= 0.0:
        raise SystemExit("--label-temp must be > 0")

    summary = run_openclaw_workflow_simulation(
        output_dir=Path(args.output_dir).expanduser(),
        embed_dim=args.embed_dim,
        epochs=args.epochs,
        rank=args.rank,
        lr=args.lr,
        label_temp=args.label_temp,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
