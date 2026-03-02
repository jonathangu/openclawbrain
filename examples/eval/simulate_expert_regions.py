#!/usr/bin/env python3
"""Synthetic expert-regions routing simulation.

This harness builds a controllable synthetic task where each Gaussian query region is
assigned to a best expert and emits soft teacher labels over all experts.

Outputs:
- train_traces.jsonl / train_labels.jsonl
- test_traces.jsonl / test_labels.jsonl
- simulation_curve.csv (heldout baselines + learned curves per epoch)
- report.md (oracle gap summary, train vs test)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from openclawbrain import Edge, Graph, Node, VectorIndex, save_state
from openclawbrain.labels import LabelRecord, from_teacher_output, write_labels_jsonl
from openclawbrain.route_model import RouteModel
from openclawbrain.trace import RouteCandidate, RouteDecisionPoint, RouteTrace, route_trace_to_json
from openclawbrain.train_route_model import _collect_points, _point_logits, _read_traces, train_route_model
from openclawbrain.store import load_state


@dataclass(frozen=True)
class QueryExample:
    query_id: str
    query_vector: np.ndarray
    best_expert_idx: int
    utilities: np.ndarray


@dataclass(frozen=True)
class SplitData:
    traces: list[RouteTrace]
    labels: list[LabelRecord]
    examples: list[QueryExample]


def _unit(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 0:
        return vec
    return vec / norm


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - float(np.max(values))
    exp_values = np.exp(shifted)
    denom = float(np.sum(exp_values))
    if denom <= 0:
        return np.ones_like(values, dtype=float) / max(1, values.shape[0])
    return exp_values / denom


def _normalized_entropy(probs: np.ndarray) -> float:
    n = int(probs.shape[0])
    if n <= 1:
        return 0.0
    entropy = -float(np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))))
    normalized = entropy / math.log(float(n))
    return max(0.0, min(1.0, normalized))


def _margin(probs: np.ndarray) -> float:
    if probs.size <= 1:
        return 1.0 if probs.size == 1 else 0.0
    order = np.sort(probs)[::-1]
    return max(0.0, min(1.0, float(order[0] - order[1])))


def _confidence(values: np.ndarray) -> tuple[float, float, float]:
    probs = _softmax(values)
    entropy = _normalized_entropy(probs)
    margin = _margin(probs)
    conf = margin if values.shape[0] <= 3 else (1.0 - entropy)
    return entropy, max(0.0, min(1.0, conf)), margin


def _build_component_table(
    *,
    rng: np.random.Generator,
    num_components: int,
    expert_vectors: np.ndarray,
    component_noise: float,
    assignment_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    num_experts, dim = expert_vectors.shape
    component_best_expert = rng.choice(num_experts, size=num_components, replace=True, p=assignment_weights)
    component_means = np.zeros((num_components, dim), dtype=float)
    for idx in range(num_components):
        best_idx = int(component_best_expert[idx])
        mean = expert_vectors[best_idx] + rng.normal(0.0, component_noise, size=dim)
        component_means[idx] = _unit(mean)
    return component_means, component_best_expert.astype(int)


def _query_utilities(
    query_vec: np.ndarray,
    best_idx: int,
    expert_vectors: np.ndarray,
) -> np.ndarray:
    cosine = expert_vectors @ query_vec
    baseline = 0.08 + (0.22 * np.clip(cosine, 0.0, 1.0))
    utilities = baseline.astype(float)
    utilities[best_idx] = 0.9 + (0.1 * max(0.0, float(cosine[best_idx])))
    return np.clip(utilities, 0.0, 1.0)


def _teacher_scores(
    *,
    query_vec: np.ndarray,
    best_idx: int,
    expert_vectors: np.ndarray,
) -> dict[str, float]:
    cosine = expert_vectors @ query_vec
    scores: dict[str, float] = {}
    for idx in range(expert_vectors.shape[0]):
        expert_id = f"expert_{idx:02d}"
        if idx == best_idx:
            value = 2.2 + (0.5 * float(cosine[idx]))
        else:
            value = -0.8 + (0.2 * float(cosine[idx]))
        scores[expert_id] = float(value)
    return scores


def _build_split(
    *,
    split: str,
    n_queries: int,
    rng: np.random.Generator,
    component_means: np.ndarray,
    component_best_expert: np.ndarray,
    component_weights: np.ndarray,
    expert_vectors: np.ndarray,
    edge_weights: np.ndarray,
    edge_relevance: np.ndarray,
    sample_noise: float,
) -> SplitData:
    traces: list[RouteTrace] = []
    labels: list[LabelRecord] = []
    examples: list[QueryExample] = []
    num_experts = int(expert_vectors.shape[0])

    for idx in range(n_queries):
        component_idx = int(rng.choice(component_means.shape[0], p=component_weights))
        best_idx = int(component_best_expert[component_idx])
        query_id = f"{split}_{idx:06d}"

        query_vec = _unit(component_means[component_idx] + rng.normal(0.0, sample_noise, size=expert_vectors.shape[1]))
        utilities = _query_utilities(query_vec, best_idx, expert_vectors)
        teacher_scores = _teacher_scores(query_vec=query_vec, best_idx=best_idx, expert_vectors=expert_vectors)
        chosen_target_id = f"expert_{best_idx:02d}"

        candidates = [
            RouteCandidate(
                target_id=f"expert_{expert_idx:02d}",
                edge_weight=float(edge_weights[expert_idx]),
                edge_relevance=float(edge_relevance[expert_idx]),
            )
            for expert_idx in range(num_experts)
        ]

        trace = RouteTrace(
            query_id=query_id,
            ts=1000.0 + float(idx),
            query_text=f"synthetic {split} query {idx}",
            seeds=[["source", 1.0]],
            fired_nodes=["source", chosen_target_id],
            traversal_config={"max_hops": 4, "max_fired_nodes": 4},
            route_policy={"route_mode": "learned"},
            query_vector=query_vec.tolist(),
            decision_points=[
                RouteDecisionPoint(
                    query_text=f"synthetic {split} query {idx}",
                    source_id="source",
                    source_preview="Synthetic source",
                    chosen_target_id=chosen_target_id,
                    candidates=candidates,
                    teacher_choose=[chosen_target_id],
                    teacher_scores=teacher_scores,
                    ts=1000.0 + float(idx),
                )
            ],
        )

        label = from_teacher_output(
            query_id=query_id,
            decision_point_idx=0,
            teacher_scores=teacher_scores,
            ts=1000.0 + float(idx),
            weight=1.0,
            metadata={"synthetic": "expert-regions", "split": split, "component": component_idx},
        )

        traces.append(trace)
        labels.append(label)
        examples.append(QueryExample(query_id=query_id, query_vector=query_vec, best_expert_idx=best_idx, utilities=utilities))

    return SplitData(traces=traces, labels=labels, examples=examples)


def _write_split(*, traces_path: Path, labels_path: Path, split_data: SplitData) -> None:
    traces_path.write_text("\n".join(route_trace_to_json(trace) for trace in split_data.traces) + "\n", encoding="utf-8")
    write_labels_jsonl(labels_path, split_data.labels)


def _build_state(
    *,
    state_path: Path,
    expert_vectors: np.ndarray,
    edge_weights: np.ndarray,
    edge_relevance: np.ndarray,
) -> None:
    graph = Graph()
    graph.add_node(Node("source", "synthetic source"))
    index = VectorIndex()
    source_vec = _unit(np.mean(expert_vectors, axis=0))
    index.upsert("source", source_vec.tolist())

    for idx in range(expert_vectors.shape[0]):
        expert_id = f"expert_{idx:02d}"
        graph.add_node(Node(expert_id, f"synthetic expert {idx}"))
        graph.add_edge(
            Edge(
                "source",
                expert_id,
                weight=float(edge_weights[idx]),
                metadata={"relevance": float(edge_relevance[idx])},
            )
        )
        index.upsert(expert_id, expert_vectors[idx].tolist())

    save_state(
        graph=graph,
        index=index,
        path=str(state_path),
        meta={"embedder_name": "hash-v1", "embedder_dim": int(expert_vectors.shape[1]), "synthetic": "expert-regions"},
    )


def _constant_feat(df: int) -> np.ndarray:
    feat = np.zeros(df, dtype=float)
    feat[-1] = 1.0
    return feat


def _reinforce_finetune(
    *,
    model: RouteModel,
    state_path: Path,
    traces_path: Path,
    utility_by_query: dict[str, np.ndarray],
    steps: int,
    lr: float,
    seed: int,
) -> None:
    if steps <= 0:
        return

    traces = _read_traces(str(traces_path))
    _graph, index, _meta = load_state(str(state_path))
    _points_total, points = _collect_points(traces, index._vectors)
    if not points:
        return

    rng = np.random.default_rng(seed)
    feat_vec = _constant_feat(model.df)
    inv_t = 1.0 / max(1e-6, float(model.T))
    lr_value = float(lr)

    for _ in range(int(steps)):
        for point_idx in rng.permutation(len(points)).tolist():
            trace, _point_num, _point, query_vector, targets, candidate_ids = points[point_idx]
            utilities = utility_by_query.get(trace.query_id)
            if utilities is None:
                continue
            logits, q_proj, target_projs = _point_logits(model, query_vector, targets)
            probs = _softmax(logits)
            action = int(rng.choice(len(candidate_ids), p=probs))

            rewards = np.asarray([float(utilities[int(candidate_id.split("_")[1])]) for candidate_id in candidate_ids], dtype=float)
            advantage = float(rewards[action] - float(np.mean(rewards)))
            if abs(advantage) <= 1e-12:
                continue

            grad = probs.copy()
            grad[action] -= 1.0
            grad *= advantage

            grad_A = np.zeros_like(model.A)
            grad_B = np.zeros_like(model.B)
            grad_w = np.zeros_like(model.w_feat)
            grad_b = 0.0

            for jdx, g_j in enumerate(grad.tolist()):
                g_scaled = float(g_j) * inv_t
                grad_A += g_scaled * np.outer(query_vector, target_projs[jdx])
                grad_B += g_scaled * np.outer(targets[jdx], q_proj)
                grad_w += g_scaled * feat_vec
                grad_b += g_scaled

            model.A -= lr_value * grad_A
            model.B -= lr_value * grad_B
            model.w_feat -= lr_value * grad_w
            model.b -= lr_value * grad_b


def _evaluate_policy(
    *,
    examples: list[QueryExample],
    model: RouteModel | None,
    expert_vectors: np.ndarray,
    edge_weights: np.ndarray,
    edge_relevance: np.ndarray,
    policy: str,
) -> dict[str, float]:
    num_experts = int(expert_vectors.shape[0])
    relevance_entropy, relevance_conf, _ = _confidence(edge_relevance)
    _ = relevance_entropy
    graph_prior_scores = (relevance_conf * edge_relevance) + ((1.0 - relevance_conf) * edge_weights)
    graph_prior_probs = _softmax(graph_prior_scores)

    rewards: list[float] = []
    accuracies: list[float] = []
    entropies: list[float] = []

    for item in examples:
        utilities = item.utilities
        best_idx = int(item.best_expert_idx)

        if policy == "random":
            rewards.append(float(np.mean(utilities)))
            accuracies.append(1.0 / num_experts)
            entropies.append(1.0)
            continue

        if policy == "oracle":
            rewards.append(float(np.max(utilities)))
            accuracies.append(1.0)
            entropies.append(0.0)
            continue

        if policy == "graph_prior_only":
            probs = graph_prior_probs
            action = int(np.argmax(graph_prior_scores))
            rewards.append(float(utilities[action]))
            accuracies.append(1.0 if action == best_idx else 0.0)
            entropies.append(_normalized_entropy(probs))
            continue

        if model is None:
            raise ValueError(f"policy '{policy}' requires a trained model")

        q_proj = model.project_query(item.query_vector)
        feat_vec = _constant_feat(model.df)
        router_scores = np.asarray(
            [
                model.score_projected(q_proj, model.project_target(expert_vectors[idx]), feat_vec)
                for idx in range(num_experts)
            ],
            dtype=float,
        )

        if policy == "qtsim_only":
            final_scores = router_scores
            probs = _softmax(final_scores)
            action = int(np.argmax(final_scores))
            rewards.append(float(utilities[action]))
            accuracies.append(1.0 if action == best_idx else 0.0)
            entropies.append(_normalized_entropy(probs))
            continue

        if policy == "learned_mixed":
            _router_entropy, router_conf, _router_margin = _confidence(router_scores)
            final_scores = (router_conf * router_scores) + ((1.0 - router_conf) * graph_prior_scores)
            probs = _softmax(final_scores)
            action = int(np.argmax(final_scores))
            rewards.append(float(utilities[action]))
            accuracies.append(1.0 if action == best_idx else 0.0)
            entropies.append(_normalized_entropy(probs))
            continue

        raise ValueError(f"unknown policy: {policy}")

    return {
        "reward": float(np.mean(rewards)) if rewards else 0.0,
        "accuracy": float(np.mean(accuracies)) if accuracies else 0.0,
        "entropy": float(np.mean(entropies)) if entropies else 0.0,
    }


def _with_gap(metrics: dict[str, float], *, oracle_reward: float, random_reward: float) -> dict[str, float]:
    denom = oracle_reward - random_reward
    if abs(denom) <= 1e-12:
        gap = 0.0
    else:
        gap = (metrics["reward"] - random_reward) / denom
    payload = dict(metrics)
    payload["oracle_gap_fraction"] = max(0.0, min(1.0, float(gap)))
    return payload


def run_expert_regions_simulation(
    *,
    output_dir: Path,
    k_experts: int = 8,
    dim: int = 16,
    num_components: int = 24,
    train_queries: int = 3000,
    test_queries: int = 1000,
    epochs: int = 12,
    rank: int = 8,
    lr: float = 0.08,
    label_temp: float = 0.6,
    component_noise: float = 0.18,
    sample_noise: float = 0.24,
    reinforce_steps: int = 0,
    reinforce_lr: float = 0.01,
    seed: int = 13,
) -> dict[str, object]:
    if k_experts < 2:
        raise ValueError("k_experts must be >= 2")
    if dim < 2:
        raise ValueError("dim must be >= 2")
    if num_components < k_experts:
        raise ValueError("num_components must be >= k_experts")
    if train_queries <= 0 or test_queries <= 0:
        raise ValueError("train_queries and test_queries must be > 0")
    if epochs <= 0:
        raise ValueError("epochs must be > 0")

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    expert_vectors = np.vstack([_unit(rng.normal(0.0, 1.0, size=dim)) for _ in range(k_experts)])
    assignment_weights = rng.dirichlet(np.full(k_experts, 0.75, dtype=float))
    component_means, component_best_expert = _build_component_table(
        rng=rng,
        num_components=num_components,
        expert_vectors=expert_vectors,
        component_noise=component_noise,
        assignment_weights=assignment_weights,
    )
    component_weights = rng.dirichlet(np.full(num_components, 0.9, dtype=float))

    # Graph prior mirrors coarse expert popularity; it is intentionally imperfect.
    expert_counts = np.bincount(component_best_expert, minlength=k_experts).astype(float)
    popularity = (expert_counts + 1.0) / float(np.sum(expert_counts + 1.0))
    edge_weights = 0.2 + (0.55 * popularity)
    edge_relevance = 0.15 + (0.7 * popularity)

    state_path = output_dir / "state.json"
    train_traces_path = output_dir / "train_traces.jsonl"
    train_labels_path = output_dir / "train_labels.jsonl"
    test_traces_path = output_dir / "test_traces.jsonl"
    test_labels_path = output_dir / "test_labels.jsonl"

    _build_state(
        state_path=state_path,
        expert_vectors=expert_vectors,
        edge_weights=edge_weights,
        edge_relevance=edge_relevance,
    )

    train_data = _build_split(
        split="train",
        n_queries=train_queries,
        rng=np.random.default_rng(seed + 1),
        component_means=component_means,
        component_best_expert=component_best_expert,
        component_weights=component_weights,
        expert_vectors=expert_vectors,
        edge_weights=edge_weights,
        edge_relevance=edge_relevance,
        sample_noise=sample_noise,
    )
    test_data = _build_split(
        split="test",
        n_queries=test_queries,
        rng=np.random.default_rng(seed + 2),
        component_means=component_means,
        component_best_expert=component_best_expert,
        component_weights=component_weights,
        expert_vectors=expert_vectors,
        edge_weights=edge_weights,
        edge_relevance=edge_relevance,
        sample_noise=sample_noise,
    )

    _write_split(traces_path=train_traces_path, labels_path=train_labels_path, split_data=train_data)
    _write_split(traces_path=test_traces_path, labels_path=test_labels_path, split_data=test_data)

    train_utils = {item.query_id: item.utilities for item in train_data.examples}

    heldout_random = _evaluate_policy(
        examples=test_data.examples,
        model=None,
        expert_vectors=expert_vectors,
        edge_weights=edge_weights,
        edge_relevance=edge_relevance,
        policy="random",
    )
    heldout_oracle = _evaluate_policy(
        examples=test_data.examples,
        model=None,
        expert_vectors=expert_vectors,
        edge_weights=edge_weights,
        edge_relevance=edge_relevance,
        policy="oracle",
    )
    heldout_graph = _evaluate_policy(
        examples=test_data.examples,
        model=None,
        expert_vectors=expert_vectors,
        edge_weights=edge_weights,
        edge_relevance=edge_relevance,
        policy="graph_prior_only",
    )

    curve_rows: list[dict[str, object]] = []
    final_model_path: Path | None = None

    for epoch in range(1, int(epochs) + 1):
        model_path = output_dir / f"route_model_epoch_{epoch:03d}.npz"
        _summary = train_route_model(
            state_path=str(state_path),
            traces_in=str(train_traces_path),
            labels_in=str(train_labels_path),
            out_path=str(model_path),
            rank=rank,
            epochs=epoch,
            lr=lr,
            label_temp=label_temp,
        )

        model = RouteModel.load_npz(model_path)
        if reinforce_steps > 0:
            _reinforce_finetune(
                model=model,
                state_path=state_path,
                traces_path=train_traces_path,
                utility_by_query=train_utils,
                steps=reinforce_steps,
                lr=reinforce_lr,
                seed=seed + epoch,
            )
            model.save_npz(model_path)

        qtsim_metrics = _evaluate_policy(
            examples=test_data.examples,
            model=model,
            expert_vectors=expert_vectors,
            edge_weights=edge_weights,
            edge_relevance=edge_relevance,
            policy="qtsim_only",
        )
        mixed_metrics = _evaluate_policy(
            examples=test_data.examples,
            model=model,
            expert_vectors=expert_vectors,
            edge_weights=edge_weights,
            edge_relevance=edge_relevance,
            policy="learned_mixed",
        )

        oracle_reward = float(heldout_oracle["reward"])
        random_reward = float(heldout_random["reward"])

        per_policy = {
            "random": _with_gap(heldout_random, oracle_reward=oracle_reward, random_reward=random_reward),
            "oracle": _with_gap(heldout_oracle, oracle_reward=oracle_reward, random_reward=random_reward),
            "graph_prior_only": _with_gap(heldout_graph, oracle_reward=oracle_reward, random_reward=random_reward),
            "qtsim_only": _with_gap(qtsim_metrics, oracle_reward=oracle_reward, random_reward=random_reward),
            "learned_mixed": _with_gap(mixed_metrics, oracle_reward=oracle_reward, random_reward=random_reward),
        }

        for policy_name, metrics in per_policy.items():
            curve_rows.append(
                {
                    "epoch": epoch,
                    "policy": policy_name,
                    "reward": float(metrics["reward"]),
                    "accuracy": float(metrics["accuracy"]),
                    "entropy": float(metrics["entropy"]),
                    "oracle_gap_fraction": float(metrics["oracle_gap_fraction"]),
                }
            )
        final_model_path = model_path

    if final_model_path is None:
        raise RuntimeError("simulation produced no model checkpoints")

    final_model = RouteModel.load_npz(final_model_path)
    train_random = _evaluate_policy(
        examples=train_data.examples,
        model=None,
        expert_vectors=expert_vectors,
        edge_weights=edge_weights,
        edge_relevance=edge_relevance,
        policy="random",
    )
    train_oracle = _evaluate_policy(
        examples=train_data.examples,
        model=None,
        expert_vectors=expert_vectors,
        edge_weights=edge_weights,
        edge_relevance=edge_relevance,
        policy="oracle",
    )
    train_learned = _evaluate_policy(
        examples=train_data.examples,
        model=final_model,
        expert_vectors=expert_vectors,
        edge_weights=edge_weights,
        edge_relevance=edge_relevance,
        policy="learned_mixed",
    )
    test_learned = _evaluate_policy(
        examples=test_data.examples,
        model=final_model,
        expert_vectors=expert_vectors,
        edge_weights=edge_weights,
        edge_relevance=edge_relevance,
        policy="learned_mixed",
    )

    train_learned_gap = _with_gap(train_learned, oracle_reward=train_oracle["reward"], random_reward=train_random["reward"])
    test_learned_gap = _with_gap(test_learned, oracle_reward=heldout_oracle["reward"], random_reward=heldout_random["reward"])

    curve_path = output_dir / "simulation_curve.csv"
    with curve_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "policy", "reward", "accuracy", "entropy", "oracle_gap_fraction"],
        )
        writer.writeheader()
        for row in curve_rows:
            writer.writerow(
                {
                    "epoch": int(row["epoch"]),
                    "policy": str(row["policy"]),
                    "reward": f"{float(row['reward']):.8f}",
                    "accuracy": f"{float(row['accuracy']):.8f}",
                    "entropy": f"{float(row['entropy']):.8f}",
                    "oracle_gap_fraction": f"{float(row['oracle_gap_fraction']):.8f}",
                }
            )

    report_path = output_dir / "report.md"
    report_path.write_text(
        "\n".join(
            [
                "# Expert-Regions Synthetic Routing Report",
                "",
                "## Setup",
                f"- Experts (K): {k_experts}",
                f"- Embedding dims (D): {dim}",
                f"- Gaussian components: {num_components}",
                f"- Train queries: {train_queries}",
                f"- Test queries: {test_queries}",
                f"- Distillation epochs: {epochs}",
                f"- Optional REINFORCE steps/epoch: {reinforce_steps}",
                "",
                "## Heldout baselines",
                f"- Oracle reward: {float(heldout_oracle['reward']):.6f}",
                f"- Random reward: {float(heldout_random['reward']):.6f}",
                f"- Graph-prior-only reward: {float(heldout_graph['reward']):.6f}",
                "",
                "## Final learned performance",
                f"- Final learned reward (test): {float(test_learned['reward']):.6f}",
                f"- Final learned accuracy (test): {float(test_learned['accuracy']):.6f}",
                f"- Gap closed (test): {100.0 * float(test_learned_gap['oracle_gap_fraction']):.2f}%",
                "",
                "## Train vs test",
                f"- Train reward: {float(train_learned['reward']):.6f}",
                f"- Test reward: {float(test_learned['reward']):.6f}",
                f"- Train gap closed: {100.0 * float(train_learned_gap['oracle_gap_fraction']):.2f}%",
                f"- Test gap closed: {100.0 * float(test_learned_gap['oracle_gap_fraction']):.2f}%",
                "",
                f"Curve CSV: `{curve_path.name}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "output_dir": str(output_dir),
        "state_path": str(state_path),
        "train_traces_path": str(train_traces_path),
        "train_labels_path": str(train_labels_path),
        "test_traces_path": str(test_traces_path),
        "test_labels_path": str(test_labels_path),
        "curve_path": str(curve_path),
        "report_path": str(report_path),
        "oracle_reward": float(heldout_oracle["reward"]),
        "random_reward": float(heldout_random["reward"]),
        "graph_prior_reward": float(heldout_graph["reward"]),
        "final_learned_reward": float(test_learned["reward"]),
        "final_learned_accuracy": float(test_learned["accuracy"]),
        "final_gap_closed": float(test_learned_gap["oracle_gap_fraction"]),
        "train_reward": float(train_learned["reward"]),
        "test_reward": float(test_learned["reward"]),
        "train_gap_closed": float(train_learned_gap["oracle_gap_fraction"]),
        "test_gap_closed": float(test_learned_gap["oracle_gap_fraction"]),
        "epochs": int(epochs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run synthetic expert-regions routing simulation.")
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "out" / "expert_regions"))
    parser.add_argument("--k-experts", type=int, default=8)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--num-components", type=int, default=24)
    parser.add_argument("--train-queries", type=int, default=3000)
    parser.add_argument("--test-queries", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--label-temp", type=float, default=0.6)
    parser.add_argument("--component-noise", type=float, default=0.18)
    parser.add_argument("--sample-noise", type=float, default=0.24)
    parser.add_argument("--reinforce-steps", type=int, default=0)
    parser.add_argument("--reinforce-lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    summary = run_expert_regions_simulation(
        output_dir=Path(args.output_dir).expanduser(),
        k_experts=args.k_experts,
        dim=args.dim,
        num_components=args.num_components,
        train_queries=args.train_queries,
        test_queries=args.test_queries,
        epochs=args.epochs,
        rank=args.rank,
        lr=args.lr,
        label_temp=args.label_temp,
        component_noise=args.component_noise,
        sample_noise=args.sample_noise,
        reinforce_steps=args.reinforce_steps,
        reinforce_lr=args.reinforce_lr,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
