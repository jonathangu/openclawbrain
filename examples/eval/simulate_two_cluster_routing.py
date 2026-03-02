#!/usr/bin/env python3
"""Synthetic two-cluster routing simulation for route model behavior.

Produces:
- simulation_curve.csv (epoch, ce_loss, accuracy)
- report.md (short narrative summary)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from openclawbrain import Edge, Graph, Node, VectorIndex, save_state
from openclawbrain.labels import LabelRecord, from_teacher_output, write_labels_jsonl
from openclawbrain.reward import RewardWeights
from openclawbrain.trace import RouteCandidate, RouteDecisionPoint, RouteTrace, route_trace_to_json
from openclawbrain.train_route_model import (
    _collect_points,
    _labels_index,
    _point_logits,
    _read_traces,
    _teacher_distribution,
    train_route_model,
)
from openclawbrain.route_model import RouteModel
from openclawbrain.store import load_state


def _unit(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 0:
        return vec
    return vec / norm


def _build_synthetic_state(state_path: Path, *, dim: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    graph = Graph()
    graph.add_node(Node("source", "Ambiguous source node"))

    target_ids = ["cluster_a_t0", "cluster_a_t1", "cluster_b_t0", "cluster_b_t1"]
    for target_id in target_ids:
        graph.add_node(Node(target_id, f"Synthetic target {target_id}"))
        graph.add_edge(Edge("source", target_id, weight=0.35, metadata={"relevance": 0.35}))

    centroid_a = _unit(np.asarray([1.0, 0.9, 0.8, 0.7, 0.0, 0.0, 0.0, 0.0], dtype=float)[:dim])
    centroid_b = _unit(np.asarray([0.0, 0.0, 0.0, 0.0, 1.0, 0.9, 0.8, 0.7], dtype=float)[:dim])

    vectors: dict[str, np.ndarray] = {
        "cluster_a_t0": _unit(centroid_a + rng.normal(0.0, 0.03, size=dim)),
        "cluster_a_t1": _unit(centroid_a + rng.normal(0.0, 0.03, size=dim)),
        "cluster_b_t0": _unit(centroid_b + rng.normal(0.0, 0.03, size=dim)),
        "cluster_b_t1": _unit(centroid_b + rng.normal(0.0, 0.03, size=dim)),
        "source": _unit((centroid_a + centroid_b) / 2.0),
    }

    index = VectorIndex()
    for node_id, vec in vectors.items():
        index.upsert(node_id, vec.tolist())

    save_state(
        graph=graph,
        index=index,
        path=str(state_path),
        meta={"embedder_name": "hash-v1", "embedder_dim": dim, "synthetic": "two-cluster"},
    )
    return vectors


def _write_traces_and_labels(
    *,
    traces_path: Path,
    labels_path: Path,
    vectors: dict[str, np.ndarray],
    samples_per_cluster: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    dim = int(next(iter(vectors.values())).shape[0])
    centroid_a = _unit((vectors["cluster_a_t0"] + vectors["cluster_a_t1"]) / 2.0)
    centroid_b = _unit((vectors["cluster_b_t0"] + vectors["cluster_b_t1"]) / 2.0)

    traces: list[RouteTrace] = []
    labels: list[LabelRecord] = []
    sample_idx = 0

    for cluster_name, centroid, preferred_targets, other_targets in (
        ("a", centroid_a, ["cluster_a_t0", "cluster_a_t1"], ["cluster_b_t0", "cluster_b_t1"]),
        ("b", centroid_b, ["cluster_b_t0", "cluster_b_t1"], ["cluster_a_t0", "cluster_a_t1"]),
    ):
        for i in range(samples_per_cluster):
            query_id = f"{cluster_name}_{i}"
            query_vec = _unit(centroid + rng.normal(0.0, 0.08, size=dim))
            chosen = preferred_targets[i % len(preferred_targets)]
            traces.append(
                RouteTrace(
                    query_id=query_id,
                    ts=1000.0 + sample_idx,
                    query_text=f"synthetic query {query_id}",
                    seeds=[["source", 1.0]],
                    fired_nodes=["source", chosen],
                    traversal_config={"max_hops": 4, "max_fired_nodes": 4},
                    route_policy={"route_mode": "learned"},
                    query_vector=query_vec.tolist(),
                    decision_points=[
                        RouteDecisionPoint(
                            query_text=f"synthetic query {query_id}",
                            source_id="source",
                            source_preview="Ambiguous source",
                            chosen_target_id=chosen,
                            candidates=[
                                RouteCandidate(target_id="cluster_a_t0", edge_weight=0.35, edge_relevance=0.35),
                                RouteCandidate(target_id="cluster_a_t1", edge_weight=0.35, edge_relevance=0.35),
                                RouteCandidate(target_id="cluster_b_t0", edge_weight=0.35, edge_relevance=0.35),
                                RouteCandidate(target_id="cluster_b_t1", edge_weight=0.35, edge_relevance=0.35),
                            ],
                            teacher_choose=[chosen],
                            teacher_scores={},
                            ts=1000.0 + sample_idx,
                        )
                    ],
                )
            )

            dense_scores = {
                preferred_targets[0]: 1.0,
                preferred_targets[1]: 0.9,
                other_targets[0]: -0.9,
                other_targets[1]: -1.0,
            }
            labels.append(
                from_teacher_output(
                    query_id=query_id,
                    decision_point_idx=0,
                    teacher_scores=dense_scores,
                    ts=1000.0 + sample_idx,
                    weight=1.0,
                    metadata={"synthetic": "two-cluster", "cluster": cluster_name},
                )
            )
            sample_idx += 1

    traces_path.write_text("\n".join(route_trace_to_json(trace) for trace in traces) + "\n", encoding="utf-8")
    write_labels_jsonl(labels_path, labels)


def _evaluate_accuracy(
    *,
    model: RouteModel,
    traces_path: Path,
    state_path: Path,
    labels_path: Path,
    label_temp: float,
) -> float:
    traces = _read_traces(str(traces_path))
    labels: list[LabelRecord] = []
    for line in labels_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            labels.append(LabelRecord.from_dict(payload))

    _graph, index, _meta = load_state(str(state_path))
    _points_total, points = _collect_points(traces, index._vectors)
    labels_by_key = _labels_index(labels)
    reward_weights = RewardWeights()

    correct = 0
    total = 0
    for trace, point_idx, point, query_vector, targets, candidate_ids in points:
        logits, _q_proj, _target_projs = _point_logits(model, query_vector, targets)
        pred_idx = int(np.argmax(logits))
        pred_target = candidate_ids[pred_idx]
        teacher_dist = _teacher_distribution(
            trace,
            point_idx,
            point,
            candidate_ids,
            labels_by_key,
            label_temp=label_temp,
            reward_weights=reward_weights,
        )
        teacher_idx = int(np.argmax(teacher_dist))
        teacher_target = candidate_ids[teacher_idx]
        if pred_target == teacher_target:
            correct += 1
        total += 1
    return float(correct / total) if total else 0.0


def run_two_cluster_simulation(
    *,
    output_dir: Path,
    dim: int = 8,
    samples_per_cluster: int = 80,
    epochs: int = 12,
    lr: float = 0.1,
    rank: int = 4,
    label_temp: float = 0.5,
    seed: int = 7,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "synthetic_state.json"
    traces_path = output_dir / "synthetic_traces.jsonl"
    labels_path = output_dir / "synthetic_labels.jsonl"

    vectors = _build_synthetic_state(state_path, dim=dim, seed=seed)
    _write_traces_and_labels(
        traces_path=traces_path,
        labels_path=labels_path,
        vectors=vectors,
        samples_per_cluster=samples_per_cluster,
        seed=seed + 1,
    )

    rows: list[dict[str, float]] = []
    initial_model = RouteModel.init_random(dq=dim, dt=dim, df=1, rank=max(1, int(rank)))
    initial_accuracy = _evaluate_accuracy(
        model=initial_model,
        traces_path=traces_path,
        state_path=state_path,
        labels_path=labels_path,
        label_temp=label_temp,
    )

    for epoch in range(1, max(1, int(epochs)) + 1):
        model_path = output_dir / f"route_model_epoch_{epoch}.npz"
        summary = train_route_model(
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
        accuracy = _evaluate_accuracy(
            model=model,
            traces_path=traces_path,
            state_path=state_path,
            labels_path=labels_path,
            label_temp=label_temp,
        )
        rows.append(
            {
                "epoch": float(epoch),
                "ce_loss": float(summary.final_ce_loss),
                "accuracy": float(accuracy),
            }
        )

    csv_path = output_dir / "simulation_curve.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "ce_loss", "accuracy"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "epoch": int(row["epoch"]),
                    "ce_loss": f"{row['ce_loss']:.8f}",
                    "accuracy": f"{row['accuracy']:.8f}",
                }
            )

    final_row = rows[-1]
    report_path = output_dir / "report.md"
    report_path.write_text(
        "\n".join(
            [
                "# Two-Cluster Routing Simulation",
                "",
                "## Setup",
                "- 1 ambiguous source node",
                "- 4 targets split into two clusters (2 per cluster)",
                "- query vectors sampled from two centroids with Gaussian noise",
                "- dense teacher labels supervise the correct cluster",
                "",
                "## Results",
                f"- Initial random-model accuracy: {initial_accuracy:.4f}",
                f"- Final accuracy (epoch {int(final_row['epoch'])}): {float(final_row['accuracy']):.4f}",
                f"- Initial CE loss (epoch 1): {float(rows[0]['ce_loss']):.6f}",
                f"- Final CE loss (epoch {int(final_row['epoch'])}): {float(final_row['ce_loss']):.6f}",
                "",
                "Loss decreases while routing accuracy improves, consistent with the expected QTsim learning behavior on clustered routing targets.",
                "",
                f"CSV curve: `{csv_path.name}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "output_dir": str(output_dir),
        "state_path": str(state_path),
        "traces_path": str(traces_path),
        "labels_path": str(labels_path),
        "csv_path": str(csv_path),
        "report_path": str(report_path),
        "initial_accuracy": initial_accuracy,
        "final_accuracy": float(final_row["accuracy"]),
        "initial_ce_loss": float(rows[0]["ce_loss"]),
        "final_ce_loss": float(final_row["ce_loss"]),
        "epochs": int(final_row["epoch"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run synthetic two-cluster routing simulation.")
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "out" / "two_cluster"))
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--samples-per-cluster", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--label-temp", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if args.dim <= 1:
        raise SystemExit("--dim must be > 1")
    if args.samples_per_cluster <= 0:
        raise SystemExit("--samples-per-cluster must be > 0")
    if args.epochs <= 0:
        raise SystemExit("--epochs must be > 0")
    if args.rank <= 0:
        raise SystemExit("--rank must be > 0")

    summary = run_two_cluster_simulation(
        output_dir=Path(args.output_dir).expanduser(),
        dim=args.dim,
        samples_per_cluster=args.samples_per_cluster,
        epochs=args.epochs,
        lr=args.lr,
        rank=args.rank,
        label_temp=args.label_temp,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
