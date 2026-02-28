"""ablation_v12.py — Ablation study for v12.2.0 paper.

Tests what each mechanism contributes:
1. PG full (apply_outcome_pg + Hebbian + decay)
2. PG no Hebbian (apply_outcome_pg, Hebbian disabled)
3. PG no decay (apply_outcome_pg + Hebbian, no decay)
4. Heuristic full (apply_outcome + Hebbian + decay)
5. Heuristic no decay (apply_outcome + Hebbian, no decay)

Uses the same v2 sim setup: 50 nodes, 3 clusters, mixed feedback, drift at 200.
"""

from __future__ import annotations

import copy
import json
import random
import statistics
from dataclasses import dataclass
from pathlib import Path

from openclawbrain.graph import Edge, Graph, Node
from openclawbrain.learn import LearningConfig, apply_outcome, apply_outcome_pg, hebbian_update
from openclawbrain.decay import DecayConfig, apply_decay

RESULT_PATH = Path(__file__).with_name("ablation_v12_results.json")


def build_graph(rng: random.Random) -> tuple[Graph, dict]:
    g = Graph()
    g.add_node(Node("hub", "Central routing hub"))
    info = {}
    for name in ["A", "B", "C"]:
        gate = f"gate_{name}"
        chain = [f"{name}_{i}" for i in range(4)]
        distractors = [f"{name}_d{i}" for i in range(5)]
        g.add_node(Node(gate, f"Gate for {name}"))
        for n in chain + distractors:
            g.add_node(Node(n, f"Node {n}"))
        g.add_edge(Edge("hub", gate, weight=0.35, kind="sibling"))
        g.add_edge(Edge(gate, chain[0], weight=0.35, kind="sibling"))
        for i in range(len(chain) - 1):
            g.add_edge(Edge(chain[i], chain[i + 1], weight=0.35, kind="sibling"))
        for src in [gate] + chain:
            for d in rng.sample(distractors, k=2):
                g.add_edge(Edge(src, d, weight=rng.uniform(0.08, 0.20), kind="sibling"))
        for d in distractors:
            tgt = rng.choice(distractors + chain)
            if tgt != d:
                g.add_edge(Edge(d, tgt, weight=rng.uniform(0.08, 0.18), kind="sibling"))
        info[name] = {"gate": gate, "chain": chain, "distractors": distractors,
                      "path": ["hub", gate] + chain}
    all_distractors = [n for cl in info.values() for n in cl["distractors"]]
    for d in rng.sample(all_distractors, k=4):
        g.add_edge(Edge("hub", d, weight=rng.uniform(0.10, 0.20), kind="sibling"))
    return g, info


def edge_weight(graph: Graph, src: str, tgt: str) -> float:
    e = graph._edges.get(src, {}).get(tgt)
    return float(e.weight) if e else 0.0


@dataclass
class Condition:
    name: str
    use_pg: bool
    use_hebbian: bool
    use_decay: bool


CONDITIONS = [
    Condition("PG full", use_pg=True, use_hebbian=True, use_decay=True),
    Condition("PG no Hebbian", use_pg=True, use_hebbian=False, use_decay=True),
    Condition("PG no decay", use_pg=True, use_hebbian=True, use_decay=False),
    Condition("Heuristic full", use_pg=False, use_hebbian=True, use_decay=True),
    Condition("Heuristic no decay", use_pg=False, use_hebbian=True, use_decay=False),
]


def run_condition(
    graph: Graph, info: dict, cond: Condition, *,
    steps: int = 400, drift_at: int = 200, seed: int = 42,
    lr: float = 0.08, discount: float = 0.95,
) -> dict:
    rng = random.Random(seed)
    cfg = LearningConfig(learning_rate=lr, discount=discount)
    # Disable Hebbian by setting increment to 0
    if not cond.use_hebbian:
        cfg = LearningConfig(learning_rate=lr, discount=discount, hebbian_increment=0.0)

    clusters = ["A", "B", "C"]
    drift_map = {"A": "B", "B": "C", "C": "A"}

    separations = []
    total_weights = []
    decay_interval = 20  # apply decay every 20 steps

    for t in range(steps):
        phase = "pre" if t < drift_at else "post"
        query_cluster = rng.choice(clusters)
        correct_cluster = query_cluster if phase == "pre" else drift_map[query_cluster]
        correct_path = info[correct_cluster]["path"]
        wrong_clusters = [c for c in clusters if c != correct_cluster]

        r = rng.random()
        if r < 0.65:
            path = correct_path
            outcome = 1.0
            per_node = None
        elif r < 0.85:
            wrong_cl = rng.choice(wrong_clusters)
            path = info[wrong_cl]["path"]
            outcome = -1.0
            per_node = None
        else:
            wrong_cl = rng.choice(wrong_clusters)
            path = correct_path[:2] + info[wrong_cl]["chain"]
            outcome = -1.0
            per_node = {}
            for idx in range(len(path) - 1):
                per_node[path[idx]] = 0.6 if idx < 2 else -1.0

        if cond.use_pg:
            apply_outcome_pg(graph, path, outcome, config=cfg,
                             per_node_outcomes=per_node, baseline=0.0, temperature=1.0)
        else:
            apply_outcome(graph, path, outcome, config=cfg,
                          per_node_outcomes=per_node)

        # Periodic decay
        if cond.use_decay and t > 0 and t % decay_interval == 0:
            apply_decay(graph, config=DecayConfig(half_life=140))

        # Metrics
        w_A = edge_weight(graph, "hub", "gate_A")
        w_B = edge_weight(graph, "hub", "gate_B")
        w_C = edge_weight(graph, "hub", "gate_C")
        gate_weights = {"A": w_A, "B": w_B, "C": w_C}
        w_correct = gate_weights[correct_cluster]
        w_wrong = [gate_weights[c] for c in clusters if c != correct_cluster]
        separations.append(w_correct - max(w_wrong))
        total_weights.append(sum(e.weight for d in graph._edges.values() for e in d.values()))

    def avg_window(xs, start, end):
        sl = xs[start:end]
        return round(statistics.mean(sl), 4) if sl else 0.0

    recovery = None
    for i in range(drift_at, steps):
        if separations[i] > 0:
            recovery = i - drift_at
            break

    return {
        "condition": cond.name,
        "separation_pre": avg_window(separations, drift_at - 40, drift_at),
        "separation_post": avg_window(separations, steps - 40, steps),
        "recovery_steps": recovery,
        "total_weight_delta": round(total_weights[-1] - total_weights[0], 2),
        "total_weight_end": round(total_weights[-1], 2),
    }


def main():
    all_results = []
    for seed in range(10):
        seed_results = {}
        for cond in CONDITIONS:
            rng = random.Random(seed)
            g, info = build_graph(rng)
            g_copy = copy.deepcopy(g)
            res = run_condition(g_copy, info, cond, seed=seed + 100)
            seed_results[cond.name] = res
        all_results.append({"seed": seed, "conditions": seed_results})

    # Aggregate
    summary = {}
    for cond in CONDITIONS:
        vals = {
            "separation_pre": [],
            "separation_post": [],
            "recovery_steps": [],
            "total_weight_delta": [],
        }
        for run in all_results:
            r = run["conditions"][cond.name]
            vals["separation_pre"].append(r["separation_pre"])
            vals["separation_post"].append(r["separation_post"])
            if r["recovery_steps"] is not None:
                vals["recovery_steps"].append(r["recovery_steps"])
            vals["total_weight_delta"].append(r["total_weight_delta"])

        summary[cond.name] = {
            k: round(statistics.mean(v), 4) if v else None
            for k, v in vals.items()
        }

    output = {"summary": summary, "runs": all_results}
    RESULT_PATH.write_text(json.dumps(output, indent=2))

    print("\n=== Ablation Study (v12.2.0) ===")
    print(f"  10 seeds, 400 steps, drift at 200\n")
    print(f"{'Condition':<25} {'Sep Pre':>10} {'Sep Post':>10} {'Recovery':>10} {'Wt Delta':>10}")
    print("-" * 67)
    for cond in CONDITIONS:
        s = summary[cond.name]
        rec = str(s["recovery_steps"]) if s["recovery_steps"] is not None else "N/A"
        print(f"{cond.name:<25} {s['separation_pre']:>10} {s['separation_post']:>10} {rec:>10} {s['total_weight_delta']:>10}")


if __name__ == "__main__":
    main()
