"""
CrabPath Activation — Spreading activation over a memory graph.

The core algorithm:
  1. Seed some nodes with initial activation
  2. Activation spreads along edges, scaled by weight
  3. Negative weights suppress (inhibit) targets
  4. Damping prevents runaway
  5. Return the top-K activated nodes

That's the whole thing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from .graph import Graph, Node


@dataclass
class Result:
    """What comes back from an activation query."""
    nodes: list[tuple[Node, float]]   # (node, activation_score), sorted descending
    inhibited: list[str]              # node IDs that were suppressed
    hops: int = 0


def activate(
    graph: Graph,
    seeds: dict[str, float],
    *,
    damping: float = 0.85,
    max_hops: int = 3,
    top_k: int = 10,
    threshold: float = 0.01,
) -> Result:
    """
    Spread activation through the graph.

    Args:
        graph: The memory graph
        seeds: {node_id: initial_activation} — where to start
        damping: How much activation decays per hop (0-1, lower = more decay)
        max_hops: Maximum propagation depth
        top_k: Number of nodes to return
        threshold: Minimum activation to keep propagating

    Returns:
        Result with top-K activated nodes and inhibited node IDs
    """
    activations: dict[str, float] = dict(seeds)
    inhibited: set[str] = set()

    for hop in range(max_hops):
        new_act: dict[str, float] = {}

        for node_id, act in activations.items():
            if act < threshold:
                continue

            for neighbor, edge in graph.neighbors(node_id):
                if edge.weight < 0:
                    # Negative weight = inhibition
                    inhibited.add(neighbor.id)
                    continue

                propagated = damping * act * edge.weight
                new_act[neighbor.id] = new_act.get(neighbor.id, 0.0) + propagated

        # Check convergence
        if not new_act:
            break

        # Merge
        for nid, act in new_act.items():
            activations[nid] = activations.get(nid, 0.0) + act

    # Apply inhibition
    for nid in inhibited:
        activations.pop(nid, None)

    # Sort and take top-K
    sorted_nodes = sorted(activations.items(), key=lambda x: x[1], reverse=True)[:top_k]

    result_nodes = []
    for nid, score in sorted_nodes:
        node = graph.get_node(nid)
        if node:
            node.access_count += 1
            node.last_accessed = time.time()
            result_nodes.append((node, score))

    return Result(nodes=result_nodes, inhibited=list(inhibited), hops=hop + 1)


def learn(
    graph: Graph,
    result: Result,
    outcome: float,
    learning_rate: float = 0.1,
) -> None:
    """
    Update edge weights based on outcome.

    Positive outcome → strengthen edges between activated nodes.
    Negative outcome → weaken them.

    Args:
        graph: The memory graph
        result: A previous activation result
        outcome: +1.0 for success, -1.0 for failure (or anything in between)
        learning_rate: How fast to adjust weights
    """
    activated_ids = [n.id for n, _ in result.nodes]

    for i, src_id in enumerate(activated_ids[:-1]):
        for tgt_id in activated_ids[i + 1:]:
            edge = graph.get_edge(src_id, tgt_id)
            if edge:
                edge.weight += learning_rate * outcome
                edge.weight = max(-10.0, min(10.0, edge.weight))  # clamp
