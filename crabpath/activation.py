"""
CrabPath Activation — Neuron-style firing over a memory graph.

The model (Leaky Integrate-and-Fire, simplified):
  1. Seed nodes receive energy (potential increases)
  2. Each step: nodes whose potential >= threshold FIRE
  3. Firing sends energy along outgoing edges: weight × signal
     - Positive weight → excitatory (adds energy to target)
     - Negative weight → inhibitory (removes energy from target)
  4. After firing, potential resets to 0 (refractory)
  5. Non-fired potentials decay each step (leak)
  6. Return nodes that fired, ranked by energy at time of firing

Zero dependencies. Pure Python.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .graph import Graph, Node


@dataclass
class Firing:
    """Result of an activation pass."""

    fired: list[tuple[Node, float]]  # (node, energy_at_firing), descending
    inhibited: list[str]  # node IDs driven below 0 by inhibition
    steps: int = 0


def activate(
    graph: Graph,
    seeds: dict[str, float],
    *,
    max_steps: int = 3,
    decay: float = 0.1,
    top_k: int = 10,
) -> Firing:
    """
    Fire neurons in the graph.

    Args:
        graph: The memory graph.
        seeds: {node_id: energy} — initial energy injection.
        max_steps: Maximum propagation rounds.
        decay: Fraction of potential that leaks each step (0 = no leak, 1 = full reset).
        top_k: Number of fired nodes to return.

    Returns:
        Firing with ranked fired nodes and inhibited node IDs.
    """
    # Start clean
    graph.reset_potentials()

    # Inject seed energy
    for node_id, energy in seeds.items():
        node = graph.get_node(node_id)
        if node:
            node.potential += energy

    fired_record: dict[str, float] = {}  # node_id -> energy at firing
    inhibited: set[str] = set()
    steps_taken = 0

    for step in range(max_steps):
        # Find neurons ready to fire
        to_fire: list[Node] = []
        for node in graph.nodes():
            if node.id not in fired_record and node.potential >= node.threshold:
                to_fire.append(node)

        if not to_fire:
            break

        steps_taken = step + 1

        # Fire each neuron
        for node in to_fire:
            signal = node.potential
            fired_record[node.id] = signal

            # Send energy along outgoing edges
            for target, edge in graph.outgoing(node.id):
                target.potential += edge.weight * signal
                if target.potential < 0:
                    inhibited.add(target.id)

            # Refractory: reset potential
            node.potential = 0.0

        # Leak: decay unfired potentials
        for node in graph.nodes():
            if node.id not in fired_record and node.potential > 0:
                node.potential *= 1.0 - decay

    # Remove inhibited nodes from fired results
    for nid in inhibited:
        fired_record.pop(nid, None)

    # Rank by energy at firing, take top_k
    ranked = sorted(fired_record.items(), key=lambda x: x[1], reverse=True)[:top_k]
    result = []
    for nid, score in ranked:
        node = graph.get_node(nid)
        if node:
            result.append((node, score))

    return Firing(fired=result, inhibited=sorted(inhibited), steps=steps_taken)


def learn(
    graph: Graph,
    result: Firing,
    outcome: float,
    rate: float = 0.1,
) -> None:
    """
    Hebbian learning: adjust weights between co-fired nodes.

    Positive outcome → strengthen connections between fired nodes.
    Negative outcome → weaken them.
    Weights clamped to [-10, 10].

    Args:
        graph: The memory graph.
        result: A previous Firing result.
        outcome: Positive = good, negative = bad.
        rate: Learning rate.
    """
    fired_ids = [n.id for n, _ in result.fired]

    for src_id in fired_ids:
        for tgt_id in fired_ids:
            if src_id == tgt_id:
                continue
            edge = graph.get_edge(src_id, tgt_id)
            if edge:
                edge.weight += rate * outcome
                edge.weight = max(-10.0, min(10.0, edge.weight))
