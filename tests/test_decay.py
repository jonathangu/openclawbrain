from __future__ import annotations

from crabpath.decay import DecayConfig, apply_decay
from crabpath.graph import Edge, Graph, Node


def test_decay_halves_weights() -> None:
    graph = Graph()
    graph.add_node(Node("a", "A"))
    graph.add_node(Node("b", "B"))
    graph.add_edge(Edge("a", "b", 1.0))

    changed = apply_decay(graph, config=DecayConfig(half_life=1, min_weight=0.0))
    assert changed == 1
    assert abs(graph._edges["a"]["b"].weight - 0.5) < 1e-9
