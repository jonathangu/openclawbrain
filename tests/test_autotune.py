from __future__ import annotations

from crabpath.autotune import GraphHealth, autotune, measure_health
from crabpath.graph import Edge, Graph, Node


def test_measure_health_counts_percentages() -> None:
    graph = Graph()
    graph.add_node(Node("a", "A", metadata={"file": "f1"}))
    graph.add_node(Node("b", "B", metadata={"file": "f1"}))
    graph.add_node(Node("c", "C", metadata={"file": "f1"}))
    graph.add_edge(Edge("a", "b", 0.9))
    graph.add_edge(Edge("b", "c", 0.5))

    health = measure_health(graph)
    assert health.reflex_pct == 0.5
    assert health.habitual_pct == 0.5
    assert health.orphan_nodes == 0


def test_autotune_recommends_actions_for_dormant_graph() -> None:
    graph = Graph()
    graph.add_node(Node("a", "A"))
    graph.add_node(Node("b", "B"))
    graph.add_edge(Edge("a", "b", 0.05))

    health = measure_health(graph)
    changes = autotune(graph, health)
    knobs = {item["knob"] for item in changes}
    assert "decay_half_life" in knobs
    assert "promotion_threshold" in knobs
