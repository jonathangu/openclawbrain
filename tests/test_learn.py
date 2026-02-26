from __future__ import annotations

from crabpath.graph import Edge, Graph, Node
from crabpath.learn import LearningConfig, apply_outcome


def test_apply_outcome_and_hebbian_update() -> None:
    graph = Graph()
    graph.add_node(Node("a", "A"))
    graph.add_node(Node("b", "B"))
    graph.add_node(Node("c", "C"))
    graph.add_edge(Edge("a", "b", 0.5))
    graph.add_edge(Edge("b", "c", 0.5))

    cfg = LearningConfig(learning_rate=0.1)
    updates = apply_outcome(graph, ["a", "b", "c"], outcome=1.0, config=cfg)
    assert "a->b" in updates
    assert graph._edges["a"]["b"].weight > 0.5
    # Hebbian creates/strengthens co-firing relations
    assert graph._edges["a"].get("c") is not None


def test_negative_outcome_creates_inhibitory() -> None:
    graph = Graph()
    graph.add_node(Node("a", "A"))
    graph.add_node(Node("b", "B"))
    graph.add_edge(Edge("a", "b", 0.5))

    apply_outcome(graph, ["a", "b"], outcome=-1.0)
    assert graph.get_node("a") is not None
    assert graph._edges["a"]["b"].weight < 0.5
    assert graph._edges["a"]["b"].kind == "inhibitory"
