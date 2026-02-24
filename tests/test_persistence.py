"""Tests for save/load."""

import tempfile
from pathlib import Path

from crabpath import Graph, Node, Edge, activate, learn


def test_save_load_roundtrip():
    g = Graph()
    g.add_node(Node(id="a", content="hello", type="fact", tags=["test"]))
    g.add_node(Node(id="b", content="world", type="rule"))
    g.add_edge(Edge(source="a", target="b", weight=0.7, type="seq"))

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    g.save(path)
    g2 = Graph.load(path)

    assert g2.node_count == 2
    assert g2.edge_count == 1
    assert g2.get_node("a").content == "hello"
    assert g2.get_node("a").type == "fact"
    assert g2.get_node("a").tags == ["test"]
    assert g2.get_edge("a", "b").weight == 0.7

    Path(path).unlink()


def test_save_load_preserves_learned_weights():
    g = Graph()
    g.add_node(Node(id="a", content="A"))
    g.add_node(Node(id="b", content="B"))
    g.add_edge(Edge(source="a", target="b", weight=0.5))

    # Learn
    result = activate(g, seeds={"a": 1.0})
    learn(g, result, outcome=1.0, learning_rate=0.2)

    new_weight = g.get_edge("a", "b").weight

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    g.save(path)
    g2 = Graph.load(path)

    assert g2.get_edge("a", "b").weight == new_weight

    Path(path).unlink()
