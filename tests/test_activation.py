"""Tests for CrabPath activation."""

from crabpath import Graph, Node, Edge, activate, learn


def _simple_graph() -> Graph:
    """A -> B -> C, with A -inhibits-> D"""
    g = Graph()
    g.add_node(Node(id="a", content="start node", tags=["start"]))
    g.add_node(Node(id="b", content="middle node"))
    g.add_node(Node(id="c", content="end node"))
    g.add_node(Node(id="d", content="bad node"))
    
    g.add_edge(Edge(source="a", target="b", weight=0.9))
    g.add_edge(Edge(source="b", target="c", weight=0.8))
    g.add_edge(Edge(source="a", target="d", weight=-1.0))  # inhibition
    return g


def test_basic_activation():
    g = _simple_graph()
    result = activate(g, seeds={"a": 1.0})
    
    assert len(result.nodes) > 0
    assert result.hops > 0


def test_activation_spreads():
    g = _simple_graph()
    result = activate(g, seeds={"a": 1.0})
    
    ids = [n.id for n, _ in result.nodes]
    assert "b" in ids  # should spread from a to b


def test_inhibition():
    g = _simple_graph()
    result = activate(g, seeds={"a": 1.0})
    
    assert "d" in result.inhibited
    ids = [n.id for n, _ in result.nodes]
    assert "d" not in ids


def test_empty_graph():
    g = Graph()
    result = activate(g, seeds={})
    assert len(result.nodes) == 0


def test_damping():
    g = _simple_graph()
    
    # High damping = more spread
    r1 = activate(g, seeds={"a": 1.0}, damping=0.99)
    # Low damping = less spread
    r2 = activate(g, seeds={"a": 1.0}, damping=0.1)
    
    # With high damping, downstream nodes should have higher activation
    scores_high = {n.id: s for n, s in r1.nodes}
    scores_low = {n.id: s for n, s in r2.nodes}
    
    if "b" in scores_high and "b" in scores_low:
        assert scores_high["b"] >= scores_low["b"]


def test_learning_strengthens():
    g = _simple_graph()
    result = activate(g, seeds={"a": 1.0})
    
    old_weight = g.get_edge("a", "b").weight
    learn(g, result, outcome=1.0)
    new_weight = g.get_edge("a", "b").weight
    
    assert new_weight >= old_weight


def test_learning_weakens():
    g = _simple_graph()
    result = activate(g, seeds={"a": 1.0})
    
    old_weight = g.get_edge("a", "b").weight
    learn(g, result, outcome=-1.0)
    new_weight = g.get_edge("a", "b").weight
    
    assert new_weight <= old_weight


def test_top_k():
    g = Graph()
    for i in range(20):
        g.add_node(Node(id=f"n{i}", content=f"node {i}"))
    # Connect all to node 0
    for i in range(1, 20):
        g.add_edge(Edge(source="n0", target=f"n{i}", weight=0.5))
    
    result = activate(g, seeds={"n0": 1.0}, top_k=5)
    assert len(result.nodes) <= 5


def test_custom_types():
    """User-defined types work fine."""
    g = Graph()
    g.add_node(Node(id="a", content="a", type="my-custom-type"))
    g.add_node(Node(id="b", content="b", type="another-type"))
    g.add_edge(Edge(source="a", target="b", weight=1.0, type="my-edge-type"))
    
    result = activate(g, seeds={"a": 1.0})
    assert len(result.nodes) > 0
