"""Tests for the CrabPath graph."""

from crabpath import Graph, Node, Edge


def test_create_graph():
    g = Graph()
    assert g.node_count == 0
    assert g.edge_count == 0


def test_add_node():
    g = Graph()
    g.add_node(Node(id="a", content="hello"))
    assert g.node_count == 1
    assert g.get_node("a").content == "hello"


def test_add_edge():
    g = Graph()
    g.add_node(Node(id="a", content="A"))
    g.add_node(Node(id="b", content="B"))
    g.add_edge(Edge(source="a", target="b", weight=0.8))
    assert g.edge_count == 1


def test_negative_edge():
    g = Graph()
    g.add_node(Node(id="rule", content="don't do X", type="rule"))
    g.add_node(Node(id="bad", content="do X", type="action"))
    g.add_edge(Edge(source="rule", target="bad", weight=-1.0))
    assert g.get_edge("rule", "bad").weight == -1.0


def test_neighbors():
    g = Graph()
    g.add_node(Node(id="a", content="A"))
    g.add_node(Node(id="b", content="B"))
    g.add_node(Node(id="c", content="C"))
    g.add_edge(Edge(source="a", target="b", weight=0.5))
    g.add_edge(Edge(source="a", target="c", weight=0.9))
    
    neighbors = g.neighbors("a")
    assert len(neighbors) == 2


def test_filter_by_type():
    g = Graph()
    g.add_node(Node(id="f1", content="fact 1", type="fact"))
    g.add_node(Node(id="f2", content="fact 2", type="fact"))
    g.add_node(Node(id="r1", content="rule 1", type="rule"))
    
    assert len(g.nodes(type="fact")) == 2
    assert len(g.nodes(type="rule")) == 1
    assert len(g.nodes()) == 3


def test_remove_node():
    g = Graph()
    g.add_node(Node(id="a", content="A"))
    g.add_node(Node(id="b", content="B"))
    g.add_edge(Edge(source="a", target="b"))
    
    g.remove_node("a")
    assert g.node_count == 1
    assert g.edge_count == 0
    assert g.get_node("a") is None


def test_repr():
    g = Graph()
    g.add_node(Node(id="a", content="A"))
    assert "nodes=1" in repr(g)
