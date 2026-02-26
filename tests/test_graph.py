from __future__ import annotations

from crabpath.graph import Edge, Graph, Node


def test_graph_add_and_get() -> None:
    graph = Graph()
    graph.add_node(Node("a", "alpha"))
    graph.add_node(Node("b", "beta", summary="s", metadata={"tag": "x"}))
    graph.add_edge(Edge("a", "b", 0.7, kind="sibling"))

    assert graph.node_count() == 2
    assert graph.edge_count() == 1
    assert graph.get_node("a") is not None
    assert graph.get_node("a").content == "alpha"
    assert graph.outgoing("a")[0][1].weight == 0.7
    assert graph.incoming("b")[0][0].id == "a"


def test_graph_save_load(tmp_path) -> None:
    graph = Graph()
    graph.add_node(Node("a", "alpha"))
    graph.add_node(Node("b", "beta"))
    graph.add_edge(Edge("a", "b", 0.5))

    path = tmp_path / "graph.json"
    graph.save(str(path))

    loaded = Graph.load(str(path))
    assert loaded.node_count() == 2
    assert loaded.edge_count() == 1
    assert loaded.get_node("b").content == "beta"
    assert loaded.outgoing("a")[0][1].weight == 0.5
