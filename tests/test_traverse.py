from __future__ import annotations

from crabpath.graph import Edge, Graph, Node
from crabpath.traverse import TraversalConfig, traverse


def _graph_for_traverse() -> Graph:
    g = Graph()
    g.add_node(Node("a", "A"))
    g.add_node(Node("b", "B"))
    g.add_node(Node("c", "C"))
    g.add_edge(Edge("a", "b", 0.9))
    g.add_edge(Edge("a", "c", 0.5))
    return g


def test_traverse_tiers() -> None:
    g = _graph_for_traverse()
    result = traverse(g, seeds=[("a", 1.0)], config=TraversalConfig(max_hops=1, beam_width=3))
    assert "a" in result.fired
    assert result.steps
    tiers = {step.tier for step in result.steps}
    assert "reflex" in tiers
    assert "habitual" in tiers

def test_route_fn_controls_habitual_nodes() -> None:
    g = Graph()
    g.add_node(Node("a", "A"))
    g.add_node(Node("b", "B"))
    g.add_node(Node("c", "C"))
    g.add_edge(Edge("a", "b", 0.4))
    g.add_edge(Edge("a", "c", 0.4))

    result = traverse(
        g,
        seeds=[("a", 1.0)],
        config=TraversalConfig(max_hops=1, beam_width=2),
        route_fn=lambda _query, cands: [node for node in cands if node == "c"],
    )
    assert "c" in result.fired
    assert "b" not in result.fired


def test_edge_damping_repeats_reduce_weight() -> None:
    g = Graph()
    g.add_node(Node("x", "X"))
    g.add_node(Node("y", "Y"))
    g.add_edge(Edge("x", "y", 0.9))
    g.add_edge(Edge("y", "x", 0.9))

    result = traverse(g, seeds=[("x", 1.0)], config=TraversalConfig(max_hops=4, beam_width=1, edge_damping=0.3))
    assert len(result.steps) >= 3
    # same directed edge used again should decay
    assert result.steps[0].effective_weight == 0.9
    assert result.steps[2].effective_weight in (0.27, 0.081, 0.024299999999999998)
