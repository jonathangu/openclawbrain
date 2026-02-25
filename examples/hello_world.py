"""
Mini example of the canonical CrabPathAgent interface.

Run: python examples/hello_world.py
"""

from __future__ import annotations

from pathlib import Path
from crabpath import CrabPathAgent, Edge, Graph, Node


GRAPH_PATH = "hello_world_graph.json"
INDEX_PATH = "hello_world_graph.index.json"


def build_graph() -> Graph:
    g = Graph()
    g.add_node(Node(id="start", content="Investigate the incident"))
    g.add_node(Node(id="check", content="Check recent logs and error messages"))
    g.add_node(Node(id="restart", content="Restart the failing service"))
    g.add_node(Node(id="verify", content="Verify service recovers"))
    g.add_node(Node(id="report", content="Report what was changed"))

    g.add_edge(Edge(source="start", target="check", weight=0.55))
    g.add_edge(Edge(source="check", target="verify", weight=0.60))
    g.add_edge(Edge(source="check", target="restart", weight=0.45))
    g.add_edge(Edge(source="restart", target="verify", weight=0.55))
    g.add_edge(Edge(source="verify", target="report", weight=0.62))
    return g


def show_weight(agent: CrabPathAgent, source: str, target: str) -> float:
    edge = agent.graph.get_edge(source, target)
    return 0.0 if edge is None else edge.weight


def main() -> None:
    graph_path = Path(GRAPH_PATH)
    index_path = Path(INDEX_PATH)

    if graph_path.exists() and index_path.exists():
        agent = CrabPathAgent(str(graph_path), str(index_path), embed_fn=None)
    else:
        agent = CrabPathAgent(str(graph_path), str(index_path), embed_fn=None)
        agent.graph = build_graph()
        agent.index.build(agent.graph, agent._embedding_fn())
        agent.save()

    print("Running query...")
    first = agent.query("service is failing, check logs first then recover", top_k=5, max_hops=3)
    print(f"First band: {first['band']} | nodes: {first['nodes']}")
    print("Context:")
    print(first["context"] or "<empty>")

    before = show_weight(agent, "check", "verify")
    print(f"Weight check->verify before: {before:.3f}")

    agent.learn(1.0)
    after_success = show_weight(agent, "check", "verify")
    print(f"Weight check->verify after success: {after_success:.3f}")

    second = agent.query("how do we verify recovery after restart", top_k=5, max_hops=3)
    print(f"Second band: {second['band']} | nodes: {second['nodes']}")
    print(f"Weight check->verify now: {show_weight(agent, 'check', 'verify'):.3f}")


if __name__ == "__main__":
    main()

