"""
CrabPath agent integration with the canonical interface.

Run: python examples/agent_memory.py
"""

from pathlib import Path

from crabpath import CrabPathAgent, Edge, Graph, Node

GRAPH_PATH = "agent_memory.json"
INDEX_PATH = "agent_memory.index.json"


def bootstrap() -> Graph:
    g = Graph()

    g.add_node(Node(id="check-config", content="git diff HEAD~1 -- config/"))
    g.add_node(Node(id="check-logs", content="tail -n 200 /var/log/svc.log"))
    g.add_node(Node(id="verify-prod", content="Test on prod before reporting fixed"))
    g.add_node(Node(id="restart-svc", content="systemctl restart app"))
    g.add_node(Node(id="no-untested", content="Never claim fixed without testing", threshold=0.5))

    g.add_edge(Edge(source="check-config", target="check-logs", weight=1.5))
    g.add_edge(Edge(source="check-logs", target="verify-prod", weight=1.2))
    g.add_edge(Edge(source="check-logs", target="restart-svc", weight=0.8))
    g.add_edge(Edge(source="no-untested", target="check-config", weight=-1.0))

    return g


if Path(GRAPH_PATH).exists() and Path(INDEX_PATH).exists():
    agent = CrabPathAgent(graph_path=GRAPH_PATH, index_path=INDEX_PATH, embed_fn=None)
    agent.load()
    print(f"Loaded graph: {agent.graph}")
else:
    graph = bootstrap()
    agent = CrabPathAgent(graph_path=GRAPH_PATH, index_path=INDEX_PATH, embed_fn=None)
    agent.graph = graph
    agent.index.build(agent.graph, agent._embedding_fn())
    agent.save()
    print(f"Bootstrapped new graph: {agent.graph}")


task = "The deployment is broken, config was changed"
result = agent.query(task, top_k=6, max_hops=3)

print(f"\nTask: {task}")
print(f"Band: {result['band']} | cosine: {result['best_cosine']:.3f}")
print(f"Nodes: {result['nodes']}")
print(f"Context chars: {result['chars']}")
print("Context:")
print(result["context"] or "<empty context>")

if result.get("auto_node", {}).get("created"):
    print(f"Created auto node: {result['auto_node']['node_id']}")

agent.learn(1.0)
print("\nApplied explicit learn(+1.0)")
for edge in [("check-config", "check-logs")]:
    updated = agent.graph.get_edge(*edge)
    if updated:
        print(f"  {edge[0]}->{edge[1]} weight now {updated.weight:.4f}")

agent.save()
print(f"\nSaved graph to {GRAPH_PATH}")

