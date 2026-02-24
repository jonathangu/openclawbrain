# 🦀 CrabPath

**Activation-driven memory graphs for AI agents. Everything evolves into this.**

---

In biology, [carcinisation](https://en.wikipedia.org/wiki/Carcinisation) is the phenomenon where crustaceans independently evolve into crab-like forms — over and over again. Nature keeps reinventing the crab because it *works*.

CrabPath is a bet that agent memory will converge the same way. Instead of loading a pile of files every session or doing naive vector search, your agent should:

1. **Spread activation** through a learned memory graph
2. **Load only what lights up** — the handful of things that matter for *this* task
3. **Learn from outcomes** — successful paths strengthen, failed paths weaken
4. **Inhibit bad patterns** — negative edges suppress known mistakes

## Install

```bash
pip install crabpath
```

## Usage

```python
from crabpath import Graph, Node, Edge, activate, learn

# Build a graph
g = Graph()

g.add_node(Node(id="check-logs", content="tail -n 200 /var/log/service.log",
                type="action", tags=["deploy"]))
g.add_node(Node(id="check-config", content="git diff HEAD~1 -- config/",
                type="action", tags=["deploy"]))
g.add_node(Node(id="no-untested-fix", content="Never claim fixed without testing",
                type="rule", tags=["deploy"]))
g.add_node(Node(id="claim-fixed", content="Tell user it's fixed",
                type="action"))

# Positive edges: these go together
g.add_edge(Edge(source="check-config", target="check-logs", weight=0.9))

# Negative edges: this blocks that
g.add_edge(Edge(source="no-untested-fix", target="claim-fixed", weight=-1.0))

# Activate
result = activate(g, seeds={"check-config": 1.0, "no-untested-fix": 0.8})

for node, score in result.nodes:
    print(f"  [{score:.2f}] {node.type}: {node.content}")

# "claim-fixed" is inhibited:
print(f"  Inhibited: {result.inhibited}")

# Learn from outcome
learn(g, result, outcome=1.0)   # success → strengthen these paths
learn(g, result, outcome=-1.0)  # failure → weaken them
```

## How It Works

CrabPath is a directed graph where:

- **Nodes** hold content (facts, rules, actions, whatever you want)
- **Edges** have weights: positive = "these go together", negative = "this blocks that"
- **Activation spreads** from seed nodes along edges, scaled by weight, with damping
- **Inhibition** is first-class: negative edges suppress nodes (the missing primitive in most memory systems)
- **Learning** adjusts weights based on outcomes: co-activated nodes during success get strengthened

That's the whole thing. Four concepts: nodes, weighted edges, spreading activation, outcome learning.

### What you build on top is up to you

CrabPath doesn't dictate your node types, edge types, or learning strategy. Use `type="fact"` or `type="quantum_flux_capacitor"` — it's a string field. The library gives you the graph and the activation dynamics. You bring the domain knowledge.

## API

### Graph

```python
g = Graph()
g.add_node(Node(id="...", content="...", type="...", tags=[...], prior=0.5))
g.add_edge(Edge(source="a", target="b", weight=0.9, type="..."))
g.get_node("a")              # Node or None
g.get_edge("a", "b")         # Edge or None
g.neighbors("a")             # [(Node, Edge), ...]
g.nodes(type="rule")         # filtered list
g.remove_node("a")           # removes node + connected edges
```

### Activation

```python
result = activate(
    graph,
    seeds={"node-a": 1.0, "node-b": 0.5},  # where to start
    damping=0.85,    # activation decay per hop (0-1)
    max_hops=3,      # propagation depth
    top_k=10,        # nodes to return
    threshold=0.01,  # minimum activation to propagate
)

result.nodes       # [(Node, score), ...] sorted by activation
result.inhibited   # [node_id, ...] suppressed by negative edges
result.hops        # number of propagation steps
```

### Learning

```python
learn(graph, result, outcome=1.0)    # +1 = success, -1 = failure
learn(graph, result, outcome=-0.5, learning_rate=0.05)
```

## The Paper

📄 **[jonathangu.com/crabpath](https://jonathangu.com/crabpath/)** — full architecture, math, experimental plan, and the biological inspiration behind the design.

## Why "CrabPath"?

🦀 Everything evolves into a crab. We think everything in agent memory evolves into this: weighted graphs, spreading activation, inhibition, compiled reflexes. CrabPath is the path everything walks.

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Citation

```bibtex
@misc{gu2026crabpath,
  title={CrabPath: Activation-Driven Memory Graphs for AI Agents},
  author={Gu, Jonathan},
  year={2026},
  url={https://github.com/jonathangu/crabpath}
}
```

---

*Built by [Jonathan Gu](https://jonathangu.com)* 🦀
