# 🦀 CrabPath

**Neuron-inspired memory graphs for AI agents. Everything evolves into this.**

---

In biology, [carcinisation](https://en.wikipedia.org/wiki/Carcinisation) is the phenomenon where crustaceans independently evolve into crab-like forms — over and over. Nature keeps reinventing the crab because it *works*.

CrabPath is a bet that agent memory will converge the same way.

## The Model

A node is a neuron. That's the whole idea.

Each node has **content** (what it knows), a **potential** (accumulated energy), and a **threshold** (when to fire). Edges are weighted pointers — positive for excitation, negative for inhibition.

When you query the graph:
1. **Seed nodes** receive energy
2. Nodes whose potential crosses their **threshold** → **fire**
3. Firing sends **weighted energy** along outgoing edges
   - Positive weight → adds energy to target (excitatory)
   - Negative weight → removes energy from target (inhibitory)
4. Fired nodes **reset** (refractory) — they don't fire again
5. Unfired potentials **decay** each step (leak)
6. Repeat until nothing fires or max steps reached

Learning is Hebbian: nodes that fire together strengthen their connections. Nodes that fire during failure weaken them.

**Zero dependencies.** Pure Python. The whole library is two files.

## Install

```bash
pip install crabpath
```

## Quick Start

```python
from crabpath import Graph, Node, Edge, activate, learn

g = Graph()

# Nodes are neurons: content + threshold
g.add_node(Node(id="check-logs", content="tail -n 200 /var/log/service.log"))
g.add_node(Node(id="check-config", content="git diff HEAD~1 -- config/"))
g.add_node(Node(id="no-untested-fix", content="Never claim fixed without testing"))
g.add_node(Node(id="claim-fixed", content="Tell user it's fixed", threshold=2.0))

# Positive edges: these go together
g.add_edge(Edge(source="check-config", target="check-logs", weight=1.5))

# Negative edges: this blocks that
g.add_edge(Edge(source="no-untested-fix", target="claim-fixed", weight=-1.0))

# Fire
result = activate(g, seeds={"check-config": 1.0, "no-untested-fix": 1.0})

for node, energy in result.fired:
    print(f"  [{energy:.2f}] {node.content}")
# "claim-fixed" is inhibited:
print(f"  Inhibited: {result.inhibited}")

# Learn from outcome
learn(g, result, outcome=1.0)   # success → strengthen
learn(g, result, outcome=-1.0)  # failure → weaken

# Persist
g.save("memory.json")
g2 = Graph.load("memory.json")
```

## API

### Node

```python
Node(
    id="...",           # unique identifier
    content="...",      # what this neuron knows
    threshold=1.0,      # fires when potential >= threshold
    potential=0.0,      # current energy (transient)
    metadata={},        # your bag of whatever (types, tags, timestamps — your call)
)
```

### Edge

```python
Edge(
    source="a",         # from node
    target="b",         # to node
    weight=1.0,         # positive = excitatory, negative = inhibitory
)
```

### Graph

```python
g = Graph()

g.add_node(node)
g.get_node("id")            # Node or None
g.remove_node("id")         # removes node + connected edges
g.nodes()                   # all nodes

g.add_edge(edge)
g.get_edge("a", "b")        # Edge or None
g.outgoing("a")             # [(target_node, edge), ...]
g.incoming("b")             # [(source_node, edge), ...]
g.edges()                   # all edges

g.reset_potentials()         # set all potentials to 0
g.save("path.json")
Graph.load("path.json")
```

### Activation

```python
result = activate(
    graph,
    seeds={"node-a": 1.0, "node-b": 0.5},
    max_steps=3,     # propagation rounds
    decay=0.1,       # potential leak per step
    top_k=10,        # max nodes to return
)

result.fired       # [(Node, energy_at_firing), ...] sorted descending
result.inhibited   # [node_id, ...] driven below 0
result.steps       # propagation rounds completed
```

### Learning

```python
learn(graph, result, outcome=1.0, rate=0.1)
# outcome > 0 → strengthen edges between co-fired nodes
# outcome < 0 → weaken them
# weights clamped to [-10, 10]
```

## Design Principles

1. **Minimum assumptions.** Nodes have `id`, `content`, `threshold`, `potential`, `metadata`. No type system, no tags, no timestamps — put what you want in metadata.
2. **Zero dependencies.** Pure Python. Plain dicts internally. No NetworkX, no numpy, no nothing.
3. **Neuron-faithful.** Leaky integrate-and-fire: accumulate, threshold, fire, propagate, refractory, decay. Positive and negative energy are first-class.
4. **Persistence is JSON.** Save/load snapshots. Defaults are omitted for compact files.

## Why "CrabPath"?

🦀 Everything evolves into a crab. We think everything in agent memory evolves into this: weighted graphs, neuron-style activation, inhibition, outcome learning. CrabPath is the path everything walks.

## The Paper

📄 **[jonathangu.com/crabpath](https://jonathangu.com/crabpath/)** — full architecture, biological inspiration, math, and experimental plan.

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Citation

```bibtex
@misc{gu2026crabpath,
  title={CrabPath: Neuron-Inspired Memory Graphs for AI Agents},
  author={Gu, Jonathan},
  year={2026},
  url={https://github.com/jonathangu/crabpath}
}
```

---

*Built by [Jonathan Gu](https://jonathangu.com)* 🦀
