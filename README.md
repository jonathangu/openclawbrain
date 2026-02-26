# 🦀 CrabPath

Memory graph for AI agents. Learns what to retrieve — and what to suppress.

## Why?

- RAG loads by similarity. CrabPath loads by learned routes.
- Positive outcomes strengthen paths. Negative outcomes create inhibitory edges.
- The graph gets smarter with every query. Unused paths decay.

## Install

pip install crabpath

Python 3.10+. Zero dependencies.

## Quick Start

```python
from crabpath import Graph, Node, Edge, traverse, TraversalConfig
from crabpath.learn import apply_outcome

g = Graph()
g.add_node(Node("a", "Deploy to production"))
g.add_node(Node("b", "Rollback procedure"))
g.add_edge(Edge("a", "b", 0.7))

result = traverse(g, seeds=[("a", 1.0)], config=TraversalConfig(max_hops=3))
print(result.fired)

apply_outcome(g, result.fired, outcome=1.0)
```

## For AI Agents (CLI)

- `crabpath init --workspace <dir> --output <path>`
- `crabpath query <text> --graph <path> --index <path> [--top N] [--json]`
- `crabpath learn --graph <path> --outcome <float> --fired-ids <id1,id2,...>`
- `crabpath health --graph <path>`

## How It Works

- `split_workspace` scans markdown, creates nodes, and sibling edges.
- `VectorIndex` stores caller-provided embeddings and performs cosine search.
- `traverse` follows reflex/habitual/dormant tiers with edge damping.
- `apply_outcome` applies policy-style updates and Hebbian co-firing.
- `apply_decay` slowly forgets weak unused edges.
- `autotune` suggests safe defaults from graph health metrics.

## Links

Paper / GitHub / PyPI
