"""
CrabPath Memory Graph — A weighted directed graph where activation spreads.

That's it. Nodes hold content. Edges have weights (positive or negative).
Activation propagates. Weights learn from outcomes.

Node types, edge types, and learning rules are yours to define.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    import networkx as nx
except ImportError:
    nx = None


@dataclass
class Node:
    """A node in the memory graph."""
    id: str
    content: str
    summary: str = ""
    tags: list[str] = field(default_factory=list)
    type: str = ""              # user-defined type (e.g. "fact", "rule", "tool")
    prior: float = 0.0          # base-level activation (higher = more likely to fire)
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    """A weighted directed edge."""
    source: str
    target: str
    weight: float = 1.0         # positive = excitatory, negative = inhibitory
    type: str = ""              # user-defined type (e.g. "association", "sequence", "blocks")
    metadata: dict[str, Any] = field(default_factory=dict)


class Graph:
    """
    A weighted directed graph for memory.

    Nodes hold content. Edges have weights.
    That's the foundation — build whatever you want on top.
    """

    def __init__(self):
        if nx is None:
            raise ImportError("CrabPath requires networkx: pip install networkx")
        self._G = nx.DiGraph()
        self._nodes: dict[str, Node] = {}
        self._edges: dict[tuple[str, str], Edge] = {}

    def add_node(self, node: Node) -> None:
        self._nodes[node.id] = node
        self._G.add_node(node.id)

    def get_node(self, node_id: str) -> Optional[Node]:
        return self._nodes.get(node_id)

    def remove_node(self, node_id: str) -> None:
        self._nodes.pop(node_id, None)
        to_remove = [k for k in self._edges if node_id in k]
        for k in to_remove:
            self._edges.pop(k)
        if node_id in self._G:
            self._G.remove_node(node_id)

    def add_edge(self, edge: Edge) -> None:
        key = (edge.source, edge.target)
        self._edges[key] = edge
        self._G.add_edge(edge.source, edge.target, weight=edge.weight)

    def get_edge(self, source: str, target: str) -> Optional[Edge]:
        return self._edges.get((source, target))

    def neighbors(self, node_id: str) -> list[tuple[Node, Edge]]:
        """Get outgoing neighbors with their edges."""
        result = []
        for (src, tgt), edge in self._edges.items():
            if src == node_id:
                node = self._nodes.get(tgt)
                if node:
                    result.append((node, edge))
        return result

    def nodes(self, type: Optional[str] = None) -> list[Node]:
        """Get all nodes, optionally filtered by type."""
        if type:
            return [n for n in self._nodes.values() if n.type == type]
        return list(self._nodes.values())

    def edges(self) -> list[Edge]:
        return list(self._edges.values())

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    def save(self, path: str) -> None:
        """Save the graph to a JSON file."""
        import json
        data = {
            "nodes": [
                {k: v for k, v in n.__dict__.items()}
                for n in self._nodes.values()
            ],
            "edges": [
                {k: v for k, v in e.__dict__.items()}
                for e in self._edges.values()
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Graph":
        """Load a graph from a JSON file."""
        import json
        with open(path) as f:
            data = json.load(f)
        g = cls()
        for nd in data["nodes"]:
            g.add_node(Node(**nd))
        for ed in data["edges"]:
            g.add_edge(Edge(**ed))
        return g

    def __repr__(self) -> str:
        return f"Graph(nodes={self.node_count}, edges={self.edge_count})"
