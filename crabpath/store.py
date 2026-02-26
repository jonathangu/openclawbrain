"""State persistence helpers for CrabPath."""

from __future__ import annotations

import json
from pathlib import Path

from .graph import Edge, Graph, Node
from .index import VectorIndex


def save_state(graph: Graph, index: VectorIndex, path: str) -> None:
    """Save graph and index together to one JSON file."""
    payload = {
        "graph": {
            "nodes": [
                {
                    "id": node.id,
                    "content": node.content,
                    "summary": node.summary,
                    "metadata": node.metadata,
                }
                for node in graph.nodes()
            ],
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "weight": edge.weight,
                    "kind": edge.kind,
                    "metadata": edge.metadata,
                }
                for source_edges in graph._edges.values()
                for edge in source_edges.values()
            ],
        },
        "index": index._vectors,
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_state(path: str) -> tuple[Graph, VectorIndex]:
    """Load graph + index from one JSON file."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))

    graph = Graph()
    graph_payload = payload.get("graph", {})
    for node_data in graph_payload.get("nodes", []):
        graph.add_node(
            Node(
                id=node_data["id"],
                content=node_data["content"],
                summary=node_data.get("summary", ""),
                metadata=node_data.get("metadata", {}),
            )
        )

    for edge_data in graph_payload.get("edges", []):
        graph.add_edge(
            Edge(
                source=edge_data["source"],
                target=edge_data["target"],
                weight=edge_data.get("weight", 0.5),
                kind=edge_data.get("kind", "sibling"),
                metadata=edge_data.get("metadata", {}),
            )
        )

    index = VectorIndex()
    for node_id, vector in payload.get("index", {}).items():
        index.upsert(node_id, vector)
    return graph, index
