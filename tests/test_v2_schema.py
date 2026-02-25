"""Tests for v2 graph schema migration and compatibility helpers."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from crabpath import Edge, EmbeddingIndex, Graph, Node


def test_node_and_edge_v2_defaults() -> None:
    n = Node(id="n", content="content")
    e = Edge(source="n", target="n")

    assert n.summary == ""
    assert n.type == "fact"
    assert e.decay_rate == 0.01
    assert e.last_followed_ts is None
    assert e.created_by == "manual"
    assert e.follow_count == 0
    assert e.skip_count == 0


def test_backward_compatible_load_from_legacy_records(tmp_path: Path) -> None:
    legacy = {
        "nodes": [
            {"id": "a", "content": "alpha", "metadata": {"type": "rule"}},
            {"id": "b", "content": "beta"},
        ],
        "edges": [
            {"source": "a", "target": "b", "weight": 0.5},
        ],
    }

    input_path = tmp_path / "legacy_graph.json"
    input_path.write_text(json.dumps(legacy), encoding="utf-8")

    graph = Graph.load(str(input_path))
    a = graph.get_node("a")
    b = graph.get_node("b")
    edge = graph.get_edge("a", "b")

    assert a is not None
    assert a.summary == ""
    assert a.type == "rule"
    assert a.potential == 0.0
    assert b is not None
    assert b.summary == ""
    assert b.type == "fact"
    assert edge is not None
    assert edge.decay_rate == 0.01
    assert edge.created_by == "manual"
    assert edge.last_followed_ts is None
    assert edge.follow_count == 0
    assert edge.skip_count == 0


def test_migration_script_backfills_fields(tmp_path: Path) -> None:
    legacy = {
        "nodes": [
            {"id": "a", "content": "alpha"},
            {"id": "b", "content": "beta", "summary": "short", "type": "goal"},
        ],
        "edges": [
            {"source": "a", "target": "b", "weight": 0.5},
        ],
    }

    input_path = tmp_path / "legacy_graph.json"
    output_path = tmp_path / "migrated_graph.json"
    input_path.write_text(json.dumps(legacy), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "scripts/migrate_graph_v2.py", str(input_path), str(output_path)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Migrated 2 nodes and 1 edges" in result.stdout
    assert output_path.exists()

    migrated = Graph.load(str(output_path))
    assert migrated.get_node("a").summary == ""
    assert migrated.get_node("b").summary == "short"
    assert migrated.get_node("b").type == "goal"
    assert migrated.get_edge("a", "b").decay_rate == 0.01
    assert migrated.get_edge("a", "b").created_by == "manual"


def test_embedding_build_indexes_summary_over_full_content() -> None:
    graph = Graph()
    graph.add_node(Node(id="summarized", content="the long text", summary="short summary"))
    graph.add_node(Node(id="full", content="only full text"))

    seen_texts: list[str] = []

    def embed_fn(texts: list[str]) -> list[list[float]]:
        seen_texts.extend(texts)
        return [[1.0, 0.0] for _ in texts]

    index = EmbeddingIndex()
    index.build(graph, embed_fn, batch_size=2)

    assert "short summary" in seen_texts
    assert "only full text" in seen_texts
    assert "the long text" not in seen_texts


def test_embedding_upsert_allows_vector_input() -> None:
    index = EmbeddingIndex()
    index.upsert("n1", [0.2, 0.4, 0.6])

    assert index.vectors["n1"] == [0.2, 0.4, 0.6]
