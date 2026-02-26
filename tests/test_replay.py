from __future__ import annotations

import json

from pathlib import Path

from crabpath.cli import main
from crabpath.graph import Edge, Graph, Node
from crabpath.replay import extract_queries, extract_queries_from_dir, replay_queries
from crabpath.traverse import TraversalConfig


def _write_graph_payload(path: Path) -> None:
    payload = {
        "graph": {
            "nodes": [
                {"id": "a", "content": "alpha chunk", "summary": "", "metadata": {"file": "a.md"}},
                {"id": "b", "content": "beta chunk", "summary": "", "metadata": {"file": "b.md"}},
            ],
            "edges": [
                {"source": "a", "target": "b", "weight": 0.5, "kind": "sibling", "metadata": {}},
            ],
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_extract_queries_openclaw_format(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "message",
                        "message": {
                            "role": "user",
                            "content": [{"type": "text", "text": "how do i deploy?"}],
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "message",
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "ignored"}],
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "message",
                        "message": {"role": "user", "content": [{"type": "text", "text": "roll back now"}]},
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    assert extract_queries(path) == ["how do i deploy?", "roll back now"]


def test_extract_queries_flat_format(tmp_path: Path) -> None:
    path = tmp_path / "session_flat.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"role": "assistant", "content": "ignore"}),
                json.dumps({"role": "user", "content": "restart service"}),
                json.dumps({"role": "user", "content": "check logs"}),
            ]
        ),
        encoding="utf-8",
    )

    assert extract_queries(path) == ["restart service", "check logs"]


def test_extract_queries_from_directory(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    (sessions / "a.jsonl").write_text(json.dumps({"role": "user", "content": "one"}), encoding="utf-8")
    (sessions / "b.jsonl").write_text(
        "\n".join([json.dumps({"role": "user", "content": "two"}), json.dumps({"role": "user", "content": "three"})]),
        encoding="utf-8",
    )
    (sessions / "ignore.txt").write_text("not jsonl", encoding="utf-8")

    assert extract_queries_from_dir(sessions) == ["one", "two", "three"]


def test_replay_strengthens_edges() -> None:
    graph = Graph()
    graph.add_node(Node("a", "alpha chunk", metadata={"file": "a.md"}))
    graph.add_node(Node("b", "beta chunk", metadata={"file": "a.md"}))
    graph.add_edge(Edge("a", "b", 0.5))

    stats = replay_queries(graph=graph, queries=["alpha"] * 10, config=TraversalConfig(max_hops=1))

    assert stats["queries_replayed"] == 10
    assert stats["edges_reinforced"] > 0
    assert graph.get_node("a") is not None
    assert graph.get_node("b") is not None
    assert graph._edges["a"]["b"].weight > 0.5


def test_replay_creates_cross_file_edges() -> None:
    graph = Graph()
    graph.add_node(Node("a", "alpha chunk", metadata={"file": "a.md"}))
    graph.add_node(Node("b", "beta chunk", metadata={"file": "b.md"}))

    stats = replay_queries(graph=graph, queries=["alpha beta"], config=TraversalConfig(max_hops=1))

    assert stats["queries_replayed"] == 1
    assert stats["cross_file_edges_created"] == 1

    assert graph._edges["b"]["a"].source == "b"
    assert graph._edges["b"]["a"].target == "a"


def test_replay_with_max_queries(tmp_path, capsys) -> None:
    graph_path = tmp_path / "graph.json"
    _write_graph_payload(graph_path)

    sessions = tmp_path / "sessions"
    sessions.mkdir()
    (sessions / "events.jsonl").write_text(
        "\n".join([json.dumps({"role": "user", "content": f"query {idx}"}) for idx in range(10)]),
        encoding="utf-8",
    )

    code = main(
        [
            "replay",
            "--graph",
            str(graph_path),
            "--sessions",
            str(sessions / "events.jsonl"),
            "--max-queries",
            "3",
            "--json",
        ]
    )
    assert code == 0

    stats = json.loads(capsys.readouterr().out.strip())
    assert stats["queries_replayed"] == 3
    assert stats["edges_reinforced"] > 0
