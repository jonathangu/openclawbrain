from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from crabpath import Edge, EmbeddingIndex, Graph, Node


def _run_cli(args: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    proc_env = os.environ.copy()
    proc_env["PYTHONPATH"] = os.getcwd()
    if env:
        proc_env.update(env)
    return subprocess.run(
        [sys.executable, "-m", "crabpath.cli", *args],
        text=True,
        capture_output=True,
        env=proc_env,
    )


def _load_json_output(raw: str) -> dict:
    return json.loads(raw.strip())


def test_query_outputs_json(tmp_path: Path) -> None:
    graph_path = tmp_path / "graph.json"
    index_path = tmp_path / "index.json"

    graph = Graph()
    graph.add_node(Node(id="deploy-config", content="deploy broke after config change"))
    graph.add_node(Node(id="ignore", content="nothing relevant"))
    graph.save(str(graph_path))
    index = EmbeddingIndex()
    index.vectors = {"deploy-config": [1.0], "ignore": [0.2]}
    index.save(str(index_path))

    result = _run_cli(
        [
            "query",
            "deploy broke after config change",
            "--graph",
            str(graph_path),
            "--index",
            str(index_path),
            "--top",
            "12",
        ],
        env={"OPENAI_API_KEY": ""},
    )
    assert result.returncode == 0
    payload = _load_json_output(result.stdout)

    assert "fired" in payload
    assert payload["fired"][0]["id"] == "deploy-config"
    assert "inhibited" in payload
    assert "guardrails" in payload


def test_learn_updates_graph_and_outputs_edges_updated(tmp_path: Path) -> None:
    graph_path = tmp_path / "graph.json"

    graph = Graph()
    graph.add_node(Node(id="a", content="start"))
    graph.add_node(Node(id="b", content="next"))
    graph.add_edge(Edge(source="a", target="b", weight=1.0))
    graph.save(str(graph_path))

    result = _run_cli(
        [
            "learn",
            "--graph",
            str(graph_path),
            "--outcome",
            "1.0",
            "--fired-ids",
            "a,b",
        ]
    )
    assert result.returncode == 0
    payload = _load_json_output(result.stdout)

    assert payload["ok"] is True
    assert payload["edges_updated"] >= 1

    updated = Graph.load(str(graph_path))
    assert updated.get_edge("a", "b").weight > 1.0


def test_snapshot_and_feedback_roundtrip(tmp_path: Path) -> None:
    graph_path = tmp_path / "graph.json"
    graph = Graph()
    graph.add_node(Node(id="a", content="seed"))
    graph.save(str(graph_path))

    snapshot_path = tmp_path / "session.events.db"
    snapshot_env = {
        "CRABPATH_SNAPSHOT_PATH": str(snapshot_path),
        "OPENAI_API_KEY": "",
    }

    snapshot = _run_cli(
        [
            "snapshot",
            "--graph",
            str(graph_path),
            "--session",
            "sess-123",
            "--turn",
            "42",
            "--fired-ids",
            "a",
        ],
        env=snapshot_env,
    )
    assert snapshot.returncode == 0
    snapshot_payload = _load_json_output(snapshot.stdout)
    assert snapshot_payload["ok"] is True
    assert snapshot_payload["snapshot_path"] == str(snapshot_path)

    feedback = _run_cli(
        [
            "feedback",
            "--session",
            "sess-123",
            "--turn-window",
            "5",
        ],
        env={**snapshot_env, "CRABPATH_CORRECTION_TURN": "45"},
    )
    assert feedback.returncode == 0
    feedback_payload = _load_json_output(feedback.stdout)

    assert feedback_payload["turn_id"] == 42
    assert feedback_payload["fired_ids"] == ["a"]
    assert feedback_payload["turns_since_fire"] == 3
    assert feedback_payload["suggested_outcome"] == -1.0


def test_stats_outputs_graph_summary(tmp_path: Path) -> None:
    graph_path = tmp_path / "graph.json"

    graph = Graph()
    graph.add_node(Node(id="a", content="start"))
    graph.add_node(Node(id="b", content="bridge"))
    graph.add_node(Node(id="c", content="end"))
    graph.add_edge(Edge(source="a", target="b", weight=0.4))
    graph.add_edge(Edge(source="b", target="c", weight=0.7))
    graph.add_edge(Edge(source="a", target="c", weight=0.2))
    graph.save(str(graph_path))

    result = _run_cli(["stats", "--graph", str(graph_path)])
    assert result.returncode == 0
    payload = _load_json_output(result.stdout)

    assert payload["nodes"] == 3
    assert payload["edges"] == 3
    assert abs(payload["avg_weight"] - 0.43333333333333335) < 1e-12
    assert payload["top_hubs"] == ["a", "b", "c"]


def test_consolidate_outputs_prune_counts(tmp_path: Path) -> None:
    graph_path = tmp_path / "graph.json"

    graph = Graph()
    graph.add_node(Node(id="a", content="a"))
    graph.add_node(Node(id="b", content="b"))
    graph.add_node(Node(id="c", content="c"))
    graph.add_node(Node(id="d", content="d"))
    graph.add_edge(Edge(source="a", target="b", weight=0.02))
    graph.add_edge(Edge(source="b", target="c", weight=0.6))
    graph.add_edge(Edge(source="c", target="a", weight=0.03))
    graph.add_edge(Edge(source="a", target="d", weight=0.09))
    graph.save(str(graph_path))

    result = _run_cli(["consolidate", "--graph", str(graph_path), "--min-weight", "0.05"])
    assert result.returncode == 0
    payload = _load_json_output(result.stdout)

    assert payload["ok"] is True
    assert payload["pruned_edges"] == 2

    consolidated = Graph.load(str(graph_path))
    assert consolidated.edge_count == 2
    assert consolidated.get_edge("b", "c") is not None


def test_query_error_when_graph_missing(tmp_path: Path) -> None:
    missing_graph = tmp_path / "missing_graph.json"
    result = _run_cli(
        [
            "query",
            "deploy broke",
            "--graph",
            str(missing_graph),
            "--index",
            str(tmp_path / "missing_index.json"),
        ],
        env={"OPENAI_API_KEY": ""},
    )

    assert result.returncode == 1
    error = _load_json_output(result.stderr)
    assert "error" in error
    assert "graph file not found" in error["error"]


def test_learn_bad_args(tmp_path: Path) -> None:
    graph_path = tmp_path / "graph.json"
    graph = Graph()
    graph.add_node(Node(id="a", content="start"))
    graph.save(str(graph_path))

    result = _run_cli(
        [
            "learn",
            "--graph",
            str(graph_path),
            "--outcome",
            "not-a-number",
            "--fired-ids",
            "a",
        ]
    )
    assert result.returncode == 1
    error = _load_json_output(result.stderr)
    assert "error" in error
