from __future__ import annotations

import json

from pathlib import Path

import pytest
from openclawbrain.cli import main
from openclawbrain.graph import Edge, Graph, Node
from openclawbrain import learn as _learn_mod
from openclawbrain.provenance import TOOL_EVIDENCE_PREFIX
from openclawbrain.replay import extract_interactions, extract_queries, extract_queries_from_dir, replay_queries
from openclawbrain.traverse import TraversalConfig




def _write_graph_payload(path: Path) -> None:
    """ write graph payload."""
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
    """test extract queries openclaw format."""
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
    """test extract queries flat format."""
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
    """test extract queries from directory."""
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    (sessions / "a.jsonl").write_text(json.dumps({"role": "user", "content": "one"}), encoding="utf-8")
    (sessions / "b.jsonl").write_text(
        "\n".join([json.dumps({"role": "user", "content": "two"}), json.dumps({"role": "user", "content": "three"})]),
        encoding="utf-8",
    )
    (sessions / "ignore.txt").write_text("not jsonl", encoding="utf-8")

    assert extract_queries_from_dir(sessions) == ["one", "two", "three"]


def test_extract_queries_from_directory_supports_nested_codex_rollouts(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    rollout_dir = sessions / "2026" / "03" / "05"
    rollout_dir.mkdir(parents=True)
    (rollout_dir / "rollout-2026-03-05T14-31-25-thread.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "timestamp": "2026-03-05T22:31:25.111Z",
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "fix the router"}],
                        },
                    }
                ),
                json.dumps(
                    {
                        "timestamp": "2026-03-05T22:31:26.111Z",
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "working on it"}],
                        },
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    assert extract_queries_from_dir(sessions) == ["fix the router"]


def test_extract_interactions_parses_user_and_assistant_messages(tmp_path: Path) -> None:
    """test extract interactions parses user and assistant messages."""
    path = tmp_path / "session.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "message",
                        "message": {
                            "role": "user",
                            "content": [{"type": "text", "text": "How do I run?"}],
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "message",
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "Use the docs."}],
                            "tool_calls": [
                                {"id": "tool-1", "type": "tool_call", "function": {"name": "lookup", "arguments": "{}"}},
                            ],
                        },
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    interactions = extract_interactions(path)
    assert len(interactions) == 1
    assert interactions[0]["query"] == "How do I run?"
    assert interactions[0]["response"] == "Use the docs."
    assert interactions[0]["tool_calls"] == [{"id": "tool-1", "name": "lookup", "arguments": "{}"}]


def test_extract_interactions_pairs_camelcase_toolcall_and_toolresult(tmp_path: Path) -> None:
    """CamelCase toolCall items and toolResult records are paired in interactions."""
    path = tmp_path / "session_toolcall.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "message",
                        "message": {
                            "role": "user",
                            "content": [{"type": "text", "text": "Search the web"}],
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "message",
                        "message": {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "toolCall",
                                    "id": "tc-1",
                                    "name": "web_search",
                                    "arguments": "{\"q\": \"alpha\"}",
                                }
                            ],
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "message",
                        "message": {
                            "role": "toolResult",
                            "toolName": "web_search",
                            "toolCallId": "tc-1",
                            "content": "result text",
                        },
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    interactions = extract_interactions(path)
    assert len(interactions) == 1
    assert interactions[0]["tool_calls"] == [{"id": "tc-1", "name": "web_search", "arguments": "{\"q\": \"alpha\"}"}]
    tool_results = interactions[0].get("tool_results")
    assert isinstance(tool_results, list)
    assert tool_results[0]["tool_call_id"] == "tc-1"
    assert tool_results[0]["tool_name"] == "web_search"
    assert tool_results[0]["content"] == "result text"


def test_extract_interactions_parses_codex_rollout_messages_and_tools(tmp_path: Path) -> None:
    path = tmp_path / "rollout.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "timestamp": "2026-03-05T22:31:25.111Z",
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "inspect current diffs"}],
                        },
                    }
                ),
                json.dumps(
                    {
                        "timestamp": "2026-03-05T22:31:25.511Z",
                        "type": "response_item",
                        "payload": {
                            "type": "function_call",
                            "name": "exec_command",
                            "arguments": '{"cmd":"git diff --stat"}',
                            "call_id": "call-1",
                        },
                    }
                ),
                json.dumps(
                    {
                        "timestamp": "2026-03-05T22:31:25.911Z",
                        "type": "response_item",
                        "payload": {
                            "type": "function_call_output",
                            "call_id": "call-1",
                            "output": "openclawbrain/cli.py | 10 +++++-----",
                        },
                    }
                ),
                json.dumps(
                    {
                        "timestamp": "2026-03-05T22:31:26.111Z",
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "I found the existing diff."}],
                        },
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    interactions = extract_interactions(path)
    assert len(interactions) == 1
    assert interactions[0]["query"] == "inspect current diffs"
    assert interactions[0]["response"] == "I found the existing diff."
    assert interactions[0]["tool_calls"] == [{"id": "call-1", "name": "exec_command", "arguments": '{"cmd":"git diff --stat"}'}]
    assert interactions[0]["tool_results"] == [
        {
            "tool_call_id": "call-1",
            "content": "openclawbrain/cli.py | 10 +++++-----",
            "line_no": 3,
            "session": "rollout.jsonl",
            "source": str(path),
        }
    ]


def test_extract_interactions_attaches_allowlisted_tool_result_for_media_stub(tmp_path: Path) -> None:
    """Allowlisted toolResult transcript text is appended to preceding media-stub user query."""
    path = tmp_path / "session_tool_result.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "message",
                        "message": {
                            "role": "user",
                            "content": [{"type": "text", "text": "[media attached: voice note (audio/ogg)]"}],
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "message",
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "tool_call", "name": "openai-whisper", "arguments": "{}"}],
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "message",
                        "message": {
                            "role": "toolResult",
                            "toolName": "openai-whisper",
                            "content": "Deploy to staging after tests pass.",
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "message",
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "I transcribed your note."}],
                        },
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    interactions = extract_interactions(path)
    assert len(interactions) == 1
    assert "[toolResult:openai-whisper] Deploy to staging after tests pass." in interactions[0]["query"]
    assert interactions[0]["response"] == "I transcribed your note."


def test_extract_queries_filtering_since_timestamp(tmp_path: Path) -> None:
    """test extract queries filtering since timestamp."""
    path = tmp_path / "session.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"role": "user", "content": "one", "ts": 100}),
                json.dumps({"role": "user", "content": "two", "ts": 200}),
                json.dumps({"role": "user", "content": "three", "ts": 300}),
            ]
        ),
        encoding="utf-8",
    )

    assert extract_queries(path, since_ts=150.0) == ["two", "three"]


def test_replay_queries_filters_by_since_ts() -> None:
    """test replay queries filters by since ts."""
    graph = Graph()
    graph.add_node(Node("a", "alpha chunk", metadata={"file": "a.md"}))
    graph.add_node(Node("b", "beta chunk", metadata={"file": "a.md"}))
    graph.add_edge(Edge("a", "b", 0.5))

    stats = replay_queries(
        graph=graph,
        queries=[("alpha", 1.0), ("alpha", 2.0), ("alpha", 3.0)],
        config=TraversalConfig(max_hops=1),
        since_ts=2.0,
    )

    assert stats["queries_replayed"] == 1
    assert stats["last_replayed_ts"] == 3.0


def test_replay_strengthens_edges() -> None:
    """test replay strengthens edges."""
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


def test_replay_queries_uses_wall_clock_timestamp_when_query_ts_missing() -> None:
    """Replay fallback uses wall-clock timestamp when no per-query ts is available."""
    graph = Graph()
    graph.add_node(Node("a", "alpha chunk", metadata={"file": "a.md"}))
    graph.add_node(Node("b", "beta chunk", metadata={"file": "a.md"}))
    graph.add_edge(Edge("a", "b", 0.5))
    stats = replay_queries(graph=graph, queries=["alpha"], config=TraversalConfig(max_hops=1))

    assert stats["queries_replayed"] == 1
    assert isinstance(stats["last_replayed_ts"], float)
    assert stats["last_replayed_ts_source"] == "wall_clock"


def test_replay_creates_cross_file_edges() -> None:
    """test replay creates cross file edges."""
    graph = Graph()
    graph.add_node(Node("a", "alpha chunk", metadata={"file": "a.md"}))
    graph.add_node(Node("b", "beta chunk", metadata={"file": "b.md"}))

    stats = replay_queries(graph=graph, queries=["alpha beta"], config=TraversalConfig(max_hops=1))

    assert stats["queries_replayed"] == 1
    assert stats["cross_file_edges_created"] == 1

    assert graph._edges["b"]["a"].source == "b"
    assert graph._edges["b"]["a"].target == "a"


def test_replay_queries_supports_outcome_fn_negative_learning() -> None:
    """test replay queries supports outcome fn negative learning."""
    graph = Graph()
    graph.add_node(Node("a", "alpha", metadata={"file": "a.md"}))
    graph.add_node(Node("bad", "bad", metadata={"file": "a.md"}))
    graph.add_edge(Edge("a", "bad", 0.8))

    stats = replay_queries(
        graph=graph,
        queries=["bad example", "normal"],
        config=TraversalConfig(max_hops=1, fire_threshold=0.0),
        outcome_fn=lambda query: -1.0 if "bad" in query else 1.0,
    )

    assert stats["queries_replayed"] == 2
    assert graph._edges["a"]["bad"].kind == "inhibitory"
    assert graph._edges["a"]["bad"].weight < 0.8


def test_replay_queries_auto_scores_if_assistant_response_matches() -> None:
    """test replay queries auto scores if assistant response matches."""
    base = Graph()
    base.add_node(Node("a", "alpha content", metadata={"file": "a.md"}))
    base.add_node(Node("b", "beta content", metadata={"file": "a.md"}))
    base.add_edge(Edge("a", "b", 0.5))

    boosted = Graph()
    boosted.add_node(Node("a", "alpha content", metadata={"file": "a.md"}))
    boosted.add_node(Node("b", "beta content", metadata={"file": "a.md"}))
    boosted.add_edge(Edge("a", "b", 0.5))

    replay_queries(
        graph=base,
        config=TraversalConfig(max_hops=1),
        queries=["alpha"],
    )
    assert base._edges["a"]["b"].weight > 0.5

    replay_queries(
        graph=boosted,
        config=TraversalConfig(max_hops=1),
        queries=[{"query": "alpha", "response": "alpha content answered", "tool_calls": []}],
    )
    assert boosted._edges["a"]["b"].weight > base._edges["a"]["b"].weight


def test_replay_creates_tool_action_edges_and_evidence_nodes() -> None:
    """Replay creates tool action nodes/edges and evidence nodes for tier-1 tools."""
    graph = Graph()
    graph.add_node(Node("a", "alpha", metadata={"file": "a.md"}))
    graph.add_node(Node("b", "beta", metadata={"file": "a.md"}))
    graph.add_edge(Edge("a", "b", 0.5))

    interactions = [
        {
            "query": "alpha",
            "response": "done",
            "tool_calls": [{"id": "tc-1", "name": "web_search", "arguments": "{\"q\": \"alpha\"}"}],
            "tool_results": [{"tool_call_id": "tc-1", "tool_name": "web_search", "content": "result text"}],
            "ts": 1.0,
            "session": "session.jsonl",
            "source": "session.jsonl",
            "line_no_start": 1,
            "line_no_end": 3,
        }
    ]

    replay_queries(graph=graph, queries=interactions, config=TraversalConfig(max_hops=1))

    tool_action_nodes = [node for node in graph.nodes() if node.id.startswith("tool_action::")]
    assert tool_action_nodes
    action_node_id = tool_action_nodes[0].id
    assert graph._edges["a"][action_node_id].kind == "tool_action"

    evidence_nodes = [node for node in graph.nodes() if node.id.startswith(TOOL_EVIDENCE_PREFIX)]
    assert evidence_nodes


def test_extract_queries_missing_file_returns_empty(tmp_path: Path) -> None:
    """Missing session file returns empty list instead of crashing."""
    missing = tmp_path / "does_not_exist.jsonl"
    assert extract_queries(missing) == []


def test_extract_interactions_missing_file_returns_empty(tmp_path: Path) -> None:
    """Missing session file returns empty list instead of crashing."""
    missing = tmp_path / "gone.jsonl"
    assert extract_interactions(missing) == []


def test_extract_queries_broken_symlink_returns_empty(tmp_path: Path) -> None:
    """Broken symlink is skipped gracefully."""
    target = tmp_path / "real.jsonl"
    link = tmp_path / "broken.jsonl"
    link.symlink_to(target)  # target does not exist → broken symlink
    assert extract_queries(link) == []


def test_extract_interactions_broken_symlink_returns_empty(tmp_path: Path) -> None:
    """Broken symlink is skipped gracefully."""
    target = tmp_path / "real.jsonl"
    link = tmp_path / "broken.jsonl"
    link.symlink_to(target)
    assert extract_interactions(link) == []


def test_decay_during_replay_reduces_unrelated_edge() -> None:
    """Decay during replay weakens edges not reinforced by replayed queries."""
    # Reset the global call counter so decay_interval triggers deterministically.
    _learn_mod._apply_outcome_call_count = 0

    graph = Graph()
    graph.add_node(Node("x", "xray topic", metadata={"file": "x.md"}))
    graph.add_node(Node("y", "yankee topic", metadata={"file": "x.md"}))
    graph.add_node(Node("z", "zulu unrelated", metadata={"file": "z.md"}))
    # x→y is the traversed edge (queries target "xray")
    graph.add_edge(Edge("x", "y", 0.5))
    # y→z is unrelated — never reinforced, should decay
    graph.add_edge(Edge("y", "z", 0.5))

    initial_yz = graph._edges["y"]["z"].weight

    replay_queries(
        graph=graph,
        queries=["xray"] * 20,
        config=TraversalConfig(max_hops=1),
        auto_decay=True,
        decay_interval=1,
    )

    assert graph._edges["y"]["z"].weight < initial_yz, (
        "unrelated edge y→z should have decayed"
    )
