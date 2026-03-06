from __future__ import annotations

import json
from pathlib import Path

from openclawbrain.daemon import _DaemonState, _handle_capture_feedback, _handle_self_learn
from openclawbrain.feedback_events import FeedbackEvent, from_dict
from openclawbrain.full_learning import _window_to_payload
from openclawbrain.graph import Edge, Graph, Node
from openclawbrain.index import VectorIndex
from openclawbrain.labels import from_self_learning_event
from openclawbrain.hasher import HashEmbedder


def _simple_graph() -> tuple[Graph, VectorIndex]:
    g = Graph()
    g.add_node(Node("a", "alpha content about deployment"))
    g.add_node(Node("b", "beta content about cleanup"))
    g.add_edge(Edge("a", "b", weight=0.5))
    idx = VectorIndex()
    embed = HashEmbedder().embed
    idx.upsert("a", embed("alpha content about deployment"))
    idx.upsert("b", embed("beta content about cleanup"))
    return g, idx


def _daemon_state(graph: Graph, index: VectorIndex) -> _DaemonState:
    return _DaemonState(graph=graph, index=index, meta={}, fired_log={}, feedback_dedup_keys=set())


def test_scanner_event_round_trips_to_canonical_contract() -> None:
    class Turn:
        def __init__(self, role: str, content: str, line_no: int, source: str) -> None:
            self.role = role
            self.content = content
            self.line_no = line_no
            self.source = source

    turns = [
        Turn("user", "that is wrong", 1, "session.jsonl"),
        Turn("assistant", "acknowledged", 2, "session.jsonl"),
    ]

    def fake_llm(_: str, __: str) -> str:
        return '{"corrections":[{"content":"Use X instead","context":"ctx","severity":"high"}],"teachings":[],"reinforcements":[]}'

    events = _window_to_payload(session="session.jsonl", window=turns, window_idx=1, total_windows=1, llm_fn=fake_llm)
    assert events
    event = from_dict(events[0])
    assert event.source_kind == "scanner"
    assert event.feedback_kind == "CORRECTION"
    assert event.session_pointer
    assert event.metadata


def test_capture_feedback_writes_canonical_feedback_event_to_journal(tmp_path: Path) -> None:
    g, idx = _simple_graph()
    state_path = tmp_path / "state.json"
    state_path.write_text('{"graph":{"nodes":[],"edges":[]},"index":{},"meta":{}}', encoding="utf-8")
    ds = _daemon_state(g, idx)
    ds.fired_log["chat:test"] = [{"chat_id": "chat:test", "query": "q", "fired_nodes": ["a", "b"], "ts": 1.0, "timestamp": 1.0}]

    payload, should_write = _handle_capture_feedback(
        daemon_state=ds,
        graph=g,
        index=idx,
        meta={},
        embed_fn=HashEmbedder().embed,
        state_path=str(state_path),
        params={
            "chat_id": "chat:test",
            "kind": "CORRECTION",
            "content": "Use deployment B instead",
            "lookback": 1,
            "message_id": "m1",
        },
    )

    assert should_write is True
    assert payload["outcome_used"] == -1.0

    journal = ((tmp_path / "journal.jsonl").read_text(encoding="utf-8").splitlines())
    records = [json.loads(line) for line in journal]
    feedback = next(r for r in records if r.get("source_kind") == "human")
    parsed = from_dict(feedback)
    assert parsed.feedback_kind == "CORRECTION"
    assert parsed.chat_id == "chat:test"
    assert parsed.message_id == "m1"
    assert parsed.fired_ids == ["a", "b"]
    assert parsed.outcome == -1.0


def test_self_learn_writes_canonical_feedback_event_and_harvest_reads_it(tmp_path: Path) -> None:
    g, idx = _simple_graph()
    state_path = tmp_path / "state.json"
    state_path.write_text('{"graph":{"nodes":[],"edges":[]},"index":{},"meta":{}}', encoding="utf-8")
    ds = _daemon_state(g, idx)

    payload, should_write = _handle_self_learn(
        daemon_state=ds,
        graph=g,
        index=idx,
        embed_fn=HashEmbedder().embed,
        state_path=str(state_path),
        params={
            "content": "This route worked reliably",
            "fired_ids": ["a", "b"],
            "outcome": 1.0,
            "node_type": "TEACHING",
        },
    )

    assert should_write is True
    assert payload["feedback_event_logged"] is True
    journal = ((tmp_path / "journal.jsonl").read_text(encoding="utf-8").splitlines())
    records = [json.loads(line) for line in journal]
    feedback = next(r for r in records if r.get("source_kind") == "self")
    parsed = from_dict(feedback)
    assert parsed.feedback_kind == "REINFORCEMENT"
    assert parsed.fired_ids == ["a", "b"]
    assert parsed.outcome == 1.0

    label = from_self_learning_event(feedback)
    assert label is not None
    assert label.candidate_scores == {"a": 1.0, "b": 1.0}
    assert label.metadata["kind"] == "self_learning"


def test_legacy_feedback_event_hash_stays_backward_compatible_without_dedup_identity() -> None:
    payload = FeedbackEvent(
        source_kind="scanner",
        feedback_kind="CORRECTION",
        content="Do X instead",
        session_pointer="session.jsonl:1-2",
    ).to_dict()
    assert payload["event_hash"].count("|") == 2
    assert payload["event_hash"].startswith("CORRECTION|")
