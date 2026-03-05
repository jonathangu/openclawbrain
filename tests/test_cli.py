from __future__ import annotations

import json
import os
import sys
import hashlib
import plistlib
import signal
from io import StringIO
from pathlib import Path

import pytest

from datetime import datetime, timezone

from openclawbrain.cli import (
    main,
    _read_replay_checkpoint_progress,
    _resolve_openclawbrain_bin,
    _filter_replay_interactions,
    _default_labels_path,
    _maybe_warn_long_running,
)
from openclawbrain.graph import Graph, Node
from openclawbrain.hasher import default_embed
from openclawbrain.journal import read_journal
from openclawbrain.index import VectorIndex
from openclawbrain.state_lock import lock_path_for_state
from openclawbrain.store import save_state


def _write_graph_payload(path: Path) -> None:
    """ write graph payload."""
    payload = {
        "graph": {
            "nodes": [
                {"id": "a", "content": "alpha", "summary": "", "metadata": {"file": "a.md"}},
                {"id": "b", "content": "beta", "summary": "", "metadata": {"file": "b.md"}},
            ],
            "edges": [
                {"source": "a", "target": "b", "weight": 0.7, "kind": "sibling", "metadata": {}},
            ],
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_index(path: Path, payload: dict[str, list[float]] | None = None) -> None:
    """ write index."""
    if payload is None:
        payload = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    path.write_text(json.dumps(payload), encoding="utf-8")


def _install_fake_local_embedder(monkeypatch):
    class FakeLocalEmbedder:
        def __init__(self, model_name: str | None = None) -> None:
            self.model_name = model_name or "BAAI/bge-large-en-v1.5"
            self.name = f"local:{self.model_name.rsplit('/', 1)[-1]}"
            self.dim = 2

        def embed(self, text: str) -> list[float]:
            lowered = text.lower()
            if "alpha" in lowered or "deploy" in lowered:
                return [1.0, 0.0]
            return [0.0, 1.0]

        def embed_batch(self, texts: list[tuple[str, str]]) -> dict[str, list[float]]:
            return {node_id: self.embed(content) for node_id, content in texts}

    import openclawbrain.cli as cli_module
    monkeypatch.setattr(cli_module, "LocalEmbedder", FakeLocalEmbedder)
    return FakeLocalEmbedder


def _write_state(
    path: Path,
    graph_payload: dict | None = None,
    index_payload: dict[str, list[float]] | None = None,
    meta: dict[str, object] | None = None,
) -> None:
    """ write state."""
    if graph_payload is None:
        graph_payload = {
            "nodes": [
                {"id": "a", "content": "alpha", "summary": "", "metadata": {"file": "a.md"}},
                {"id": "b", "content": "beta", "summary": "", "metadata": {"file": "b.md"}},
            ],
            "edges": [
                {"source": "a", "target": "b", "weight": 0.7, "kind": "sibling", "metadata": {}},
            ],
        }
    if index_payload is None:
        index_payload = {
            "a": default_embed("alpha"),
            "b": default_embed("beta"),
        }
    if meta is None:
        meta = {"embedder_name": "hash-v1", "embedder_dim": 1024}
    path.write_text(json.dumps({"graph": graph_payload, "index": index_payload, "meta": meta}), encoding="utf-8")


def test_long_running_warning_prints_when_interactive(monkeypatch, capsys) -> None:
    """interactive sessions print a long-running warning."""
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True)
    _maybe_warn_long_running()
    err = capsys.readouterr().err
    assert "Note: init/build-all may take a long time; do not run under a short timeout." in err


def test_long_running_warning_suppressed_when_not_tty(monkeypatch, capsys) -> None:
    """non-interactive sessions suppress the long-running warning."""
    monkeypatch.setattr(sys.stderr, "isatty", lambda: False)
    _maybe_warn_long_running()
    err = capsys.readouterr().err
    assert err == ""


def test_init_command_creates_workspace_graph(tmp_path, monkeypatch) -> None:
    """test init command creates workspace graph."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "a.md").write_text("## A\nHello", encoding="utf-8")
    output = tmp_path

    monkeypatch.setenv("OPENCLAWBRAIN_FASTEMBED_STUB", "1")
    code = main(["init", "--workspace", str(workspace), "--output", str(output), "--embedder", "local", "--llm", "none"])
    assert code == 0

    graph_path = output / "graph.json"
    texts_path = output / "texts.json"
    index_path = output / "index.json"
    assert graph_path.exists()
    assert texts_path.exists()
    assert index_path.exists()
    graph_data = json.loads(graph_path.read_text(encoding="utf-8"))
    graph_payload = graph_data["graph"] if "graph" in graph_data else graph_data
    assert len(graph_payload["nodes"]) == 1
    assert graph_data.get("meta", {}).get("embedder_name") == "local:bge-large-en-v1.5"
    state_data = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert state_data["meta"]["embedder_name"] == "local:bge-large-en-v1.5"
    texts_data = json.loads(texts_path.read_text(encoding="utf-8"))
    assert len(texts_data) == 1


def test_init_command_with_empty_workspace(tmp_path, monkeypatch) -> None:
    """test init command with empty workspace."""
    workspace = tmp_path / "empty"
    workspace.mkdir()
    output = tmp_path / "out"
    output.mkdir()

    monkeypatch.setenv("OPENCLAWBRAIN_FASTEMBED_STUB", "1")
    code = main(["init", "--workspace", str(workspace), "--output", str(output), "--embedder", "local", "--llm", "none"])
    assert code == 0
    graph_file = json.loads((output / "graph.json").read_text(encoding="utf-8"))
    graph_data = graph_file["graph"] if "graph" in graph_file else graph_file
    assert graph_data["nodes"] == []
    assert graph_data["edges"] == []
    assert (output / "index.json").exists()


def test_init_sets_authority_metadata_for_mapped_files(tmp_path, monkeypatch) -> None:
    """init assigns authority metadata for files in DEFAULT_AUTHORITY_MAP."""
    import openclawbrain.cli as cli_module

    monkeypatch.setattr(
        cli_module,
        "DEFAULT_AUTHORITY_MAP",
        {
            "SOUL.md": "constitutional",
            "AGENTS.md": "constitutional",
            "USER.md": "canonical",
        },
    )

    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "SOUL.md").write_text("Constitutional guidance", encoding="utf-8")
    (workspace / "AGENTS.md").write_text("Agent operating rules", encoding="utf-8")
    (workspace / "USER.md").write_text("User preferences", encoding="utf-8")
    output = tmp_path / "out"
    output.mkdir()

    monkeypatch.setenv("OPENCLAWBRAIN_FASTEMBED_STUB", "1")
    code = main(["init", "--workspace", str(workspace), "--output", str(output), "--embedder", "local", "--llm", "none"])
    assert code == 0

    state_data = json.loads((output / "state.json").read_text(encoding="utf-8"))
    nodes = state_data["graph"]["nodes"]
    workspace_prefix = f"{workspace.name}/"

    def authorities_for(file_name: str) -> set[str]:
        expected_file = f"{workspace_prefix}{file_name}"
        return {
            str(node.get("metadata", {}).get("authority"))
            for node in nodes
            if node.get("metadata", {}).get("file") == expected_file
        }

    assert "constitutional" in authorities_for("SOUL.md")
    assert "constitutional" in authorities_for("AGENTS.md")
    assert "canonical" in authorities_for("USER.md")


def test_init_resume_skips_completed_embeddings(tmp_path, monkeypatch) -> None:
    """init resumes embeddings without re-embedding completed chunks."""
    import openclawbrain.cli as cli_module

    workspace = tmp_path / "ws"
    workspace.mkdir()
    for idx in range(4):
        (workspace / f"file{idx}.md").write_text(f"## File {idx}\nHello {idx}", encoding="utf-8")
    output = tmp_path / "out"
    output.mkdir()

    seen_ids: set[str] = set()
    crash_after = 2
    crash_state = {"crashed": False}

    def embed_batch_fn(items: list[tuple[str, str]]) -> dict[str, list[float]]:
        for node_id, _ in items:
            if node_id in seen_ids:
                raise AssertionError(f"re-embedded {node_id}")
        if not crash_state["crashed"] and len(seen_ids) >= crash_after:
            crash_state["crashed"] = True
            raise RuntimeError("simulated crash")
        vectors = {node_id: [0.1, 0.2, 0.3] for node_id, _ in items}
        for node_id, _ in items:
            seen_ids.add(node_id)
        return vectors

    def embed_fn(_text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    def resolve_embedder(_args: object, _meta: dict[str, object]):
        return embed_fn, embed_batch_fn, "test-embedder", 3, None

    monkeypatch.setattr(cli_module, "_resolve_embedder", resolve_embedder)

    with pytest.raises(RuntimeError, match="simulated crash"):
        cli_module.main(
            [
                "init",
                "--workspace",
                str(workspace),
                "--output",
                str(output),
                "--embedder",
                "local",
                "--llm",
                "none",
                "--checkpoint-every",
                "2",
                "--resume",
            ]
        )

    code = cli_module.main(
        [
            "init",
            "--workspace",
            str(workspace),
            "--output",
            str(output),
            "--embedder",
            "local",
            "--llm",
            "none",
            "--checkpoint-every",
            "2",
            "--resume",
        ]
    )
    assert code == 0
    assert len(seen_ids) == 4
    state_data = json.loads((output / "state.json").read_text(encoding="utf-8"))
    assert len(state_data["graph"]["nodes"]) == 4


def test_resolve_openclawbrain_bin_prefers_sys_argv(tmp_path, monkeypatch) -> None:
    """resolve openclawbrain bin prefers sys.argv[0] when executable."""
    monkeypatch.chdir(tmp_path)
    bin_path = Path("openclawbrain")
    bin_path.write_text("#!/bin/sh\n", encoding="utf-8")
    os.chmod(bin_path, 0o755)

    monkeypatch.setattr(sys, "argv", [str(bin_path)])
    monkeypatch.setattr("openclawbrain.cli.shutil.which", lambda _: "/opt/homebrew/bin/openclawbrain")

    assert _resolve_openclawbrain_bin() == str(bin_path)


def test_default_labels_path_resolves_from_state_path(tmp_path) -> None:
    state_path = tmp_path / "brain" / "state.json"
    expected = state_path.parent / "labels.jsonl"
    assert _default_labels_path(str(state_path)) == expected


def test_query_command_returns_json_with_fired_nodes(tmp_path, capsys, monkeypatch) -> None:
    """test query command returns json with fired nodes."""
    _install_fake_local_embedder(monkeypatch)
    graph_path = tmp_path / "graph.json"
    index_path = tmp_path / "index.json"
    _write_graph_payload(graph_path)
    _write_index(
        index_path,
        {
            "a": [1.0, 0.0],
            "b": [0.0, 1.0],
        },
    )

    code = main(
        [
            "query",
            "alpha",
            "--graph",
            str(graph_path),
            "--index",
            str(index_path),
            "--top",
            "2",
            "--embedder",
            "local",
            "--json",
        ]
    )
    assert code == 0
    out = json.loads(capsys.readouterr().out.strip())
    assert out["fired"]
    assert "a" in out["fired"]
    assert out["tier_thresholds"] == {
        "reflex": ">= 0.6",
        "habitual": "0.15 - 0.6",
        "dormant": "< 0.15",
        "inhibitory": "< -0.01",
    }


def test_query_auto_embeds(tmp_path, capsys, monkeypatch) -> None:
    """test query auto embeds."""
    _install_fake_local_embedder(monkeypatch)
    graph_path = tmp_path / "graph.json"
    index_path = tmp_path / "index.json"
    _write_graph_payload(graph_path)
    _write_index(
        index_path,
        {
            "a": [1.0, 0.0],
            "b": [0.0, 1.0],
        },
    )

    code = main(
        [
            "query",
            "alpha",
            "--graph",
            str(graph_path),
            "--index",
            str(index_path),
            "--top",
            "2",
            "--embedder",
            "local",
            "--json",
        ]
    )
    assert code == 0
    out = json.loads(capsys.readouterr().out.strip())
    assert out["fired"]
    assert out["fired"][0] == "a"


def test_query_uses_vector_from_stdin(tmp_path, capsys, monkeypatch) -> None:
    """test query uses vector from stdin."""
    graph_path = tmp_path / "graph.json"
    index_path = tmp_path / "index.json"
    _write_graph_payload(graph_path)
    _write_index(index_path)

    monkeypatch.setattr("sys.stdin", StringIO(json.dumps([1.0, 0.0])))

    code = main(
        [
            "query",
            "seed",
            "--graph",
            str(graph_path),
            "--index",
            str(index_path),
            "--top",
            "1",
            "--query-vector-stdin",
            "--json",
        ]
    )
    assert code == 0
    out = json.loads(capsys.readouterr().out.strip())
    assert out["fired"][0] == "a"


def test_cli_state_flag_query(tmp_path, capsys) -> None:
    """test cli state flag query."""
    state_path = tmp_path / "state.json"
    _write_state(state_path)

    code = main(
        [
            "query",
            "alpha",
            "--state",
            str(state_path),
            "--top",
            "2",
            "--json",
        ]
    )
    assert code == 0
    out = json.loads(capsys.readouterr().out.strip())
    assert out["fired"]
    assert "a" in out["fired"]


def test_query_max_context_chars_cap_in_query_command(tmp_path, capsys) -> None:
    """test query max context chars cap in query command."""
    state_path = tmp_path / "state.json"
    graph_payload = {
        "nodes": [
            {"id": "a", "content": "alpha " * 80, "summary": "", "metadata": {"file": "a.md"}},
            {"id": "b", "content": "zeta " * 80, "summary": "", "metadata": {"file": "b.md"}},
        ],
        "edges": [{"source": "a", "target": "b", "weight": 0.95, "kind": "sibling", "metadata": {}}],
    }
    _write_state(state_path, graph_payload=graph_payload)

    code = main(
        [
            "query",
            "alpha",
            "--state",
            str(state_path),
            "--top",
            "1",
            "--max-context-chars",
            "120",
            "--json",
        ]
    )
    assert code == 0
    out = json.loads(capsys.readouterr().out.strip())
    assert len(out["context"]) <= 120


def test_query_journal_written_to_state_directory(tmp_path) -> None:
    """test query journal written to state directory."""
    state_path = tmp_path / "state.json"
    _write_state(state_path)

    main(
        [
            "query",
            "alpha",
            "--state",
            str(state_path),
            "--json",
        ]
    )
    journal_path = tmp_path / "journal.jsonl"
    assert journal_path.exists()

    entries = read_journal(journal_path=str(journal_path))
    assert entries
    assert entries[-1]["type"] == "query"


def test_cli_state_replay_uses_last_replayed_ts(tmp_path, capsys) -> None:
    """test cli state replay uses last replayed ts."""
    state_path = tmp_path / "state.json"
    _write_state(state_path)

    sessions = tmp_path / "sessions.jsonl"
    sessions.write_text(
        "\n".join(
            [
                json.dumps({"role": "user", "content": "alpha", "ts": 1.0}),
                json.dumps({"role": "user", "content": "alpha", "ts": 2.0}),
                json.dumps({"role": "user", "content": "alpha", "ts": 3.0}),
            ]
        ),
        encoding="utf-8",
    )

    code = main(["replay", "--state", str(state_path), "--sessions", str(sessions), "--edges-only", "--json"])
    assert code == 0
    first = json.loads(capsys.readouterr().out.strip())
    assert first["queries_replayed"] == 3
    assert json.loads(state_path.read_text(encoding="utf-8"))["meta"]["last_replayed_ts"] == 3.0

    sessions.write_text(
        "\n".join(
            [
                json.dumps({"role": "user", "content": "alpha", "ts": 4.0}),
                json.dumps({"role": "user", "content": "alpha", "ts": 5.0}),
            ]
        ),
        encoding="utf-8",
    )
    code = main(["replay", "--state", str(state_path), "--sessions", str(sessions), "--edges-only", "--json"])
    assert code == 0
    second = json.loads(capsys.readouterr().out.strip())
    assert second["queries_replayed"] == 2
    assert json.loads(state_path.read_text(encoding="utf-8"))["meta"]["last_replayed_ts"] == 5.0


def test_cli_state_replay_uses_wall_clock_ts_when_session_ts_missing(tmp_path, capsys) -> None:
    """Replay persists wall-clock fallback timestamp and source when query ts is absent."""
    state_path = tmp_path / "state.json"
    _write_state(state_path)
    sessions = tmp_path / "sessions.jsonl"
    sessions.write_text(
        json.dumps({"role": "user", "content": "alpha"}),
        encoding="utf-8",
    )

    code = main(["replay", "--state", str(state_path), "--sessions", str(sessions), "--edges-only", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["queries_replayed"] == 1
    assert isinstance(payload["last_replayed_ts"], float)
    assert payload["last_replayed_ts_source"] == "wall_clock"

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert isinstance(state["meta"]["last_replayed_ts"], float)
    assert state["meta"]["last_replayed_ts_source"] == "wall_clock"

    code = main(["status", "--state", str(state_path), "--json"])
    assert code == 0
    status_payload = json.loads(capsys.readouterr().out.strip())
    assert status_payload["last_replayed"] != "never"


def test_cli_replay_discovers_reset_session_files(tmp_path, capsys) -> None:
    """test replay discovers and processes reset session files."""
    state_path = tmp_path / "state.json"
    _write_state(state_path)

    sessions = tmp_path / "sessions"
    sessions.mkdir()
    (sessions / "current.jsonl").write_text(
        json.dumps({"role": "user", "content": "alpha", "ts": 1.0}),
        encoding="utf-8",
    )
    (sessions / "current.jsonl.reset.2026-02-27").write_text(
        json.dumps({"role": "user", "content": "alpha", "ts": 2.0}),
        encoding="utf-8",
    )

    code = main(["replay", "--state", str(state_path), "--sessions", str(sessions), "--edges-only", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["queries_replayed"] == 2
    assert json.loads(state_path.read_text(encoding="utf-8"))["meta"]["last_replayed_ts"] == 2.0


def test_cli_replay_fails_cleanly_when_state_lock_held_unless_force(tmp_path, capsys, monkeypatch) -> None:
    """replay exits with a clean lock error unless --force is set."""
    fcntl = pytest.importorskip("fcntl")
    monkeypatch.delenv("OPENCLAWBRAIN_STATE_LOCK_FORCE", raising=False)
    monkeypatch.delenv("OCB_STATE_LOCK_FORCE", raising=False)
    state_path = tmp_path / "state.json"
    _write_state(state_path)

    sessions = tmp_path / "sessions.jsonl"
    sessions.write_text(json.dumps({"role": "user", "content": "alpha", "ts": 1.0}), encoding="utf-8")

    lock_path = lock_path_for_state(state_path)
    fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(SystemExit, match="state write lock is already held"):
            main(["replay", "--state", str(state_path), "--sessions", str(sessions), "--edges-only", "--json"])

        code = main(["replay", "--state", str(state_path), "--sessions", str(sessions), "--edges-only", "--force", "--json"])
        assert code == 0
        payload = json.loads(capsys.readouterr().out.strip())
        assert payload["queries_replayed"] == 1
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def test_cli_replay_with_fast_learning_writes_learning_events_and_injects_nodes(
    tmp_path,
    capsys,
    monkeypatch,
) -> None:
    """test cli replay with fast learning writes learning events and injects nodes."""
    state_path = tmp_path / "state.json"
    _write_state(state_path)

    sessions = tmp_path / "sessions.jsonl"
    sessions.write_text(
        "\n".join(
            [
                json.dumps({"role": "user", "content": "do not do that", "ts": 1.0}),
                json.dumps({"role": "user", "content": "how to deploy", "ts": 2.0}),
            ]
        ),
        encoding="utf-8",
    )

    from openclawbrain import full_learning as learning_module

    def fake_llm(_: str, __: str) -> str:
        return json.dumps(
            {
                "corrections": [
                    {
                        "content": "Never answer with sensitive data.",
                        "context": "user corrected behavior",
                        "severity": "high",
                    }
                ],
                "teachings": [],
                "reinforcements": [],
            }
        )

    monkeypatch.setattr(learning_module, "openai_llm_fn", fake_llm)

    code = main(
        [
            "replay",
            "--state",
            str(state_path),
            "--sessions",
            str(sessions),
            "--fast-learning",
            "--llm",
            "none",
            "--ignore-checkpoint",
            "--json",
        ]
    )
    assert code == 0
    first = json.loads(capsys.readouterr().out.strip())

    assert "fast_learning" in first
    assert first["fast_learning"]["events_injected"] >= 1
    assert first["fast_learning"]["events_appended"] >= 1

    first_state = json.loads(state_path.read_text(encoding="utf-8"))
    learning_nodes = [node for node in first_state["graph"]["nodes"] if str(node["id"]).startswith("learning::")]
    assert learning_nodes

    events_path = Path(first["fast_learning"]["learning_events_path"])
    assert events_path.exists()
    initial_events = len(events_path.read_text(encoding="utf-8").splitlines())
    assert first["fast_learning"]["learning_events_path"] == str(state_path.parent / "learning_events.jsonl")

    code = main(
        [
            "replay",
            "--state",
            str(state_path),
            "--sessions",
            str(sessions),
            "--fast-learning",
            "--llm",
            "none",
            "--ignore-checkpoint",
            "--json",
        ]
    )
    assert code == 0
    second = json.loads(capsys.readouterr().out.strip())
    assert second["fast_learning"]["events_appended"] == 0
    assert second["fast_learning"]["events_injected"] == 0
    assert len(events_path.read_text(encoding="utf-8").splitlines()) == initial_events

    second_state = json.loads(state_path.read_text(encoding="utf-8"))
    second_learning_nodes = [
        node for node in second_state["graph"]["nodes"] if str(node["id"]).startswith("learning::")
    ]
    assert len(second_learning_nodes) == len(learning_nodes)


def test_query_command_text_output_includes_node_ids(tmp_path, capsys, monkeypatch) -> None:
    """test query text output includes node ids."""
    _install_fake_local_embedder(monkeypatch)
    graph_path = tmp_path / "graph.json"
    index_path = tmp_path / "index.json"
    graph_payload = {
        "graph": {
            "nodes": [
                {
                    "id": "deploy.md::0",
                    "content": "How to create a hotfix",
                    "summary": "",
                    "metadata": {"file": "deploy.md"},
                },
                {
                    "id": "deploy.md::1",
                    "content": "CI must pass for hotfixes",
                    "summary": "",
                    "metadata": {"file": "deploy.md"},
                },
            ],
            "edges": [
                {
                    "source": "deploy.md::0",
                    "target": "deploy.md::1",
                    "weight": 0.85,
                    "kind": "sibling",
                    "metadata": {},
                }
            ],
        }
    }
    graph_path.write_text(json.dumps(graph_payload), encoding="utf-8")
    _write_index(index_path, {"deploy.md::0": [1.0, 0.0], "deploy.md::1": [0.0, 1.0]})

    code = main(
        [
            "query",
            "deploy hotfix",
            "--graph",
            str(graph_path),
            "--index",
            str(index_path),
            "--top",
            "2",
            "--embedder",
            "local",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "deploy.md::0" in out
    assert "deploy.md::1" in out


def test_cli_rejects_hash_embedder_flag(tmp_path, capsys) -> None:
    """CLI rejects hash embedder selection."""
    state_path = tmp_path / "state.json"
    _write_state(state_path)

    with pytest.raises(SystemExit):
        main(["query", "alpha", "--state", str(state_path), "--embedder", "hash", "--top", "2", "--json"])

    err = capsys.readouterr().err
    assert "invalid choice" in err


def test_query_command_error_on_missing_graph(tmp_path) -> None:
    """test query command error on missing graph."""
    index_path = tmp_path / "index.json"
    _write_index(index_path)
    with pytest.raises(SystemExit):
        main(["query", "seed", "--graph", str(tmp_path / "missing.json"), "--index", str(index_path)])


def test_query_command_keywords_without_index(tmp_path, capsys) -> None:
    """test query command keywords without index."""
    graph_path = tmp_path / "graph.json"
    _write_graph_payload(graph_path)

    code = main(["query", "alpha", "--graph", str(graph_path), "--top", "2", "--json"])
    assert code == 0
    out = json.loads(capsys.readouterr().out.strip())
    assert out["fired"]
    assert out["fired"][0] == "a"


def test_learn_command_updates_graph_weights(tmp_path) -> None:
    """test learn command updates graph weights."""
    graph_path = tmp_path / "graph.json"
    _write_graph_payload(graph_path)

    code = main(["learn", "--graph", str(graph_path), "--outcome", "1.0", "--fired-ids", "a,b"])
    assert code == 0

    payload = json.loads(graph_path.read_text(encoding="utf-8"))
    assert payload["graph"]["edges"][0]["weight"] > 0.7


def test_cli_state_flag_learn(tmp_path) -> None:
    """test cli state flag learn."""
    state_path = tmp_path / "state.json"
    _write_state(state_path)

    original = json.loads(state_path.read_text(encoding="utf-8"))
    original_weight = original["graph"]["edges"][0]["weight"]
    code = main(
        [
            "learn",
            "--state",
            str(state_path),
            "--outcome",
            "1.0",
            "--fired-ids",
            "a,b",
        ]
    )
    assert code == 0

    updated = json.loads(state_path.read_text(encoding="utf-8"))
    assert updated["graph"]["edges"][0]["weight"] > original_weight


def test_cli_state_flag_health(tmp_path, capsys) -> None:
    """test cli state flag health."""
    state_path = tmp_path / "state.json"
    _write_state(state_path)

    code = main(["health", "--state", str(state_path), "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    expected = {
        "dormant_pct",
        "habitual_pct",
        "reflex_pct",
        "cross_file_edge_pct",
        "orphan_nodes",
        "nodes",
        "edges",
    }
    assert expected.issubset(set(payload.keys()))


def test_learn_command_supports_json_output(tmp_path, capsys) -> None:
    """test learn command supports json output."""
    graph_path = tmp_path / "graph.json"
    _write_graph_payload(graph_path)

    code = main(["learn", "--graph", str(graph_path), "--outcome", "-1.0", "--fired-ids", "a,b", "--json"])
    assert code == 0
    data = json.loads(capsys.readouterr().out.strip())
    assert data["edges_updated"] >= 1
    assert data["max_weight_delta"] >= 0.0


def test_merge_command_suggests_and_applies(tmp_path, capsys) -> None:
    """test merge command suggests and applies."""
    graph_path = tmp_path / "graph.json"
    payload = {
        "graph": {
            "nodes": [
                {"id": "a", "content": "alpha", "summary": "", "metadata": {}},
                {"id": "b", "content": "beta", "summary": "", "metadata": {}},
                {"id": "c", "content": "gamma", "summary": "", "metadata": {}},
            ],
            "edges": [
                {"source": "a", "target": "b", "weight": 0.95, "kind": "sibling", "metadata": {}},
                {"source": "b", "target": "a", "weight": 0.95, "kind": "sibling", "metadata": {}},
                {"source": "a", "target": "c", "weight": 0.2, "kind": "sibling", "metadata": {}},
                {"source": "c", "target": "a", "weight": 0.1, "kind": "sibling", "metadata": {}},
            ],
        }
    }
    graph_path.write_text(json.dumps(payload), encoding="utf-8")

    code = main(["merge", "--graph", str(graph_path), "--json"])
    assert code == 0
    out = json.loads(capsys.readouterr().out.strip())
    assert "suggestions" in out
    assert out["suggestions"]

    updated = json.loads(graph_path.read_text(encoding="utf-8"))
    if "graph" in updated:
        graph_payload = updated["graph"]
    else:
        graph_payload = updated
    assert "nodes" in graph_payload
    assert len(graph_payload["nodes"]) <= 3


def test_health_command_outputs_all_metrics(tmp_path, capsys) -> None:
    """test health command outputs all metrics."""
    graph_path = tmp_path / "graph.json"
    _write_graph_payload(graph_path)
    code = main(["health", "--graph", str(graph_path), "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    expected = {
        "dormant_pct",
        "habitual_pct",
        "reflex_pct",
        "cross_file_edge_pct",
        "orphan_nodes",
        "nodes",
        "edges",
    }
    assert expected.issubset(set(payload.keys()))


def test_health_command_text_output_is_readable(tmp_path, capsys) -> None:
    """test health command has human-readable non-json output."""
    graph_path = tmp_path / "graph.json"
    _write_graph_payload(graph_path)

    code = main(["health", "--graph", str(graph_path)])
    assert code == 0
    out = capsys.readouterr().out
    assert "Brain health:" in out
    assert "Nodes:" in out
    assert "Edges:" in out
    assert "Cross-file edges:" in out


def test_cli_help_text_for_commands() -> None:
    """test cli help text for commands."""
    for command in [
        "init",
        "query",
        "learn",
        "merge",
        "health",
        "connect",
        "replay",
        "harvest",
        "journal",
        "sync",
        "serve",
    ]:
        with pytest.raises(SystemExit):
            main([command, "--help"])


def test_serve_command_prints_banner_and_calls_socket_server(monkeypatch, capsys) -> None:
    """serve should print a startup banner and delegate to socket_server.main."""
    called: list[list[str]] = []

    def fake_default_socket_path(state_path: str) -> str:
        return f"/tmp/{Path(state_path).parent.name}/daemon.sock"

    def fake_socket_server_main(argv: list[str] | None = None) -> int:
        called.append(list(argv or []))
        return 0

    from openclawbrain import socket_server

    monkeypatch.setattr(socket_server, "_default_socket_path", fake_default_socket_path)
    monkeypatch.setattr(socket_server, "main", fake_socket_server_main)

    code = main(["serve", "--state", "~/agent/state.json"])
    assert code == 0
    assert called == [[
        "--state",
        str(Path("~/agent/state.json").expanduser()),
        "--embed-model",
        "auto",
        "--max-prompt-context-chars",
        "30000",
        "--max-fired-nodes",
        "30",
        "--route-mode",
        "learned",
        "--route-top-k",
        "5",
        "--route-alpha-sim",
        "0.5",
        "--route-use-relevance",
        "true",
        "--route-enable-stop",
        "false",
        "--route-stop-margin",
        "0.1",
    ]]

    err = capsys.readouterr().err
    assert "OpenClawBrain socket service (foreground)" in err
    assert "socket path: /tmp/agent/daemon.sock" in err
    assert f"state path: {Path('~/agent/state.json').expanduser()}" in err
    assert "query status: openclawbrain serve status --state" in err
    assert "stop: Ctrl-C" in err


def test_serve_install_dry_run_prints_launchd_plist(monkeypatch, tmp_path, capsys) -> None:
    """serve install --dry-run should print a launchd plist with executable-based ProgramArguments."""
    state_path = tmp_path / "main" / "state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({"nodes": []}), encoding="utf-8")

    import openclawbrain.cli as cli_module

    monkeypatch.setattr(cli_module.sys, "platform", "darwin")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("OPENCLAWBRAIN_SERVE_WRAPPER", raising=False)

    code = main([
        "serve",
        "install",
        "--state",
        str(state_path),
        "--socket-path",
        str(tmp_path / "daemon.sock"),
        "--dry-run",
    ])
    assert code == 0

    payload_xml, _sep, _commands = capsys.readouterr().out.partition("Planned launchctl commands:")
    payload = plistlib.loads(payload_xml.encode("utf-8"))
    assert payload["Label"] == "com.openclawbrain.main"

    program_arguments = payload["ProgramArguments"]
    assert len(program_arguments) > 6
    assert program_arguments[0] == sys.executable
    assert program_arguments[1] == "-m"
    assert program_arguments[2] == "openclawbrain.cli"
    assert program_arguments[3] == "serve"
    assert program_arguments[4] == "start"
    assert "--state" in program_arguments
    assert str(state_path) in program_arguments


def test_serve_install_uses_openclaw_wrapper(monkeypatch, tmp_path, capsys) -> None:
    """serve install --dry-run should prefer OpenClaw wrapper when available."""
    state_path = tmp_path / "main" / "state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({"nodes": []}), encoding="utf-8")

    wrapper_path = tmp_path / ".openclaw" / "scripts" / "openclawbrain-serve"
    wrapper_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper_path.write_text("#!/bin/sh\n", encoding="utf-8")

    import openclawbrain.cli as cli_module

    monkeypatch.setattr(cli_module.sys, "platform", "darwin")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("OPENCLAWBRAIN_SERVE_WRAPPER", raising=False)

    code = main([
        "serve",
        "install",
        "--state",
        str(state_path),
        "--socket-path",
        str(tmp_path / "daemon.sock"),
        "--dry-run",
    ])
    assert code == 0

    payload_xml, _sep, _commands = capsys.readouterr().out.partition("Planned launchctl commands:")
    payload = plistlib.loads(payload_xml.encode("utf-8"))

    program_arguments = payload["ProgramArguments"]
    assert program_arguments[0] == str(wrapper_path)
    assert "--state" in program_arguments
    assert str(state_path) in program_arguments


def _extract_plist_from_output(output: str) -> dict:
    _, _, tail = output.partition("Planned launchctl commands:")
    xml_start = tail.find("<?xml")
    payload_xml = tail[xml_start:] if xml_start >= 0 else tail
    return plistlib.loads(payload_xml.encode("utf-8"))


def test_loop_install_dry_run_prints_launchd_plist(monkeypatch, tmp_path, capsys) -> None:
    """loop install --dry-run should print a launchd plist with executable-based ProgramArguments."""
    state_path = tmp_path / "main" / "state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({"nodes": []}), encoding="utf-8")

    import openclawbrain.cli as cli_module

    monkeypatch.setattr(cli_module.sys, "platform", "darwin")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("OPENCLAWBRAIN_LOOP_PYTHON", raising=False)
    monkeypatch.delenv("OPENCLAWBRAIN_PYTHON", raising=False)

    code = main([
        "loop",
        "install",
        "--state",
        str(state_path),
        "--dry-run",
    ])
    assert code == 0

    payload = _extract_plist_from_output(capsys.readouterr().out)
    assert payload["Label"] == "com.openclawbrain.loop.main"

    program_arguments = payload["ProgramArguments"]
    assert program_arguments[0] == sys.executable
    assert program_arguments[1] == "-m"
    assert program_arguments[2] == "openclawbrain.cli"
    assert program_arguments[3] == "loop"
    assert program_arguments[4] == "run"
    assert "--state" in program_arguments
    assert str(state_path) in program_arguments


def test_loop_install_dry_run_prefers_openclaw_venv_python(monkeypatch, tmp_path, capsys) -> None:
    """loop install --dry-run should prefer the OpenClaw venv python when present."""
    state_path = tmp_path / "main" / "state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({"nodes": []}), encoding="utf-8")

    venv_python = tmp_path / ".openclaw" / "venvs" / "openclawbrain" / "bin" / "python"
    venv_python.parent.mkdir(parents=True, exist_ok=True)
    venv_python.write_text("#!/bin/sh\n", encoding="utf-8")

    import openclawbrain.cli as cli_module

    monkeypatch.setattr(cli_module.sys, "platform", "darwin")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("OPENCLAWBRAIN_LOOP_PYTHON", raising=False)
    monkeypatch.delenv("OPENCLAWBRAIN_PYTHON", raising=False)

    code = main([
        "loop",
        "install",
        "--state",
        str(state_path),
        "--dry-run",
    ])
    assert code == 0

    payload = _extract_plist_from_output(capsys.readouterr().out)
    program_arguments = payload["ProgramArguments"]
    assert program_arguments[0] == str(venv_python)


def test_loop_install_dry_run_includes_env_file(monkeypatch, tmp_path, capsys) -> None:
    """loop install --dry-run should include EnvironmentVariables from --env-file."""
    state_path = tmp_path / "main" / "state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({"nodes": []}), encoding="utf-8")

    env_file = tmp_path / "service.env"
    env_file.write_text("OPENAI_API_KEY=secret\nOPENCLAWBRAIN_DEFAULT_LLM=openai", encoding="utf-8")

    import openclawbrain.cli as cli_module

    monkeypatch.setattr(cli_module.sys, "platform", "darwin")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("OPENCLAWBRAIN_LOOP_PYTHON", raising=False)
    monkeypatch.delenv("OPENCLAWBRAIN_PYTHON", raising=False)

    code = main([
        "loop",
        "install",
        "--state",
        str(state_path),
        "--env-file",
        str(env_file),
        "--dry-run",
    ])
    assert code == 0

    payload = _extract_plist_from_output(capsys.readouterr().out)
    environment = payload["EnvironmentVariables"]
    assert environment["OPENAI_API_KEY"] == "secret"
    assert environment["OPENCLAWBRAIN_DEFAULT_LLM"] == "openai"


def test_loop_install_fast_flags_in_program_arguments(monkeypatch, tmp_path, capsys) -> None:
    """loop install --fast should bake fast flags into ProgramArguments."""
    state_path = tmp_path / "main" / "state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({"nodes": []}), encoding="utf-8")

    import openclawbrain.cli as cli_module

    monkeypatch.setattr(cli_module.sys, "platform", "darwin")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("OPENCLAWBRAIN_LOOP_PYTHON", raising=False)
    monkeypatch.delenv("OPENCLAWBRAIN_PYTHON", raising=False)

    code = main([
        "loop",
        "install",
        "--state",
        str(state_path),
        "--fast",
        "--dry-run",
    ])
    assert code == 0

    payload = _extract_plist_from_output(capsys.readouterr().out)
    program_arguments = payload["ProgramArguments"]

    def _assert_flag_value(flag: str, value: str) -> None:
        idx = program_arguments.index(flag)
        assert program_arguments[idx + 1] == value

    assert "--include-tool-results" in program_arguments
    assert "--advance-offsets-on-skip" in program_arguments
    assert "--maintain" in program_arguments
    assert "--harvest-labels" in program_arguments
    assert "--enable-teacher" in program_arguments
    assert "--enable-async-route-pg" in program_arguments
    assert "--enable-train-route-model" in program_arguments
    assert "--enable-dreaming" in program_arguments
    _assert_flag_value("--replay-priority", "tool")
    _assert_flag_value("--replay-max-interactions", "500")
    _assert_flag_value("--tool-result-max-chars", "20000")
    _assert_flag_value("--since-hours", "24.0")
    _assert_flag_value("--max-queries", "60")
    _assert_flag_value("--sample-rate", "0.1")
    _assert_flag_value("--max-candidates-per-node", "8")
    _assert_flag_value("--max-decision-points", "200")
    _assert_flag_value("--dream-max-queries", "40")
    _assert_flag_value("--dream-max-decision-points", "160")


def test_loop_pause_serve_when_locked_uses_launchctl(monkeypatch, tmp_path) -> None:
    """Loop pause-serve should bootout + bootstrap when the state lock is held."""
    state_path = tmp_path / "main" / "state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({"nodes": []}), encoding="utf-8")

    import openclawbrain.cli as cli_module

    monkeypatch.setattr(cli_module.Path, "home", lambda: tmp_path)
    plist_path = tmp_path / "Library" / "LaunchAgents" / "com.openclawbrain.main.plist"
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    plist_path.write_text("plist", encoding="utf-8")

    calls: list[list[str]] = []

    def fake_launchctl(argv: list[str]) -> int:
        calls.append(argv)
        return 0

    def fake_wait(_path: Path, timeout_seconds: int) -> bool:
        return timeout_seconds > 0

    ready, resume_cmd, reason = cli_module._maybe_pause_serve_for_state_lock(
        state_path=str(state_path),
        pause_when_locked=True,
        timeout_seconds=30,
        run_launchctl=fake_launchctl,
        wait_for_unlock=fake_wait,
        platform="darwin",
    )

    assert ready is True
    assert reason is None
    assert resume_cmd is not None
    assert calls[0][:2] == ["launchctl", "bootout"]
    fake_launchctl(resume_cmd)
    assert calls[1][:2] == ["launchctl", "bootstrap"]


def test_loop_pause_serve_kills_orphan_daemon(monkeypatch, tmp_path) -> None:
    """Loop pause-serve should terminate orphan daemon if lock persists after bootout."""
    state_path = tmp_path / "main" / "state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({"nodes": []}), encoding="utf-8")

    import openclawbrain.cli as cli_module

    monkeypatch.setattr(cli_module.Path, "home", lambda: tmp_path)
    plist_path = tmp_path / "Library" / "LaunchAgents" / "com.openclawbrain.main.plist"
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    plist_path.write_text("plist", encoding="utf-8")

    calls: list[list[str]] = []

    def fake_launchctl(argv: list[str]) -> int:
        calls.append(argv)
        return 0

    wait_results = [False, False, False, True]

    def fake_wait(_path: Path, _timeout_seconds: int) -> bool:
        return wait_results.pop(0)

    owner_pid = 4242

    def fake_get_pid(lock_path: Path) -> int | None:
        assert lock_path == lock_path_for_state(state_path)
        return owner_pid

    def fake_ps(pid: int) -> str | None:
        assert pid == owner_pid
        return f"python -m openclawbrain daemon --state {state_path} --other"

    kills: list[tuple[int, int]] = []

    def fake_kill(pid: int, sig: int) -> None:
        kills.append((pid, sig))

    monkeypatch.setattr(cli_module.os, "kill", fake_kill)

    ready, resume_cmd, reason = cli_module._maybe_pause_serve_for_state_lock(
        state_path=str(state_path),
        pause_when_locked=True,
        timeout_seconds=30,
        run_launchctl=fake_launchctl,
        wait_for_unlock=fake_wait,
        get_lock_owner_pid=fake_get_pid,
        get_process_command=fake_ps,
        platform="darwin",
    )

    assert ready is True
    assert reason is None
    assert resume_cmd is not None
    assert calls[0][:2] == ["launchctl", "bootout"]
    assert len(calls) == 1
    assert kills == [(owner_pid, signal.SIGTERM), (owner_pid, signal.SIGKILL)]


def test_loop_pause_serve_disabled_skips(monkeypatch, tmp_path) -> None:
    """Loop pause-serve disabled should skip when the state lock is held."""
    state_path = tmp_path / "main" / "state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({"nodes": []}), encoding="utf-8")

    import openclawbrain.cli as cli_module

    def fake_wait(_path: Path, _timeout_seconds: int) -> bool:
        return False

    ready, resume_cmd, reason = cli_module._maybe_pause_serve_for_state_lock(
        state_path=str(state_path),
        pause_when_locked=False,
        timeout_seconds=30,
        run_launchctl=lambda _argv: 0,
        wait_for_unlock=fake_wait,
        platform="darwin",
    )

    assert ready is False
    assert resume_cmd is None
    assert reason == "state_lock_held"


def test_serve_install_dry_run_includes_env_file(monkeypatch, tmp_path, capsys) -> None:
    """serve install --dry-run should include EnvironmentVariables from --env-file."""
    state_path = tmp_path / "main" / "state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({"nodes": []}), encoding="utf-8")

    env_file = tmp_path / "service.env"
    env_file.write_text("OPENAI_API_KEY=secret\nOPENCLAWBRAIN_DEFAULT_LLM=openai", encoding="utf-8")

    import openclawbrain.cli as cli_module

    monkeypatch.setattr(cli_module.sys, "platform", "darwin")

    code = main([
        "serve",
        "install",
        "--state",
        str(state_path),
        "--env-file",
        str(env_file),
        "--dry-run",
    ])
    assert code == 0

    payload_xml, _sep, _commands = capsys.readouterr().out.partition("Planned launchctl commands:")
    payload = plistlib.loads(payload_xml.encode("utf-8"))
    environment = payload["EnvironmentVariables"]
    assert environment["OPENAI_API_KEY"] == "secret"
    assert environment["OPENCLAWBRAIN_DEFAULT_LLM"] == "openai"


def test_serve_uninstall_dry_run_prints_bootout_command(monkeypatch, tmp_path, capsys) -> None:
    state_path = tmp_path / "main" / "state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({"nodes": []}), encoding="utf-8")

    import openclawbrain.cli as cli_module

    monkeypatch.setattr(cli_module.sys, "platform", "darwin")

    code = main([
        "serve",
        "uninstall",
        "--state",
        str(state_path),
        "--dry-run",
    ])
    assert code == 0

    output = capsys.readouterr().out
    assert "Planned launchctl commands:" in output
    assert f"launchctl bootout gui/{os.getuid()}" in output
    assert f"{Path.home()}/Library/LaunchAgents/com.openclawbrain.main.plist" in output


def test_serve_install_on_non_darwin_fails(monkeypatch, tmp_path, capsys) -> None:
    state_path = tmp_path / "main" / "state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({"nodes": []}), encoding="utf-8")

    import openclawbrain.cli as cli_module

    monkeypatch.setattr(cli_module.sys, "platform", "linux")
    code = main(["serve", "install", "--state", str(state_path)])
    assert code == 1
    assert "launchd lifecycle commands are supported on macOS only." in capsys.readouterr().err


def test_serve_command_passes_explicit_socket_path(monkeypatch) -> None:
    """serve should pass --socket-path through to the socket server."""
    called: list[list[str]] = []

    def fake_socket_server_main(argv: list[str] | None = None) -> int:
        called.append(list(argv or []))
        return 0

    from openclawbrain import socket_server

    monkeypatch.setattr(socket_server, "main", fake_socket_server_main)

    code = main(["serve", "--state", "/tmp/state.json", "--socket-path", "/tmp/d.sock"])
    assert code == 0
    assert called == [[
        "--state",
        "/tmp/state.json",
        "--socket-path",
        "/tmp/d.sock",
        "--embed-model",
        "auto",
        "--max-prompt-context-chars",
        "30000",
        "--max-fired-nodes",
        "30",
        "--route-mode",
        "learned",
        "--route-top-k",
        "5",
        "--route-alpha-sim",
        "0.5",
        "--route-use-relevance",
        "true",
        "--route-enable-stop",
        "false",
        "--route-stop-margin",
        "0.1",
    ]]


def test_socket_health_status_missing_socket_returns_error(tmp_path) -> None:
    """Socket health helper should fail fast when the socket path is missing."""
    from openclawbrain import cli as cli_module

    ok, health, error = cli_module._socket_health_status(str(tmp_path / "missing.sock"))
    assert ok is False
    assert health is None
    assert isinstance(error, str) and "socket missing" in error


def test_socket_health_status_uses_client_ping(monkeypatch, tmp_path) -> None:
    """Socket health helper should return daemon health payload when ping succeeds."""
    from openclawbrain import cli as cli_module

    socket_path = tmp_path / "daemon.sock"
    socket_path.write_text("", encoding="utf-8")

    class FakeClient:
        def __init__(self, socket_path: str, timeout: float = 30.0) -> None:
            self.socket_path = socket_path
            self.timeout = timeout

        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def health(self) -> dict[str, object]:
            return {"nodes": 2, "edges": 1}

    import openclawbrain.socket_client as socket_client_module

    monkeypatch.setattr(socket_client_module, "OCBClient", FakeClient)

    ok, health, error = cli_module._socket_health_status(str(socket_path))
    assert ok is True
    assert health == {"nodes": 2, "edges": 1}
    assert error is None


def test_openclaw_install_runs_expected_commands(monkeypatch, tmp_path) -> None:
    hooks_dir = tmp_path / "openclawbrain-context-injector"
    hooks_dir.mkdir(parents=True, exist_ok=True)

    import openclawbrain.cli as cli_module
    from types import SimpleNamespace

    calls: list[list[str]] = []

    def fake_run(argv: list[str]) -> SimpleNamespace:
        calls.append(argv)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli_module, "_run_subprocess_command", fake_run)
    monkeypatch.setattr(cli_module, "_resolve_hooks_path", lambda _: hooks_dir)

    code = main([
        "openclaw",
        "install",
        "--agent",
        "main",
        "--yes",
        "--hooks-path",
        str(hooks_dir),
    ])
    assert code == 0
    assert calls[0][:4] == [sys.executable, "-m", "openclawbrain.cli", "serve"]
    assert calls[1][:3] == ["openclaw", "hooks", "install"]
    assert calls[2][:3] == ["openclaw", "hooks", "enable"]
    assert calls[3][:4] == [sys.executable, "-m", "openclawbrain.cli", "loop"]
    assert calls[4][:4] == [sys.executable, "-m", "openclawbrain.cli", "harvest"]
    assert calls[5][:4] == [sys.executable, "-m", "openclawbrain.cli", "async-route-pg"]
    assert calls[-1][:3] == ["openclaw", "gateway", "restart"]


def test_serve_status_payload_reports_ping_failure(monkeypatch, tmp_path) -> None:
    """Serve status payload should include socket existence and ping error details."""
    from openclawbrain import cli as cli_module

    socket_path = tmp_path / "daemon.sock"
    socket_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(
        cli_module,
        "_socket_health_status",
        lambda _path: (False, None, "boom"),
    )

    payload = cli_module._serve_status_payload(str(tmp_path / "state.json"), str(socket_path))
    assert payload["socket_exists"] is True
    assert payload["daemon_running"] is False
    assert payload["health"] is None
    assert payload["error"] == "boom"


def test_serve_status_subcommand_uses_health_ping(monkeypatch, capsys) -> None:
    """`serve status` should report running when socket ping succeeds."""
    from openclawbrain import cli as cli_module

    monkeypatch.setattr(
        cli_module,
        "_serve_status_payload",
        lambda _state, _socket: {
            "state_path": "/tmp/state.json",
            "socket_path": "/tmp/daemon.sock",
            "socket_exists": True,
            "daemon_running": True,
            "health": {"nodes": 4, "edges": 3, "dormant_pct": 0.1},
            "error": None,
        },
    )

    code = main(["serve", "status", "--state", "/tmp/state.json"])
    assert code == 0
    out = capsys.readouterr().out
    assert "serve status: running" in out
    assert "nodes=4" in out
    assert "edges=3" in out


def test_report_handles_missing_daemon_health(monkeypatch, tmp_path, capsys) -> None:
    """report should keep local stats and warn when daemon health is unavailable."""
    from openclawbrain import cli as cli_module

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))

    state_dir = tmp_path / "main"
    state_dir.mkdir()
    state_path = state_dir / "state.json"
    _write_state(state_path)
    socket_path = Path(cli_module._default_daemon_socket_path(str(state_path)))
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    socket_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(cli_module, "_socket_health_status", lambda _path: (False, None, "timeout"))

    code = main(["report", "--state", str(state_path)])
    assert code == 0
    out = capsys.readouterr().out
    assert "warning: health unavailable (daemon timeout)" in out
    assert "orphans:" in out


def test_report_surfaces_route_model_health_fields(monkeypatch, tmp_path, capsys) -> None:
    from openclawbrain import cli as cli_module

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))

    state_dir = tmp_path / "main"
    state_dir.mkdir()
    state_path = state_dir / "state.json"
    _write_state(state_path)
    socket_path = Path(cli_module._default_daemon_socket_path(str(state_path)))
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    socket_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(
        cli_module,
        "_socket_health_status",
        lambda _path: (
            True,
            {
                "route_model_present": False,
                "route_mode_configured": "learned",
                "route_mode_effective": "edge+sim",
                "route_model_error": "load_failed: bad zip",
                "route_model_path": str(state_dir / "route_model.npz"),
            },
            None,
        ),
    )

    code = main(["report", "--state", str(state_path)])
    assert code == 0
    out = capsys.readouterr().out
    assert "route_model_error: load_failed: bad zip" in out
    assert "warning: learned routing configured but effective mode is degraded" in out


def test_serve_stop_subcommand_sends_shutdown(monkeypatch, capsys) -> None:
    """`serve stop` should send daemon shutdown request over socket when reachable."""
    from openclawbrain import cli as cli_module
    import openclawbrain.socket_client as socket_client_module

    monkeypatch.setattr(cli_module, "_socket_health_status", lambda _socket: (True, {"nodes": 1}, None))
    captured: dict[str, object] = {}

    class FakeClient:
        def __init__(self, socket_path: str, timeout: float = 30.0) -> None:
            captured["socket_path"] = socket_path
            captured["timeout"] = timeout

        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def request(self, method: str, params: dict[str, object] | None = None) -> dict[str, object]:
            captured["method"] = method
            captured["params"] = dict(params or {})
            return {"shutdown": True}

    monkeypatch.setattr(socket_client_module, "OCBClient", FakeClient)

    code = main(["serve", "stop", "--state", "/tmp/state.json", "--socket-path", "/tmp/d.sock"])
    assert code == 0
    assert captured["socket_path"] == "/tmp/d.sock"
    assert captured["method"] == "shutdown"
    assert captured["params"] == {}
    out = capsys.readouterr().out
    assert "Shutdown request sent to /tmp/d.sock" in out


def test_daemon_profile_defaults_with_cli_override(monkeypatch, tmp_path) -> None:
    """CLI daemon flags should override profile values."""
    profile_path = tmp_path / "brainprofile.json"
    profile_path.write_text(
        json.dumps(
            {
                "policy": {
                    "max_prompt_context_chars": 11111,
                    "max_fired_nodes": 17,
                    "route_mode": "edge",
                    "route_top_k": 3,
                    "route_alpha_sim": 0.25,
                    "route_use_relevance": False,
                },
                "embedder": {"embed_model": "hash"},
            }
        ),
        encoding="utf-8",
    )
    captured: list[list[str]] = []

    def fake_daemon_main(argv: list[str] | None = None) -> int:
        captured.append(list(argv or []))
        return 0

    import openclawbrain.daemon as daemon_module

    monkeypatch.setattr(daemon_module, "main", fake_daemon_main)

    code = main([
        "daemon",
        "--state",
        "/tmp/state.json",
        "--profile",
        str(profile_path),
        "--embed-model",
        "auto",
        "--route-mode",
        "edge+sim",
        "--max-fired-nodes",
        "99",
    ])
    assert code == 0
    assert captured
    argv = captured[0]
    assert "--embed-model" in argv and argv[argv.index("--embed-model") + 1] == "auto"
    assert "--route-mode" in argv and argv[argv.index("--route-mode") + 1] == "edge+sim"
    assert "--max-fired-nodes" in argv and argv[argv.index("--max-fired-nodes") + 1] == "99"
    # Non-overridden fields still come from profile.
    assert "--max-prompt-context-chars" in argv and argv[argv.index("--max-prompt-context-chars") + 1] == "11111"


def test_daemon_command_defaults_route_mode_learned(monkeypatch) -> None:
    captured: list[list[str]] = []

    def fake_daemon_main(argv: list[str] | None = None) -> int:
        captured.append(list(argv or []))
        return 0

    import openclawbrain.daemon as daemon_module

    monkeypatch.setattr(daemon_module, "main", fake_daemon_main)

    code = main(["daemon", "--state", "/tmp/state.json"])
    assert code == 0
    assert captured
    argv = captured[0]
    assert "--route-mode" in argv and argv[argv.index("--route-mode") + 1] == "learned"


def test_inject_command_defaults_connect_min_sim_for_hash_embedder(tmp_path, capsys) -> None:
    """test inject uses a zero min similarity threshold by default for hash embeds."""
    state_path = tmp_path / "state.json"
    _write_state(state_path)

    code = main(
        [
            "inject",
            "--state",
            str(state_path),
            "--id",
            "fix::ci",
            "--content",
            "alpha",
            "--type",
            "CORRECTION",
            "--json",
        ]
    )
    assert code == 0

    out = json.loads(capsys.readouterr().out.strip())
    assert out["connected_to"]
    assert out["inhibitory_edges_created"] > 0


def test_cli_replay_default_mode_is_full(tmp_path, capsys, monkeypatch) -> None:
    """replay without mode/legacy flags defaults to full and prints a note."""
    state_path = tmp_path / "state.json"
    _write_state(state_path)

    sessions = tmp_path / "sessions.jsonl"
    sessions.write_text(
        "\n".join(
            [
                json.dumps({"role": "user", "content": "do not do that", "ts": 1.0}),
                json.dumps({"role": "user", "content": "how to deploy", "ts": 2.0}),
            ]
        ),
        encoding="utf-8",
    )

    import openclawbrain.cli as cli_module

    monkeypatch.setattr(cli_module, "run_fast_learning", lambda **_: {"events_injected": 0})
    monkeypatch.setattr(cli_module, "run_harvest", lambda **_: {"harvested": 0})

    code = main(
        [
            "replay",
            "--state",
            str(state_path),
            "--sessions",
            str(sessions),
            "--json",
        ]
    )
    assert code == 0
    captured = capsys.readouterr()
    result = json.loads(captured.out.strip())
    assert result["queries_replayed"] == 2
    assert "fast_learning" in result
    assert "harvest" in result
    assert "defaulting to --mode full" in captured.err


def test_cli_replay_edges_only_skips_learning(tmp_path, capsys) -> None:
    """replay --edges-only does not run fast_learning or harvest."""
    state_path = tmp_path / "state.json"
    _write_state(state_path)

    sessions = tmp_path / "sessions.jsonl"
    sessions.write_text(
        json.dumps({"role": "user", "content": "alpha", "ts": 1.0}),
        encoding="utf-8",
    )

    code = main(
        [
            "replay",
            "--state",
            str(state_path),
            "--sessions",
            str(sessions),
            "--edges-only",
            "--json",
        ]
    )
    assert code == 0
    result = json.loads(capsys.readouterr().out.strip())

    assert "fast_learning" not in result
    assert "harvest" not in result
    assert result["queries_replayed"] == 1


def test_cli_replay_tool_result_flags_forwarded_json_and_text(tmp_path, capsys, monkeypatch) -> None:
    """replay toolResult controls are forwarded to the replay loader."""
    state_path = tmp_path / "state.json"
    _write_state(state_path)
    sessions = tmp_path / "sessions.jsonl"
    sessions.write_text('{"role":"user","content":"alpha","ts":1.0}\n', encoding="utf-8")

    import openclawbrain.cli as cli_module

    seen: list[dict[str, object]] = []

    def fake_load_interactions_for_replay(session_paths, since_lines=None, **kwargs):
        seen.append(dict(kwargs))
        interactions = [
            {
                "query": "alpha",
                "response": None,
                "tool_calls": [],
                "ts": 1.0,
                "source": str(sessions),
                "line_no": 1,
            }
        ]
        return interactions, {str(sessions): 1}

    monkeypatch.setattr(cli_module, "load_interactions_for_replay", fake_load_interactions_for_replay)

    code = main(
        [
            "replay",
            "--state",
            str(state_path),
            "--sessions",
            str(sessions),
            "--edges-only",
            "--no-include-tool-results",
            "--tool-result-allowlist",
            "image,summarize",
            "--tool-result-max-chars",
            "123",
            "--json",
        ]
    )
    assert code == 0
    # JSON output should parse.
    json.loads(capsys.readouterr().out.strip())
    assert seen[-1]["include_tool_results"] is False
    assert seen[-1]["tool_result_allowlist"] == {"image", "summarize"}
    assert seen[-1]["tool_result_max_chars"] == 123

    code = main(
        [
            "replay",
            "--state",
            str(state_path),
            "--sessions",
            str(sessions),
            "--edges-only",
            "--include-tool-results",
            "--tool-result-allowlist",
            "openai-whisper",
            "--tool-result-max-chars",
            "456",
        ]
    )
    assert code == 0
    assert seen[-1]["include_tool_results"] is True
    assert seen[-1]["tool_result_allowlist"] == {"openai-whisper"}
    assert seen[-1]["tool_result_max_chars"] == 456


def test_cli_replay_progress_events_jsonl(tmp_path, capsys) -> None:
    """replay emits JSONL progress events when --json and --progress-every are set."""
    state_path = tmp_path / "state.json"
    _write_state(state_path)
    sessions = tmp_path / "sessions.jsonl"
    sessions.write_text(
        "\n".join(
            [
                json.dumps({"role": "user", "content": "alpha", "ts": 1.0}),
                json.dumps({"role": "assistant", "content": "ok", "ts": 1.1}),
                json.dumps({"role": "user", "content": "beta", "ts": 2.0}),
                json.dumps({"role": "assistant", "content": "ok", "ts": 2.1}),
            ]
        ),
        encoding="utf-8",
    )

    code = main(
        [
            "replay",
            "--state",
            str(state_path),
            "--sessions",
            str(sessions),
            "--edges-only",
            "--progress-every",
            "1",
            "--json",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert '{"type": "progress", "phase": "replay"' in out


def test_cli_replay_fast_learning_progress_events_jsonl(tmp_path, capsys, monkeypatch) -> None:
    """replay emits fast-learning JSONL progress events when enabled."""
    state_path = tmp_path / "state.json"
    _write_state(state_path)
    sessions = tmp_path / "sessions.jsonl"
    sessions.write_text(
        "\n".join(
            [
                json.dumps({"role": "user", "content": "that is wrong", "ts": 1.0}),
                json.dumps({"role": "assistant", "content": "ok", "ts": 1.1}),
            ]
        ),
        encoding="utf-8",
    )

    from openclawbrain import full_learning as learning_module

    def fake_llm(_: str, __: str) -> str:
        return json.dumps(
            {
                "corrections": [{"content": "Use the runbook.", "context": "ctx", "severity": "high"}],
                "teachings": [],
                "reinforcements": [],
            }
        )

    monkeypatch.setattr(learning_module, "openai_llm_fn", fake_llm)

    code = main(
        [
            "replay",
            "--state",
            str(state_path),
            "--sessions",
            str(sessions),
            "--fast-learning",
            "--llm",
            "none",
            "--stop-after-fast-learning",
            "--progress-every",
            "1",
            "--json",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert '{"type": "progress", "phase": "fast_learning"' in out


def test_read_replay_checkpoint_progress_minimal_fast_learning() -> None:
    """Helper formats a minimal fast-learning checkpoint progress line."""
    checkpoint = {
        "fast_learning": {
            "windows_processed": 2,
            "windows_total": 10,
            "status": "running",
        }
    }
    line = _read_replay_checkpoint_progress(checkpoint, agent_id="agent-1")
    assert line == "[build-all] agent=agent-1 replay_progress fast_learning=2/10 status=running"


def test_cli_replay_show_checkpoint_json_uses_new_schema_fixture(tmp_path, capsys) -> None:
    """replay --show-checkpoint emits stable JSON status for new schema checkpoints."""
    state_path = tmp_path / "state.json"
    _write_state(state_path)
    fixture = Path(__file__).parent / "fixtures" / "checkpoints" / "new_schema.json"
    checkpoint_path = tmp_path / "replay_checkpoint.json"
    checkpoint_path.write_text(fixture.read_text(encoding="utf-8"), encoding="utf-8")

    code = main(
        [
            "replay",
            "--state",
            str(state_path),
            "--checkpoint",
            str(checkpoint_path),
            "--show-checkpoint",
            "--resume",
            "--json",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["type"] == "checkpoint_status"
    assert payload["checkpoint_path"] == str(checkpoint_path)
    assert payload["schema_version"] == 1
    assert payload["fast_learning"]["windows_processed"] == 4
    assert payload["fast_learning"]["windows_total"] == 10
    assert payload["fast_learning"]["status"] == "running"
    assert payload["replay"]["queries_processed"] == 8
    assert payload["replay"]["queries_total"] == 16
    assert payload["replay"]["merge_batches"] == 2
    assert payload["resume"]["would_take_effect"] is True


def test_cli_replay_show_checkpoint_text_includes_fast_learning_counters(tmp_path, capsys) -> None:
    """replay --show-checkpoint prints fast-learning counter fields when present."""
    state_path = tmp_path / "state.json"
    _write_state(state_path)
    checkpoint_path = tmp_path / "replay_checkpoint.json"
    checkpoint_path.write_text(
        json.dumps(
            {
                "version": 1,
                "fast_learning": {
                    "status": "running",
                    "windows_processed": 4,
                    "windows_total": 10,
                    "windows_candidate": 12,
                    "windows_sent_to_llm": 10,
                    "windows_skipped_low_signal": 2,
                    "windows_skipped_existing_pointer": 0,
                    "updated_at": 1700000100.0,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    code = main(
        [
            "replay",
            "--state",
            str(state_path),
            "--checkpoint",
            str(checkpoint_path),
            "--show-checkpoint",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "Fast learning: 4/10 windows, status=running" in out
    assert "windows_candidate=12" in out
    assert "windows_sent_to_llm=10" in out
    assert "windows_skipped_low_signal=2" in out
    assert "windows_skipped_existing_pointer=0" in out


def test_cli_replay_show_checkpoint_text_legacy_fixture_warns(tmp_path, capsys) -> None:
    """replay --show-checkpoint warns when legacy top-level sessions are used."""
    state_path = tmp_path / "state.json"
    _write_state(state_path)
    fixture = Path(__file__).parent / "fixtures" / "checkpoints" / "legacy_schema.json"
    checkpoint_path = tmp_path / "replay_checkpoint.json"
    checkpoint_path.write_text(fixture.read_text(encoding="utf-8"), encoding="utf-8")

    with pytest.warns(UserWarning, match="legacy top-level 'sessions' offsets"):
        code = main(
            [
                "replay",
                "--state",
                str(state_path),
                "--checkpoint",
                str(checkpoint_path),
                "--show-checkpoint",
                "--resume",
            ]
        )
    assert code == 0
    out = capsys.readouterr().out
    assert f"Checkpoint: {checkpoint_path}" in out
    assert "Schema version: 0" in out
    assert "Resume would take effect: True (resume=True, ignore_checkpoint=False)" in out


def test_cli_replay_startup_banner_printed_for_text_mode(tmp_path, capsys) -> None:
    """replay prints startup banner in non-JSON mode."""
    state_path = tmp_path / "state.json"
    _write_state(state_path)
    sessions = tmp_path / "sessions.jsonl"
    sessions.write_text(
        "\n".join(
            [
                json.dumps({"role": "user", "content": "alpha", "ts": 1.0}),
                json.dumps({"role": "assistant", "content": "ok", "ts": 1.1}),
            ]
        ),
        encoding="utf-8",
    )

    code = main(
        [
            "replay",
            "--state",
            str(state_path),
            "--sessions",
            str(sessions),
            "--edges-only",
        ]
    )
    assert code == 0
    err = capsys.readouterr().err
    assert "Replay startup:" in err
    assert "mode: edges-only" in err
    assert "resume: False" in err
    assert "fresh: False" in err
    assert "phases: replay" in err


def test_cli_replay_quiet_suppresses_banners_and_progress(tmp_path, capsys) -> None:
    """quiet replay suppresses status banners and progress lines on stderr."""
    state_path = tmp_path / "state.json"
    _write_state(state_path)
    sessions = tmp_path / "sessions.jsonl"
    sessions.write_text(
        "\n".join(
            [
                json.dumps({"role": "user", "content": "alpha", "ts": 1.0}),
                json.dumps({"role": "assistant", "content": "ok", "ts": 1.1}),
                json.dumps({"role": "user", "content": "beta", "ts": 2.0}),
                json.dumps({"role": "assistant", "content": "ok", "ts": 2.1}),
            ]
        ),
        encoding="utf-8",
    )

    code = main(
        [
            "replay",
            "--state",
            str(state_path),
            "--sessions",
            str(sessions),
            "--edges-only",
            "--progress-every",
            "1",
            "--quiet",
        ]
    )
    assert code == 0
    err = capsys.readouterr().err
    assert "Loaded " not in err
    assert "[replay]" not in err


def test_cli_replay_alias_extract_learning_events(tmp_path, capsys, monkeypatch) -> None:
    """--extract-learning-events is accepted as alias for --fast-learning."""
    from openclawbrain.cli import _build_parser
    parser = _build_parser()
    args = parser.parse_args([
        "replay", "--state", "/tmp/x.json", "--sessions", "/tmp/s",
        "--extract-learning-events", "--stop-after-fast-learning",
    ])
    assert args.fast_learning is True


def test_cli_replay_alias_full_pipeline(tmp_path, capsys, monkeypatch) -> None:
    """--full-pipeline is accepted as alias for --full-learning."""
    from openclawbrain.cli import _build_parser
    parser = _build_parser()
    args = parser.parse_args([
        "replay", "--state", "/tmp/x.json", "--sessions", "/tmp/s",
        "--full-pipeline",
    ])
    assert args.full_learning is True


def test_cli_replay_mode_resolution_maps_legacy_flags() -> None:
    """legacy replay flags map cleanly to replay --mode values."""
    from openclawbrain.cli import _build_parser, _resolve_replay_mode

    parser = _build_parser()
    default_args = parser.parse_args(["replay", "--state", "/tmp/x.json", "--sessions", "/tmp/s"])
    assert _resolve_replay_mode(default_args) == ("full", True)

    edges_args = parser.parse_args(["replay", "--state", "/tmp/x.json", "--sessions", "/tmp/s", "--edges-only"])
    assert _resolve_replay_mode(edges_args) == ("edges-only", False)

    fast_args = parser.parse_args(["replay", "--state", "/tmp/x.json", "--sessions", "/tmp/s", "--fast-learning"])
    assert _resolve_replay_mode(fast_args) == ("fast-learning", False)

    full_args = parser.parse_args(["replay", "--state", "/tmp/x.json", "--sessions", "/tmp/s", "--full-learning"])
    assert _resolve_replay_mode(full_args) == ("full", False)


def test_cli_replay_filters_parse() -> None:
    """replay parser accepts budgeted/prioritized replay filtering flags."""
    from openclawbrain.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args(
        [
            "replay",
            "--state",
            "/tmp/x.json",
            "--sessions",
            "/tmp/s",
            "--replay-since-hours",
            "1.5",
            "--replay-max-interactions",
            "42",
            "--replay-sample-rate",
            "0.25",
            "--replay-priority",
            "tool",
            "--advance-offsets-on-skip",
        ]
    )
    assert args.replay_since_hours == 1.5
    assert args.replay_max_interactions == 42
    assert args.replay_sample_rate == 0.25
    assert args.replay_priority == "tool"
    assert args.advance_offsets_on_skip is True


def test_cli_replay_fresh_aliases_parse() -> None:
    """--fresh/--no-checkpoint set fresh-start semantics."""
    from openclawbrain.cli import _build_parser

    parser = _build_parser()
    args_fresh = parser.parse_args(["replay", "--state", "/tmp/x.json", "--sessions", "/tmp/s", "--fresh"])
    assert args_fresh.fresh is True

    args_no_checkpoint = parser.parse_args(
        ["replay", "--state", "/tmp/x.json", "--sessions", "/tmp/s", "--no-checkpoint"]
    )
    assert args_no_checkpoint.fresh is True


def test_cli_replay_llm_model_parse() -> None:
    """replay accepts --llm-model when --llm ollama."""
    from openclawbrain.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args(
        [
            "replay",
            "--state",
            "/tmp/x.json",
            "--sessions",
            "/tmp/s",
            "--llm",
            "ollama",
            "--llm-model",
            "qwen3.5:9b",
        ]
    )
    assert args.llm == "ollama"
    assert args.llm_model == "qwen3.5:9b"


def test_cli_async_route_pg_teacher_ollama_parses() -> None:
    """--teacher ollama is accepted by async-route-pg."""
    from openclawbrain.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args(["async-route-pg", "--state", "/tmp/x.json", "--teacher", "ollama"])
    assert args.teacher == "ollama"


def test_cli_build_all_parser_accepts_subcommand() -> None:
    """build-all subcommand parses with defaults."""
    from openclawbrain.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args(["build-all"])
    assert args.command == "build-all"
    assert args.parallel_agents == 1
    assert args.require_local_embedder is False


def test_cli_bootstrap_parser_defaults() -> None:
    """bootstrap subcommand parses with defaults."""
    from openclawbrain.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args(["bootstrap", "--agent", "main"])
    assert args.command == "bootstrap"
    assert args.agent == "main"
    assert args.fast is True


def test_cmd_status_json_includes_embedder_dim_and_index_dim(tmp_path, capsys) -> None:
    """status --json includes embedder and index dimensions."""
    state_path = tmp_path / "state.json"
    _write_state(
        state_path,
        index_payload={"a": [0.0, 1.0, 0.0], "b": [1.0, 0.0, 0.0]},
        meta={"embedder_name": "local:bge-large-en-v1.5", "embedder_dim": 3},
    )
    code = main(["status", "--state", str(state_path), "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["embedder_dim"] == 3
    assert payload["index_dim"] == 3


def test_build_all_preflight_guard_blocks_dim_mismatch_without_reembed(tmp_path) -> None:
    """build-all preflight fails when embedder_dim and index_dim mismatch and reembed is disabled."""
    import openclawbrain.cli as cli_module

    state_path = tmp_path / "state.json"
    status_before = tmp_path / "status_before.json"
    state_path.write_text(json.dumps({"nodes": []}), encoding="utf-8")
    status_before.write_text(
        json.dumps(
            {
                "embedder_name": "local:bge-large-en-v1.5",
                "embedder_dim": 1536,
                "index_dim": 1024,
            }
        ),
        encoding="utf-8",
    )
    code, message = cli_module._evaluate_build_all_preflight(
        state_path=str(state_path),
        status_before_path=str(status_before),
        reembed=False,
        require_local_embedder=False,
    )
    assert code == 1
    assert message is not None
    assert "Re-run with --reembed" in message


def test_build_all_preflight_guard_allows_dim_mismatch_with_reembed(tmp_path, capsys) -> None:
    """build-all preflight continues on mismatch when reembed is enabled."""
    import openclawbrain.cli as cli_module

    state_path = tmp_path / "state.json"
    status_before = tmp_path / "status_before.json"
    state_path.write_text(json.dumps({"nodes": []}), encoding="utf-8")
    status_before.write_text(
        json.dumps(
            {
                "embedder_name": "local:bge-large-en-v1.5",
                "embedder_dim": 1536,
                "index_dim": 1024,
            }
        ),
        encoding="utf-8",
    )
    code, message = cli_module._evaluate_build_all_preflight(
        state_path=str(state_path),
        status_before_path=str(status_before),
        reembed=True,
        require_local_embedder=False,
    )
    assert code == 0
    assert message is None
    assert "continuing because --reembed is enabled" in capsys.readouterr().out


def test_build_all_preflight_guard_blocks_openai_without_reembed_when_local_required(tmp_path) -> None:
    """build-all local-only preflight fails for openai states when --reembed is disabled."""
    import openclawbrain.cli as cli_module

    state_path = tmp_path / "state.json"
    status_before = tmp_path / "status_before.json"
    state_path.write_text(json.dumps({"nodes": []}), encoding="utf-8")
    status_before.write_text(
        json.dumps(
            {
                "embedder_name": "openai-text-embedding-3-small",
                "embedder_dim": 3,
                "index_dim": 3,
            }
        ),
        encoding="utf-8",
    )
    code, message = cli_module._evaluate_build_all_preflight(
        state_path=str(state_path),
        status_before_path=str(status_before),
        reembed=False,
        require_local_embedder=True,
    )
    assert code == 1
    assert message is not None
    assert "OpenAI embedder" in message


def test_build_all_root_manifest_written_before_agents_run(tmp_path, monkeypatch) -> None:
    """Build-all root manifest can be written immediately via internal helpers."""
    import openclawbrain.cli as cli_module

    monkeypatch.setattr(cli_module.Path, "home", lambda: tmp_path)
    run_ts = datetime(2026, 3, 3, 8, 9, 10, 12000, tzinfo=timezone.utc)
    ts_label = run_ts.strftime("%Y%m%dT%H%M%S") + f"{run_ts.microsecond // 1000:03d}Z"
    events_jsonl = tmp_path / ".openclawbrain" / "scratch" / f"build-all.{ts_label}.events.jsonl"
    manifest_path = tmp_path / ".openclawbrain" / "scratch" / f"build-all.{ts_label}.manifest.json"

    parser = cli_module._build_parser()
    args = parser.parse_args(["build-all"])
    ocb_bin = cli_module._resolve_openclawbrain_bin()

    payload = cli_module._build_all_root_manifest_payload(
        run_id=ts_label,
        run_ts=run_ts,
        args=args,
        ocb_bin=ocb_bin,
        agent_ids=[],
        parallel_agents=1,
        events_jsonl=events_jsonl,
        status="running",
        agents=[],
    )
    cli_module._write_json_atomic(manifest_path, payload)
    assert manifest_path.exists()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert data["run_id"] == ts_label
    assert data["status"] == "running"
    assert data["agents"] == []


def test_cmd_build_all_events_jsonl_contains_run_and_agent_events(tmp_path, monkeypatch) -> None:
    """build-all creates JSONL events with run/agent lifecycle records."""
    import openclawbrain.cli as cli_module

    monkeypatch.setattr(cli_module.Path, "home", lambda: tmp_path)
    (tmp_path / ".openclawbrain" / "agent-1").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".openclaw" / "agents" / "agent-1").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".openclaw" / "agents" / "agent-1" / "sessions").write_text("", encoding="utf-8")
    (tmp_path / ".openclaw" / "openclaw.json").write_text(
        json.dumps({"agents": {"list": [{"id": "agent-1"}]}}),
        encoding="utf-8",
    )
    _write_state(tmp_path / ".openclawbrain" / "agent-1" / "state.json")

    def fake_run_logged_command(cmd: list[str], **kwargs) -> int:
        stdout_path = kwargs.get("stdout_path")
        if isinstance(stdout_path, Path) and stdout_path.name.endswith("status_before.json"):
            stdout_path.write_text(
                json.dumps(
                    {
                        "embedder_name": "local:bge-large-en-v1.5",
                        "embedder_dim": 3,
                        "index_dim": 3,
                    }
                ),
                encoding="utf-8",
            )
        return 0

    monkeypatch.setattr(cli_module, "_run_logged_command", fake_run_logged_command)
    monkeypatch.setattr(cli_module, "_run_logged_replay_command_with_watchdog", lambda *args, **kwargs: 0)

    events_path = tmp_path / ".openclawbrain" / "scratch" / "events.jsonl"
    code = cli_module.main(
        [
            "build-all",
            "--agents",
            "agent-1",
            "--events-jsonl",
            str(events_path),
        ]
    )
    assert code == 0

    lines = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert any(item.get("type") == "run_start" for item in lines)
    assert any(item.get("type") == "agent_start" for item in lines)


def test_cli_build_all_forwards_workers_and_llm_model_to_replay_and_records_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """build-all forwards --workers and --llm-model through the replay subprocess."""
    import openclawbrain.cli as cli_module

    monkeypatch.setattr(cli_module.Path, "home", lambda: tmp_path)
    (tmp_path / ".openclawbrain" / "agent-1").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".openclaw" / "agents" / "agent-1").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".openclaw" / "agents" / "agent-1" / "sessions").write_text("", encoding="utf-8")
    (tmp_path / ".openclaw" / "openclaw.json").write_text(
        json.dumps({"agents": {"list": [{"id": "agent-1"}]}}),
        encoding="utf-8",
    )
    _write_state(tmp_path / ".openclawbrain" / "agent-1" / "state.json")

    captured_replay_cmd: list[list[str]] = []

    def fake_run_logged_command(cmd: list[str], **kwargs) -> int:
        stdout_path = kwargs.get("stdout_path")
        if isinstance(stdout_path, Path) and stdout_path.name.endswith("status_before.json"):
            stdout_path.write_text(
                json.dumps(
                    {
                        "embedder_name": "local:bge-large-en-v1.5",
                        "embedder_dim": 3,
                        "index_dim": 3,
                    }
                ),
                encoding="utf-8",
            )
        return 0

    def fake_run_logged_replay_command(cmd: list[str], **kwargs) -> int:
        captured_replay_cmd.append(cmd)
        return 0

    monkeypatch.setattr(cli_module, "_run_logged_command", fake_run_logged_command)
    monkeypatch.setattr(cli_module, "_run_logged_replay_command_with_watchdog", fake_run_logged_replay_command)

    code = cli_module.main(
        [
            "build-all",
            "--agents",
            "agent-1",
            "--llm",
            "ollama",
            "--llm-model",
            "qwen3.5:9b",
            "--workers",
            "8",
            "--replay-since-hours",
            "2.5",
            "--replay-max-interactions",
            "10",
            "--replay-sample-rate",
            "0.3",
            "--replay-priority",
            "tool",
            "--advance-offsets-on-skip",
        ]
    )
    assert code == 0

    assert captured_replay_cmd, "replay command was not invoked"
    replay_cmd = captured_replay_cmd[0]
    assert "--workers" in replay_cmd and "8" in replay_cmd
    assert "--llm-model" in replay_cmd and "qwen3.5:9b" in replay_cmd
    assert "--replay-since-hours" in replay_cmd and "2.5" in replay_cmd
    assert "--replay-max-interactions" in replay_cmd and "10" in replay_cmd
    assert "--replay-sample-rate" in replay_cmd and "0.3" in replay_cmd
    assert "--replay-priority" in replay_cmd and "tool" in replay_cmd
    assert "--advance-offsets-on-skip" in replay_cmd

    manifest_paths = sorted((tmp_path / ".openclawbrain" / "scratch").glob("build-all.*.manifest.json"))
    assert manifest_paths
    manifest = json.loads(manifest_paths[0].read_text(encoding="utf-8"))
    assert manifest["args"]["workers"] == 8
    assert manifest["args"]["llm_model"] == "qwen3.5:9b"
    assert manifest["args"]["replay_since_hours"] == 2.5
    assert manifest["args"]["replay_max_interactions"] == 10
    assert manifest["args"]["replay_sample_rate"] == 0.3
    assert manifest["args"]["replay_priority"] == "tool"


def test_replay_watchdog_restarts_and_fallback(tmp_path: Path) -> None:
    """watchdog restarts wedged replay and falls back to edges-only."""
    import openclawbrain.cli as cli_module

    checkpoint_path = tmp_path / "replay_checkpoint.json"
    checkpoint_path.write_text(
        json.dumps({"replay": {"queries_processed": 0, "queries_total": 10, "updated_at": 1.0}}),
        encoding="utf-8",
    )
    log_path = tmp_path / "replay.log"
    watchdog_path = tmp_path / "replay_watchdog.jsonl"

    spawned_cmds: list[list[str]] = []

    class FakeProc:
        def __init__(self, returncode: int | None = None) -> None:
            self.returncode = returncode

        def poll(self):
            return self.returncode

        def wait(self, timeout: float | None = None):
            if self.returncode is None:
                self.returncode = 143
            return self.returncode

        def terminate(self):
            self.returncode = 143

        def kill(self):
            self.returncode = 137

    def fake_spawn(cmd: list[str], **_kwargs):
        spawned_cmds.append(cmd)
        if "--mode" in cmd and cmd[cmd.index("--mode") + 1] == "edges-only":
            return FakeProc(returncode=0)
        return FakeProc()

    class FakeClock:
        def __init__(self) -> None:
            self.value = 0.0

        def monotonic(self) -> float:
            return self.value

        def sleep(self, seconds: float) -> None:
            self.value += seconds

    clock = FakeClock()

    code = cli_module._run_logged_replay_command_with_watchdog(
        ["openclawbrain", "replay", "--mode", "full"],
        log_path=log_path,
        step_name="replay",
        checkpoint_path=checkpoint_path,
        agent_id="agent-1",
        run_id="run-1",
        watchdog_path=watchdog_path,
        progress_interval_seconds=0,
        stall_timeout_seconds=3,
        stall_max_restarts=1,
        stall_fallback_mode="edges-only",
        env=None,
        spawn=fake_spawn,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    assert code == 0
    assert len(spawned_cmds) == 3
    assert "--mode" in spawned_cmds[-1]
    assert spawned_cmds[-1][spawned_cmds[-1].index("--mode") + 1] == "edges-only"

    events = [json.loads(line) for line in watchdog_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert any(item.get("event") == "stall_detected" for item in events)
    assert any(item.get("event") == "restart" for item in events)
    assert any(item.get("event") == "fallback" for item in events)


def test_cli_build_all_replay_filters_parse() -> None:
    """build-all parser accepts replay filter forwarding flags."""
    from openclawbrain.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args(
        [
            "build-all",
            "--replay-since-hours",
            "4.2",
            "--replay-max-interactions",
            "55",
            "--replay-sample-rate",
            "0.8",
            "--replay-priority",
            "tool",
            "--advance-offsets-on-skip",
        ]
    )
    assert args.replay_since_hours == 4.2
    assert args.replay_max_interactions == 55
    assert args.replay_sample_rate == 0.8
    assert args.replay_priority == "tool"
    assert args.advance_offsets_on_skip is True


def _expected_replay_sample_ratio(source: str, line_no: int) -> float:
    """Compute expected deterministic replay sample ratio."""
    key = f"{source}:{line_no}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return (int(digest[:16], 16) % 1_000_000_007) / 1_000_000_007


def test_filter_replay_interactions_applies_priority_then_since_then_sample_then_max() -> None:
    """filtering helper preserves order and applies filters in priority since sample max."""
    interactions: list[dict[str, object]] = [
        {"source": "s1.jsonl", "line_no": 1, "ts": 1.0, "tool_calls": [], "tool_results": []},
        {"source": "s1.jsonl", "line_no": 2, "ts": 1001.0, "tool_calls": [{"name": "x"}], "tool_results": []},
        {"source": "s1.jsonl", "line_no": 3, "ts": None, "tool_calls": [], "tool_results": [{"name": "ocr"}]},
        {"source": "s1.jsonl", "line_no": 4, "ts": 900.0, "tool_calls": [], "tool_results": []},
    ]
    _, summary = _filter_replay_interactions(
        interactions,
        now_ts=1000.0,
        since_hours=0.001,
        sample_rate=1.0,
        max_interactions=2,
        priority="tool",
    )
    assert summary["loaded_total"] == 4
    assert summary["after_priority"] == 2
    assert summary["after_since"] == 2
    assert summary["after_sample"] == 2
    assert summary["after_max"] == 2


def test_filter_replay_interactions_with_sample_and_max_preserves_order() -> None:
    """sample and max filters keep most recent filtered matches in original order."""
    interactions: list[dict[str, object]] = [
        {"source": "s1.jsonl", "line_no": 1, "ts": 1.0, "tool_calls": [{"name": "a"}], "tool_results": []},
        {"source": "s2.jsonl", "line_no": 2, "ts": 2.0, "tool_calls": [{"name": "a"}], "tool_results": []},
        {"source": "s3.jsonl", "line_no": 3, "ts": 3.0, "tool_calls": [{"name": "a"}], "tool_results": []},
        {"source": "s4.jsonl", "line_no": 4, "ts": 4.0, "tool_calls": [{"name": "a"}], "tool_results": []},
    ]
    sample_rate = 0.5
    expected_sampled = [
        item
        for item in interactions
        if _expected_replay_sample_ratio(item["source"], item["line_no"]) < sample_rate  # type: ignore[index]
    ]
    expected_max = expected_sampled[-2:] if len(expected_sampled) > 2 else expected_sampled
    filtered, summary = _filter_replay_interactions(
        interactions,
        now_ts=10.0,
        since_hours=None,
        sample_rate=sample_rate,
        max_interactions=2,
        priority="all",
    )
    assert filtered == expected_max
    assert summary["loaded_total"] == 4
    assert summary["after_priority"] == 4
    assert summary["after_since"] == 4
    assert summary["after_sample"] == len(expected_sampled)
    assert summary["after_max"] == len(expected_max)


def test_build_all_agent_discovery_falls_back_to_main(tmp_path, monkeypatch) -> None:
    """Missing openclaw.json falls back to main agent id."""
    import openclawbrain.cli as cli_module

    monkeypatch.setattr(cli_module.Path, "home", lambda: tmp_path)
    assert cli_module._discover_agent_ids() == ["main"]



def test_cli_replay_writes_checkpoint_and_resume_uses_it(tmp_path, capsys) -> None:
    """replay writes checkpoint and resume only replays new lines."""
    state_path = tmp_path / "state.json"
    _write_state(state_path)
    checkpoint_path = tmp_path / "replay_checkpoint.json"
    sessions = tmp_path / "sessions.jsonl"
    sessions.write_text(
        "\n".join(
            [
                json.dumps({"role": "user", "content": "alpha", "ts": 1.0}),
                json.dumps({"role": "assistant", "content": "ok", "ts": 1.1}),
            ]
        ),
        encoding="utf-8",
    )

    code = main(
        [
            "replay",
            "--state",
            str(state_path),
            "--sessions",
            str(sessions),
            "--edges-only",
            "--checkpoint",
            str(checkpoint_path),
            "--checkpoint-every",
            "1",
            "--resume",
            "--json",
        ]
    )
    assert code == 0
    first_payload = json.loads(capsys.readouterr().out.strip())
    assert first_payload["queries_replayed"] == 1
    assert checkpoint_path.exists()
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    replay_checkpoint = checkpoint.get("replay", {})
    assert replay_checkpoint.get("sessions")

    with sessions.open("a", encoding="utf-8") as handle:
        handle.write("\n" + json.dumps({"role": "user", "content": "beta", "ts": 2.0}))
        handle.write("\n" + json.dumps({"role": "assistant", "content": "ok", "ts": 2.1}))

    code = main(
        [
            "replay",
            "--state",
            str(state_path),
            "--sessions",
            str(sessions),
            "--edges-only",
            "--checkpoint",
            str(checkpoint_path),
            "--resume",
            "--json",
        ]
    )
    assert code == 0
    second_payload = json.loads(capsys.readouterr().out.strip())
    assert second_payload["queries_replayed"] == 1
    assert second_payload["last_replayed_ts"] == 2.1


def test_cli_replay_resume_legacy_checkpoint_warns_and_uses_offsets(tmp_path, capsys) -> None:
    """replay resume falls back to legacy top-level sessions offsets."""
    state_path = tmp_path / "state.json"
    _write_state(state_path)
    sessions = tmp_path / "sessions.jsonl"
    sessions.write_text(
        "\n".join(
            [
                json.dumps({"role": "user", "content": "alpha", "ts": 1.0}),
                json.dumps({"role": "assistant", "content": "ok", "ts": 1.1}),
                json.dumps({"role": "user", "content": "beta", "ts": 2.0}),
                json.dumps({"role": "assistant", "content": "ok", "ts": 2.1}),
            ]
        ),
        encoding="utf-8",
    )
    fixture = Path(__file__).parent / "fixtures" / "checkpoints" / "legacy_schema.json"
    checkpoint_path = tmp_path / "replay_checkpoint.json"
    checkpoint = json.loads(fixture.read_text(encoding="utf-8"))
    checkpoint["sessions"] = {str(sessions): 2}
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")

    with pytest.warns(UserWarning, match="replay checkpoint missing phase-scoped sessions"):
        code = main(
            [
                "replay",
                "--state",
                str(state_path),
                "--sessions",
                str(sessions),
                "--edges-only",
                "--checkpoint",
                str(checkpoint_path),
                "--resume",
                "--json",
            ]
        )
    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["queries_replayed"] == 1
    assert payload["last_replayed_ts"] == 2.1


def test_cli_replay_stop_after_fast_learning(tmp_path, capsys, monkeypatch) -> None:
    """stop-after-fast-learning exits before replay and harvest."""
    state_path = tmp_path / "state.json"
    _write_state(state_path)
    sessions = tmp_path / "sessions.jsonl"
    sessions.write_text(
        "\n".join(
            [
                json.dumps({"role": "user", "content": "do not do that", "ts": 1.0}),
                json.dumps({"role": "assistant", "content": "ok", "ts": 1.1}),
            ]
        ),
        encoding="utf-8",
    )

    from openclawbrain import full_learning as learning_module

    def fake_llm(_: str, __: str) -> str:
        return json.dumps(
            {
                "corrections": [{"content": "Never do that.", "context": "ctx", "severity": "high"}],
                "teachings": [],
                "reinforcements": [],
            }
        )

    monkeypatch.setattr(learning_module, "openai_llm_fn", fake_llm)

    code = main(
        [
            "replay",
            "--state",
            str(state_path),
            "--sessions",
            str(sessions),
            "--fast-learning",
            "--llm",
            "none",
            "--stop-after-fast-learning",
            "--ignore-checkpoint",
            "--json",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["stopped_after_fast_learning"] is True
    assert "fast_learning" in payload
    assert "harvest" not in payload
    assert "queries_replayed" not in payload


def test_cli_replay_fast_learning_resume_legacy_checkpoint_warns(tmp_path, capsys, monkeypatch) -> None:
    """fast-learning resume falls back to legacy top-level sessions offsets."""
    state_path = tmp_path / "state.json"
    _write_state(state_path)
    sessions = tmp_path / "sessions.jsonl"
    sessions.write_text(
        "\n".join(
            [
                json.dumps({"role": "user", "content": "first", "ts": 1.0}),
                json.dumps({"role": "assistant", "content": "ok", "ts": 1.1}),
                json.dumps({"role": "user", "content": "second", "ts": 2.0}),
                json.dumps({"role": "assistant", "content": "ok", "ts": 2.1}),
            ]
        ),
        encoding="utf-8",
    )

    from openclawbrain import full_learning as learning_module

    def fake_llm(_: str, __: str) -> str:
        return json.dumps({"corrections": [], "teachings": [], "reinforcements": []})

    monkeypatch.setattr(learning_module, "openai_llm_fn", fake_llm)

    fixture = Path(__file__).parent / "fixtures" / "checkpoints" / "legacy_schema.json"
    checkpoint_path = tmp_path / "replay_checkpoint.json"
    checkpoint = json.loads(fixture.read_text(encoding="utf-8"))
    checkpoint["sessions"] = {str(sessions): 2}
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")

    with pytest.warns(UserWarning, match="fast_learning checkpoint missing phase-scoped sessions"):
        code = main(
            [
                "replay",
                "--state",
                str(state_path),
                "--sessions",
                str(sessions),
                "--fast-learning",
                "--llm",
                "none",
                "--stop-after-fast-learning",
                "--checkpoint",
                str(checkpoint_path),
                "--resume",
                "--json",
            ]
        )
    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["stopped_after_fast_learning"] is True


def test_cli_replay_parallel_mode_v0(tmp_path, capsys) -> None:
    """replay supports simple parallel reducer mode."""
    state_path = tmp_path / "state.json"
    _write_state(state_path)
    sessions = tmp_path / "sessions.jsonl"
    sessions.write_text(
        "\n".join(
            [
                json.dumps({"role": "user", "content": "alpha", "ts": 1.0}),
                json.dumps({"role": "assistant", "content": "ok", "ts": 1.1}),
                json.dumps({"role": "user", "content": "beta", "ts": 2.0}),
                json.dumps({"role": "assistant", "content": "ok", "ts": 2.1}),
                json.dumps({"role": "user", "content": "alpha beta", "ts": 3.0}),
                json.dumps({"role": "assistant", "content": "ok", "ts": 3.1}),
            ]
        ),
        encoding="utf-8",
    )

    code = main(
        [
            "replay",
            "--state",
            str(state_path),
            "--sessions",
            str(sessions),
            "--edges-only",
            "--replay-workers",
            "2",
            "--checkpoint-every",
            "1",
            "--json",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["queries_replayed"] == 3
    assert payload["replay_workers"] == 2
    assert payload["merge_batches"] >= 1


def test_cli_init_auto_embedder_prefers_local(tmp_path, monkeypatch) -> None:
    """init with auto embedder uses local embedder metadata when available."""

    class FakeLocalEmbedder:
        name = "local:bge-large-en-v1.5"
        dim = 3

        def __init__(self, model_name: str | None = None) -> None:
            self.model_name = model_name or "BAAI/bge-large-en-v1.5"

        def embed(self, _text: str) -> list[float]:
            return [1.0, 0.0, 0.0]

        def embed_batch(self, texts: list[tuple[str, str]]) -> dict[str, list[float]]:
            return {node_id: [1.0, 0.0, 0.0] for node_id, _content in texts}

    import openclawbrain.cli as cli_module
    monkeypatch.setattr(cli_module, "LocalEmbedder", FakeLocalEmbedder)

    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "a.md").write_text("## A\nHello", encoding="utf-8")
    output = tmp_path / "out"
    output.mkdir()

    code = main(["init", "--workspace", str(workspace), "--output", str(output)])
    assert code == 0

    state_data = json.loads((output / "state.json").read_text(encoding="utf-8"))
    assert state_data["meta"]["embedder_name"] == "local:bge-large-en-v1.5"
    assert state_data["meta"]["embedder_dim"] == 3
    assert state_data["meta"]["embedder_model"] == "BAAI/bge-large-en-v1.5"
    assert (output / "route_model.npz").exists()


def test_reembed_rewrites_embedder_meta_and_index_dims(tmp_path, monkeypatch) -> None:
    """reembed should recompute vectors and update embedder metadata."""
    class FakeLocalEmbedder:
        def __init__(self, model_name: str | None = None) -> None:
            self.model_name = model_name or "BAAI/bge-large-en-v1.5"
            self.name = f"local:{self.model_name.rsplit('/', 1)[-1]}"
            self.dim = 3

        def embed(self, text: str) -> list[float]:
            return [1.0, 0.0, 0.0] if "alpha" in text.lower() else [0.0, 1.0, 0.0]

        def embed_batch(self, texts: list[tuple[str, str]]) -> dict[str, list[float]]:
            return {node_id: self.embed(content) for node_id, content in texts}

    import openclawbrain.cli as cli_module
    monkeypatch.setattr(cli_module, "LocalEmbedder", FakeLocalEmbedder)

    graph = Graph()
    graph.add_node(Node(id="a", content="alpha", summary="", metadata={}))
    graph.add_node(Node(id="b", content="beta", summary="", metadata={}))
    index = VectorIndex()
    index.upsert("a", [0.0, 1.0])
    index.upsert("b", [1.0, 0.0])
    state_path = tmp_path / "state.json"
    save_state(
        graph=graph,
        index=index,
        path=str(state_path),
        meta={"embedder_name": "openai-text-embedding-3-small", "embedder_dim": 2},
    )

    code = main(
        [
            "reembed",
            "--state",
            str(state_path),
            "--embedder",
            "local",
            "--embed-model",
            "BAAI/bge-large-en-v1.5",
            "--json",
        ]
    )
    assert code == 0

    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["meta"]["embedder_name"] == "local:bge-large-en-v1.5"
    assert payload["meta"]["embedder_dim"] == 3
    assert payload["meta"]["embedder_model"] == "BAAI/bge-large-en-v1.5"
    index_payload = payload["index"]
    assert len(index_payload["a"]) == 3
    assert len(index_payload["b"]) == 3
