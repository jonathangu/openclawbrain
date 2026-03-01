from __future__ import annotations

import json
from pathlib import Path

from openclawbrain.full_learning import _load_checkpoint, _checkpoint_phase_offsets


FIXTURES = Path(__file__).parent / "fixtures" / "checkpoints"


def _write_checkpoint_from_fixture(tmp_path: Path, fixture_name: str, session_path: Path) -> Path:
    raw = json.loads((FIXTURES / fixture_name).read_text(encoding="utf-8"))

    def _replace_session_token(value: object) -> object:
        if isinstance(value, dict):
            out: dict[str, object] = {}
            for key, item in value.items():
                next_key = str(session_path) if key == "__SESSION__" else key
                out[next_key] = _replace_session_token(item)
            return out
        if isinstance(value, list):
            return [_replace_session_token(item) for item in value]
        return value

    checkpoint_path = tmp_path / fixture_name
    checkpoint_path.write_text(
        json.dumps(_replace_session_token(raw), indent=2),
        encoding="utf-8",
    )
    return checkpoint_path


def test_replay_resume_prefers_phase_scoped_sessions(tmp_path: Path, capsys) -> None:
    """Replay resume uses replay.sessions when available."""
    session_path = tmp_path / "sessions.jsonl"
    checkpoint_path = _write_checkpoint_from_fixture(
        tmp_path,
        "new_phase_scoped_sessions.json",
        session_path,
    )
    checkpoint = _load_checkpoint(checkpoint_path)

    offsets, used_legacy = _checkpoint_phase_offsets(checkpoint, phase="replay")

    assert offsets == {str(session_path): 2}
    assert "legacy checkpoint 'sessions'" not in capsys.readouterr().err


def test_replay_resume_falls_back_to_legacy_sessions(tmp_path: Path, capsys) -> None:
    """Replay resume falls back to top-level sessions when replay.sessions is missing."""
    session_path = tmp_path / "sessions.jsonl"
    checkpoint_path = _write_checkpoint_from_fixture(
        tmp_path,
        "mixed_partial_sessions.json",
        session_path,
    )
    checkpoint = _load_checkpoint(checkpoint_path)

    offsets, used_legacy = _checkpoint_phase_offsets(checkpoint, phase="replay")

    assert offsets == {str(session_path): 2}
    assert used_legacy is True


def test_fast_learning_resume_prefers_phase_scoped_sessions(tmp_path: Path, capsys) -> None:
    """Fast-learning resume uses fast_learning.sessions when available."""
    session_path = tmp_path / "sessions.jsonl"
    checkpoint_path = _write_checkpoint_from_fixture(
        tmp_path,
        "new_phase_scoped_sessions.json",
        session_path,
    )
    checkpoint = _load_checkpoint(checkpoint_path)

    offsets, used_legacy = _checkpoint_phase_offsets(checkpoint, phase="fast_learning")

    assert offsets == {str(session_path): 3}
    assert "legacy checkpoint 'sessions'" not in capsys.readouterr().err


def test_fast_learning_resume_falls_back_to_legacy_sessions_with_warning(tmp_path: Path, capsys) -> None:
    """Fast-learning resume falls back to top-level sessions and emits a warning."""
    session_path = tmp_path / "sessions.jsonl"
    checkpoint_path = _write_checkpoint_from_fixture(
        tmp_path,
        "legacy_top_level_sessions.json",
        session_path,
    )
    checkpoint = _load_checkpoint(checkpoint_path)

    offsets, used_legacy = _checkpoint_phase_offsets(checkpoint, phase="fast_learning")

    assert offsets == {str(session_path): 2}
    assert used_legacy is True
