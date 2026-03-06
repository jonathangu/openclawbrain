from __future__ import annotations

import json
import os
import sqlite3
import sys
from collections.abc import Iterable
from pathlib import Path


SESSION_FILE_PATTERNS = ("*.jsonl", "*.jsonl.reset.*", "*.jsonl.deleted.*")
CODEX_ROLLOUT_GLOB = "**/rollout-*.jsonl"


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    ordered: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        resolved = str(path.expanduser())
        if resolved in seen:
            continue
        seen.add(resolved)
        ordered.append(path)
    return ordered


def _default_codex_home() -> Path:
    raw = os.environ.get("CODEX_HOME")
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".codex"


def _resolve_codex_rollouts_from_sqlite(db_path: Path) -> list[Path]:
    if not db_path.exists() or not db_path.is_file():
        return []

    query = "SELECT rollout_path FROM threads WHERE rollout_path IS NOT NULL AND rollout_path != '' ORDER BY updated_at, id"
    try:
        with sqlite3.connect(str(db_path)) as conn:
            rows = conn.execute(query).fetchall()
    except sqlite3.Error:
        return []

    rollouts: list[Path] = []
    for (raw_path,) in rows:
        if not isinstance(raw_path, str) or not raw_path.strip():
            continue
        path = Path(raw_path).expanduser()
        if path.exists() and path.is_file():
            rollouts.append(path)
    return _dedupe_paths(rollouts)


def _resolve_codex_rollouts(codex_home: Path | None = None) -> list[Path]:
    home = (codex_home or _default_codex_home()).expanduser()
    if not home.exists() or not home.is_dir():
        return []

    rollouts: list[Path] = []
    for db_path in sorted(home.glob("state_*.sqlite")):
        rollouts.extend(_resolve_codex_rollouts_from_sqlite(db_path))

    if rollouts:
        return _dedupe_paths(rollouts)

    sessions_root = home / "sessions"
    if not sessions_root.exists() or not sessions_root.is_dir():
        return []
    return _dedupe_paths(path for path in sorted(sessions_root.glob(CODEX_ROLLOUT_GLOB)) if path.is_file())


def _sessions_index_requires_codex_rollouts(path: Path, payload: dict[str, object]) -> bool:
    if path.parent.name == "sessions" and path.parent.parent.name == "codex":
        return True

    for entry in payload.values():
        if not isinstance(entry, dict):
            continue
        acp = entry.get("acp")
        if not isinstance(acp, dict):
            continue
        backend = str(acp.get("backend", "")).strip().lower()
        agent = str(acp.get("agent", "")).strip().lower()
        if backend in {"acpx", "acp"} or agent == "codex":
            return True
    return False


def _resolve_sessions_index(path: Path) -> list[Path]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, dict):
        return []

    resolved: list[Path] = []
    for entry in payload.values():
        if not isinstance(entry, dict):
            continue
        for key in ("sessionFile", "session_file", "rollout_path"):
            raw = entry.get(key)
            if not isinstance(raw, str) or not raw.strip():
                continue
            candidate = Path(raw).expanduser()
            if candidate.exists() and candidate.is_file():
                resolved.append(candidate)
                break

    if _sessions_index_requires_codex_rollouts(path, payload):
        resolved.extend(_resolve_codex_rollouts())

    return _dedupe_paths(resolved)


def _collect_from_directory(path: Path) -> list[Path]:
    resolved: list[Path] = []
    for pattern in SESSION_FILE_PATTERNS:
        resolved.extend(match for match in sorted(path.glob(pattern)) if match.is_file())

    sessions_index = path / "sessions.json"
    if sessions_index.exists() and sessions_index.is_file():
        resolved.extend(_resolve_sessions_index(sessions_index))

    nested_rollouts = [match for match in sorted(path.glob(CODEX_ROLLOUT_GLOB)) if match.is_file()]
    resolved.extend(nested_rollouts)
    return _dedupe_paths(resolved)


def collect_session_files(session_paths: str | Path | Iterable[str | Path]) -> list[Path]:
    """Resolve session paths into replayable files.

    Supports plain OpenClaw session JSONL files, rotated JSONL sidecars,
    `sessions.json` indices, Codex rollout directories, and direct Codex state
    sqlite databases whose `threads.rollout_path` columns reference session logs.
    Missing paths are skipped with a warning; if no files remain, the caller gets
    a helpful `SystemExit`.
    """

    if isinstance(session_paths, (str, Path)):
        session_paths = [session_paths]

    collected: list[Path] = []
    skipped: list[Path] = []
    for item in session_paths:
        src = Path(item).expanduser()
        if src.is_dir():
            collected.extend(_collect_from_directory(src))
            continue
        if src.is_file():
            if src.name == "sessions.json":
                collected.extend(_resolve_sessions_index(src))
                continue
            if src.suffix == ".sqlite":
                rollouts = _resolve_codex_rollouts_from_sqlite(src)
                if rollouts:
                    collected.extend(rollouts)
                    continue
            collected.append(src)
            continue
        print(f"warning: skipping missing session path: {src}", file=sys.stderr)
        skipped.append(src)

    deduped = _dedupe_paths(collected)
    if not deduped and skipped:
        raise SystemExit(f"no valid session files found ({len(skipped)} missing/broken path(s) skipped)")
    if not deduped and not skipped:
        raise SystemExit("no valid session files found")
    return deduped
