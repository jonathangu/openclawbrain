"""Feedback helpers for delayed learning attribution."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

DEFAULT_SNAPSHOT_PATH = "crabpath_events.db"


def snapshot_path(graph_path: str | None = None) -> Path:
    """Resolve snapshot location.

    Args:
        graph_path: Optional graph path to scope the events file. If omitted,
            use CRABPATH_SNAPSHOT_PATH if set, else package default.
    """
    env_path = os.getenv("CRABPATH_SNAPSHOT_PATH")
    if env_path:
        return Path(env_path)
    if graph_path:
        return Path(graph_path).with_suffix(".events.db")
    return Path(DEFAULT_SNAPSHOT_PATH)


def _load_raw_snapshots(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _save_raw_snapshots(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def _coerce_turn_id(record: dict[str, Any]) -> Optional[int]:
    turn_id = record.get("turn_id")
    try:
        return int(turn_id)
    except (TypeError, ValueError):
        return None


def map_correction_to_snapshot(
    session_id: str,
    turn_window: int = 5,
) -> Optional[dict[str, Any]]:
    """Find the latest attributable snapshot for a session.

    The window is interpreted as max turn distance from the most recent
    non-attributed snapshot.
    """
    path = snapshot_path()
    records = _load_raw_snapshots(path)
    candidates = [r for r in records if r.get("session_id") == session_id]
    if not candidates:
        return None

    parsed = []
    for record in candidates:
        turn = _coerce_turn_id(record)
        if turn is None:
            continue
        parsed.append((turn, record))
    if not parsed:
        return None

    parsed.sort(key=lambda item: item[0], reverse=True)
    latest_turn = parsed[0][0]
    env_turn = os.getenv("CRABPATH_CORRECTION_TURN")
    if env_turn is None:
        correction_turn = latest_turn
    else:
        try:
            correction_turn = int(env_turn)
        except ValueError:
            correction_turn = latest_turn

    for turn, record in parsed:
        if correction_turn - turn <= turn_window:
            record = dict(record)
            record["turns_since_fire"] = max(0, correction_turn - turn)
            return record

    return None


def auto_outcome(corrections_count: int, turns_since_fire: int) -> float:
    """Generate a coarse outcome score from delayed feedback signals.

    Args:
        corrections_count: Number of correction hits attached to this firing context.
        turns_since_fire: Distance from the firing turn to the feedback event.
    """
    if corrections_count > 0:
        return -1.0
    if turns_since_fire >= 5:
        return 0.3
    return 0.0
