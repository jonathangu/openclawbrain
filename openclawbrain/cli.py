"""Pure graph-operations CLI for OpenClawBrain."""

from __future__ import annotations

import argparse
import functools
import socket
import json
import os
import hashlib
import time
import warnings
import threading
import signal
from datetime import datetime, timezone, timedelta
import sys
import tempfile
import shutil
import subprocess
import shlex
import plistlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Callable, Iterable
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

from .connect import apply_connections, suggest_connections
from .autotune import measure_health
from .graph import Edge, Graph, Node
from .index import VectorIndex
from .inject import inject_correction, inject_node
from .compact import compact_daily_notes
from .journal import (
    log_event,
    log_health,
    log_learn,
    log_query,
    log_replay,
    journal_stats,
    read_journal,
)
from .learn import apply_outcome_pg
from .merge import apply_merge, suggest_merges
from .replay import (
    DEFAULT_TOOL_RESULT_ALLOWLIST,
    DEFAULT_TOOL_RESULT_MAX_CHARS,
    extract_interactions,
    extract_query_records,
    extract_query_records_from_dir,
    extract_queries,
    extract_queries_from_dir,
    replay_queries,
    replay_queries_parallel,
)
from .split import split_workspace, _resolve_workspace_id, _sibling_weight
from .hasher import HashEmbedder
from .traverse import TraversalConfig, TraversalResult, traverse
from .sync import DEFAULT_AUTHORITY_MAP, SyncReport, sync_workspace
from .local_embedder import LocalEmbedder, resolve_local_model
from .route_model import RouteModel
from .full_learning import (
    REPLAY_CHECKPOINT_FILENAME,
    _checkpoint_phase_offsets,
    collect_session_files,
    default_checkpoint_path,
    load_interactions_for_replay,
    run_fast_learning,
    run_harvest,
    event_log_entries,
    _load_checkpoint,
    _persist_state,
    _save_checkpoint,
)
from ._util import _tokenize
from .maintain import run_maintenance
from .reward import RewardSource, RewardWeights
from .labels import (
    LabelRecord,
    append_labels_jsonl,
    from_self_learning_event,
    from_teacher_output,
    read_labels_jsonl,
    write_labels_jsonl,
)
from .trace import RouteTrace, route_trace_to_json
from .state_lock import StateLockError, lock_path_for_state, state_write_lock
from .store import load_state, save_state, resolve_default_state_path
from ._batch import batch_or_single_embed
from .ops.async_route_pg import run_async_route_pg
from .profile import BrainProfile, BrainProfileError
from .train_route_model import train_route_model, write_summary_json
from . import __version__

DEFAULT_STATE_PROFILE = "main"
REPLAY_MODES = ("edges-only", "fast-learning", "full")
LOOP_LOG_FILENAME = "loop.log"
LOOP_EVENTS_FILENAME = "loop.events.jsonl"
LOOP_LOCK_FILENAME = "loop.lock"
LOOP_CHECKPOINT_FILENAME = "loop.checkpoint.json"
LOOP_MANIFEST_FILENAME = "loop.manifest.json"
LOOP_STDOUT_FILENAME = "loop.stdout.log"
LOOP_STDERR_FILENAME = "loop.stderr.log"
INIT_ROUTE_SINCE_HOURS = 100000
INIT_ROUTE_MAX_QUERIES = 100000
INIT_ROUTE_SAMPLE_RATE = 1.0
INIT_ROUTE_TRACES_FILENAME = "init.route_traces.jsonl"
INIT_CHECKPOINT_DIRNAME = "init_checkpoints"
INIT_SPLIT_MANIFEST_FILENAME = "split.jsonl"
INIT_MANIFEST_META_FILENAME = "manifest.json"
INIT_VECTORS_FILENAME = "vectors.jsonl"
INIT_PROGRESS_FILENAME = "progress.json"
INIT_COMPLETE_FILENAME = "complete.json"
REPLAY_HELP_EPILOG = (
    "Replay modes and rough cost profile:\n"
    "  edges-only    Fastest/cheapest. No LLM calls, no harvest.\n"
    "  fast-learning LLM-bound transcript mining + injection only. Usually the slowest and highest-cost phase.\n"
    "  full          Fast-learning + edge replay + harvest. Highest end-to-end time and API spend. Default.\n\n"
    "Checkpoint semantics:\n"
    "  --resume uses saved per-session offsets.\n"
    "  --fresh / --no-checkpoint starts from scratch even if a checkpoint exists.\n"
    "  --checkpoint chooses the checkpoint file path."
)

_DEFAULT_DAEMON_HEALTH_TIMEOUT = 30.0
FAST_BOOT_REPLAY_MAX_INTERACTIONS = 500
FAST_BOOT_REPLAY_PRIORITY = "tool"
FAST_BOOT_TOOL_RESULT_MAX_CHARS = 20_000
FAST_BOOT_ASYNC_SINCE_HOURS = 24.0
FAST_BOOT_ASYNC_MAX_QUERIES = 60
FAST_BOOT_ASYNC_SAMPLE_RATE = 0.1
FAST_BOOT_ASYNC_MAX_CANDIDATES = 8
FAST_BOOT_ASYNC_MAX_DECISION_POINTS = 200
FAST_BOOT_DREAM_SINCE_HOURS = 24.0
FAST_BOOT_DREAM_MAX_QUERIES = 40
FAST_BOOT_DREAM_SAMPLE_RATE = 0.1
FAST_BOOT_DREAM_MAX_CANDIDATES = 8
FAST_BOOT_DREAM_MAX_DECISION_POINTS = 160
FAST_BOOT_REPLAY_STALL_TIMEOUT_SECONDS = 900
FAST_BOOT_REPLAY_STALL_MAX_RESTARTS = 1
FAST_BOOT_REPLAY_STALL_FALLBACK_MODE = "edges-only"
FAST_BOOT_DREAM_STALL_TIMEOUT_SECONDS = 900
FAST_BOOT_DREAM_STALL_MAX_RESTARTS = 1


def _daemon_health_timeout() -> float:
    """Resolve daemon health timeout with env override."""
    raw = os.getenv("OCB_DAEMON_HEALTH_TIMEOUT")
    if raw is None:
        return _DEFAULT_DAEMON_HEALTH_TIMEOUT
    try:
        return float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_DAEMON_HEALTH_TIMEOUT


def _write_json_atomic(path: Path, payload: object) -> None:
    """Write JSON payload via a temporary file and atomic rename."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2))
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError:
            pass
    temp.replace(target)


def _subprocess_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    if extra:
        env.update(extra)
    return env


def _read_json_optional(path: Path) -> dict[str, object] | None:
    """Read a JSON file if it exists, otherwise return None."""
    target = Path(path)
    if not target.exists():
        return None
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _sync_report_path(state_path: str) -> Path:
    """Resolve sync report path for a given state file."""
    return Path(state_path).expanduser().parent / "sync.report.json"


def _maintain_report_path(state_path: str) -> Path:
    """Resolve maintenance report path for a given state file."""
    return Path(state_path).expanduser().parent / "maintain.report.json"


def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
    """Append one JSON object as one line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")


def _append_jsonl_lines_fsync(path: Path, payloads: list[dict[str, object]]) -> None:
    """Append multiple JSONL lines and fsync."""
    if not payloads:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload, default=str) + "\n")
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError:
            pass


def _init_checkpoint_paths(output_dir: Path) -> dict[str, Path]:
    """Resolve init checkpoint paths."""
    root = output_dir / "scratch" / INIT_CHECKPOINT_DIRNAME
    return {
        "root": root,
        "manifest": root / INIT_SPLIT_MANIFEST_FILENAME,
        "manifest_meta": root / INIT_MANIFEST_META_FILENAME,
        "vectors": root / INIT_VECTORS_FILENAME,
        "progress": root / INIT_PROGRESS_FILENAME,
        "complete": root / INIT_COMPLETE_FILENAME,
    }


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    """Read a JSONL file, skipping malformed lines."""
    if not path.exists():
        return []
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
    return records


def _write_init_manifest(
    manifest_path: Path,
    meta_path: Path,
    entries: list[dict[str, object]],
    meta: dict[str, object],
) -> None:
    """Write init split manifest and metadata."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, default=str) + "\n")
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError:
            pass
    _write_json_atomic(meta_path, meta)


def _build_graph_from_manifest(
    entries: list[dict[str, object]],
    workspace_id: str | None,
) -> tuple[Graph, dict[str, str]]:
    """Rebuild graph/texts from an init split manifest."""
    graph = Graph()
    texts: dict[str, str] = {}
    file_chunks: dict[str, list[tuple[int, str]]] = {}
    prefix = f"{workspace_id}/" if workspace_id else ""
    for entry in entries:
        node_id = entry.get("id")
        text = entry.get("text")
        metadata = entry.get("metadata")
        summary = entry.get("summary")
        if not isinstance(node_id, str) or not isinstance(text, str):
            continue
        if not isinstance(metadata, dict):
            metadata = {}
        if not isinstance(summary, str):
            summary = text.splitlines()[0] if text.splitlines() else ""
        node = Node(id=node_id, content=text, summary=summary, metadata=dict(metadata))
        graph.add_node(node)
        texts[node_id] = text
        file_name = metadata.get("file")
        chunk_idx = metadata.get("chunk")
        if isinstance(file_name, str) and isinstance(chunk_idx, int):
            file_chunks.setdefault(file_name, []).append((chunk_idx, node_id))

    for file_name, chunks in file_chunks.items():
        if not chunks:
            continue
        chunks.sort(key=lambda item: item[0])
        rel = file_name[len(prefix) :] if prefix and file_name.startswith(prefix) else file_name
        for source_offset, (_, source_id) in enumerate(chunks[:-1]):
            target_id = chunks[source_offset + 1][1]
            weight = _sibling_weight(rel, source_offset)
            graph.add_edge(Edge(source=source_id, target=target_id, weight=weight, kind="sibling"))
            graph.add_edge(Edge(source=target_id, target=source_id, weight=weight, kind="sibling"))
    return graph, texts


def _iso_now() -> str:
    """UTC timestamp string."""
    return datetime.now(timezone.utc).isoformat()


def _build_all_root_manifest_payload(
    *,
    run_id: str,
    run_ts: datetime,
    args: argparse.Namespace,
    ocb_prefix: list[str],
    agent_ids: list[str],
    parallel_agents: int,
    events_jsonl: Path,
    stall_audit_jsonl: Path | None,
    status: str = "running",
    agents: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Build the build-all root manifest content."""
    succeeded = len([item for item in (agents or []) if int(item.get("exit_code", 1)) == 0])
    failed = len([item for item in (agents or []) if int(item.get("exit_code", 1)) != 0])
    completed = len(agents or [])
    return {
        "run_id": run_id,
        "status": status,
        "timestamp": run_ts.isoformat(),
        "openclawbrain_bin": " ".join(shlex.quote(part) for part in ocb_prefix),
        "command": "build-all",
        "args": {
            "agents": getattr(args, "agents", None),
            "parallel_agents": parallel_agents,
            "reembed": bool(args.reembed),
            "require_local_embedder": bool(args.require_local_embedder),
            "embed_model": str(args.embed_model),
            "mode": str(args.mode),
            "llm": str(args.llm),
            "workers": int(args.workers) if args.workers is not None else None,
            "llm_model": getattr(args, "llm_model", None),
            "resume": bool(args.resume),
            "include_tool_results": bool(args.include_tool_results),
            "checkpoint_every_seconds": int(args.checkpoint_every_seconds),
            "replay_progress_interval_seconds": int(args.replay_progress_interval_seconds),
            "replay_since_hours": args.replay_since_hours,
            "replay_max_interactions": args.replay_max_interactions,
            "replay_sample_rate": float(args.replay_sample_rate),
            "replay_priority": str(args.replay_priority),
            "advance_offsets_on_skip": bool(args.advance_offsets_on_skip),
            "tool_result_max_chars": getattr(args, "tool_result_max_chars", None),
            "step_stall_timeout_seconds": int(getattr(args, "step_stall_timeout_seconds", 0)),
            "enable_async_teacher": bool(args.enable_async_teacher),
            "skip_init_route_model": bool(args.skip_init_route_model),
            "events_jsonl": str(events_jsonl),
            "stall_audit_jsonl": str(stall_audit_jsonl) if stall_audit_jsonl else None,
        },
        "agents": agents or [],
        "summary": {
            "total_agents": len(agent_ids),
            "completed": completed,
            "succeeded": succeeded,
            "failed": failed,
        },
    }


def _evaluate_build_all_preflight(
    *,
    state_path: str,
    status_before_path: str,
    reembed: bool,
    require_local_embedder: bool,
) -> tuple[int, str | None]:
    """Evaluate build-all preflight checks from a status snapshot.

    Returns (exit_code, optional_error). exit_code 0 indicates continue.
    """
    try:
        payload = _load_json(status_before_path)
    except SystemExit as exc:
        message = f"failed to read status_before snapshot: {exc}"
        print(f"[build-all] preflight: {message}")
        return 1, message

    embedder_name = payload.get("embedder_name")
    embedder_dim = payload.get("embedder_dim")
    index_dim = payload.get("index_dim")

    if (
        require_local_embedder
        and isinstance(embedder_name, str)
        and embedder_name.startswith("openai")
        and not reembed
    ):
        message = (
            "state is using an OpenAI embedder. "
            "Re-run with --reembed to switch to local embeddings, "
            f"or run: openclawbrain reembed --state {state_path}"
        )
        print(f"[build-all] preflight: {message}")
        return 1, message

    if isinstance(embedder_dim, int) and isinstance(index_dim, int) and embedder_dim != index_dim:
        if reembed:
            print(
                f"[build-all] preflight: embedder_dim mismatch "
                f"(index={index_dim}, meta={embedder_dim}); continuing because --reembed is enabled."
            )
        else:
            message = (
                "embedder_dim differs from index_dim. "
                "Re-run with --reembed or run: "
                f"openclawbrain reembed --state {state_path}"
            )
            print(f"[build-all] preflight: {message}")
            return 1, message

    return 0, None


def _parse_positive_float(value: str) -> float:
    """Parse a strictly positive float CLI argument."""
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {value}")
    return parsed


def _parse_replay_sample_rate(value: str) -> float:
    """Parse replay sample-rate as (0, 1]."""
    parsed = float(value)
    if not (0 < parsed <= 1):
        raise argparse.ArgumentTypeError(f"must be in (0, 1], got {value}")
    return parsed


def _parse_positive_int(value: str) -> int:
    """Parse a strictly positive integer CLI argument."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {value}")
    return parsed


def _filter_replay_interactions(
    interactions: list[dict[str, object]],
    *,
    now_ts: float,
    since_hours: float | None = None,
    max_interactions: int | None = None,
    sample_rate: float = 1.0,
    priority: str = "all",
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Apply replay interaction filters and return summary counts.

    Filters are applied in order: priority -> since -> sample -> max.
    """

    def _has_tool_data(item: dict[str, object]) -> bool:
        tool_calls = item.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            return True
        tool_results = item.get("tool_results")
        if isinstance(tool_results, list) and tool_results:
            return True
        return False

    def _sample_keep(item: dict[str, object]) -> bool:
        source = item.get("source")
        line_no = item.get("line_no")
        if sample_rate >= 1:
            return True
        if not isinstance(source, str) or not isinstance(line_no, (int, float)):
            return True
        key = f"{source}:{int(line_no)}"
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        ratio = (int(digest[:16], 16) % 1_000_000_007) / 1_000_000_007
        return ratio < sample_rate

    loaded_total = len(interactions)

    filtered = list(interactions)
    if priority == "tool":
        filtered = [item for item in filtered if _has_tool_data(item)]
    after_priority = len(filtered)

    after_since = after_priority
    if since_hours is not None:
        cutoff = now_ts - (float(since_hours) * 3600)
        filtered = [
            item
            for item in filtered
            if (ts := item.get("ts")) is None
            or not isinstance(ts, (int, float))
            or float(ts) >= cutoff
        ]
        after_since = len(filtered)

    if sample_rate < 1:
        filtered = [item for item in filtered if _sample_keep(item)]
    after_sample = len(filtered)

    if max_interactions is not None:
        filtered = filtered[-int(max_interactions) :] if max_interactions > 0 else []
    after_max = len(filtered)

    summary = {
        "loaded_total": loaded_total,
        "after_priority": after_priority,
        "after_since": after_since,
        "after_sample": after_sample,
        "after_max": after_max,
    }
    return filtered, summary


def _resolve_state_path(
    explicit_state: str | None,
    *,
    allow_default: bool = False,
    profile: str = DEFAULT_STATE_PROFILE,
) -> str | None:
    """Resolve explicit state path or default profile location."""
    if explicit_state is not None:
        return str(Path(explicit_state).expanduser())
    if allow_default:
        return resolve_default_state_path(profile)
    return None


def _state_path_for_agent(agent_id: str) -> str:
    """Return the default state path for an agent id."""
    return str((Path.home() / ".openclawbrain" / agent_id / "state.json").expanduser())


def _resolve_effective_state_path(
    args: argparse.Namespace,
    *,
    allow_default_state: bool = False,
) -> str | None:
    """Resolve CLI state path, optionally allowing the default profile state."""
    use_default_state = allow_default_state and getattr(args, "graph", None) is None
    return _resolve_state_path(getattr(args, "state", None), allow_default=use_default_state)


def _resolve_replay_mode(args: argparse.Namespace) -> tuple[str, bool]:
    """Resolve replay mode from explicit mode and legacy booleans.

    Returns `(mode, defaulted)` where `defaulted=True` means no mode/legacy flag
    was provided and we selected the explicit default.
    """
    explicit_mode = str(args.mode) if getattr(args, "mode", None) else None
    legacy_selected = [
        mode
        for mode, enabled in (
            ("edges-only", bool(getattr(args, "edges_only", False))),
            ("fast-learning", bool(getattr(args, "fast_learning", False))),
            ("full", bool(getattr(args, "full_learning", False))),
        )
        if enabled
    ]
    if len(legacy_selected) > 1:
        raise SystemExit("conflicting replay flags: choose one of --edges-only, --fast-learning, or --full-learning")
    legacy_mode = legacy_selected[0] if legacy_selected else None
    if explicit_mode is not None and legacy_mode is not None and explicit_mode != legacy_mode:
        raise SystemExit(
            f"conflicting replay flags: --mode {explicit_mode} does not match legacy flag selecting {legacy_mode}"
        )
    if explicit_mode is not None:
        return explicit_mode, False
    if legacy_mode is not None:
        return legacy_mode, False
    return "full", True


def _load_profile(profile_path: str | None) -> BrainProfile | None:
    if profile_path is None:
        return None
    try:
        return BrainProfile.load(profile_path)
    except BrainProfileError as exc:
        raise SystemExit(f"invalid brain profile: {exc}") from None


def _coalesce(cli_value: str | None, profile_value: str | None, default: str) -> str:
    if cli_value is not None:
        return cli_value
    if profile_value is not None:
        return profile_value
    return default


def _resolve_openclawbrain_bin() -> str:
    """Resolve the installed openclawbrain CLI binary."""
    argv0 = Path(sys.argv[0]).expanduser()
    if argv0.is_file() and os.access(argv0, os.X_OK):
        return str(argv0)
    candidate = shutil.which("openclawbrain")
    if candidate:
        return candidate
    return str(argv0)


def _resolve_subprocess_python() -> str:
    override = os.environ.get("OPENCLAWBRAIN_SUBPROCESS_PYTHON") or os.environ.get("OPENCLAWBRAIN_PYTHON")
    if isinstance(override, str) and override.strip():
        return str(Path(override).expanduser())
    return sys.executable


def _resolve_subprocess_prefix() -> list[str]:
    return [_resolve_subprocess_python(), "-m", "openclawbrain.cli"]


def _discover_agent_ids(config_path: Path | None = None) -> list[str]:
    """Discover agent ids from ~/.openclaw/openclaw.json or fallback to main."""
    cfg_path = config_path or (Path.home() / ".openclaw" / "openclaw.json")
    if not cfg_path.exists():
        return ["main"]
    try:
        payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid openclaw config: {cfg_path} ({exc})") from exc

    agents = payload.get("agents", {})
    if not isinstance(agents, dict):
        raise SystemExit(f"invalid openclaw config: {cfg_path} (agents must be an object)")
    agent_list = agents.get("list", [])
    if not isinstance(agent_list, list):
        raise SystemExit(f"invalid openclaw config: {cfg_path} (agents.list must be a list)")

    resolved: list[str] = []
    for entry in agent_list:
        if not isinstance(entry, dict):
            continue
        agent_id = entry.get("id") or entry.get("name")
        if isinstance(agent_id, str):
            cleaned = agent_id.strip()
            if cleaned and cleaned not in resolved:
                resolved.append(cleaned)

    if not resolved:
        raise SystemExit(f"no agents found in {cfg_path}")
    return resolved


def _resolve_agent_workspace(agent_id: str, config_path: Path | None = None) -> Path:
    """Resolve workspace path for an OpenClaw agent id."""
    cfg_path = config_path or (Path.home() / ".openclaw" / "openclaw.json")
    workspace: Path | None = None
    if cfg_path.exists():
        try:
            payload = json.loads(cfg_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"invalid openclaw config: {cfg_path} ({exc})") from exc

        agents = payload.get("agents", {})
        agent_list = agents.get("list", []) if isinstance(agents, dict) else []
        if not isinstance(agent_list, list):
            raise SystemExit(f"invalid openclaw config: {cfg_path} (agents.list must be a list)")
        for entry in agent_list:
            if not isinstance(entry, dict):
                continue
            entry_id = entry.get("id") or entry.get("name")
            if not isinstance(entry_id, str):
                continue
            if entry_id.strip() != agent_id:
                continue
            raw_workspace = entry.get("workspace")
            if isinstance(raw_workspace, str) and raw_workspace.strip():
                workspace = Path(raw_workspace).expanduser()
                break

    if workspace is None:
        fallback = Path.home() / ".openclaw" / "workspace"
        if fallback.exists():
            return fallback
        raise SystemExit(
            f"workspace not found for agent '{agent_id}' in {cfg_path}; "
            "add workspace in openclaw.json or pass --workspace"
        )
    return workspace


def _apply_fast_loop_defaults(args: argparse.Namespace) -> None:
    """Apply fast-boot defaults for the always-learning loop."""
    args.replay_max_interactions = FAST_BOOT_REPLAY_MAX_INTERACTIONS
    args.replay_priority = FAST_BOOT_REPLAY_PRIORITY
    args.advance_offsets_on_skip = True
    args.include_tool_results = True
    args.tool_result_max_chars = FAST_BOOT_TOOL_RESULT_MAX_CHARS
    args.maintain = True
    args.skip_maintain = False
    args.harvest_labels = True
    args.enable_teacher = True
    args.enable_async_route_pg = True
    args.enable_train_route_model = True
    args.enable_dreaming = True
    args.since_hours = FAST_BOOT_ASYNC_SINCE_HOURS
    args.max_queries = FAST_BOOT_ASYNC_MAX_QUERIES
    args.sample_rate = FAST_BOOT_ASYNC_SAMPLE_RATE
    args.max_candidates_per_node = FAST_BOOT_ASYNC_MAX_CANDIDATES
    args.max_decision_points = FAST_BOOT_ASYNC_MAX_DECISION_POINTS
    args.dream_since_hours = FAST_BOOT_DREAM_SINCE_HOURS
    args.dream_max_queries = FAST_BOOT_DREAM_MAX_QUERIES
    args.dream_sample_rate = FAST_BOOT_DREAM_SAMPLE_RATE
    args.dream_max_candidates_per_node = FAST_BOOT_DREAM_MAX_CANDIDATES
    args.dream_max_decision_points = FAST_BOOT_DREAM_MAX_DECISION_POINTS
    args.replay_stall_timeout_seconds = FAST_BOOT_REPLAY_STALL_TIMEOUT_SECONDS
    args.replay_stall_max_restarts = FAST_BOOT_REPLAY_STALL_MAX_RESTARTS
    args.replay_stall_fallback_mode = FAST_BOOT_REPLAY_STALL_FALLBACK_MODE
    args.dream_stall_timeout_seconds = FAST_BOOT_DREAM_STALL_TIMEOUT_SECONDS
    args.dream_stall_max_restarts = FAST_BOOT_DREAM_STALL_MAX_RESTARTS
    args.mode = "full"
    args.skip_if_locked = True
    args.pause_serve_when_locked = True


def _resolve_agent_ids(args: argparse.Namespace) -> list[str]:
    """Resolve agent ids from CLI override or openclaw config."""
    raw = getattr(args, "agents", None)
    if raw:
        resolved = [part.strip() for part in raw.split(",") if part.strip()]
        if not resolved:
            raise SystemExit("--agents specified but empty after parsing")
        return resolved
    return _discover_agent_ids()


def _wait_for_state_unlock(state_path: Path, timeout_seconds: int) -> bool:
    """Wait for the POSIX state lock to clear. Returns True if unlocked."""
    try:
        import fcntl  # type: ignore
    except Exception:  # pragma: no cover - non-POSIX fallback
        return True

    lock_path = lock_path_for_state(state_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    backoff = 1.0
    start = time.time()
    while True:
        fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
        acquired = False
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                return True
            except BlockingIOError:
                elapsed = time.time() - start
                if elapsed >= timeout_seconds:
                    return False
        finally:
            if acquired:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                except OSError:
                    pass
            os.close(fd)

        time.sleep(backoff)
        if backoff < 30.0:
            backoff = min(30.0, backoff * 2)


def _state_lock_available(state_path: str | Path) -> bool:
    """Return True if the state write lock is currently available."""
    return _wait_for_state_unlock(Path(state_path).expanduser(), 0)


def _read_state_lock_owner(lock_path: Path) -> dict[str, object]:
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _get_state_lock_owner_pid(lock_path: Path) -> int | None:
    owner = _read_state_lock_owner(lock_path)
    pid = owner.get("pid")
    return pid if isinstance(pid, int) else None


def _get_process_command(pid: int) -> str | None:
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None
    if result.returncode != 0:
        return None
    command = result.stdout.strip()
    return command or None


def _lock_owner_matches(command: str, state_path: str | Path) -> bool:
    expected_state = f"--state {Path(state_path).expanduser()}"
    return "openclawbrain daemon" in command and expected_state in command


def _run_logged_command(
    cmd: list[str],
    *,
    log_path: Path,
    step_name: str,
    stdout_path: Path | None = None,
) -> int:
    """Run a subprocess command and append stdout/stderr to the log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write(f"\n== {step_name} ==\n")
        log_handle.write(f"command: {' '.join(shlex.quote(part) for part in cmd)}\n")
        log_handle.flush()
        if stdout_path is not None:
            stdout_path.parent.mkdir(parents=True, exist_ok=True)
            with stdout_path.open("w", encoding="utf-8") as out_handle:
                proc = subprocess.run(
                    cmd,
                    stdout=out_handle,
                    stderr=log_handle,
                    text=True,
                    check=False,
                    env=_subprocess_env(),
                )
        else:
            proc = subprocess.run(
                cmd,
                stdout=log_handle,
                stderr=log_handle,
                text=True,
                check=False,
                env=_subprocess_env(),
            )
    return int(proc.returncode)


def _heartbeat_from_paths(paths: list[Path]) -> tuple[tuple[str, float, int], ...] | None:
    entries: list[tuple[str, float, int]] = []
    for path in paths:
        try:
            stat = path.stat()
        except OSError:
            continue
        entries.append((path.name, float(stat.st_mtime), int(stat.st_size)))
    return tuple(entries) if entries else None


def _run_logged_command_with_watchdog(
    cmd: list[str],
    *,
    log_path: Path,
    step_name: str,
    stdout_path: Path | None,
    stall_timeout_seconds: int,
    heartbeat_fn: Callable[[], object | None] | None,
    kill_grace_seconds: int = 10,
) -> tuple[int, bool]:
    if stall_timeout_seconds <= 0:
        return _run_logged_command(cmd, log_path=log_path, step_name=step_name, stdout_path=stdout_path), False

    log_path.parent.mkdir(parents=True, exist_ok=True)
    last_heartbeat_at = time.monotonic()
    last_token: object | None = None
    stalled = False

    with log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write(f"\n== {step_name} ==\n")
        log_handle.write(f"command: {' '.join(shlex.quote(part) for part in cmd)}\n")
        log_handle.flush()
        out_handle = None
        try:
            if stdout_path is not None:
                stdout_path.parent.mkdir(parents=True, exist_ok=True)
                out_handle = stdout_path.open("w", encoding="utf-8")
                proc = subprocess.Popen(
                    cmd,
                    stdout=out_handle,
                    stderr=log_handle,
                    text=True,
                    env=_subprocess_env(),
                )
            else:
                proc = subprocess.Popen(
                    cmd,
                    stdout=log_handle,
                    stderr=log_handle,
                    text=True,
                    env=_subprocess_env(),
                )
            while True:
                returncode = proc.poll()
                if returncode is not None:
                    return int(returncode), stalled
                token = heartbeat_fn() if heartbeat_fn else None
                if token is not None and token != last_token:
                    last_token = token
                    last_heartbeat_at = time.monotonic()
                if (time.monotonic() - last_heartbeat_at) > stall_timeout_seconds:
                    stalled = True
                    log_handle.write(
                        f"[stall] {step_name} heartbeat stalled for {stall_timeout_seconds}s; terminating\n"
                    )
                    log_handle.flush()
                    proc.terminate()
                    try:
                        returncode = proc.wait(timeout=max(1, int(kill_grace_seconds)))
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        returncode = proc.wait()
                    return int(returncode), stalled
                time.sleep(1.0)
        finally:
            if out_handle is not None:
                out_handle.close()


def _progress_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _read_replay_checkpoint_progress(
    checkpoint: dict[str, object],
    *,
    agent_id: str,
) -> str | None:
    parts: list[str] = []

    fast_payload = checkpoint.get("fast_learning")
    if isinstance(fast_payload, dict):
        processed = _progress_int(fast_payload.get("windows_processed"))
        total = _progress_int(fast_payload.get("windows_total"))
        status = fast_payload.get("status")
        status_text = str(status) if isinstance(status, str) else "unknown"
        processed_text = str(processed) if processed is not None else "?"
        total_text = str(total) if total is not None else "?"
        parts.append(f"fast_learning={processed_text}/{total_text} status={status_text}")

    replay_payload = checkpoint.get("replay")
    if isinstance(replay_payload, dict):
        processed = _progress_int(replay_payload.get("queries_processed"))
        total = _progress_int(replay_payload.get("queries_total"))
        merge_batches = _progress_int(replay_payload.get("merge_batches"))
        status = replay_payload.get("status")
        processed_text = str(processed) if processed is not None else "?"
        total_text = str(total) if total is not None else "?"
        part = f"replay={processed_text}/{total_text}"
        if merge_batches is not None:
            part = f"{part} merge_batches={merge_batches}"
        if isinstance(status, str):
            part = f"{part} status={status}"
        parts.append(part)

    if not parts:
        return None
    return f"[build-all] agent={agent_id} replay_progress " + " | ".join(parts)


def _replay_checkpoint_heartbeat(checkpoint_path: Path, *, agent_id: str) -> object | None:
    try:
        if not checkpoint_path.exists():
            return None
        checkpoint = _load_checkpoint(checkpoint_path)
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(checkpoint, dict):
        return None
    try:
        stat = checkpoint_path.stat()
        mtime = float(stat.st_mtime)
        size = int(stat.st_size)
    except OSError:
        mtime = 0.0
        size = 0
    progress = _read_replay_checkpoint_progress(checkpoint, agent_id=agent_id)
    return (mtime, size, progress)


def _monitor_replay_checkpoint(
    *,
    checkpoint_path: Path,
    agent_id: str,
    interval_seconds: int,
    stop_event: threading.Event,
) -> None:
    last_line: str | None = None
    while True:
        try:
            if not checkpoint_path.exists():
                checkpoint = None
            else:
                checkpoint = _load_checkpoint(checkpoint_path)
        except Exception:  # noqa: BLE001
            checkpoint = None
        if isinstance(checkpoint, dict):
            line = _read_replay_checkpoint_progress(checkpoint, agent_id=agent_id)
        else:
            line = None
        if line and line != last_line:
            print(line)
            last_line = line
        if stop_event.wait(interval_seconds):
            return


def _checkpoint_progress_snapshot(checkpoint_path: Path) -> dict[str, object]:
    mtime: float | None = None
    if checkpoint_path.exists():
        try:
            mtime = checkpoint_path.stat().st_mtime
        except OSError:
            mtime = None
    try:
        checkpoint = _load_checkpoint(checkpoint_path)
    except Exception:  # noqa: BLE001
        checkpoint = None

    updated_at: float | None = None
    processed: int | None = None
    total: int | None = None

    if isinstance(checkpoint, dict):
        fast_payload = checkpoint.get("fast_learning")
        if isinstance(fast_payload, dict):
            fast_updated = fast_payload.get("updated_at")
            if isinstance(fast_updated, (int, float)):
                updated_at = float(fast_updated)
            fast_processed = _progress_int(fast_payload.get("windows_processed"))
            fast_total = _progress_int(fast_payload.get("windows_total"))
            if fast_processed is not None:
                processed = fast_processed
            if fast_total is not None:
                total = fast_total

        replay_payload = checkpoint.get("replay")
        if isinstance(replay_payload, dict):
            replay_updated = replay_payload.get("updated_at")
            if isinstance(replay_updated, (int, float)):
                updated_at = max(updated_at or 0.0, float(replay_updated))
            replay_processed = _progress_int(replay_payload.get("queries_processed"))
            replay_total = _progress_int(replay_payload.get("queries_total"))
            if replay_processed is not None:
                processed = replay_processed
            if replay_total is not None:
                total = replay_total

    return {
        "mtime": mtime,
        "updated_at": updated_at,
        "processed": processed,
        "total": total,
    }


def _progress_snapshot_changed(
    previous: dict[str, object] | None,
    current: dict[str, object] | None,
) -> bool:
    if previous is None:
        return current is not None
    if current is None:
        return False
    for key in ("mtime", "updated_at", "processed", "total"):
        if previous.get(key) != current.get(key):
            return True
    return False


def _append_replay_watchdog_event(
    *,
    watchdog_path: Path,
    run_id: str,
    agent_id: str,
    event: str,
    checkpoint_path: Path,
    snapshot: dict[str, object] | None,
    extra: dict[str, object] | None = None,
) -> None:
    payload: dict[str, object] = {
        "run_id": run_id,
        "agent_id": agent_id,
        "event": event,
        "checkpoint_path": str(checkpoint_path),
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    if snapshot is not None:
        payload["processed"] = snapshot.get("processed")
        payload["total"] = snapshot.get("total")
    if extra:
        payload.update(extra)
    _append_jsonl(watchdog_path, payload)


def _append_watchdog_event(
    *,
    watchdog_path: Path,
    payload: dict[str, object],
) -> None:
    payload = dict(payload)
    payload["ts"] = datetime.now(timezone.utc).isoformat()
    _append_jsonl(watchdog_path, payload)


def _latest_mtime(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        if path.is_file():
            return path.stat().st_mtime
        if path.is_dir():
            latest: float | None = None
            for entry in path.iterdir():
                try:
                    mtime = entry.stat().st_mtime
                except OSError:
                    continue
                if latest is None or mtime > latest:
                    latest = mtime
            return latest or path.stat().st_mtime
    except OSError:
        return None
    return None


def _progress_snapshot_for_paths(paths: list[Path]) -> dict[str, object]:
    latest: float | None = None
    for path in paths:
        mtime = _latest_mtime(path)
        if mtime is None:
            continue
        latest = mtime if latest is None or mtime > latest else latest
    return {"mtime": latest}


def _run_logged_command_with_watchdog(
    cmd: list[str],
    *,
    log_path: Path,
    step_name: str,
    watchdog_path: Path,
    run_id: str,
    progress_paths: list[Path],
    stall_timeout_seconds: int,
    stall_max_restarts: int,
    env: dict[str, str] | None = None,
    spawn: Callable[..., subprocess.Popen] = subprocess.Popen,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> int:
    if stall_timeout_seconds <= 0 or stall_max_restarts < 0:
        return _run_logged_command(cmd, log_path=log_path, step_name=step_name)

    restarts = 0
    attempt = 0
    base_cmd = list(cmd)

    while True:
        attempt += 1
        attempt_label = f"{step_name} (attempt {attempt})" if attempt > 1 else step_name
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log_handle:
            log_handle.write(f"\n== {attempt_label} ==\n")
            log_handle.write(f"command: {' '.join(shlex.quote(part) for part in base_cmd)}\n")
            log_handle.flush()
            try:
                proc = spawn(base_cmd, stdout=log_handle, stderr=log_handle, text=True, env=env)
            except FileNotFoundError:
                return 127

            last_snapshot = _progress_snapshot_for_paths(progress_paths)
            last_progress_at = monotonic()
            stalled = False

            while True:
                returncode = proc.poll()
                snapshot = _progress_snapshot_for_paths(progress_paths)
                if _progress_snapshot_changed(last_snapshot, snapshot):
                    last_progress_at = monotonic()
                    last_snapshot = snapshot
                if returncode is not None:
                    returncode = int(returncode)
                    break
                if monotonic() - last_progress_at >= float(stall_timeout_seconds):
                    stalled = True
                    _append_watchdog_event(
                        watchdog_path=watchdog_path,
                        payload={
                            "run_id": run_id,
                            "event": "stall_detected",
                            "step": step_name,
                            "command": " ".join(shlex.quote(part) for part in base_cmd),
                            "progress_paths": [str(path) for path in progress_paths],
                            "last_mtime": last_snapshot.get("mtime"),
                        },
                    )
                    returncode = _terminate_process(proc, timeout_seconds=10.0)
                    break
                sleep(max(1.0, min(5.0, stall_timeout_seconds / 6)))

        if not stalled:
            return int(returncode)

        if restarts < stall_max_restarts:
            restarts += 1
            _append_watchdog_event(
                watchdog_path=watchdog_path,
                payload={
                    "run_id": run_id,
                    "event": "restart",
                    "step": step_name,
                    "restart_count": restarts,
                    "command": " ".join(shlex.quote(part) for part in base_cmd),
                },
            )
            continue

        _append_watchdog_event(
            watchdog_path=watchdog_path,
            payload={
                "run_id": run_id,
                "event": "give_up",
                "step": step_name,
                "command": " ".join(shlex.quote(part) for part in base_cmd),
            },
        )
        return 0

def _terminate_process(proc: subprocess.Popen, timeout_seconds: float) -> int:
    proc.terminate()
    try:
        return int(proc.wait(timeout=timeout_seconds))
    except subprocess.TimeoutExpired:
        proc.kill()
        return int(proc.wait())


def _run_logged_replay_command(
    cmd: list[str],
    *,
    log_path: Path,
    step_name: str,
    checkpoint_path: Path,
    agent_id: str,
    progress_interval_seconds: int,
    env: dict[str, str] | None = None,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stop_event = threading.Event()
    monitor: threading.Thread | None = None
    with log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write(f"\n== {step_name} ==\n")
        log_handle.write(f"command: {' '.join(shlex.quote(part) for part in cmd)}\n")
        log_handle.flush()
        if progress_interval_seconds > 0:
            monitor = threading.Thread(
                target=_monitor_replay_checkpoint,
                kwargs={
                    "checkpoint_path": checkpoint_path,
                    "agent_id": agent_id,
                    "interval_seconds": progress_interval_seconds,
                    "stop_event": stop_event,
                },
                daemon=True,
            )
            monitor.start()
        proc = subprocess.Popen(cmd, stdout=log_handle, stderr=log_handle, text=True, env=_subprocess_env(env))
        try:
            returncode = proc.wait()
        finally:
            stop_event.set()
            if monitor is not None:
                monitor.join(timeout=max(1.0, float(progress_interval_seconds)))
    return int(returncode)


def _run_logged_replay_command_with_watchdog(
    cmd: list[str],
    *,
    log_path: Path,
    step_name: str,
    checkpoint_path: Path,
    agent_id: str,
    run_id: str,
    watchdog_path: Path,
    progress_interval_seconds: int,
    stall_timeout_seconds: int,
    stall_max_restarts: int,
    stall_fallback_mode: str,
    env: dict[str, str] | None = None,
    spawn: Callable[..., subprocess.Popen] = subprocess.Popen,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> int:
    if stall_timeout_seconds <= 0 or stall_max_restarts < 0:
        return _run_logged_replay_command(
            cmd,
            log_path=log_path,
            step_name=step_name,
            checkpoint_path=checkpoint_path,
            agent_id=agent_id,
            progress_interval_seconds=progress_interval_seconds,
            env=env,
        )

    restarts = 0
    attempt = 0
    base_cmd = list(cmd)
    fallback_used = False
    merged_env = _subprocess_env(env)

    while True:
        attempt += 1
        log_path.parent.mkdir(parents=True, exist_ok=True)
        stop_event = threading.Event()
        monitor: threading.Thread | None = None
        attempt_label = f"{step_name} (attempt {attempt})" if attempt > 1 else step_name
        with log_path.open("a", encoding="utf-8") as log_handle:
            log_handle.write(f"\n== {attempt_label} ==\n")
            log_handle.write(f"command: {' '.join(shlex.quote(part) for part in base_cmd)}\n")
            log_handle.flush()
            if progress_interval_seconds > 0:
                monitor = threading.Thread(
                    target=_monitor_replay_checkpoint,
                    kwargs={
                        "checkpoint_path": checkpoint_path,
                        "agent_id": agent_id,
                        "interval_seconds": progress_interval_seconds,
                        "stop_event": stop_event,
                    },
                    daemon=True,
                )
                monitor.start()
            try:
                proc = spawn(base_cmd, stdout=log_handle, stderr=log_handle, text=True, env=merged_env)
            except FileNotFoundError:
                stop_event.set()
                if monitor is not None:
                    monitor.join(timeout=max(1.0, float(progress_interval_seconds)))
                return 127

            last_snapshot = _checkpoint_progress_snapshot(checkpoint_path)
            last_progress_at = monotonic()
            stalled = False

            try:
                while True:
                    returncode = proc.poll()
                    snapshot = _checkpoint_progress_snapshot(checkpoint_path)
                    if _progress_snapshot_changed(last_snapshot, snapshot):
                        last_progress_at = monotonic()
                        last_snapshot = snapshot
                    if returncode is not None:
                        returncode = int(returncode)
                        break
                    if monotonic() - last_progress_at >= float(stall_timeout_seconds):
                        stalled = True
                        _append_replay_watchdog_event(
                            watchdog_path=watchdog_path,
                            run_id=run_id,
                            agent_id=agent_id,
                            event="stall_detected",
                            checkpoint_path=checkpoint_path,
                            snapshot=last_snapshot,
                        )
                        returncode = _terminate_process(proc, timeout_seconds=10.0)
                        break
                    sleep(max(1.0, min(5.0, stall_timeout_seconds / 6)))
            finally:
                stop_event.set()
                if monitor is not None:
                    monitor.join(timeout=max(1.0, float(progress_interval_seconds)))

        if not stalled:
            return int(returncode)

        if restarts < stall_max_restarts:
            restarts += 1
            _append_replay_watchdog_event(
                watchdog_path=watchdog_path,
                run_id=run_id,
                agent_id=agent_id,
                event="restart",
                checkpoint_path=checkpoint_path,
                snapshot=last_snapshot,
                extra={"restart_count": restarts},
            )
            continue

        if not fallback_used and stall_fallback_mode == "edges-only":
            fallback_used = True
            fallback_cmd = []
            skip_next = False
            replaced_mode = False
            for part in base_cmd:
                if skip_next:
                    skip_next = False
                    continue
                if part == "--mode":
                    fallback_cmd.append(part)
                    fallback_cmd.append("edges-only")
                    skip_next = True
                    replaced_mode = True
                    continue
                if part in {"--edges-only", "--fast-learning", "--full-learning"}:
                    continue
                fallback_cmd.append(part)
            if not replaced_mode:
                fallback_cmd.extend(["--mode", "edges-only"])
            base_cmd = fallback_cmd
            _append_replay_watchdog_event(
                watchdog_path=watchdog_path,
                run_id=run_id,
                agent_id=agent_id,
                event="fallback",
                checkpoint_path=checkpoint_path,
                snapshot=last_snapshot,
                extra={"fallback_mode": "edges-only"},
            )
            stall_timeout_seconds = max(0, int(stall_timeout_seconds))
            restarts = 0
            if stall_timeout_seconds <= 0:
                return _run_logged_replay_command(
                    base_cmd,
                    log_path=log_path,
                    step_name=f"{step_name} (fallback)",
                    checkpoint_path=checkpoint_path,
                    agent_id=agent_id,
                    progress_interval_seconds=progress_interval_seconds,
                    env=env,
                )
            continue

        _append_replay_watchdog_event(
            watchdog_path=watchdog_path,
            run_id=run_id,
            agent_id=agent_id,
            event="give_up",
            checkpoint_path=checkpoint_path,
            snapshot=last_snapshot,
            extra={"fallback_mode": stall_fallback_mode},
        )
        return 0


def _build_all_agent_pipeline(
    *,
    agent_id: str,
    args: argparse.Namespace,
    ocb_prefix: list[str],
    ts_label: str,
    run_ts: datetime,
    stall_audit_path: Path | None,
    emit_event: Callable[[dict[str, object]], None],
) -> dict[str, object]:
    """Run build-all pipeline for a single agent."""
    root = Path.home() / ".openclawbrain"
    agent_root = root / agent_id
    scratch = agent_root / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)

    prefix = f"build-all.{ts_label}"
    log_path = scratch / f"{prefix}.log"
    status_before_path = scratch / f"{prefix}.status_before.json"
    status_after_path = scratch / f"{prefix}.status_after.json"
    maintain_path = scratch / f"{prefix}.maintain.json"
    async_route_path = scratch / f"{prefix}.async-route-pg.json"
    train_route_path = scratch / f"{prefix}.train-route-model.json"
    init_harvest_path = scratch / f"{prefix}.init.harvest.json"
    traces_out_path = scratch / f"{prefix}.route_traces.jsonl"
    route_model_out_path = scratch / f"{prefix}.route_model.npz"
    init_async_route_path = scratch / f"{prefix}.init.async-route-pg.json"
    init_train_route_path = scratch / f"{prefix}.init.train-route-model.json"
    init_traces_path = scratch / INIT_ROUTE_TRACES_FILENAME
    init_route_model_out_path = agent_root / "route_model.npz"
    agent_manifest_path = scratch / f"{prefix}.manifest.json"

    state_path = agent_root / "state.json"
    sessions_path = Path.home() / ".openclaw" / "agents" / agent_id / "sessions"
    checkpoint_path = agent_root / REPLAY_CHECKPOINT_FILENAME
    labels_path = _default_labels_path(str(state_path))

    steps: list[dict[str, object]] = []
    exit_code = 0
    last_error: str | None = None
    step_started: dict[str, float] = {}
    agent_started_at = time.perf_counter()

    artifact_paths = {
        "log_path": str(log_path),
        "manifest_path": str(agent_manifest_path),
    }

    def set_last_error(message: str | None) -> None:
        nonlocal last_error
        if message is not None and message:
            last_error = message

    def emit_event_payload(event: dict[str, object], *, with_artifact_paths: bool = True) -> None:
        payload = {
            "run_id": ts_label,
            "agent_id": agent_id,
            "ts": datetime.now(timezone.utc).isoformat(),
            **event,
        }
        if with_artifact_paths:
            payload["artifact_paths"] = dict(artifact_paths)
        emit_event(payload)

    def emit_stall_audit(payload: dict[str, object]) -> None:
        if stall_audit_path is None:
            return
        row = {
            "run_id": ts_label,
            "agent_id": agent_id,
            "ts": datetime.now(timezone.utc).isoformat(),
            **payload,
        }
        _append_jsonl(stall_audit_path, row)

    def emit_step_start(step: str, *, stdout_path: Path | None = None) -> None:
        step_started[step] = time.perf_counter()
        event: dict[str, object] = {"type": "step_start", "step": step}
        if stdout_path is not None:
            event["artifact_paths"] = {**artifact_paths, "stdout_path": str(stdout_path)}
        emit_event_payload(event, with_artifact_paths=stdout_path is None)

    def emit_step_end(
        step: str,
        *,
        status: str,
        code: int | None = None,
        error: str | None = None,
        stdout_path: Path | None = None,
    ) -> None:
        started = step_started.get(step, time.perf_counter())
        event: dict[str, object] = {
            "type": "step_end",
            "step": step,
            "status": status,
            "duration_seconds": max(0.0, time.perf_counter() - started),
        }
        if code is not None:
            event["exit_code"] = int(code)
        if error is not None:
            event["error"] = error
        if stdout_path is not None:
            event["artifact_paths"] = {**artifact_paths, "stdout_path": str(stdout_path)}
        emit_event_payload(event, with_artifact_paths=stdout_path is None)
        if code is not None:
            step_started.pop(step, None)

    def emit_skipped(step: str, *, reason: str | None = None, code: int = 0, stdout_path: Path | None = None) -> None:
        emit_step_start(step, stdout_path=stdout_path)
        steps.append({"step": step, "status": "skipped"})
        emit_step_end(step, status="skipped", code=code, error=reason, stdout_path=stdout_path)
        if code != 0:
            set_last_error(reason)

    def emit_status(step: str, status: str, extra: str | None = None) -> None:
        base = f"[build-all] agent={agent_id} step={step} status={status}"
        if extra is not None:
            base = f"{base} {extra}"
        print(base)

    def record(step: str, status: str, code: int | None = None) -> None:
        payload: dict[str, object] = {"step": step, "status": status}
        if code is not None:
            payload["exit_code"] = int(code)
        steps.append(payload)

    def run_step(
        step: str,
        cmd: list[str],
        *,
        stdout_path: Path | None = None,
        heartbeat_fn: Callable[[], object | None] | None = None,
    ) -> int:
        emit_step_start(step, stdout_path=stdout_path)
        emit_status(step, "running")
        stall_timeout = max(0, int(getattr(args, "step_stall_timeout_seconds", 0)))
        max_retries = 1
        attempt = 0
        stalled = False
        while True:
            if stall_timeout > 0:
                code, stalled = _run_logged_command_with_watchdog(
                    cmd,
                    log_path=log_path,
                    step_name=step,
                    stdout_path=stdout_path,
                    stall_timeout_seconds=stall_timeout,
                    heartbeat_fn=heartbeat_fn,
                )
            else:
                code = _run_logged_command(cmd, log_path=log_path, step_name=step, stdout_path=stdout_path)
                stalled = False
            if not stalled:
                break
            attempt += 1
            emit_event_payload(
                {
                    "type": "step_stall",
                    "step": step,
                    "attempt": attempt,
                    "timeout_seconds": stall_timeout,
                }
            )
            emit_stall_audit(
                {
                    "type": "step_stall",
                    "step": step,
                    "attempt": attempt,
                    "timeout_seconds": stall_timeout,
                    "command": cmd,
                }
            )
            if attempt > max_retries:
                break
            emit_status(step, "retrying", f"stall_timeout={stall_timeout}s attempt={attempt}/{max_retries}")
        if stalled and attempt > max_retries:
            reason = f"stall_timeout_{stall_timeout}s"
            emit_status(step, "skipped", reason)
            record(step, "skipped", 0)
            emit_step_end(step, status="skipped", code=0, error=reason, stdout_path=stdout_path)
            return 0
        if code == 0:
            emit_status(step, "ok")
            record(step, "ok", code)
            emit_step_end(step, status="ok", code=code, stdout_path=stdout_path)
        else:
            emit_status(step, "failed", f"exit={code}")
            record(step, "failed", code)
            error = f"{step} failed with exit {code}"
            set_last_error(error)
            emit_step_end(step, status="failed", code=code, error=error, stdout_path=stdout_path)
        return int(code)

    def run_replay_step(replay_cmd: list[str]) -> int:
        emit_step_start("replay")
        emit_status("replay", "running")
        replay_env = dict(os.environ)
        replay_env["PYTHONUNBUFFERED"] = "1"
        watchdog_path = scratch / "replay_watchdog.jsonl"
        code = _run_logged_replay_command_with_watchdog(
            replay_cmd,
            log_path=log_path,
            step_name="replay",
            checkpoint_path=checkpoint_path,
            agent_id=agent_id,
            run_id=ts_label,
            watchdog_path=watchdog_path,
            progress_interval_seconds=max(0, int(args.replay_progress_interval_seconds)),
            stall_timeout_seconds=max(0, int(args.replay_stall_timeout_seconds)),
            stall_max_restarts=max(0, int(args.replay_stall_max_restarts)),
            stall_fallback_mode=str(args.replay_stall_fallback_mode),
            env=replay_env,
        )
        if code == 0:
            emit_status("replay", "ok")
            record("replay", "ok", code)
            emit_step_end("replay", status="ok", code=code)
        else:
            emit_status("replay", "failed", f"exit={code}")
            record("replay", "failed", code)
            set_last_error(f"replay failed with exit {code}")
            emit_step_end("replay", status="failed", code=code, error=f"replay failed with exit {code}")
        return int(code)

    emit_event_payload(
        {
            "type": "agent_start",
            "artifact_paths": {
                "log_path": str(log_path),
                "manifest_path": str(agent_manifest_path),
            },
        },
        with_artifact_paths=True,
    )

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write("== OpenClawBrain build-all ==\n")
        log_handle.write(f"agent: {agent_id}\n")
        log_handle.write(f"state: {state_path}\n")
        log_handle.write(f"sessions: {sessions_path}\n")
        log_handle.write(f"timestamp: {run_ts.isoformat()}\n")

    if not state_path.exists():
        emit_status("preflight", "failed", f"missing_state={state_path}")
        set_last_error("missing state")
        exit_code = 2
        emit_skipped("status_before", reason="missing state", code=exit_code, stdout_path=status_before_path)
    elif not sessions_path.exists():
        emit_status("preflight", "failed", f"missing_sessions={sessions_path}")
        set_last_error("missing sessions")
        exit_code = 2
        emit_skipped("status_before", reason="missing sessions", code=exit_code, stdout_path=status_before_path)
    elif not _wait_for_state_unlock(state_path, int(args.state_lock_timeout_seconds)):
        emit_status("preflight", "failed", "state_lock_timeout")
        set_last_error("state lock timeout")
        exit_code = 3
        emit_skipped("status_before", reason="state lock timeout", code=exit_code, stdout_path=status_before_path)
    else:
        steps.append(
            {
                "step": "preflight",
                "status": "ok",
                "state_path": str(state_path),
                "sessions_path": str(sessions_path),
            }
        )

    if exit_code == 0:
        exit_code = run_step(
            "status_before",
            [*ocb_prefix, "status", "--state", str(state_path), "--json"],
            stdout_path=status_before_path,
        )

    if exit_code == 0:
        preflight_code, preflight_error = _evaluate_build_all_preflight(
            state_path=str(state_path),
            status_before_path=str(status_before_path),
            reembed=bool(args.reembed),
            require_local_embedder=bool(args.require_local_embedder),
        )
        if preflight_code != 0:
            exit_code = preflight_code
            set_last_error(preflight_error)

    if exit_code == 0 and args.reembed:
        exit_code = run_step(
            "reembed",
            [
                *ocb_prefix,
                "reembed",
                "--state",
                str(state_path),
                "--embedder",
                "local",
                "--embed-model",
                str(args.embed_model),
            ],
            heartbeat_fn=lambda: _heartbeat_from_paths([log_path]),
        )
    elif exit_code == 0:
        emit_skipped("reembed", reason="disabled", code=0)

    if exit_code == 0:
        replay_cmd = [
            *ocb_prefix,
            "replay",
            "--state",
            str(state_path),
            "--sessions",
            str(sessions_path),
            "--mode",
            str(args.mode),
            "--llm",
            str(args.llm),
        ]
        if args.workers is not None:
            replay_cmd.extend(["--workers", str(int(args.workers))])
        if getattr(args, "llm_model", None) is not None:
            replay_cmd.extend(["--llm-model", str(args.llm_model)])
        replay_cmd += [
            "--checkpoint-every-seconds",
            str(args.checkpoint_every_seconds),
        ]
        if args.replay_since_hours is not None:
            replay_cmd.extend(["--replay-since-hours", str(args.replay_since_hours)])
        if args.replay_max_interactions is not None:
            replay_cmd.extend(["--replay-max-interactions", str(args.replay_max_interactions)])
        if args.replay_sample_rate != 1.0:
            replay_cmd.extend(["--replay-sample-rate", str(args.replay_sample_rate)])
        if args.replay_priority != "all":
            replay_cmd.extend(["--replay-priority", str(args.replay_priority)])
        advance_offsets_on_skip = args.advance_offsets_on_skip
        if advance_offsets_on_skip is None:
            advance_offsets_on_skip = str(args.mode) == "full"
        if advance_offsets_on_skip:
            replay_cmd.append("--advance-offsets-on-skip")
        if args.include_tool_results:
            replay_cmd.append("--include-tool-results")
            tool_max = (
                args.tool_result_max_chars
                if getattr(args, "tool_result_max_chars", None) is not None
                else DEFAULT_TOOL_RESULT_MAX_CHARS
            )
            replay_cmd.extend(["--tool-result-max-chars", str(int(tool_max))])
        else:
            replay_cmd.append("--no-include-tool-results")
        if args.resume:
            replay_cmd.append("--resume")

        exit_code = run_replay_step(replay_cmd)
    else:
        emit_skipped("replay", reason=last_error, code=exit_code)

    if exit_code == 0:
        exit_code = run_step(
            "maintain",
            [
                *ocb_prefix,
                "maintain",
                "--state",
                str(state_path),
                "--tasks",
                "health,decay,scale,split,merge,prune,connect",
                "--llm",
                "none",
                "--embedder",
                "local",
                "--json",
            ],
            stdout_path=maintain_path,
        )
    else:
        emit_skipped("maintain", reason=last_error, code=exit_code, stdout_path=maintain_path)

    if exit_code == 0 and not args.skip_init_route_model:
        preserved = [record for record in read_labels_jsonl(labels_path) if record.reward_source != RewardSource.HARVESTER]
        harvest_cmd = [
            *ocb_prefix,
            "harvest",
            "--state",
            str(state_path),
            "--dry-run",
            "--labels-out",
            str(labels_path),
            "--json",
        ]
        exit_code = run_step("init_harvest_labels", harvest_cmd, stdout_path=init_harvest_path)
        if exit_code == 0 and preserved:
            append_labels_jsonl(labels_path, preserved)
        if exit_code == 0:
            init_async_cmd = [
                *ocb_prefix,
                "async-route-pg",
                "--state",
                str(state_path),
                "--teacher",
                str(args.teacher),
                "--teacher-model",
                str(args.teacher_model),
                "--since-hours",
                str(INIT_ROUTE_SINCE_HOURS),
                "--max-queries",
                str(INIT_ROUTE_MAX_QUERIES),
                "--sample-rate",
                str(INIT_ROUTE_SAMPLE_RATE),
                "--max-candidates-per-node",
                str(args.max_candidates_per_node),
                "--max-decision-points",
                str(args.max_decision_points),
                "--score-scale",
                str(args.score_scale),
                "--reward-source",
                str(args.reward_source),
                "--labels-out",
                str(labels_path),
                "--traces-out",
                str(init_traces_path),
                "--include-query-vector",
                "--apply",
                "--json",
            ]
            if args.reward_weights:
                init_async_cmd.extend(["--reward-weights", str(args.reward_weights)])
            if args.write_relevance_metadata:
                init_async_cmd.append("--write-relevance-metadata")
            else:
                init_async_cmd.append("--no-write-relevance-metadata")

            exit_code = run_step(
                "init_async_route_pg",
                init_async_cmd,
                stdout_path=init_async_route_path,
                heartbeat_fn=lambda: _heartbeat_from_paths([init_async_route_path, log_path]),
            )
            if exit_code == 0 and init_traces_path.exists() and init_traces_path.stat().st_size > 0:
                init_train_cmd = [
                    *ocb_prefix,
                    "train-route-model",
                    "--state",
                    str(state_path),
                    "--traces-in",
                    str(init_traces_path),
                    "--labels-in",
                    str(labels_path),
                    "--out",
                    str(init_route_model_out_path),
                    "--json",
                ]
                exit_code = run_step(
                    "init_train_route_model",
                    init_train_cmd,
                    stdout_path=init_train_route_path,
                    heartbeat_fn=lambda: _heartbeat_from_paths([init_train_route_path, log_path]),
                )
            else:
                skip_reason = "missing traces" if exit_code == 0 else "init async route failed"
                emit_skipped("init_train_route_model", reason=skip_reason, code=exit_code, stdout_path=init_train_route_path)
        else:
            emit_skipped("init_async_route_pg", reason="init harvest failed", code=exit_code, stdout_path=init_async_route_path)
            emit_skipped("init_train_route_model", reason="init harvest failed", code=exit_code, stdout_path=init_train_route_path)
    elif exit_code == 0:
        emit_skipped("init_harvest_labels", reason="disabled", code=0, stdout_path=init_harvest_path)
        emit_skipped("init_async_route_pg", reason="disabled", code=0, stdout_path=init_async_route_path)
        emit_skipped("init_train_route_model", reason="disabled", code=0, stdout_path=init_train_route_path)

    if exit_code == 0 and args.enable_async_teacher:
        async_cmd = [
            *ocb_prefix,
            "async-route-pg",
            "--state",
            str(state_path),
            "--teacher",
            str(args.teacher),
            "--teacher-model",
            str(args.teacher_model),
            "--since-hours",
            str(args.since_hours),
            "--max-queries",
            str(args.max_queries),
            "--sample-rate",
            str(args.sample_rate),
            "--max-candidates-per-node",
            str(args.max_candidates_per_node),
            "--max-decision-points",
            str(args.max_decision_points),
            "--score-scale",
            str(args.score_scale),
            "--reward-source",
            str(args.reward_source),
            "--labels-out",
            str(labels_path),
            "--traces-out",
            str(traces_out_path),
            "--apply",
            "--json",
        ]
        if args.reward_weights:
            async_cmd.extend(["--reward-weights", str(args.reward_weights)])
        if args.write_relevance_metadata:
            async_cmd.append("--write-relevance-metadata")
        else:
            async_cmd.append("--no-write-relevance-metadata")

        exit_code = run_step(
            "async_route_pg",
            async_cmd,
            stdout_path=async_route_path,
            heartbeat_fn=lambda: _heartbeat_from_paths([async_route_path, log_path]),
        )
        if exit_code == 0 and traces_out_path.exists() and traces_out_path.stat().st_size > 0:
            train_cmd = [
                *ocb_prefix,
                "train-route-model",
                "--state",
                str(state_path),
                "--traces-in",
                str(traces_out_path),
                "--labels-in",
                str(labels_path),
                "--out",
                str(route_model_out_path),
                "--json",
            ]
            exit_code = run_step(
                "train_route_model",
                train_cmd,
                stdout_path=train_route_path,
                heartbeat_fn=lambda: _heartbeat_from_paths([train_route_path, log_path]),
            )
        else:
            skip_reason = "missing traces" if exit_code == 0 else "async route failed"
            emit_skipped("train_route_model", reason=skip_reason, code=exit_code, stdout_path=train_route_path)
    elif exit_code == 0:
        emit_skipped("async_route_pg", reason="disabled", code=0, stdout_path=async_route_path)
        emit_skipped("train_route_model", reason="disabled", code=0, stdout_path=train_route_path)
    else:
        emit_skipped("async_route_pg", reason=last_error, code=exit_code, stdout_path=async_route_path)
        emit_skipped("train_route_model", reason=last_error, code=exit_code, stdout_path=train_route_path)

    if exit_code == 0:
        exit_code = run_step(
            "status_after",
            [*ocb_prefix, "status", "--state", str(state_path), "--json"],
            stdout_path=status_after_path,
        )
    else:
        emit_skipped("status_after", reason=last_error, code=exit_code, stdout_path=status_after_path)

    manifest = {
        "agent_id": agent_id,
        "state_path": str(state_path),
        "sessions_path": str(sessions_path),
        "timestamp": run_ts.isoformat(),
        "exit_code": int(exit_code),
        "log_path": str(log_path),
        "artifacts": {
            "status_before": str(status_before_path),
            "status_after": str(status_after_path),
            "maintain": str(maintain_path),
            "async_route_pg": str(async_route_path),
            "train_route_model": str(train_route_path),
            "traces_out": str(traces_out_path),
            "route_model_out": str(route_model_out_path),
        },
        "steps": steps,
    }
    _write_json_atomic(agent_manifest_path, manifest)

    emit_event_payload(
        {
            "type": "agent_end",
            "status": "ok" if exit_code == 0 else "failed",
            "exit_code": int(exit_code),
            "duration_seconds": max(0.0, time.perf_counter() - agent_started_at),
            "step": "agent_end",
            **({"error": str(last_error)} if last_error else {}),
            "artifact_paths": {
                "log_path": str(log_path),
                "manifest_path": str(agent_manifest_path),
            },
        },
        with_artifact_paths=True,
    )

    return {
        "agent_id": agent_id,
        "exit_code": int(exit_code),
        "manifest_path": str(agent_manifest_path),
        "log_path": str(log_path),
        "artifacts": manifest["artifacts"],
        "steps": steps,
        **({"error": str(last_error)} if last_error else {}),
    }

def _coalesce_int(cli_value: int | None, profile_value: int | None, default: int) -> int:
    if cli_value is not None:
        return cli_value
    if profile_value is not None:
        return profile_value
    return default


def _coalesce_float(cli_value: float | None, profile_value: float | None, default: float) -> float:
    if cli_value is not None:
        return cli_value
    if profile_value is not None:
        return profile_value
    return default


def _coalesce_route_use_relevance(
    cli_value: str | None,
    profile_value: bool | None,
    default: bool,
) -> bool:
    if cli_value is not None:
        return cli_value.strip().lower() == "true"
    if profile_value is not None:
        return profile_value
    return default


def _coalesce_route_enable_stop(
    cli_value: str | None,
    profile_value: bool | None,
    default: bool,
) -> bool:
    if cli_value is not None:
        return cli_value.strip().lower() == "true"
    if profile_value is not None:
        return profile_value
    return default


def _coalesce_assert_learned(
    cli_value: bool | None,
    profile_value: bool | None,
    default: bool,
) -> bool:
    if cli_value is not None:
        return bool(cli_value)
    if profile_value is not None:
        return bool(profile_value)
    return default


@contextmanager
def _command_state_write_lock(
    args: argparse.Namespace,
    *,
    allow_default_state: bool = False,
    explicit_state_path: str | Path | None = None,
    command_hint: str | None = None,
) -> Iterator[None]:
    """Acquire the single-writer lock for mutating commands."""
    if explicit_state_path is None:
        state_path = _resolve_effective_state_path(args, allow_default_state=allow_default_state)
    else:
        state_path = str(Path(explicit_state_path).expanduser())

    if state_path is None:
        yield
        return

    try:
        with state_write_lock(
            state_path,
            force=bool(getattr(args, "force", False)),
            command_hint=command_hint,
        ):
            yield
    except StateLockError as exc:
        raise SystemExit(str(exc)) from None


def _state_lock_context_for_command(args: argparse.Namespace):
    """Return per-command state write lock context manager when required."""
    command = getattr(args, "command", None)
    if command is None:
        return nullcontext()

    command_hint = f"openclawbrain {command}"
    if command == "init":
        output_dir = Path(args.output).expanduser()
        if output_dir.suffix == ".json" and not output_dir.is_dir():
            output_dir = output_dir.parent
        return _command_state_write_lock(
            args,
            explicit_state_path=output_dir / "state.json",
            command_hint=command_hint,
        )

    allow_default_state_map = {
        "learn": True,
        "maintain": True,
        "compact": True,
        "anchor": True,
        "connect": True,
        "merge": True,
        "inject": True,
        "self-learn": False,
        "self-correct": False,
        "replay": True,
        "harvest": False,
        "reembed": True,
        "sync": True,
        "async-route-pg": False,
    }
    if command in allow_default_state_map:
        return _command_state_write_lock(
            args,
            allow_default_state=allow_default_state_map[command],
            command_hint=command_hint,
        )
    return nullcontext()


def _build_parser() -> argparse.ArgumentParser:
    """ build parser."""
    parser = argparse.ArgumentParser(prog="openclawbrain")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    boot = sub.add_parser("bootstrap", help="Fast-boot a brain and install serve+loop services.")
    boot.add_argument("--agent", required=True, help="Agent id (required)")
    boot.add_argument("--workspace", help="Workspace root (overrides openclaw.json)")
    boot.add_argument("--fast", action=argparse.BooleanOptionalAction, default=True)
    boot.add_argument("--env-file", help="Optional .env file to pass into launchd service plists")

    i = sub.add_parser("init")
    i.add_argument("--workspace", required=True)
    i.add_argument("--output", required=True)
    i.add_argument("--sessions")
    i.add_argument("--embedder", choices=["local", "openai", "auto"], default="auto")
    i.add_argument("--embed-model", default=None)
    i.add_argument("--llm", choices=["none", "openai", "ollama", "auto"], default="auto")
    # LLM-splitting controls (default: use LLM only for larger/complex files)
    i.add_argument("--llm-split-min-chars", type=int, default=20000)
    i.add_argument("--llm-split-mode", choices=["auto", "all", "off"], default="auto")
    i.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    i.add_argument("--checkpoint-every", type=int, default=200)
    i.add_argument("--json", action="store_true")

    q = sub.add_parser("query")
    q.add_argument("text")
    q.add_argument("--state")
    q.add_argument("--graph")
    q.add_argument("--index")
    q.add_argument("--top", type=int, default=10)
    q.add_argument("--query-vector-stdin", action="store_true")
    q.add_argument("--embedder", choices=["local", "openai"], default=None)
    q.add_argument("--max-context-chars", type=int, default=None)
    q.add_argument("--provenance", action=argparse.BooleanOptionalAction, default=False)
    q.add_argument("--json", action="store_true")

    l = sub.add_parser("learn")
    l.add_argument("--state")
    l.add_argument("--graph")
    l.add_argument("--outcome", type=float, required=True)
    l.add_argument("--fired-ids", required=True)
    l.add_argument("--json", action="store_true")

    m = sub.add_parser("merge")
    m.add_argument("--state")
    m.add_argument("--graph")
    m.add_argument("--llm", choices=["none", "openai", "ollama"], default="none")
    m.add_argument("--json", action="store_true")

    a = sub.add_parser("anchor")
    a.add_argument("--state")
    a.add_argument("--node-id")
    a.add_argument("--authority", choices=["constitutional", "canonical"])
    a.add_argument("--remove", action="store_true")
    a.add_argument("--list", action="store_true")
    a.add_argument("--json", action="store_true")

    c = sub.add_parser("connect")
    c.add_argument("--state")
    c.add_argument("--graph")
    c.add_argument("--llm", choices=["none", "openai", "ollama"], default="none")
    c.add_argument("--json", action="store_true")

    p = sub.add_parser("maintain")
    p.add_argument("--state")
    p.add_argument("--tasks", default="health,decay,merge,prune")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-merges", type=int, default=5)
    p.add_argument("--prune-below", type=float, default=0.01)
    p.add_argument("--llm", choices=["none", "openai", "ollama"], default="none")
    p.add_argument("--embedder", choices=["local", "openai"], default=None)
    p.add_argument("--force", action="store_true", help="Bypass state lock (expert use)")
    p.add_argument("--json", action="store_true")

    z = sub.add_parser("compact")
    z.add_argument("--state")
    z.add_argument("--memory-dir", required=True)
    z.add_argument("--max-age-days", type=int, default=7)
    z.add_argument("--target-lines", type=int, default=15)
    z.add_argument("--llm", choices=["none", "openai", "ollama"], default="none")
    z.add_argument("--dry-run", action="store_true")
    z.add_argument("--force", action="store_true", help="Bypass state lock (expert use)")
    z.add_argument("--json", action="store_true")

    reembed = sub.add_parser("reembed", help="Rebuild embeddings + index for an existing state.json")
    reembed.add_argument("--state", required=True)
    reembed.add_argument("--embedder", choices=["local"], default="local")
    reembed.add_argument("--embed-model", default=None)
    reembed.add_argument("--backup", action=argparse.BooleanOptionalAction, default=True)
    reembed.add_argument("--json", action="store_true")

    d = sub.add_parser(
        "daemon",
        help="Low-level NDJSON worker over stdio (typically run behind `openclawbrain serve`)",
    )
    d.add_argument("--state")
    d.add_argument("--profile")
    d.add_argument("--embed-model", default=None)
    d.add_argument("--max-prompt-context-chars", type=int, default=None)
    d.add_argument("--max-fired-nodes", type=int, default=None)
    d.add_argument("--route-mode", choices=["off", "edge", "edge+sim", "learned"], default=None)
    d.add_argument("--route-top-k", type=int, default=None)
    d.add_argument("--route-alpha-sim", type=float, default=None)
    d.add_argument("--route-use-relevance", choices=["true", "false"], default=None)
    d.add_argument("--route-enable-stop", choices=["true", "false"], default=None)
    d.add_argument("--route-stop-margin", type=float, default=None)
    d.add_argument(
        "--assert-learned",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Error if effective routing mode is not learned.",
    )
    d.add_argument("--route-model", default=None)
    d.add_argument("--auto-save-interval", type=int, default=10)

    serve = sub.add_parser("serve", help="Canonical operator socket service lifecycle (`start|status|stop|install|uninstall`)")
    serve.add_argument(
        "serve_action",
        nargs="?",
        choices=["start", "status", "stop", "install", "uninstall"],
        default="start",
        help="Lifecycle action (default: start)",
    )
    serve.add_argument("--state", help="Path to state.json")
    serve.add_argument("--profile")
    serve.add_argument("--socket-path", help="Override Unix socket path (default: ~/.openclawbrain/<agent>/daemon.sock)")
    serve.add_argument("--embed-model", default=None)
    serve.add_argument("--max-prompt-context-chars", type=int, default=None)
    serve.add_argument("--max-fired-nodes", type=int, default=None)
    serve.add_argument("--route-mode", choices=["off", "edge", "edge+sim", "learned"], default=None)
    serve.add_argument("--route-top-k", type=int, default=None)
    serve.add_argument("--route-alpha-sim", type=float, default=None)
    serve.add_argument("--route-use-relevance", choices=["true", "false"], default=None)
    serve.add_argument("--route-enable-stop", choices=["true", "false"], default=None)
    serve.add_argument("--route-stop-margin", type=float, default=None)
    serve.add_argument(
        "--assert-learned",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Error if effective routing mode is not learned.",
    )
    serve.add_argument("--route-model", default=None)
    serve.add_argument(
        "--foreground",
        action="store_true",
        default=True,
        help="Run in foreground mode (default: true)",
    )
    serve.add_argument("--label", help="Override launchd label")
    serve.add_argument("--plist-path", help="Override launchd plist path")
    serve.add_argument("--env-file", help="Optional .env file for launchd EnvironmentVariables")
    serve.add_argument("--dry-run", action="store_true", help="Render launchd configuration only")
    serve.add_argument(
        "--launchd",
        action="store_true",
        help="Print a launchd plist template for this service and exit",
    )
    serve.add_argument(
        "--systemd",
        action="store_true",
        help="Print a systemd unit template for this service and exit",
    )

    x = sub.add_parser("inject")
    x.add_argument("--state")
    x.add_argument("--id", required=True)
    x.add_argument("--content", required=True)
    x.add_argument(
        "--type",
        choices=["CORRECTION", "TEACHING", "DIRECTIVE"],
        default="TEACHING",
    )
    x.add_argument("--summary")
    x.add_argument("--connect-top-k", type=int, default=3)
    x.add_argument("--connect-min-sim", type=float, default=None)
    x.add_argument("--embedder", choices=["local", "openai"], default=None)
    x.add_argument("--vector-stdin", action="store_true")
    x.add_argument("--json", action="store_true")

    sl = sub.add_parser("self-learn")
    sl.add_argument("--state", required=True)
    sl.add_argument("--content", required=True, help="The lesson learned")
    sl.add_argument("--fired-ids", default="", help="Comma-separated node IDs to penalize/reinforce")
    sl.add_argument("--outcome", type=float, default=-1.0, help="Negative=correct mistake, 0=neutral, positive=reinforce success")
    sl.add_argument("--type", choices=["CORRECTION", "TEACHING"], default="CORRECTION", dest="node_type")
    sl.add_argument("--json", dest="json_output", action="store_true")

    sc = sub.add_parser("self-correct")
    sc.add_argument("--state", required=True)
    sc.add_argument("--content", required=True, help="Alias for self-learn")
    sc.add_argument("--fired-ids", default="")
    sc.add_argument("--outcome", type=float, default=-1.0)
    sc.add_argument("--type", choices=["CORRECTION", "TEACHING"], default="CORRECTION", dest="node_type")
    sc.add_argument("--json", dest="json_output", action="store_true")

    r = sub.add_parser(
        "replay",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=REPLAY_HELP_EPILOG,
    )
    r.add_argument("--state")
    r.add_argument("--graph")
    r.add_argument("--sessions", nargs="+")
    r.add_argument(
        "--mode",
        choices=REPLAY_MODES,
        default=None,
        help=(
            "Replay mode: edges-only (cheap), fast-learning (LLM mining only), "
            "or full (fast-learning + replay + harvest; default)."
        ),
    )
    r.add_argument(
        "--fast-learning",
        "--extract-learning-events",
        dest="fast_learning",
        action="store_true",
        help=(
            "Run only LLM transcript mining + learning node injection "
            "(alias: --extract-learning-events). Note: this can be the slowest step because it is LLM-bound."
        ),
    )
    r.add_argument(
        "--full-learning",
        "--full-pipeline",
        dest="full_learning",
        action="store_true",
        help="Run full pipeline: fast-learning + edge replay + harvest (alias: --full-pipeline).",
    )

    r.add_argument("--edges-only", action="store_true")
    r.add_argument("--llm", choices=["none", "openai", "ollama", "auto"], default="auto")
    r.add_argument("--llm-model")
    r.add_argument("--show-checkpoint", action="store_true")
    r.add_argument("--decay-during-replay", action="store_true")
    r.add_argument("--decay-interval", type=int, default=10)
    r.add_argument(
        "--workers",
        type=int,
        default=4,
        help="LLM workers for fast-learning transcript extraction.",
    )
    r.add_argument("--window-radius", type=int, default=8)
    r.add_argument("--max-windows", type=int, default=6)
    r.add_argument("--hard-max-turns", type=int, default=120)
    r.add_argument("--backup", action=argparse.BooleanOptionalAction, default=True)
    r.add_argument("--resume", action="store_true")
    r.add_argument("--checkpoint", default=None)
    r.add_argument(
        "--fresh",
        "--no-checkpoint",
        dest="fresh",
        action="store_true",
        help="Ignore saved checkpoint offsets and start replay from the beginning.",
    )
    r.add_argument("--ignore-checkpoint", action="store_true", help=argparse.SUPPRESS)
    r.add_argument("--include-tool-results", action=argparse.BooleanOptionalAction, default=True)
    r.add_argument(
        "--replay-since-hours",
        type=_parse_positive_float,
        default=None,
        help="Keep only interactions with ts >= now - since_hours*3600.",
    )
    r.add_argument(
        "--replay-max-interactions",
        type=_parse_positive_int,
        default=None,
        help="Keep only the most recent N interactions after filtering.",
    )
    r.add_argument(
        "--replay-sample-rate",
        type=_parse_replay_sample_rate,
        default=1.0,
        help="Deterministic interaction sampling rate in (0, 1] (default 1).",
    )
    r.add_argument(
        "--replay-priority",
        choices=("all", "tool"),
        default="all",
        help="Filter interactions by tool content: all (default) or tool only.",
    )
    r.add_argument(
        "--advance-offsets-on-skip",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Advance replay offsets immediately when filtering skips interactions "
            "instead of only updating offsets for processed interactions."
        ),
    )
    r.add_argument(
        "--tool-edges",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create tool action/evidence edges during replay (default: on).",
    )
    r.add_argument(
        "--tool-result-allowlist",
        default=",".join(sorted(DEFAULT_TOOL_RESULT_ALLOWLIST)),
        help="Comma-separated tool names whose toolResult text may be attached for media stubs.",
    )
    r.add_argument("--tool-result-max-chars", type=int, default=DEFAULT_TOOL_RESULT_MAX_CHARS)
    r.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help=(
            "Emit replay progress every N interactions "
            "(non-JSON mode also emits heartbeat progress every 30s unless --quiet)."
        ),
    )
    r.add_argument("--checkpoint-every-seconds", type=int, default=60)
    r.add_argument("--checkpoint-every", type=int, default=0, help="Checkpoint every K replay windows/merge batches")
    r.add_argument("--stop-after-fast-learning", action="store_true")
    r.add_argument(
        "--replay-workers",
        type=int,
        default=1,
        help=(
            "Edge replay workers. 1 keeps strict sequential replay behavior; "
            ">1 uses deterministic shard/merge approximation for higher throughput."
        ),
    )
    r.add_argument("--persist-state-every-seconds", type=int, default=0)
    r.add_argument("--traces-out", default=None)
    r.add_argument("--labels-out", default=None)
    r.add_argument("--quiet", action="store_true", help="Suppress replay banners and progress output.")
    r.add_argument("--force", action="store_true", help="Bypass state lock (expert use)")
    r.add_argument("--json", action="store_true")

    hcmd = sub.add_parser("harvest")
    hcmd.add_argument("--state", required=True)
    hcmd.add_argument("--events")
    hcmd.add_argument("--tasks", default="split,merge,soft_prune,prune,connect,scale")
    hcmd.add_argument("--dry-run", action="store_true")
    hcmd.add_argument("--max-merges", type=int, default=5)
    hcmd.add_argument("--prune-below", type=float, default=0.01)
    hcmd.add_argument("--backup", action=argparse.BooleanOptionalAction, default=True)
    hcmd.add_argument("--traces-out", default=None)
    hcmd.add_argument("--labels-out", default=None)
    hcmd.add_argument("--json", action="store_true")

    h = sub.add_parser("health")
    h.add_argument("--state")
    h.add_argument("--graph")
    h.add_argument("--json", action="store_true")

    rep = sub.add_parser("report", help="Daily brain update summary")
    rep.add_argument("--state")

    ra = sub.add_parser("route-audit", help="Learned routing audit snapshot")
    ra.add_argument("--state")
    ra.add_argument("--json", action="store_true")

    ar = sub.add_parser("async-route-pg")
    ar.add_argument("--state", required=True)
    ar.add_argument("--since-hours", type=float, default=24.0)
    ar.add_argument("--max-queries", type=int, default=200)
    ar.add_argument("--sample-rate", type=float, default=0.1)
    ar.add_argument("--max-candidates-per-node", type=int, default=12)
    ar.add_argument("--teacher", choices=["openai", "ollama", "none"], default="openai")
    ar.add_argument("--teacher-model", default="gpt-5-mini")
    ar.add_argument("--apply", action="store_true")
    ar.add_argument("--json", action="store_true")
    ar.add_argument("--write-relevance-metadata", action=argparse.BooleanOptionalAction, default=True)
    ar.add_argument("--score-scale", type=float, default=0.3)
    ar.add_argument("--max-decision-points", type=int, default=500)
    ar.add_argument("--traces-out", default=None)
    ar.add_argument("--traces-in", default=None)
    ar.add_argument("--labels-out", default=None)
    ar.add_argument("--include-query-vector", action="store_true")
    ar.add_argument(
        "--reward-source",
        choices=[
            RewardSource.HUMAN.value,
            RewardSource.SELF.value,
            RewardSource.HARVESTER.value,
            RewardSource.TEACHER.value,
        ],
        default=RewardSource.TEACHER.value,
    )
    ar.add_argument("--reward-weights", default=None, help="Comma-separated weights: human=1.0,self=0.6,harvester=0.3,teacher=0.1")

    dream = sub.add_parser("dream")
    dream.add_argument("--state", required=True)
    dream.add_argument("--interval-seconds", type=int, default=900)
    dream.add_argument("--once", action="store_true", help="Run a single dreaming cycle and exit.")
    dream.add_argument("--apply", action=argparse.BooleanOptionalAction, default=True)
    dream.add_argument("--skip-if-locked", action=argparse.BooleanOptionalAction, default=True)
    dream.add_argument("--since-hours", type=float, default=24.0)
    dream.add_argument("--max-queries", type=int, default=200)
    dream.add_argument("--sample-rate", type=float, default=0.1)
    dream.add_argument("--max-candidates-per-node", type=int, default=12)
    dream.add_argument("--max-decision-points", type=int, default=500)
    dream.add_argument("--teacher", choices=["openai", "ollama", "none"], default="openai")
    dream.add_argument("--teacher-model", default="gpt-5-mini")
    dream.add_argument("--score-scale", type=float, default=0.3)
    dream.add_argument(
        "--reward-source",
        choices=[
            RewardSource.HUMAN.value,
            RewardSource.SELF.value,
            RewardSource.HARVESTER.value,
            RewardSource.TEACHER.value,
        ],
        default=RewardSource.TEACHER.value,
    )
    dream.add_argument("--reward-weights", default=None, help="Comma-separated weights: human=1.0,self=0.6,harvester=0.3,teacher=0.1")
    dream.add_argument("--traces-dir", default=None)
    dream.add_argument("--labels-out", default=None)
    dream.add_argument("--json", action="store_true")

    loop = sub.add_parser(
        "loop",
        help="Always-learning loop runner + scheduler (`run|install|uninstall|status`)",
    )
    loop.add_argument(
        "loop_action",
        nargs="?",
        choices=["run", "install", "uninstall", "status"],
        default="run",
        help="Loop action (default: run)",
    )
    loop.add_argument("--agent", help="Agent id (defaults to main; overrides --state)")
    loop.add_argument("--state", help="Path to state.json (defaults to main profile)")
    loop.add_argument("--sessions", help="Path to OpenClaw sessions directory")
    loop.add_argument("--mode", choices=REPLAY_MODES, default="full")
    loop.add_argument("--llm", choices=["none", "openai", "ollama", "auto"], default="auto")
    loop.add_argument("--llm-model")
    loop.add_argument("--workers", type=int, default=4)
    loop.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    loop.add_argument("--include-tool-results", action=argparse.BooleanOptionalAction, default=True)
    loop.add_argument("--tool-result-max-chars", type=int, default=None)
    loop.add_argument("--advance-offsets-on-skip", action=argparse.BooleanOptionalAction, default=None)
    loop.add_argument("--checkpoint-every-seconds", type=int, default=60)
    loop.add_argument("--replay-progress-interval-seconds", type=int, default=30)
    loop.add_argument("--replay-max-interactions", type=_parse_positive_int, default=None)
    loop.add_argument("--replay-priority", choices=("all", "tool"), default="all")
    loop.add_argument(
        "--maintain",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run maintenance pass (default: on)",
    )
    loop.add_argument("--skip-maintain", action="store_true", help="Skip maintenance pass (legacy alias)")
    loop.add_argument("--maintain-tasks", default="health,decay,prune,merge")
    loop.add_argument("--maintain-llm", choices=["none", "openai", "ollama"], default="none")
    loop.add_argument("--maintain-embedder", choices=["local", "openai"], default="local")
    loop.add_argument("--harvest-labels", action=argparse.BooleanOptionalAction, default=True, help="Refresh labels from harvest (default: on)")
    loop.add_argument("--enable-teacher", action=argparse.BooleanOptionalAction, default=True)
    loop.add_argument("--enable-async-route-pg", action=argparse.BooleanOptionalAction, default=True)
    loop.add_argument("--since-hours", type=float, default=24.0)
    loop.add_argument("--max-queries", type=int, default=200)
    loop.add_argument("--sample-rate", type=float, default=0.1)
    loop.add_argument("--max-candidates-per-node", type=int, default=12)
    loop.add_argument("--max-decision-points", type=int, default=500)
    loop.add_argument("--teacher", choices=["openai", "ollama", "none"], default="openai")
    loop.add_argument("--teacher-model", default="gpt-5-mini")
    loop.add_argument("--score-scale", type=float, default=0.3)
    loop.add_argument("--enable-train-route-model", action=argparse.BooleanOptionalAction, default=True)
    loop.add_argument("--train-route-model-out", default=None)
    loop.add_argument("--enable-dreaming", action=argparse.BooleanOptionalAction, default=True)
    loop.add_argument("--dream-since-hours", type=float, default=24.0)
    loop.add_argument("--dream-max-queries", type=int, default=200)
    loop.add_argument("--dream-sample-rate", type=float, default=0.1)
    loop.add_argument("--dream-max-candidates-per-node", type=int, default=12)
    loop.add_argument("--dream-max-decision-points", type=int, default=500)
    loop.add_argument(
        "--reward-source",
        choices=[
            RewardSource.HUMAN.value,
            RewardSource.SELF.value,
            RewardSource.HARVESTER.value,
            RewardSource.TEACHER.value,
        ],
        default=RewardSource.TEACHER.value,
    )
    loop.add_argument("--reward-weights", default=None)
    loop.add_argument("--write-relevance-metadata", action=argparse.BooleanOptionalAction, default=True)
    loop.add_argument("--skip-if-locked", action=argparse.BooleanOptionalAction, default=True)
    loop.add_argument("--pause-serve-when-locked", action=argparse.BooleanOptionalAction, default=True)
    loop.add_argument("--pause-serve-timeout-seconds", type=int, default=30)
    loop.add_argument("--replay-stall-timeout-seconds", type=int, default=900)
    loop.add_argument("--replay-stall-max-restarts", type=int, default=1)
    loop.add_argument(
        "--replay-stall-fallback-mode",
        choices=["off", "edges-only"],
        default="edges-only",
    )
    loop.add_argument("--dream-stall-timeout-seconds", type=int, default=900)
    loop.add_argument("--dream-stall-max-restarts", type=int, default=1)
    loop.add_argument("--hourly-interval-seconds", type=int, default=3600)
    loop.add_argument("--nightly-hour", type=int, default=2)
    loop.add_argument("--nightly-minute", type=int, default=30)
    loop.add_argument("--fast", action=argparse.BooleanOptionalAction, default=False)
    loop.add_argument("--dry-run", action="store_true")
    loop.add_argument("--launchd", action="store_true", help="Print launchd plist for loop service")
    loop.add_argument("--systemd", action="store_true", help="Print systemd unit template for loop service")
    loop.add_argument("--env-file", help="Optional .env file for launchd EnvironmentVariables")

    dreaming = sub.add_parser("dreaming")
    dreaming.add_argument("--state", required=True)
    dreaming.add_argument("--interval-seconds", type=int, default=900)
    dreaming.add_argument("--once", action="store_true", help="Run a single dreaming cycle and exit.")
    dreaming.add_argument("--apply", action=argparse.BooleanOptionalAction, default=True)
    dreaming.add_argument("--skip-if-locked", action=argparse.BooleanOptionalAction, default=True)
    dreaming.add_argument("--since-hours", type=float, default=24.0)
    dreaming.add_argument("--max-queries", type=int, default=200)
    dreaming.add_argument("--sample-rate", type=float, default=0.1)
    dreaming.add_argument("--max-candidates-per-node", type=int, default=12)
    dreaming.add_argument("--max-decision-points", type=int, default=500)
    dreaming.add_argument("--teacher", choices=["openai", "ollama", "none"], default="openai")
    dreaming.add_argument("--teacher-model", default="gpt-5-mini")
    dreaming.add_argument("--score-scale", type=float, default=0.3)
    dreaming.add_argument(
        "--reward-source",
        choices=[
            RewardSource.HUMAN.value,
            RewardSource.SELF.value,
            RewardSource.HARVESTER.value,
            RewardSource.TEACHER.value,
        ],
        default=RewardSource.TEACHER.value,
    )
    dreaming.add_argument("--reward-weights", default=None, help="Comma-separated weights: human=1.0,self=0.6,harvester=0.3,teacher=0.1")
    dreaming.add_argument("--traces-dir", default=None)
    dreaming.add_argument("--labels-out", default=None)
    dreaming.add_argument("--json", action="store_true")

    tr = sub.add_parser("train-route-model")
    tr.add_argument("--state", required=True)
    tr.add_argument("--traces-in", required=True)
    tr.add_argument("--labels-in", default=None)
    tr.add_argument("--out", required=True)
    tr.add_argument("--rank", type=int, default=16)
    tr.add_argument("--epochs", type=int, default=3)
    tr.add_argument("--lr", type=float, default=0.01)
    tr.add_argument("--label-temp", type=float, default=0.5)
    tr.add_argument("--reward-weights", default=None)
    tr.add_argument("--json", action="store_true")

    j = sub.add_parser("journal")
    j.add_argument("--state")
    j.add_argument("--last", type=int, default=10)
    j.add_argument("--stats", action="store_true")
    j.add_argument("--json", action="store_true")

    status_p = sub.add_parser("status")
    status_p.add_argument("--state", required=True)
    status_p.add_argument("--json", action="store_true")

    doctor = sub.add_parser("doctor")
    doctor.add_argument("--state", required=True)

    info = sub.add_parser("info")
    info_group = info.add_mutually_exclusive_group(required=True)
    info_group.add_argument("--state")
    info_group.add_argument("--graph")
    info.add_argument("--json", action="store_true")

    sync = sub.add_parser("sync")
    sync.add_argument("--state")
    sync.add_argument("--workspace", action="append")
    sync.add_argument("--workspaces", help="Comma-separated workspace roots")
    sync.add_argument("--embedder", choices=["openai", "local"], default=None)
    sync.add_argument(
        "--authority-map",
        help="JSON object mapping file name -> authority level",
    )
    sync.add_argument("--dry-run", action="store_true")
    sync.add_argument("--force", action="store_true", help="Bypass state lock (expert use)")
    sync.add_argument("--json", action="store_true")

    build_all = sub.add_parser(
        "build-all",
        help="Run the default unattended brain-building pipeline for all agents.",
    )
    build_all.add_argument("--agents", help="Comma-separated agent ids to run (default: discover from openclaw.json)")
    build_all.add_argument("--parallel-agents", type=int, default=1)
    build_all.add_argument("--reembed", action=argparse.BooleanOptionalAction, default=True)
    build_all.add_argument(
        "--require-local-embedder",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    build_all.add_argument("--embed-model", default="BAAI/bge-large-en-v1.5")
    build_all.add_argument("--mode", choices=REPLAY_MODES, default="full")
    build_all.add_argument("--llm", choices=["none", "openai", "ollama", "auto"], default="auto")
    build_all.add_argument("--workers", type=int, default=None)
    build_all.add_argument("--llm-model")
    build_all.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    build_all.add_argument("--include-tool-results", action=argparse.BooleanOptionalAction, default=True)
    build_all.add_argument("--tool-result-max-chars", type=int, default=None)
    build_all.add_argument("--replay-since-hours", type=_parse_positive_float, default=None)
    build_all.add_argument("--replay-max-interactions", type=_parse_positive_int, default=None)
    build_all.add_argument(
        "--replay-sample-rate",
        type=_parse_replay_sample_rate,
        default=1.0,
    )
    build_all.add_argument("--replay-priority", choices=("all", "tool"), default="all")
    build_all.add_argument(
        "--advance-offsets-on-skip",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    build_all.add_argument("--replay-stall-timeout-seconds", type=int, default=900)
    build_all.add_argument("--replay-stall-max-restarts", type=int, default=2)
    build_all.add_argument(
        "--replay-stall-fallback-mode",
        choices=("full", "edges-only"),
        default="edges-only",
    )
    build_all.add_argument("--checkpoint-every-seconds", type=int, default=60)
    build_all.add_argument("--replay-progress-interval-seconds", type=int, default=15)
    build_all.add_argument("--state-lock-timeout-seconds", type=int, default=3600)
    build_all.add_argument("--step-stall-timeout-seconds", type=int, default=0)
    build_all.add_argument("--enable-async-teacher", action="store_true")
    build_all.add_argument("--since-hours", type=float, default=24.0)
    build_all.add_argument("--max-queries", type=int, default=200)
    build_all.add_argument("--sample-rate", type=float, default=0.1)
    build_all.add_argument("--max-candidates-per-node", type=int, default=12)
    build_all.add_argument("--max-decision-points", type=int, default=500)
    build_all.add_argument("--teacher", choices=["openai", "ollama", "none"], default="openai")
    build_all.add_argument("--teacher-model", default="gpt-5-mini")
    build_all.add_argument("--score-scale", type=float, default=0.3)
    build_all.add_argument(
        "--reward-source",
        choices=[
            RewardSource.HUMAN.value,
            RewardSource.SELF.value,
            RewardSource.HARVESTER.value,
            RewardSource.TEACHER.value,
        ],
        default=RewardSource.TEACHER.value,
    )
    build_all.add_argument("--reward-weights", default=None)
    build_all.add_argument("--write-relevance-metadata", action=argparse.BooleanOptionalAction, default=True)
    build_all.add_argument(
        "--skip-init-route-model",
        action="store_true",
        help="Skip one-shot route model init (harvest + async-route-pg + train-route-model).",
    )
    build_all.add_argument(
        "--events-jsonl",
        default=None,
        help="Optional path for build-all JSONL event stream. Defaults to ~/.openclawbrain/scratch/build-all.<ts>.events.jsonl",
    )

    openclaw = sub.add_parser(
        "openclaw",
        help="OpenClaw integration helper (`install|uninstall|status`)",
    )
    openclaw.add_argument(
        "action",
        choices=["install", "uninstall", "status"],
        help="OpenClaw integration action",
    )
    openclaw.add_argument("--agent", default="main", help="Agent id (default: main)")
    openclaw.add_argument("--state", help="Explicit state.json path (overrides --agent)")
    openclaw.add_argument("--hooks-path", help="Path to openclawbrain-context-injector hook directory")
    openclaw.add_argument("--env-file", help="Optional .env file to pass into launchd service plists")
    openclaw.add_argument(
        "--skip-init-route-model",
        action="store_true",
        help="Skip one-shot route model init (harvest + async-route-pg + train-route-model).",
    )
    openclaw.add_argument("--yes", action="store_true", help="Skip confirmation prompts")
    return parser


def _load_payload(path: str) -> dict:
    """ load payload."""
    payload_path = Path(os.path.expanduser(path))
    if payload_path.is_dir():
        payload_path = payload_path / "graph.json"
    if not payload_path.exists():
        raise SystemExit(f"missing graph file: {path}")
    return json.loads(payload_path.read_text(encoding="utf-8"))


def _load_graph(path: str) -> Graph:
    """ load graph."""
    payload = _load_payload(path)
    payload = payload["graph"] if "graph" in payload else payload
    graph = Graph()
    for node_data in payload.get("nodes", []):
        graph.add_node(
            Node(node_data["id"], node_data["content"], node_data.get("summary", ""), node_data.get("metadata", {}))
        )
    for edge_data in payload.get("edges", []):
        graph.add_edge(
            Edge(
                edge_data["source"],
                edge_data["target"],
                edge_data.get("weight", 0.5),
                edge_data.get("kind", "sibling"),
                edge_data.get("metadata", {}),
            )
        )
    return graph


def _resolve_graph_index(
    args: argparse.Namespace,
    *,
    allow_default_state: bool = False,
) -> tuple[Graph, VectorIndex | None, dict[str, object]]:
    """ resolve graph index."""
    use_default_state = allow_default_state and getattr(args, "graph", None) is None
    state_path = _resolve_state_path(args.state, allow_default=use_default_state)
    if state_path is not None:
        state_file = Path(state_path).expanduser()
        if not state_file.exists():
            raise SystemExit(f"state file not found: {state_file}")
        graph, index, meta = load_state(state_path)
        return graph, index, meta
    if args.graph is None:
        raise SystemExit("--state or --graph is required")

    graph = _load_graph(args.graph)
    index_arg = getattr(args, "index", None)
    index = _load_index(index_arg) if index_arg is not None else None
    return graph, index, {}


def _ensure_route_model_exists(state_path: str) -> None:
    """Ensure route_model.npz exists beside state.json, creating identity model if missing."""
    route_model_path = Path(state_path).expanduser().parent / "route_model.npz"
    if route_model_path.exists():
        return
    _, _index, meta = load_state(str(Path(state_path).expanduser()))
    embedder_dim = meta.get("embedder_dim")
    if not isinstance(embedder_dim, int) or embedder_dim <= 0:
        raise SystemExit("could not determine embedder dimension for route_model initialization")
    RouteModel.init_identity(d=embedder_dim, df=1).save_npz(route_model_path)
    print(f"wrote default route_model.npz: {route_model_path}", file=sys.stderr)


def _graph_payload(graph: Graph) -> dict:
    """ graph payload."""
    return {
        "nodes": [
            {
                "id": n.id,
                "content": n.content,
                "summary": n.summary,
                "metadata": n.metadata,
            }
            for n in graph.nodes()
        ],
        "edges": [
            {"source": e.source, "target": e.target, "weight": e.weight, "kind": e.kind, "metadata": e.metadata}
            for source in graph._edges.values()
            for e in source.values()
        ],
    }


def _write_graph(
    path: str | Path,
    graph: Graph,
    *,
    include_meta: bool = False,
    meta: dict[str, object] | None = None,
) -> None:
    """Write graph payload to a JSON file."""
    destination = Path(path).expanduser()
    if destination.is_dir():
        destination = destination / "graph.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = _graph_payload(graph)
    if include_meta:
        payload = {"graph": payload, "meta": meta or {}}
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_query_vector_from_stdin() -> list[float]:
    """ load query vector from stdin."""
    data = sys.stdin.read().strip()
    if not data:
        raise SystemExit("query vector JSON required on stdin")
    payload = json.loads(data)
    if not isinstance(payload, list):
        raise SystemExit("query vector stdin payload must be a JSON array")
    return [float(v) for v in payload]


def _default_daemon_socket_path(state_path: str) -> str:
    """Resolve the default daemon socket path for a state file."""
    from .socket_server import _default_socket_path as _server_default_socket_path

    return str(Path(_server_default_socket_path(state_path)).expanduser())


def _daemon_socket_status(state_path: str) -> tuple[bool, str]:
    """Check whether the daemon socket is accepting connections."""
    socket_path = _default_daemon_socket_path(state_path)
    test_path = Path(socket_path)
    if not test_path.exists():
        return False, str(test_path)

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(0.25)
    try:
        sock.connect(str(test_path))
        return True, str(test_path)
    except OSError:
        return False, str(test_path)
    finally:
        sock.close()


def _socket_health_status(
    socket_path: str,
    *,
    timeout: float | None = None,
) -> tuple[bool, dict[str, object] | None, str | None]:
    """Check socket existence + daemon health call via socket protocol."""
    resolved_socket = str(Path(socket_path).expanduser())
    if not Path(resolved_socket).exists():
        return False, None, f"socket missing: {resolved_socket}"

    from .socket_client import OCBClient

    if timeout is None:
        timeout = _daemon_health_timeout()
    try:
        with OCBClient(resolved_socket, timeout=timeout) as client:
            health = client.health()
        return True, health, None
    except Exception as exc:  # noqa: BLE001
        return False, None, str(exc)


def _serve_status_payload(state_path: str, socket_path: str) -> dict[str, object]:
    """Build serve-status payload from socket path and ping health response."""
    resolved_state = str(Path(state_path).expanduser())
    resolved_socket = str(Path(socket_path).expanduser())
    socket_exists = Path(resolved_socket).exists()
    ping_ok, health, error = _socket_health_status(resolved_socket)
    return {
        "state_path": resolved_state,
        "socket_path": resolved_socket,
        "socket_exists": socket_exists,
        "daemon_running": bool(ping_ok),
        "health": health,
        "error": error,
    }


def _derive_serve_label(state_path: str) -> str:
    state_parent = Path(state_path).expanduser().parent
    return f"com.openclawbrain.{state_parent.name or 'main'}"


def _derive_loop_label(state_path: str, flavor: str | None = None) -> str:
    state_parent = Path(state_path).expanduser().parent
    base = f"com.openclawbrain.loop.{state_parent.name or 'main'}"
    return f"{base}.{flavor}" if flavor else base


def _derive_launchd_plist_path(label: str) -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"


def _loop_state_root(state_path: str) -> Path:
    return Path(state_path).expanduser().parent


def _loop_log_path(state_path: str) -> Path:
    return _loop_state_root(state_path) / LOOP_LOG_FILENAME


def _loop_events_path(state_path: str) -> Path:
    return _loop_state_root(state_path) / LOOP_EVENTS_FILENAME


def _loop_lock_path(state_path: str) -> Path:
    return _loop_state_root(state_path) / LOOP_LOCK_FILENAME


def _loop_checkpoint_path(state_path: str) -> Path:
    return _loop_state_root(state_path) / LOOP_CHECKPOINT_FILENAME


def _loop_manifest_path(state_path: str) -> Path:
    return _loop_state_root(state_path) / LOOP_MANIFEST_FILENAME


def _loop_stdout_path(state_path: str) -> Path:
    return _loop_state_root(state_path) / LOOP_STDOUT_FILENAME


def _loop_stderr_path(state_path: str) -> Path:
    return _loop_state_root(state_path) / LOOP_STDERR_FILENAME


def _try_acquire_loop_lock(lock_path: Path) -> tuple[int | None, bool]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        import fcntl  # type: ignore

        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fd, True
    except BlockingIOError:
        os.close(fd)
        return None, False
    except Exception:  # noqa: BLE001
        try:
            os.close(fd)
        except OSError:
            pass
        return None, False


def _loop_lock_held(lock_path: Path) -> bool:
    fd, acquired = _try_acquire_loop_lock(lock_path)
    if acquired and fd is not None:
        try:
            import fcntl  # type: ignore

            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(fd)
        return False
    return True


def _maybe_pause_serve_for_state_lock(
    *,
    state_path: str,
    pause_when_locked: bool,
    timeout_seconds: int,
    run_launchctl: Callable[[list[str]], int] | None = None,
    wait_for_unlock: Callable[[Path, int], bool] = _wait_for_state_unlock,
    get_lock_owner_pid: Callable[[Path], int | None] | None = None,
    get_process_command: Callable[[int], str | None] | None = None,
    platform: str | None = None,
) -> tuple[bool, list[str] | None, str | None]:
    """Attempt to pause launchd-managed serve daemon if state lock is held."""
    if run_launchctl is None:
        run_launchctl = _run_launchctl_returncode
    if get_lock_owner_pid is None:
        get_lock_owner_pid = _get_state_lock_owner_pid
    if get_process_command is None:
        get_process_command = _get_process_command
    state_path_obj = Path(state_path).expanduser()
    if wait_for_unlock(state_path_obj, 0):
        return True, None, None

    if not pause_when_locked:
        return False, None, "state_lock_held"

    effective_platform = platform or sys.platform
    if effective_platform != "darwin":
        return False, None, "state_lock_held"

    label = _derive_serve_label(state_path)
    plist_path = _derive_launchd_plist_path(label)
    if not plist_path.exists():
        return False, None, "state_lock_held"

    uid = os.getuid()
    bootout_cmd = ["launchctl", "bootout", f"gui/{uid}", str(plist_path)]
    bootstrap_cmd = ["launchctl", "bootstrap", f"gui/{uid}", str(plist_path)]

    bootout_rc = run_launchctl(bootout_cmd)
    if bootout_rc != 0:
        return False, None, "state_lock_held"

    grace_seconds = min(3, max(0, int(timeout_seconds)))
    if wait_for_unlock(state_path_obj, grace_seconds):
        return True, bootstrap_cmd, None

    try:
        lock_path = lock_path_for_state(state_path_obj)
        owner_pid = get_lock_owner_pid(lock_path)
        if owner_pid is None:
            raise ValueError("state lock owner pid missing")
        command = get_process_command(owner_pid)
        if command is None or not _lock_owner_matches(command, state_path_obj):
            raise ValueError("state lock owner mismatch")

        os.kill(owner_pid, signal.SIGTERM)
        if wait_for_unlock(state_path_obj, max(0, int(timeout_seconds))):
            return True, bootstrap_cmd, None

        os.kill(owner_pid, signal.SIGKILL)
        if wait_for_unlock(state_path_obj, max(0, int(timeout_seconds))):
            return True, bootstrap_cmd, None
    except Exception:
        run_launchctl(bootstrap_cmd)
        return False, None, "state_lock_held"

    run_launchctl(bootstrap_cmd)
    return False, None, "state_lock_held"



def _parse_env_file(env_path: str) -> dict[str, str]:
    path = Path(env_path).expanduser()
    if not path.exists():
        raise SystemExit(f"env file not found: {path}")
    env_vars: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if (
            (len(value) >= 2)
            and ((value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")))
        ):
            value = value[1:-1]
        env_vars[key] = value
    return env_vars


def _run_launchctl(argv: list[str], *, ignore_errors: bool = False) -> None:
    result = subprocess.run(argv, check=not ignore_errors)
    if ignore_errors and result.returncode != 0:
        return


def _run_launchctl_returncode(argv: list[str]) -> int:
    try:
        return int(subprocess.run(argv, check=False).returncode)
    except FileNotFoundError:
        return 127


def _serve_start_arguments(
    *,
    state_path: str,
    socket_path: str | None,
    embed_model: str,
    max_prompt_context_chars: int,
    max_fired_nodes: int,
    route_mode: str,
    route_top_k: int,
    route_alpha_sim: float,
    route_use_relevance: bool,
    route_enable_stop: bool,
    route_stop_margin: float,
    assert_learned: bool,
    route_model: str | None,
) -> list[str]:
    argv = ["--state", state_path]
    if socket_path:
        argv.extend(["--socket-path", socket_path])
    argv.extend([
        "--embed-model",
        embed_model,
        "--max-prompt-context-chars",
        str(max_prompt_context_chars),
        "--max-fired-nodes",
        str(max_fired_nodes),
        "--route-mode",
        route_mode,
        "--route-top-k",
        str(route_top_k),
        "--route-alpha-sim",
        str(route_alpha_sim),
        "--route-use-relevance",
        "true" if route_use_relevance else "false",
        "--route-enable-stop",
        "true" if route_enable_stop else "false",
        "--route-stop-margin",
        str(route_stop_margin),
    ])
    if assert_learned:
        argv.append("--assert-learned")
    if route_model:
        argv.extend(["--route-model", str(route_model)])
    return argv


def _render_launchd_plist(
    *,
    label: str,
    state_path: str,
    program_arguments: list[str],
    env_vars: dict[str, str] | None = None,
) -> str:
    """Render a minimal launchd plist for `openclawbrain serve start`."""
    state_parent = Path(state_path).expanduser().parent
    stdout_path = state_parent / "daemon.stdout.log"
    stderr_path = state_parent / "daemon.stderr.log"
    payload: dict[str, object] = {
        "Label": label,
        "ProgramArguments": program_arguments,
        "RunAtLoad": True,
        "KeepAlive": True,
        "StandardOutPath": str(stdout_path),
        "StandardErrorPath": str(stderr_path),
    }
    if env_vars:
        payload["EnvironmentVariables"] = env_vars
    payload_bytes = plistlib.dumps(payload, sort_keys=False)
    return payload_bytes.decode("utf-8")


def _resolve_serve_launchd_program_arguments(start_argv: list[str]) -> list[str]:
    wrapper_override = os.environ.get("OPENCLAWBRAIN_SERVE_WRAPPER")
    if isinstance(wrapper_override, str) and wrapper_override.strip():
        wrapper_path = Path(wrapper_override).expanduser()
        if wrapper_path.is_file():
            return [str(wrapper_path)] + start_argv

    wrapper_path = Path.home() / ".openclaw" / "scripts" / "openclawbrain-serve"
    if wrapper_path.is_file():
        return [str(wrapper_path)] + start_argv

    return [
        sys.executable,
        "-m",
        "openclawbrain.cli",
        "serve",
        "start",
    ] + start_argv


def _resolve_loop_python() -> str:
    override = os.environ.get("OPENCLAWBRAIN_LOOP_PYTHON") or os.environ.get("OPENCLAWBRAIN_PYTHON")
    if isinstance(override, str) and override.strip():
        return str(Path(override).expanduser())

    venv_python = Path.home() / ".openclaw" / "venvs" / "openclawbrain" / "bin" / "python"
    if venv_python.is_file():
        return str(venv_python)

    return sys.executable


def _render_loop_launchd_plist(
    *,
    label: str,
    program_arguments: list[str],
    stdout_path: Path,
    stderr_path: Path,
    schedule: dict[str, object],
    env_vars: dict[str, str] | None = None,
) -> str:
    payload: dict[str, object] = {
        "Label": label,
        "ProgramArguments": program_arguments,
        "RunAtLoad": True,
        "KeepAlive": True,
        "StandardOutPath": str(stdout_path),
        "StandardErrorPath": str(stderr_path),
        **schedule,
    }
    if env_vars:
        payload["EnvironmentVariables"] = env_vars
    payload_bytes = plistlib.dumps(payload, sort_keys=False)
    return payload_bytes.decode("utf-8")


def _render_systemd_unit(*, state_path: str, socket_path: str) -> str:
    """Render a minimal systemd unit for `openclawbrain serve start`."""
    state_parent = Path(state_path).expanduser().parent
    return "\n".join(
        [
            "[Unit]",
            "Description=OpenClawBrain socket service",
            "After=network.target",
            "",
            "[Service]",
            "Type=simple",
            f"WorkingDirectory={state_parent}",
            (
                "ExecStart=/usr/bin/env openclawbrain serve start "
                f"--state {Path(state_path).expanduser()} --socket-path {Path(socket_path).expanduser()}"
            ),
            "Restart=always",
            "RestartSec=2",
            "",
            "[Install]",
            "WantedBy=multi-user.target",
        ]
    )


def _render_loop_systemd_templates(
    *,
    state_path: str,
    sessions_path: str,
    loop_cmd: str,
) -> str:
    return "\n".join(
        [
            "# /etc/systemd/system/openclawbrain-loop.service",
            "[Unit]",
            "Description=OpenClawBrain always-learning loop",
            "After=network-online.target",
            "",
            "[Service]",
            "Type=simple",
            f"WorkingDirectory={Path(state_path).expanduser().parent}",
            f"ExecStart={loop_cmd}",
            "Restart=always",
            "RestartSec=10",
            "",
            "[Install]",
            "WantedBy=multi-user.target",
            "# Enable:",
            "#   sudo systemctl daemon-reload",
            "#   sudo systemctl enable --now openclawbrain-loop.service",
            "# Logs:",
            "#   journalctl -u openclawbrain-loop.service -n 200 --no-pager",
            f"# Sessions path assumed: {sessions_path}",
        ]
    )


def _last_replayed_display(value: object) -> str:
    """Format last replay timestamp."""
    if not isinstance(value, (int, float)):
        return "never"
    return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()


def _infer_index_dim(index: VectorIndex) -> int | None:
    """Infer vector dimension from any vector in the index."""
    for vector in index._vectors.values():
        return len(vector)
    return None


def _status_payload(state_path: str, meta: dict[str, object], graph: Graph, index: VectorIndex) -> dict[str, object]:
    """Build status payload details."""
    embedder_dim = meta.get("embedder_dim")
    index_dim = _infer_index_dim(index)

    health = measure_health(graph)
    inhibitory_edges = 0
    for source_edges in graph._edges.values():
        for edge in source_edges.values():
            if edge.kind == "inhibitory":
                inhibitory_edges += 1

    constitutional_nodes = sum(
        1 for node in graph.nodes() if node.metadata.get("authority") == "constitutional"
    )
    canonical_nodes = sum(1 for node in graph.nodes() if node.metadata.get("authority") == "canonical")

    daemon_running, daemon_socket = _daemon_socket_status(state_path)

    decay_half_life = meta.get("decay_half_life")
    if isinstance(decay_half_life, int):
        decay_value = decay_half_life
    elif isinstance(decay_half_life, float):
        decay_value = round(decay_half_life, 2)
    else:
        decay_value = "n/a"

    payload = {
        "version": __version__,
        "state": state_path,
        "nodes": graph.node_count(),
        "edges": graph.edge_count(),
        "reflex_pct": health.reflex_pct * 100,
        "habitual_pct": health.habitual_pct * 100,
        "dormant_pct": health.dormant_pct * 100,
        "inhibitory_edges": inhibitory_edges,
        "constitutional_nodes": constitutional_nodes,
        "canonical_nodes": canonical_nodes,
        "decay_half_life": decay_value,
        "last_replayed": _last_replayed_display(meta.get("last_replayed_ts")),
        "embedder_name": meta.get("embedder_name", "unknown"),
        "embedder_dim": embedder_dim if isinstance(embedder_dim, int) else "unknown",
        "index_dim": index_dim,
        "daemon_running": daemon_running,
        "daemon_socket_path": daemon_socket,
    }
    payload["daemon_running"] = daemon_running
    payload["daemon_socket"] = daemon_socket
    return payload


def _load_session_queries(
    session_paths: str | Iterable[str],
    since_ts: float | None = None,
) -> list[str]:
    """ load session queries."""
    if isinstance(session_paths, str):
        session_paths = [session_paths]
    queries: list[str] = []
    for session_path in session_paths:
        path = Path(session_path).expanduser()
        if path.is_dir():
            queries.extend(extract_queries_from_dir(path, since_ts=since_ts))
        elif path.is_file():
            queries.extend(extract_queries(path, since_ts=since_ts))
        else:
            raise SystemExit(f"invalid sessions path: {path}")
    return queries


def _resolve_journal_path(args: argparse.Namespace, *, allow_default_state: bool = False) -> str | None:
    """ resolve journal path."""
    use_default_state = allow_default_state and getattr(args, "graph", None) is None
    state_path = _resolve_state_path(args.state, allow_default=use_default_state)
    if state_path is not None:
        path = Path(state_path).expanduser()
        return str(path.parent / "journal.jsonl")
    if getattr(args, "graph", None) is not None:
        graph_path = Path(args.graph).expanduser()
        if graph_path.is_dir():
            return str(graph_path / "journal.jsonl")
        return str(graph_path.parent / "journal.jsonl")
    return None


def _default_dream_traces_dir(state_path: str) -> Path:
    resolved_state = Path(state_path).expanduser()
    agent = resolved_state.parent.name or "default"
    return Path.home() / ".openclawbrain" / agent / "scratch"


def _default_labels_path(state_path: str) -> Path:
    """Resolve default labels.jsonl path for a state file."""
    return Path(state_path).expanduser().parent / "labels.jsonl"


def _refresh_labels_from_harvest(labels_path: Path, harvest_output_path: Path) -> None:
    """Replace labels file with harvest output, preserving non-harvester labels."""
    preserved = [
        record for record in read_labels_jsonl(labels_path)
        if record.reward_source != RewardSource.HARVESTER
    ]
    if preserved:
        append_labels_jsonl(harvest_output_path, preserved)
    harvest_output_path.replace(labels_path)


def _load_session_query_records(session_paths: str | Iterable[str], since_ts: float | None = None) -> list[tuple[str, float | None]]:
    """ load session query records."""
    if isinstance(session_paths, str):
        session_paths = [session_paths]
    records: list[tuple[str, float | None]] = []
    for session_path in session_paths:
        path = Path(session_path).expanduser()
        if path.is_dir():
            records.extend(extract_query_records_from_dir(path, since_ts=since_ts))
        elif path.is_file():
            records.extend(extract_query_records(path, since_ts=since_ts))
        else:
            raise SystemExit(f"invalid sessions path: {path}")
    return records


def _write_route_traces_jsonl(path: str, traces: list[RouteTrace]) -> None:
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for trace in traces:
            handle.write(route_trace_to_json(trace) + "\n")


def _load_session_interactions(
    session_paths: str | Iterable[str],
    since_ts: float | None = None,
    *,
    include_tool_results: bool = True,
    tool_result_allowlist: set[str] | list[str] | tuple[str, ...] | None = None,
    tool_result_max_chars: int = DEFAULT_TOOL_RESULT_MAX_CHARS,
) -> list[dict[str, object]]:
    """ load session interactions."""
    if isinstance(session_paths, str):
        session_paths = [session_paths]

    session_files: list[Path] = []
    invalid_paths: list[str] = []

    for session_path in session_paths:
        try:
            session_files.extend(collect_session_files(session_path))
        except SystemExit:
            warnings.warn(f"invalid sessions path: {Path(session_path).expanduser()}")
            invalid_paths.append(str(session_path))

    if not session_files:
        raise SystemExit(f"invalid sessions path: {invalid_paths[0]}")

    interactions: list[dict[str, object]] = []
    for session_file in session_files:
        interactions.extend(
            extract_interactions(
                session_file,
                since_ts=since_ts,
                include_tool_results=include_tool_results,
                tool_result_allowlist=tool_result_allowlist,
                tool_result_max_chars=tool_result_max_chars,
            )
        )
    return interactions


def _parse_tool_result_allowlist(raw: str | None) -> set[str]:
    """Parse comma-separated allowlist text into normalized lower-case names."""
    if raw is None:
        return set(DEFAULT_TOOL_RESULT_ALLOWLIST)
    names = {part.strip().lower() for part in raw.split(",")}
    return {name for name in names if name}


def _state_meta(
    meta: dict[str, object] | None,
    fallback_name: str | None = None,
    fallback_dim: int | None = None,
    fallback_model: str | None = None,
) -> dict[str, object]:
    """ state meta."""
    base = dict(meta or {})
    embedder_name, embedder_dim = _state_embedder_meta(base)
    if fallback_name is not None:
        base["embedder_name"] = embedder_name or fallback_name
    if fallback_dim is not None:
        base["embedder_dim"] = embedder_dim if embedder_dim is not None else fallback_dim
    if fallback_model is not None:
        embedder_model = _state_embedder_model(base)
        if embedder_model is None:
            base["embedder_model"] = fallback_model
    return base


def _keyword_seeds(graph: Graph, text: str, top_k: int) -> list[tuple[str, float]]:
    """ keyword seeds."""
    query_tokens = _tokenize(text)
    if not query_tokens:
        return []
    scores = [
        (node.id, len(query_tokens & _tokenize(node.content)) / len(query_tokens))
        for node in graph.nodes()
    ]
    scores.sort(key=lambda item: (item[1], item[0]), reverse=True)
    return scores[:top_k]


def _load_index(path: str) -> VectorIndex:
    """ load index."""
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit("index payload must be a JSON object")
    index = VectorIndex()
    for node_id, vector in payload.items():
        if not isinstance(vector, list):
            raise SystemExit("index payload vectors must be arrays")
        index.upsert(str(node_id), [float(v) for v in vector])
    return index


def _state_embedder_meta(meta: dict[str, object]) -> tuple[str | None, int | None]:
    """ state embedder meta."""
    embedder_name = meta.get("embedder_name")
    if not isinstance(embedder_name, str):
        embedder_name = meta.get("embedder")
        if not isinstance(embedder_name, str):
            embedder_name = None

    embedder_dim = meta.get("embedder_dim")
    if not isinstance(embedder_dim, int):
        embedder_dim = None

    return embedder_name, embedder_dim


def _state_embedder_model(meta: dict[str, object]) -> str | None:
    """Resolve stored embedder model name if present."""
    embedder_model = meta.get("embedder_model")
    if isinstance(embedder_model, str) and embedder_model.strip():
        return embedder_model.strip()
    return None


def _resolve_embedder(
    args: argparse.Namespace, meta: dict[str, object]
) -> tuple[
    callable[[str], list[float]],
    callable[[list[tuple[str, str]]], dict[str, list[float]]],
    str,
    int,
    str | None,
]:
    """ resolve embedder."""
    openai_name = "openai-text-embedding-3-small"
    embedder_name, _ = _state_embedder_meta(meta)
    embed_model = getattr(args, "embed_model", None)

    if args.embedder == "hash":
        raise SystemExit("hash embedder is not selectable via CLI; use local/openai or migrate legacy state")

    if args.embedder == "auto":
        local_model = resolve_local_model(meta, embed_model=embed_model)
        embedder = LocalEmbedder(model_name=local_model)
        return embedder.embed, embedder.embed_batch, embedder.name, embedder.dim, local_model

    if args.embedder == "local":
        local_model = resolve_local_model(meta, embed_model=embed_model)
        embedder = LocalEmbedder(model_name=local_model)
        return embedder.embed, embedder.embed_batch, embedder.name, embedder.dim, local_model

    use_openai = args.embedder == "openai" or (args.embedder is None and embedder_name == openai_name)
    use_local = args.embedder is None and isinstance(embedder_name, str) and embedder_name.startswith("local:")
    if use_openai:
        from .openai_embeddings import OpenAIEmbedder

        _, prior_dim = _state_embedder_meta(meta)
        embedder = OpenAIEmbedder(dimensions=prior_dim)
    elif use_local:
        local_model = resolve_local_model(meta, embed_model=embed_model)
        embedder = LocalEmbedder(model_name=local_model)
    else:
        if embedder_name == HashEmbedder().name:
            embedder = HashEmbedder()
        else:
            local_model = resolve_local_model(meta, embed_model=embed_model)
            embedder = LocalEmbedder(model_name=local_model)

    resolved_model = local_model if isinstance(embedder, LocalEmbedder) else None
    return embedder.embed, embedder.embed_batch, embedder.name, embedder.dim, resolved_model


def _resolve_llm(args: argparse.Namespace) -> tuple[Callable[[str, str], str] | None, Callable[[list[dict]], list[dict]] | None]:
    """Resolve optional LLM callbacks."""
    llm = getattr(args, "llm", None)
    initial_llm = str(llm).strip().lower() if llm is not None else "auto"
    llm_model = getattr(args, "llm_model", None)
    if llm == "auto":
        default_llm = os.environ.get("OPENCLAWBRAIN_DEFAULT_LLM")
        if isinstance(default_llm, str):
            normalized = default_llm.strip().lower()
            if normalized in {"none", "openai", "ollama", "openrouter"}:
                llm = normalized
        if llm == "auto":
            if os.environ.get("OPENCLAWBRAIN_OLLAMA_MODEL") or os.environ.get("OLLAMA_MODEL"):
                llm = "ollama"
            elif os.environ.get("OPENAI_API_KEY"):
                llm = "openai"
            else:
                return None, None
    if llm in (None, "none"):
        return None, None
    if llm in {"openai", "openrouter"}:
        from .openai_llm import openai_llm_batch_fn, openai_llm_fn

        return openai_llm_fn, openai_llm_batch_fn
    if llm == "ollama":
        from .ollama_llm import ollama_llm_batch_fn, ollama_llm_fn

        if initial_llm == "ollama" and isinstance(llm_model, str) and llm_model:
            ollama_llm_fn = functools.partial(ollama_llm_fn, model=llm_model)
        return ollama_llm_fn, ollama_llm_batch_fn
    return None, None


def _load_json(path: str) -> dict:
    """ load json."""
    try:
        payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSON in state file: {path}") from exc
    except OSError as exc:
        raise SystemExit(f"missing state file: {path}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"state file payload must be an object: {path}")
    return payload


def _state_payload(meta_path: str) -> tuple[dict, dict[str, object], dict[str, list[float]], Graph]:
    """ state payload."""
    payload = _load_json(meta_path)
    graph_payload = payload.get("graph", payload)
    if not isinstance(graph_payload, dict):
        raise SystemExit("state file graph payload must be an object")

    index_payload = payload.get("index", {})
    if not isinstance(index_payload, dict):
        raise SystemExit("state index payload must be an object")

    graph = _load_graph(meta_path)
    return payload, graph_payload, index_payload, graph


def _check_result(ok: bool, label: str, details: str = "") -> bool:
    """ check result."""
    print(f"{label}: {'PASS' if ok else 'FAIL'}" + (f" ({details})" if details else ""))
    return ok


def _maybe_warn_long_running() -> None:
    """warn about long-running commands when in an interactive session."""
    if not sys.stderr.isatty():
        return
    print(
        "Note: init/build-all may take a long time; do not run under a short timeout.",
        file=sys.stderr,
    )


def _journal_entry_count(journal_path: str | None) -> int | None:
    """ journal entry count."""
    if journal_path is None:
        return None
    path = Path(journal_path)
    if not path.exists():
        return None
    return len(read_journal(journal_path=str(path)))


def _ensure_hash_embedder_compat(meta: dict[str, object]) -> None:
    """ ensure hash embedder compat."""
    embedder_name, embedder_dim = _state_embedder_meta(meta)
    if embedder_dim is None:
        return

    hash_dim = HashEmbedder().dim
    if embedder_dim != hash_dim:
        raise SystemExit(
            f"Index was built with {embedder_name} (dim={embedder_dim}). "
            "CLI hash embedder uses dim=1024. Dimension mismatch. "
            "Use --query-vector-stdin with matching embedder."
        )


def _result_payload(result: TraversalResult) -> dict:
    """ result payload."""
    return {
        "fired": result.fired,
        "steps": [step.__dict__ for step in result.steps],
        "context": result.context,
        "tier_thresholds": result.tier_summary,
    }


def _query_text_output(result: TraversalResult, graph: Graph, max_context_chars: int | None = None) -> str:
    """format query output with node IDs."""
    if not result.fired:
        return "(No matches.)"

    rendered: list[str] = []
    used_chars = 0
    for idx, node_id in enumerate(dict.fromkeys(result.fired)):
        node = graph.get_node(node_id)
        if node is None:
            continue
        block = f"{node_id}\n{'~' * len(node_id)}\n{node.content}"
        if max_context_chars is None:
            rendered.append(block)
            continue

        if max_context_chars <= 0:
            break
        separator = "\n\n"
        if not rendered:
            if len(block) > max_context_chars:
                if max_context_chars > 3:
                    rendered.append(block[: max_context_chars - 3] + "...")
                else:
                    rendered.append(block[:max_context_chars])
                break
            rendered.append(block)
            used_chars = len(block)
            continue

        if used_chars + len(separator) >= max_context_chars:
            break
        available = max_context_chars - used_chars - len(separator)
        if available <= 0:
            break
        if len(block) <= available:
            rendered.append(block)
            used_chars += len(separator) + len(block)
        else:
            if available > 3:
                rendered.append(block[:available - 3] + "...")
            else:
                rendered.append(block[:available])
            break

    return "\n\n".join(rendered)


def cmd_init(args: argparse.Namespace) -> int:
    """cmd init."""
    _maybe_warn_long_running()
    output_dir = Path(args.output).expanduser()
    if output_dir.suffix == ".json" and not output_dir.is_dir():
        output_dir = output_dir.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_paths = _init_checkpoint_paths(output_dir)
    checkpoint_root = checkpoint_paths["root"]
    checkpoint_every = int(getattr(args, "checkpoint_every", 0) or 0)
    if checkpoint_every <= 0:
        raise SystemExit("--checkpoint-every must be >= 1")
    if not getattr(args, "resume", True):
        if checkpoint_root.exists():
            shutil.rmtree(checkpoint_root)

    old_state_path = output_dir / "state.json"
    prior_meta: dict[str, object] = {}
    preserved_nodes: list[Node] = []
    old_index: VectorIndex | None = None
    if old_state_path.is_file():
        old_graph, old_index, prior_meta = load_state(str(old_state_path))
        preserved_nodes = [
            node
            for node in old_graph.nodes()
            if node.metadata.get("type") in {"CORRECTION", "TEACHING", "DIRECTIVE"}
        ]

    manifest_path = checkpoint_paths["manifest"]
    manifest_meta_path = checkpoint_paths["manifest_meta"]
    vectors_path = checkpoint_paths["vectors"]
    progress_path = checkpoint_paths["progress"]
    complete_path = checkpoint_paths["complete"]
    resume_enabled = bool(getattr(args, "resume", True))
    workspace_id = _resolve_workspace_id(args.workspace, None)
    manifest_meta = _read_json_optional(manifest_meta_path) if resume_enabled else None
    resume_manifest = False
    if resume_enabled and manifest_path.exists():
        meta_workspace_id = manifest_meta.get("workspace_id") if isinstance(manifest_meta, dict) else None
        if isinstance(meta_workspace_id, str) and meta_workspace_id and meta_workspace_id != workspace_id:
            print("Split checkpoint workspace mismatch; rebuilding.", file=sys.stderr)
        else:
            entries = _read_jsonl(manifest_path)
            if entries:
                print("Phase 1/4: Loading split checkpoint...", file=sys.stderr)
                graph, texts = _build_graph_from_manifest(entries, workspace_id)
                resume_manifest = True

    if not resume_manifest:
        print("Phase 1/4: Splitting workspace...", file=sys.stderr)
        llm_fn, llm_batch_fn = _resolve_llm(args)

        def _should_use_llm(rel: str, text: str) -> bool:
            mode = getattr(args, "llm_split_mode", "auto")
            if mode == "off":
                return False
            if mode == "all":
                return True
            # auto: only use LLM for larger/complex files
            min_chars = int(getattr(args, "llm_split_min_chars", 20000) or 20000)
            if len(text) >= min_chars:
                return True
            # Also use LLM for markdown/docs that likely benefit even if smaller
            lower = rel.lower()
            if lower.endswith((".md", ".rst")) and (
                "docs/" in lower or lower.endswith("agents.md") or lower.endswith("tools.md")
            ):
                return True
            return False

        graph, texts = split_workspace(
            args.workspace,
            llm_fn=llm_fn,
            llm_batch_fn=llm_batch_fn,
            should_use_llm_for_file=_should_use_llm,
        )
        for node in graph.nodes():
            file_name = node.metadata.get("file")
            if not isinstance(file_name, str) or not file_name:
                continue
            authority = DEFAULT_AUTHORITY_MAP.get(Path(file_name).name)
            if authority is not None:
                node.metadata["authority"] = authority

        manifest_entries: list[dict[str, object]] = []
        for node_id, text in texts.items():
            node = graph.get_node(node_id)
            if node is None:
                continue
            manifest_entries.append(
                {
                    "id": node.id,
                    "text": text,
                    "summary": node.summary,
                    "metadata": dict(node.metadata),
                }
            )
        _write_init_manifest(
            manifest_path,
            manifest_meta_path,
            manifest_entries,
            {
                "created_at": _iso_now(),
                "workspace": str(Path(args.workspace).expanduser()),
                "workspace_id": workspace_id,
                "node_count": len(texts),
            },
        )

    split_progress = _read_json_optional(progress_path) or {}
    split_progress.update(
        {
            "phase": "split",
            "split_completed_at": _iso_now(),
            "node_count": len(texts),
        }
    )
    _write_json_atomic(progress_path, split_progress)

    print("Phase 2/4: Embedding texts...", file=sys.stderr)
    embedder_fn, embed_batch_fn, embedder_name, embedder_dim, embedder_model = _resolve_embedder(args, prior_meta)
    if resume_enabled:
        existing_progress = _read_json_optional(progress_path) or {}
        prior_name = existing_progress.get("embedder_name")
        prior_dim = existing_progress.get("embedder_dim")
        if isinstance(prior_name, str) and prior_name and prior_name != embedder_name:
            raise SystemExit(
                f"init checkpoint embedder mismatch: {prior_name} vs {embedder_name} (use --no-resume to rebuild)"
            )
        if isinstance(prior_dim, int) and prior_dim and prior_dim != embedder_dim:
            raise SystemExit(
                f"init checkpoint dimension mismatch: {prior_dim} vs {embedder_dim} (use --no-resume to rebuild)"
            )
    print(
        f"Embedding {len(texts)} texts ({embedder_name}, dim={embedder_dim})",
        file=sys.stderr,
    )
    index_vectors: dict[str, list[float]] = {}
    index = VectorIndex()
    done_ids: set[str] = set()
    if resume_enabled and vectors_path.exists():
        for record in _read_jsonl(vectors_path):
            node_id = record.get("id")
            vector = record.get("vector")
            if not isinstance(node_id, str) or not isinstance(vector, list):
                continue
            try:
                cast_vector = [float(v) for v in vector]
            except (TypeError, ValueError):
                continue
            index_vectors[node_id] = cast_vector
            done_ids.add(node_id)
            index.upsert(node_id, cast_vector)

    embedding_started_at = None
    existing_progress = _read_json_optional(progress_path) or {}
    if isinstance(existing_progress.get("embedding_started_at"), str):
        embedding_started_at = str(existing_progress["embedding_started_at"])
    if embedding_started_at is None:
        embedding_started_at = _iso_now()
    existing_progress.update(
        {
            "phase": "embedding",
            "embedding_started_at": embedding_started_at,
            "updated_at": _iso_now(),
            "total": len(texts),
            "done": len(done_ids),
            "checkpoint_every": checkpoint_every,
            "embedder_name": embedder_name,
            "embedder_dim": embedder_dim,
            "embedder_model": embedder_model,
        }
    )
    _write_json_atomic(progress_path, existing_progress)

    ordered_items = list(texts.items())
    pending_items = [(node_id, text) for node_id, text in ordered_items if node_id not in done_ids]
    for batch_start in range(0, len(pending_items), checkpoint_every):
        batch = pending_items[batch_start : batch_start + checkpoint_every]
        if not batch:
            continue
        batch_vectors = batch_or_single_embed(batch, embedder_fn, embed_batch_fn)
        payloads: list[dict[str, object]] = []
        for node_id, _text in batch:
            vector = batch_vectors.get(node_id)
            if vector is None:
                continue
            index_vectors[node_id] = vector
            done_ids.add(node_id)
            index.upsert(node_id, vector)
            payloads.append({"id": node_id, "vector": vector})
        _append_jsonl_lines_fsync(vectors_path, payloads)
        next_index = len(texts)
        for idx, (node_id, _text) in enumerate(ordered_items):
            if node_id not in done_ids:
                next_index = idx
                break
        progress_payload = _read_json_optional(progress_path) or {}
        progress_payload.update(
            {
                "phase": "embedding",
                "updated_at": _iso_now(),
                "total": len(texts),
                "done": len(done_ids),
                "next_index": next_index,
            }
        )
        _write_json_atomic(progress_path, progress_payload)

    progress_payload = _read_json_optional(progress_path) or {}
    progress_payload.update(
        {
            "phase": "embedding_complete",
            "embedding_completed_at": _iso_now(),
            "total": len(texts),
            "done": len(done_ids),
            "next_index": len(texts),
        }
    )
    _write_json_atomic(progress_path, progress_payload)

    replay_stats: dict[str, object] = {}
    if args.sessions is not None:
        print("Phase 3/4: Replaying sessions...", file=sys.stderr)
        interactions = _load_session_interactions(args.sessions, since_ts=None)
        print(
            f"Loaded {len(interactions)} interactions from session files",
            file=sys.stderr,
        )
        replay_stats = replay_queries(graph=graph, queries=interactions)

    if preserved_nodes:
        connect_min_sim = 0.0 if embedder_name == HashEmbedder().name else 0.3
        for node in preserved_nodes:
            inject_node(
                graph=graph,
                index=index,
                node_id=node.id,
                content=node.content,
                summary=node.summary,
                metadata=dict(node.metadata),
                vector=old_index._vectors.get(node.id) if old_index is not None else None,
                embed_fn=None if old_index is not None else embedder_fn,
                connect_top_k=3,
                connect_min_sim=connect_min_sim,
            )
        print(f"Preserved {len(preserved_nodes)} injected nodes from previous state")

    print("Phase 4/5: Saving state...", file=sys.stderr)
    graph_path = output_dir / "graph.json"
    text_path = output_dir / "texts.json"
    state_meta = _state_meta(
        prior_meta,
        fallback_name=embedder_name,
        fallback_dim=embedder_dim,
        fallback_model=embedder_model,
    )
    if replay_stats.get("last_replayed_ts") is not None:
        state_meta["last_replayed_ts"] = replay_stats["last_replayed_ts"]
        source = replay_stats.get("last_replayed_ts_source")
        if isinstance(source, str):
            state_meta["last_replayed_ts_source"] = source
        else:
            state_meta.pop("last_replayed_ts_source", None)
    else:
        state_meta.pop("last_replayed_ts", None)
        state_meta.pop("last_replayed_ts_source", None)

    _write_graph(graph_path, graph, include_meta=True, meta=state_meta)
    save_state(
        graph=graph,
        index=index,
        path=output_dir / "state.json",
        meta=state_meta,
    )
    route_model_path = output_dir / "route_model.npz"
    resolved_embedder_dim = state_meta.get("embedder_dim")
    if not isinstance(resolved_embedder_dim, int):
        resolved_embedder_dim = embedder_dim
    if isinstance(resolved_embedder_dim, int) and resolved_embedder_dim > 0:
        RouteModel.init_identity(d=resolved_embedder_dim, df=1).save_npz(route_model_path)
    else:
        raise SystemExit("could not determine embedder dimension for route_model initialization")
    index_path = output_dir / "index.json"
    index_path.write_text(json.dumps(index_vectors, indent=2), encoding="utf-8")
    text_path.write_text(json.dumps(texts, indent=2), encoding="utf-8")
    print("Phase 5/5: Wrote default route_model.npz", file=sys.stderr)

    completion_payload = _read_json_optional(progress_path) or {}
    completion_payload.update({"phase": "complete", "completed_at": _iso_now()})
    _write_json_atomic(progress_path, completion_payload)
    _write_json_atomic(
        complete_path,
        {
            "completed_at": completion_payload["completed_at"],
            "state_path": str(output_dir / "state.json"),
        },
    )

    if args.json:
        print(json.dumps({"graph": str(graph_path), "texts": str(text_path)}))
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    """cmd query."""
    graph, index, meta = _resolve_graph_index(args, allow_default_state=True)
    if args.top <= 0:
        raise SystemExit("--top must be >= 1")

    if args.query_vector_stdin:
        if index is None:
            raise SystemExit("query-vector-stdin requires --index")
        query_vec = _load_query_vector_from_stdin()
        seeds = index.search(query_vec, top_k=args.top)
    elif index is not None:
        embed_fn, _, embedder_name, _, _ = _resolve_embedder(args, meta)
        if embedder_name == HashEmbedder().name:
            _ensure_hash_embedder_compat(meta)
        query_vec = embed_fn(args.text)
        seeds = index.search(query_vec, top_k=args.top)
    else:
        seeds = _keyword_seeds(graph, args.text, args.top)

    result = traverse(
        graph=graph,
        seeds=seeds,
        config=TraversalConfig(
            max_hops=15,
            max_context_chars=args.max_context_chars,
            include_provenance=bool(args.provenance),
        ),
        query_text=args.text,
    )
    log_query(
        query_text=args.text,
        fired_ids=result.fired,
        node_count=graph.node_count(),
        journal_path=_resolve_journal_path(args, allow_default_state=True),
    )
    if args.json:
        print(json.dumps(_result_payload(result)))
    else:
        print(_query_text_output(result=result, graph=graph, max_context_chars=args.max_context_chars))
    return 0


def cmd_learn(args: argparse.Namespace) -> int:
    """cmd learn."""
    allow_default_state = getattr(args, "graph", None) is None
    graph, index, meta = _resolve_graph_index(args, allow_default_state=allow_default_state)
    state_path = _resolve_state_path(args.state, allow_default=allow_default_state)
    fired_ids = [value.strip() for value in args.fired_ids.split(",") if value.strip()]
    if not fired_ids:
        raise SystemExit("provide at least one fired id")

    updates = apply_outcome_pg(
        graph=graph,
        fired_nodes=fired_ids,
        outcome=args.outcome,
        baseline=0.0,
        temperature=1.0,
    )
    if state_path is not None:
        state_meta = _state_meta(meta)
        save_state(graph=graph, index=index or VectorIndex(), path=state_path, meta=state_meta)
    updates_abs = [abs(delta) for delta in updates.values()]
    summary = {
        "edges_updated": len(updates),
        "max_weight_delta": max(updates_abs) if updates_abs else 0.0,
    }
    if state_path is None:
        payload = {"graph": _graph_payload(graph)}
        Path(args.graph).expanduser().write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log_learn(fired_ids=fired_ids, outcome=args.outcome, journal_path=_resolve_journal_path(args, allow_default_state=allow_default_state))
    print(json.dumps(summary, indent=2) if args.json else f"updated {state_path or args.graph}")
    return 0


def cmd_merge(args: argparse.Namespace) -> int:
    """cmd merge."""
    allow_default_state = getattr(args, "graph", None) is None
    graph, index, meta = _resolve_graph_index(args, allow_default_state=allow_default_state)
    state_path = _resolve_state_path(args.state, allow_default=allow_default_state)
    llm_fn, llm_batch_fn = _resolve_llm(args)
    suggestions = suggest_merges(graph, llm_fn=llm_fn, llm_batch_fn=llm_batch_fn)
    applied = []
    for source_id, target_id in suggestions:
        if graph.get_node(source_id) and graph.get_node(target_id):
            merged = apply_merge(graph, source_id, target_id)
            applied.append({"from": [source_id, target_id], "to": [merged]})
    if state_path is not None:
        state_meta = _state_meta(meta)
        save_state(graph=graph, index=index or VectorIndex(), path=state_path, meta=state_meta)
    else:
        _write_graph(args.graph, graph)
    payload = {"suggestions": [{"from": [s, t]} for s, t in suggestions], "applied": applied}
    print(json.dumps(payload) if args.json else f"Applied merges: {len(applied)}")
    return 0


def cmd_connect(args: argparse.Namespace) -> int:
    """cmd connect."""
    allow_default_state = getattr(args, "graph", None) is None
    graph, index, meta = _resolve_graph_index(args, allow_default_state=allow_default_state)
    state_path = _resolve_state_path(args.state, allow_default=allow_default_state)
    llm_fn, llm_batch_fn = _resolve_llm(args)
    suggestions = suggest_connections(graph, llm_fn=llm_fn, llm_batch_fn=llm_batch_fn)
    added = apply_connections(graph=graph, connections=suggestions)
    if state_path is not None:
        state_meta = _state_meta(meta)
        save_state(graph=graph, index=index or VectorIndex(), path=state_path, meta=state_meta)
    else:
        _write_graph(args.graph, graph)
    payload = {
        "suggestions": [
            {"source_id": s, "target_id": t, "weight": w, "reason": r} for s, t, w, r in suggestions
        ],
        "added": added,
    }
    print(json.dumps(payload) if args.json else f"Added edges: {added}")
    return 0


def cmd_anchor(args: argparse.Namespace) -> int:
    """cmd anchor."""
    allow_default_state = getattr(args, "graph", None) is None
    graph, index, meta = _resolve_graph_index(args, allow_default_state=allow_default_state)
    state_path = _resolve_state_path(args.state, allow_default=allow_default_state)
    if args.list:
        nodes = [
            {"node_id": node.id, "authority": node.metadata.get("authority", "overlay")}
            for node in graph.nodes()
            if node.metadata.get("authority") in {"constitutional", "canonical"}
        ]
        payload = {"nodes": nodes, "count": len(nodes)}
        print(json.dumps(payload, indent=2) if args.json else "\n".join(f"{node['node_id']}: {node['authority']}" for node in nodes) or "No anchored nodes.")
        return 0

    if not args.node_id:
        raise SystemExit("--node-id required unless --list is set")
    node = graph.get_node(args.node_id)
    if node is None:
        raise SystemExit(f"node not found: {args.node_id}")

    current_authority = node.metadata.get("authority", "overlay")
    if args.remove:
        if "authority" in node.metadata:
            node.metadata.pop("authority", None)
            current_authority = "overlay"
            if state_path is not None:
                save_state(graph=graph, index=index or VectorIndex(), path=state_path, meta=_state_meta(meta))
        payload = {"node_id": args.node_id, "authority": current_authority}
        print(json.dumps(payload) if args.json else f"{args.node_id} authority: {current_authority}")
        return 0

    if args.authority:
        node.metadata["authority"] = args.authority
        current_authority = args.authority
        if state_path is not None:
            save_state(graph=graph, index=index or VectorIndex(), path=state_path, meta=_state_meta(meta))

    payload = {"node_id": args.node_id, "authority": current_authority}
    print(json.dumps(payload) if args.json else f"{args.node_id} authority: {current_authority}")


def cmd_compact(args: argparse.Namespace) -> int:
    """cmd compact."""
    state_path = _resolve_state_path(args.state, allow_default=True)
    if state_path is None:
        raise SystemExit("--state is required for compact")

    _, _, meta = _resolve_graph_index(args, allow_default_state=True)

    embed_args = SimpleNamespace(embedder=None)
    embed_fn, _, _, _, _ = _resolve_embedder(embed_args, meta)
    llm_fn, _ = _resolve_llm(args)

    if embed_fn is None:
        raise SystemExit("embedding callback missing")

    report = compact_daily_notes(
        state_path=state_path,
        memory_dir=args.memory_dir,
        max_age_days=args.max_age_days,
        target_lines=args.target_lines,
        embed_fn=embed_fn,
        llm_fn=llm_fn,
        journal_path=_resolve_journal_path(args, allow_default_state=True),
        dry_run=args.dry_run,
    )
    if args.json:
        print(json.dumps(asdict(report), indent=2))
    else:
        print(f"Compaction report for {state_path}")
        print(f"  scanned: {report.files_scanned}")
        print(f"  compacted: {report.files_compacted}")
        print(f"  skipped: {report.files_skipped}")
        print(f"  nodes_injected: {report.nodes_injected}")
        print(f"  lines: {report.lines_before} -> {report.lines_after}")
    return 0


def cmd_reembed(args: argparse.Namespace) -> int:
    """Rebuild embeddings for every node and rewrite index + embedder metadata."""
    state_path = _resolve_state_path(args.state, allow_default=False)
    if state_path is None:
        raise SystemExit("--state is required for reembed")

    graph, _index, meta = load_state(state_path)
    embed_fn, embed_batch_fn, embedder_name, embedder_dim, embedder_model = _resolve_embedder(args, meta)

    if embedder_name == HashEmbedder().name:
        raise SystemExit("hash embedder is not permitted for reembed; use --embedder local")

    texts = [(node.id, node.content) for node in graph.nodes()]
    vectors: dict[str, list[float]] = {}
    total = len(texts)
    batch_size = 128
    start = time.perf_counter()
    for offset in range(0, total, batch_size):
        chunk = texts[offset : offset + batch_size]
        if embed_batch_fn is not None:
            batch_vectors = embed_batch_fn(chunk)
        else:
            batch_vectors = batch_or_single_embed(
                chunk,
                embed_fn=embed_fn,
                embed_batch_fn=None,
            )
        vectors.update(batch_vectors)
        if not args.json:
            completed = min(offset + len(chunk), total)
            pct = (100.0 * completed / total) if total > 0 else 100.0
            elapsed_seconds = time.perf_counter() - start
            rate = (completed / elapsed_seconds) if elapsed_seconds > 0 else 0.0
            eta_seconds = ((total - completed) / rate) if rate > 0 else None
            extras: list[str] = [
                f"elapsed={float(elapsed_seconds):.1f}s",
                f"rate={float(rate):.1f}/s",
            ]
            if isinstance(eta_seconds, (int, float)):
                extras.append(f"eta={float(eta_seconds):.1f}s")
            extra_text = f" {' '.join(extras)}" if extras else ""
            print(
                f"[reembed] {completed}/{total} ({pct:.1f}%)" + extra_text,
                file=sys.stderr,
            )
    if len(vectors) != graph.node_count():
        raise SystemExit(
            f"reembed failed: expected {graph.node_count()} embeddings, got {len(vectors)}"
        )

    index = VectorIndex()
    for node_id, vector in vectors.items():
        index.upsert(node_id, vector)

    state_meta = dict(meta)
    state_meta["embedder_name"] = embedder_name
    state_meta["embedder_dim"] = embedder_dim
    if embedder_model is not None:
        state_meta["embedder_model"] = embedder_model
    _persist_state(
        graph=graph,
        index=index,
        meta=state_meta,
        state_path=state_path,
        backup=bool(args.backup),
    )

    payload = {
        "state": state_path,
        "nodes": graph.node_count(),
        "embedder_name": state_meta.get("embedder_name"),
        "embedder_dim": state_meta.get("embedder_dim"),
        "embedder_model": state_meta.get("embedder_model"),
        "backup": bool(args.backup),
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(
            "reembed ok: "
            f"nodes={payload['nodes']} "
            f"embedder={payload['embedder_name']} "
            f"dim={payload['embedder_dim']}"
        )
    return 0


def cmd_inject(args: argparse.Namespace) -> int:
    """cmd inject."""
    graph, index, meta = _resolve_graph_index(args, allow_default_state=True)
    state_path = _resolve_state_path(args.state, allow_default=True)
    if state_path is None:
        raise SystemExit("--state is required for inject")
    if index is None:
        index = VectorIndex()
    embed_fn, _, embedder_name, _, _ = _resolve_embedder(args, meta)

    if args.vector_stdin:
        vector = _load_query_vector_from_stdin()
    else:
        if embedder_name == HashEmbedder().name:
            _ensure_hash_embedder_compat(meta)
        vector = None

    if args.summary is None:
        from ._util import _first_line

        summary = _first_line(args.content)
    else:
        summary = args.summary

    if args.connect_min_sim is not None:
        connect_min_sim = args.connect_min_sim
    else:
        connect_min_sim = 0.0 if embedder_name == "hash-v1" else 0.3

    node_type = args.type
    metadata = {"source": "cli_inject", "type": node_type}
    if node_type == "CORRECTION":
        payload = inject_correction(
            graph=graph,
            index=index,
            node_id=args.id,
            content=args.content,
            summary=summary,
            metadata=metadata,
            vector=vector,
            embed_fn=None if args.vector_stdin else embed_fn,
            connect_top_k=args.connect_top_k,
            connect_min_sim=connect_min_sim,
        )
    else:
        payload = inject_node(
            graph=graph,
            index=index,
            node_id=args.id,
            content=args.content,
            summary=summary,
            metadata=metadata,
            vector=vector,
            embed_fn=None if args.vector_stdin else embed_fn,
            connect_top_k=args.connect_top_k,
            connect_min_sim=connect_min_sim,
        )

    state_meta = _state_meta(
        meta,
    )
    save_state(graph=graph, index=index, path=state_path, meta=state_meta)

    print(json.dumps(payload, indent=2) if args.json else f"Injected {payload['node_id']}")
    return 0


def cmd_self_correct(args: argparse.Namespace) -> int:
    """cmd self-correct."""
    graph, index, meta = _resolve_graph_index(args, allow_default_state=False)
    state_path = _resolve_state_path(args.state, allow_default=False)
    if state_path is None:
        raise SystemExit("--state is required for self-correct")
    if index is None:
        index = VectorIndex()
    embed_fn, _, embedder_name, _, _ = _resolve_embedder(args, meta)
    if embedder_name == HashEmbedder().name:
        _ensure_hash_embedder_compat(meta)

    edges_updated = 0
    fired_ids = [value.strip() for value in args.fired_ids.split(",") if value.strip()]
    if fired_ids and args.outcome != 0:
        updates = apply_outcome_pg(
            graph=graph,
            fired_nodes=fired_ids,
            outcome=args.outcome,
            baseline=0.0,
            temperature=1.0,
        )
        edges_updated = len(updates)
        log_learn(
            fired_ids=fired_ids,
            outcome=args.outcome,
            journal_path=_resolve_journal_path(args, allow_default_state=True),
            metadata={"source": "self"},
        )

    from .daemon import _correction_node_id

    summary = args.content.split("\n", 1)[0] or args.content
    node_id = _correction_node_id(args.content)
    node_existed = graph.get_node(node_id) is not None

    metadata = {"source": "self", "type": args.node_type, "auto": True}

    if args.node_type == "CORRECTION":
        payload = inject_correction(
            graph=graph,
            index=index,
            node_id=node_id,
            content=args.content,
            summary=summary,
            metadata=metadata,
            embed_fn=embed_fn,
            connect_top_k=3,
            connect_min_sim=0.0 if embedder_name == HashEmbedder().name else 0.3,
        )
    else:
        payload = inject_node(
            graph=graph,
            index=index,
            node_id=node_id,
            content=args.content,
            summary=summary,
            metadata=metadata,
            embed_fn=embed_fn,
            connect_top_k=3,
            connect_min_sim=0.0 if embedder_name == HashEmbedder().name else 0.3,
        )

    state_meta = _state_meta(meta)
    save_state(graph=graph, index=index, path=state_path, meta=state_meta)

    output = {
        "node_id": payload["node_id"],
        "node_injected": not node_existed,
        "edges_updated": edges_updated,
        "fired_ids_penalized": fired_ids,
    }

    print(json.dumps(output, indent=2) if args.json_output else f"Self-correct updated {state_path}")
    return 0


def _checkpoint_time_display(value: object) -> str:
    """Format checkpoint timestamps for human output."""
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()
        except (OverflowError, OSError, ValueError):
            return str(value)
    if value is None:
        return "n/a"
    return str(value)


def _replay_phase_plan(
    *,
    run_fast: bool,
    run_full: bool,
    stop_after_fast_learning: bool,
) -> list[str]:
    """Return replay phases that will run for the current arguments."""
    phases: list[str] = []
    if run_fast or run_full:
        phases.append("fast_learning")
    if not stop_after_fast_learning:
        phases.append("replay")
    if run_full and not stop_after_fast_learning:
        phases.append("harvest")
    return phases


def _build_checkpoint_status_payload(
    *,
    checkpoint_path: str,
    checkpoint_data: dict[str, object],
    run_fast: bool,
    run_full: bool,
    stop_after_fast_learning: bool,
    resume_requested: bool,
    ignore_checkpoint: bool,
) -> tuple[dict[str, object], bool, bool]:
    """Build stable checkpoint status payload for replay CLI output."""
    fast_offsets, fast_legacy = _checkpoint_phase_offsets(checkpoint_data, phase="fast_learning")
    replay_offsets, replay_legacy = _checkpoint_phase_offsets(checkpoint_data, phase="replay")
    if fast_legacy and fast_offsets:
        warnings.warn(
            "fast_learning checkpoint missing phase-scoped sessions; falling back to legacy top-level 'sessions' offsets",
            stacklevel=2,
        )
    if replay_legacy and replay_offsets:
        warnings.warn(
            "replay checkpoint missing phase-scoped sessions; falling back to legacy top-level 'sessions' offsets",
            stacklevel=2,
        )

    fast_raw = checkpoint_data.get("fast_learning")
    if not isinstance(fast_raw, dict):
        fast_raw = {}
    replay_raw = checkpoint_data.get("replay")
    if not isinstance(replay_raw, dict):
        replay_raw = {}

    phase_plan = _replay_phase_plan(
        run_fast=run_fast,
        run_full=run_full,
        stop_after_fast_learning=stop_after_fast_learning,
    )
    resume_would_take_effect = (
        resume_requested
        and not ignore_checkpoint
        and (bool(fast_offsets) or bool(replay_offsets))
    )

    payload: dict[str, object] = {
        "type": "checkpoint_status",
        "checkpoint_path": checkpoint_path,
        "schema_version": int(checkpoint_data.get("version", 1)),
        "resume": {
            "requested": bool(resume_requested),
            "ignore_checkpoint": bool(ignore_checkpoint),
            "would_take_effect": bool(resume_would_take_effect),
        },
        "phases": phase_plan,
        "fast_learning": {
            "status": str(fast_raw.get("status", "unknown")),
            "windows_processed": int(fast_raw.get("windows_processed", 0))
            if isinstance(fast_raw.get("windows_processed"), (int, float))
            else 0,
            "windows_total": int(fast_raw.get("windows_total", 0))
            if isinstance(fast_raw.get("windows_total"), (int, float))
            else 0,
            "updated_at": fast_raw.get("updated_at"),
            "session_offsets_count": len(fast_offsets),
            "legacy_offsets_fallback": bool(fast_legacy and fast_offsets),
        },
        "replay": {
            "queries_processed": int(replay_raw.get("queries_processed", 0)) if isinstance(replay_raw.get("queries_processed"), (int, float)) else 0,
            "queries_total": int(replay_raw.get("queries_total", 0)) if isinstance(replay_raw.get("queries_total"), (int, float)) else 0,
            "merge_batches": int(replay_raw.get("merge_batches", 0)) if isinstance(replay_raw.get("merge_batches"), (int, float)) else 0,
            "updated_at": replay_raw.get("updated_at"),
            "session_offsets_count": len(replay_offsets),
            "legacy_offsets_fallback": bool(replay_legacy and replay_offsets),
        },
    }
    for counter_name in (
        "windows_candidate",
        "windows_sent_to_llm",
        "windows_skipped_low_signal",
        "windows_skipped_existing_pointer",
    ):
        value = fast_raw.get(counter_name)
        if isinstance(value, (int, float)):
            payload["fast_learning"][counter_name] = int(value)
    return payload, bool(fast_legacy and fast_offsets), bool(replay_legacy and replay_offsets)


def cmd_replay(args: argparse.Namespace) -> int:
    """cmd replay."""
    state_path = _resolve_state_path(args.state, allow_default=True)
    mode, mode_defaulted = _resolve_replay_mode(args)
    run_fast = mode in {"fast-learning", "full"}
    run_full = mode == "full"
    ignore_checkpoint = bool(args.fresh or args.ignore_checkpoint)

    checkpoint_path = args.checkpoint or (str(default_checkpoint_path(str(state_path))) if state_path is not None else None)
    phase_plan = _replay_phase_plan(
        run_fast=run_fast,
        run_full=run_full,
        stop_after_fast_learning=bool(args.stop_after_fast_learning),
    )

    if args.show_checkpoint:
        if checkpoint_path is None:
            raise SystemExit("--checkpoint or --state is required for --show-checkpoint")
        checkpoint_data = _load_checkpoint(checkpoint_path)
        status_payload, _, _ = _build_checkpoint_status_payload(
            checkpoint_path=checkpoint_path,
            checkpoint_data=checkpoint_data,
            run_fast=run_fast,
            run_full=run_full,
            stop_after_fast_learning=bool(args.stop_after_fast_learning),
            resume_requested=bool(args.resume),
            ignore_checkpoint=ignore_checkpoint,
        )
        if args.json:
            print(json.dumps(status_payload, indent=2))
            return 0
        fast_status = status_payload["fast_learning"] if isinstance(status_payload.get("fast_learning"), dict) else {}
        replay_status = status_payload["replay"] if isinstance(status_payload.get("replay"), dict) else {}
        resume_status = status_payload["resume"] if isinstance(status_payload.get("resume"), dict) else {}
        print(f"Checkpoint: {status_payload['checkpoint_path']}")
        print(f"Schema version: {status_payload.get('schema_version', 1)}")
        print(
            "Fast learning: "
            f"{fast_status.get('windows_processed', 0)}/{fast_status.get('windows_total', 0)} windows, "
            f"status={fast_status.get('status', 'unknown')}, "
            f"updated_at={_checkpoint_time_display(fast_status.get('updated_at'))}"
        )
        fast_counter_bits = []
        for counter_name in (
            "windows_candidate",
            "windows_sent_to_llm",
            "windows_skipped_low_signal",
            "windows_skipped_existing_pointer",
        ):
            if counter_name in fast_status:
                fast_counter_bits.append(f"{counter_name}={fast_status[counter_name]}")
        if fast_counter_bits:
            print(f"  Fast learning counters: {', '.join(fast_counter_bits)}")
        print(
            "Replay: "
            f"{replay_status.get('queries_processed', 0)}/{replay_status.get('queries_total', 0)} queries, "
            f"merge_batches={replay_status.get('merge_batches', 0)}, "
            f"updated_at={_checkpoint_time_display(replay_status.get('updated_at'))}"
        )
        print(
            "Resume would take effect: "
            f"{bool(resume_status.get('would_take_effect', False))} "
            f"(resume={bool(resume_status.get('requested', False))}, ignore_checkpoint={bool(resume_status.get('ignore_checkpoint', False))})"
        )
        print(f"Mode: {mode}")
        print(f"Phases: {', '.join(phase_plan) if phase_plan else 'none'}")
        return 0

    if not args.sessions:
        raise SystemExit("--sessions is required unless --show-checkpoint is set")
    if (run_fast or run_full) and state_path is None:
        raise SystemExit("--state is required for fast/full learning replay mode")

    graph, index, meta = _resolve_graph_index(args, allow_default_state=True)
    tool_result_allowlist = _parse_tool_result_allowlist(args.tool_result_allowlist)
    tool_result_max_chars = max(0, int(args.tool_result_max_chars))
    include_tool_results = bool(args.include_tool_results)

    if not args.json:
        print("Replay startup:", file=sys.stderr)
        print(f"  mode: {mode}", file=sys.stderr)
        print(f"  checkpoint: {checkpoint_path if checkpoint_path is not None else 'none'}", file=sys.stderr)
        print(f"  resume: {bool(args.resume)}", file=sys.stderr)
        print(f"  fresh: {ignore_checkpoint}", file=sys.stderr)
        print(f"  phases: {', '.join(phase_plan) if phase_plan else 'none'}", file=sys.stderr)

    checkpoint_every_windows = max(0, int(args.checkpoint_every))
    checkpoint_every_seconds = max(0, int(args.checkpoint_every_seconds))
    persist_state_every_seconds = max(0, int(args.persist_state_every_seconds))
    progress_every = max(0, int(args.progress_every))
    replay_workers = max(1, int(args.replay_workers))
    quiet = bool(args.quiet)
    timed_progress_interval_seconds = 30
    if mode_defaulted and not quiet:
        print(
            "Replay mode note: no mode specified; defaulting to --mode full.",
            file=sys.stderr,
        )

    def _emit_progress(payload: dict[str, object]) -> None:
        if quiet:
            return
        if args.json:
            print(json.dumps(payload))
            return
        completed = int(payload.get("completed", 0))
        total = int(payload.get("total", 0))
        pct = (100.0 * completed / total) if total > 0 else 100.0
        phase = payload.get("phase", "replay")
        elapsed_seconds = payload.get("elapsed_seconds")
        rate = payload.get("rate")
        eta_seconds = payload.get("eta_seconds")
        extras: list[str] = []
        if isinstance(elapsed_seconds, (int, float)):
            extras.append(f"elapsed={float(elapsed_seconds):.1f}s")
        if isinstance(rate, (int, float)):
            extras.append(f"rate={float(rate):.2f}/s")
        if isinstance(eta_seconds, (int, float)):
            extras.append(f"eta={float(eta_seconds):.1f}s")
        extra_text = f" {' '.join(extras)}" if extras else ""
        print(f"[{phase}] {completed}/{total} ({pct:.1f}%){extra_text}", file=sys.stderr)

    fast_stats: dict[str, object] | None = None
    llm_fn, _ = _resolve_llm(args)
    if run_fast or run_full:
        fast_stats = run_fast_learning(
            state_path=str(state_path),
            session_paths=args.sessions,
            workers=max(1, args.workers),
            window_radius=max(1, args.window_radius),
            max_windows=max(1, args.max_windows),
            hard_max_turns=max(1, args.hard_max_turns),
            checkpoint_path=checkpoint_path,
            resume=bool(args.resume),
            ignore_checkpoint=ignore_checkpoint,
            backup=bool(args.backup),
            include_tool_results=include_tool_results,
            tool_result_allowlist=tool_result_allowlist,
            tool_result_max_chars=tool_result_max_chars,
            checkpoint_every=checkpoint_every_windows,
            checkpoint_every_seconds=checkpoint_every_seconds,
            on_progress=_emit_progress if progress_every > 0 else None,
            progress_every_windows=progress_every,
            progress_every_seconds=10 if progress_every > 0 else 0,
            llm_fn=llm_fn,
        )
        graph, index, meta = load_state(str(state_path))
        if args.stop_after_fast_learning:
            output: dict[str, object] = {
                "stopped_after_fast_learning": True,
                "fast_learning": fast_stats,
            }
            if args.json:
                print(json.dumps(output, indent=2))
            elif not quiet:
                print("Completed fast-learning; stopped before replay/harvest.")
            return 0

    checkpoint_data = _load_checkpoint(checkpoint_path) if (checkpoint_path and args.resume and not ignore_checkpoint) else {"version": 1}
    replay_since_lines, replay_legacy_fallback = _checkpoint_phase_offsets(checkpoint_data, phase="replay")
    if replay_legacy_fallback and replay_since_lines:
        warnings.warn(
            "replay checkpoint missing phase-scoped sessions; falling back to legacy top-level 'sessions' offsets",
            stacklevel=2,
        )
    if ignore_checkpoint or not args.resume:
        replay_since_lines = {}

    interactions, replay_offsets = load_interactions_for_replay(
        args.sessions,
        since_lines=replay_since_lines if args.resume and not ignore_checkpoint else {},
        include_tool_results=include_tool_results,
        tool_result_allowlist=tool_result_allowlist,
        tool_result_max_chars=tool_result_max_chars,
    )
    interactions, filter_summary = _filter_replay_interactions(
        interactions,
        now_ts=time.time(),
        since_hours=args.replay_since_hours,
        max_interactions=args.replay_max_interactions,
        sample_rate=float(args.replay_sample_rate),
        priority=str(args.replay_priority),
    )
    if not quiet:
        print(
            f"Loaded {filter_summary['loaded_total']} interactions from session files",
            file=sys.stderr,
        )
        print(
            "Replay filter summary: "
            f"loaded_total={filter_summary['loaded_total']} "
            f"after_priority={filter_summary['after_priority']} "
            f"after_since={filter_summary['after_since']} "
            f"after_sample={filter_summary['after_sample']} "
            f"after_max={filter_summary['after_max']}",
            file=sys.stderr,
        )
    filtering_active = (
        args.replay_since_hours is not None
        or args.replay_max_interactions is not None
        or float(args.replay_sample_rate) < 1.0
        or args.replay_priority == "tool"
    )
    if filtering_active and not bool(args.advance_offsets_on_skip):
        warnings.warn(
            "Replay filtering is active with --advance-offsets-on-skip disabled. "
            "Subsequent --resume runs will require repeated runs to eventually cover all interactions.",
            stacklevel=2,
        )
    auto_decay = bool(args.decay_during_replay) or run_full
    decay_interval = max(1, args.decay_interval)

    total_interactions = len(interactions)
    processed_interactions = 0
    progress_mark = 0
    last_timed_progress_at = time.monotonic()
    merge_batches = 0
    state_dirty = False
    replay_offsets_done = dict(
        replay_offsets
        if (bool(args.advance_offsets_on_skip) and filtering_active)
        else replay_since_lines
    )
    last_checkpoint_at = time.monotonic()
    last_persist_at = time.monotonic()
    replay_latest_ts: float | None = None
    replay_latest_ts_source: str | None = None
    stats = {
        "queries_replayed": 0,
        "edges_reinforced": 0,
        "cross_file_edges_created": 0,
        "last_replayed_ts": None,
        "last_replayed_ts_source": None,
    }

    def _update_offsets_from_batch(batch: list[dict[str, object]]) -> None:
        for item in batch:
            source = item.get("source")
            line_no = item.get("line_no")
            if not isinstance(source, str) or not isinstance(line_no, (int, float)):
                continue
            prev = replay_offsets_done.get(source, 0)
            if int(line_no) > prev:
                replay_offsets_done[source] = int(line_no)

    def _checkpoint_if_due(force: bool = False) -> None:
        nonlocal last_checkpoint_at
        if not checkpoint_path:
            return
        due_by_count = checkpoint_every_windows > 0 and merge_batches > 0 and merge_batches % checkpoint_every_windows == 0
        due_by_time = checkpoint_every_seconds > 0 and (time.monotonic() - last_checkpoint_at) >= checkpoint_every_seconds
        if not force and not (due_by_count or due_by_time):
            return
        _save_checkpoint(
            checkpoint_path,
            phase="replay",
            session_offsets=replay_offsets_done,
            extra={
                "queries_processed": processed_interactions,
                "queries_total": total_interactions,
                "merge_batches": merge_batches,
                "updated_at": time.time(),
            },
        )
        last_checkpoint_at = time.monotonic()

    def _persist_if_due(force: bool = False) -> None:
        nonlocal last_persist_at, state_dirty
        if state_path is None or not state_dirty:
            return
        due = force or (persist_state_every_seconds > 0 and (time.monotonic() - last_persist_at) >= persist_state_every_seconds)
        if not due:
            return
        state_meta = _state_meta(meta)
        if replay_latest_ts is not None:
            state_meta["last_replayed_ts"] = replay_latest_ts
            if replay_latest_ts_source is not None:
                state_meta["last_replayed_ts_source"] = replay_latest_ts_source
        _persist_state(
            graph=graph,
            index=index if index is not None else VectorIndex(),
            meta=state_meta,
            state_path=str(state_path),
            backup=bool(args.backup),
        )
        last_persist_at = time.monotonic()
        state_dirty = False

    def _emit_periodic_progress() -> None:
        nonlocal progress_mark
        if progress_every <= 0:
            return
        while processed_interactions - progress_mark >= progress_every:
            progress_mark += progress_every
            _emit_progress(
                {
                    "type": "progress",
                    "phase": "replay",
                    "completed": processed_interactions,
                    "total": total_interactions,
                    "merge_batches": merge_batches,
                }
            )

    def _emit_timed_progress(force: bool = False) -> None:
        nonlocal last_timed_progress_at
        if args.json or quiet or total_interactions <= 0:
            return
        now = time.monotonic()
        if not force and (now - last_timed_progress_at) < timed_progress_interval_seconds:
            return
        last_timed_progress_at = now
        _emit_progress(
            {
                "type": "progress",
                "phase": "replay",
                "completed": processed_interactions,
                "total": total_interactions,
                "merge_batches": merge_batches,
            }
        )

    if replay_workers > 1:
        merge_every = checkpoint_every_windows if checkpoint_every_windows > 0 else 50

        def _on_merge(event: dict[str, object]) -> None:
            nonlocal processed_interactions, merge_batches, state_dirty, replay_latest_ts, replay_latest_ts_source
            merged_queries = int(event.get("merged_queries", processed_interactions))
            merge_batches = int(event.get("merge_batches", merge_batches))
            processed_interactions = merged_queries
            last_ts = event.get("last_replayed_ts")
            if isinstance(last_ts, (int, float)):
                candidate_ts = float(last_ts)
                if replay_latest_ts is None or candidate_ts >= replay_latest_ts:
                    replay_latest_ts = candidate_ts
                    source = event.get("last_replayed_ts_source")
                    if isinstance(source, str):
                        replay_latest_ts_source = source
            state_dirty = True
            _checkpoint_if_due()
            _persist_if_due()
            _emit_periodic_progress()
            _emit_timed_progress()

        parallel_stats = replay_queries_parallel(
            graph=graph,
            queries=interactions,
            workers=replay_workers,
            merge_every=merge_every,
            verbose=(not args.json and not quiet),
            auto_decay=auto_decay,
            decay_interval=decay_interval,
            on_merge=_on_merge,
            tool_edges=bool(args.tool_edges),
        )
        stats["queries_replayed"] = int(parallel_stats.get("queries_replayed", 0))
        stats["edges_reinforced"] = int(parallel_stats.get("edges_reinforced", 0))
        stats["cross_file_edges_created"] = int(parallel_stats.get("cross_file_edges_created", 0))
        stats["last_replayed_ts"] = parallel_stats.get("last_replayed_ts")
        stats["last_replayed_ts_source"] = parallel_stats.get("last_replayed_ts_source")
        merge_batches = int(parallel_stats.get("merge_batches", merge_batches))
        if isinstance(stats["last_replayed_ts"], (int, float)):
            replay_latest_ts = float(stats["last_replayed_ts"])
            if isinstance(stats["last_replayed_ts_source"], str):
                replay_latest_ts_source = stats["last_replayed_ts_source"]
        processed_interactions = int(stats["queries_replayed"])
        state_dirty = bool(processed_interactions)
        replay_offsets_done.update({key: int(value) for key, value in replay_offsets.items()})
        stats["replay_workers"] = replay_workers
        stats["merge_batches"] = merge_batches
    else:
        window_size = checkpoint_every_windows if checkpoint_every_windows > 0 else 50
        for start in range(0, len(interactions), window_size):
            batch = interactions[start : start + window_size]
            replay_batch = replay_queries(
                graph=graph,
                queries=batch,
                verbose=False,
                auto_decay=auto_decay,
                decay_interval=decay_interval,
                tool_edges=bool(args.tool_edges),
            )
            merge_batches += 1
            processed_interactions += len(batch)
            _update_offsets_from_batch(batch)
            stats["queries_replayed"] += int(replay_batch.get("queries_replayed", 0))
            stats["edges_reinforced"] += int(replay_batch.get("edges_reinforced", 0))
            stats["cross_file_edges_created"] += int(replay_batch.get("cross_file_edges_created", 0))
            batch_last_ts = replay_batch.get("last_replayed_ts")
            if isinstance(batch_last_ts, (int, float)):
                candidate_ts = float(batch_last_ts)
                if replay_latest_ts is None or candidate_ts >= replay_latest_ts:
                    replay_latest_ts = candidate_ts
                    batch_last_ts_source = replay_batch.get("last_replayed_ts_source")
                    if isinstance(batch_last_ts_source, str):
                        replay_latest_ts_source = batch_last_ts_source
            stats["last_replayed_ts"] = replay_latest_ts
            stats["last_replayed_ts_source"] = replay_latest_ts_source
            state_dirty = state_dirty or bool(replay_batch.get("queries_replayed", 0))
            _checkpoint_if_due()
            _persist_if_due()
            _emit_periodic_progress()
            _emit_timed_progress()

    _checkpoint_if_due(force=True)
    _persist_if_due(force=True)
    _emit_timed_progress(force=True)

    harvest_stats: dict[str, object] | None = None
    if run_full:
        harvest_stats = run_harvest(
            state_path=str(state_path),
            tasks=["decay", "scale", "split", "merge", "soft_prune", "prune", "connect"],
            backup=bool(args.backup),
        )
        graph, index, meta = load_state(str(state_path))

    if state_path is not None:
        state_meta = _state_meta(meta)
        if stats.get("last_replayed_ts") is not None:
            state_meta["last_replayed_ts"] = stats["last_replayed_ts"]
            source = stats.get("last_replayed_ts_source")
            if isinstance(source, str):
                state_meta["last_replayed_ts_source"] = source
        _persist_state(
            graph=graph,
            index=index if index is not None else VectorIndex(),
            meta=state_meta,
            state_path=str(state_path),
            backup=bool(args.backup),
        ) if stats.get("queries_replayed") else None
    else:
        _write_graph(args.graph, graph)

    log_replay(
        queries_replayed=stats["queries_replayed"],
        edges_reinforced=stats["edges_reinforced"],
        cross_file_created=stats["cross_file_edges_created"],
        journal_path=_resolve_journal_path(args, allow_default_state=True),
    )

    if args.traces_out:
        replay_traces = [
            RouteTrace(
                query_id=f"replay:{idx}",
                ts=float(item.get("ts", 0.0) or 0.0),
                query_text=str(item.get("query", "") or ""),
                chat_id=str(item["chat_id"]) if isinstance(item.get("chat_id"), str) else None,
                seeds=[],
                fired_nodes=[],
                traversal_config={},
                route_policy={"route_mode": "off"},
                query_vector=None,
                decision_points=[],
            )
            for idx, item in enumerate(interactions)
            if isinstance(item, dict)
        ]
        _write_route_traces_jsonl(str(args.traces_out), replay_traces)

    if args.labels_out:
        write_labels_jsonl(str(args.labels_out), [])

    output: dict[str, object] = dict(stats)
    if fast_stats is not None:
        output["fast_learning"] = fast_stats
    if harvest_stats is not None:
        output["harvest"] = harvest_stats

    if args.json:
        print(json.dumps(output, indent=2))
        return 0

    message_parts = [
        f"Replayed {stats['queries_replayed']}/{len(interactions)} queries, {stats['cross_file_edges_created']} cross-file edges created"
    ]
    if run_fast:
        message_parts.append(
            f"learning events: {fast_stats['events_injected']} injected, {fast_stats['events_appended']} appended"
            if fast_stats is not None
            else ""
        )
    if run_full:
        maintenance = harvest_stats["maintenance"] if harvest_stats is not None else {}
        message_parts.append(
            f"harvest: tasks={len(maintenance.get('tasks_run', [])) if isinstance(maintenance, dict) else 0}, "
            f"damped_edges={harvest_stats.get('damped_edges', 0) if harvest_stats is not None else 0}"
        )
    print("\n".join(part for part in message_parts if part))
    return 0


def cmd_harvest(args: argparse.Namespace) -> int:
    """cmd harvest."""
    state_path = _resolve_state_path(args.state, allow_default=False)
    if state_path is None:
        raise SystemExit("--state is required for harvest")

    requested_tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    report = run_harvest(
        state_path=state_path,
        events_path=args.events,
        tasks=requested_tasks or None,
        dry_run=args.dry_run,
        max_merges=args.max_merges,
        prune_below=args.prune_below,
        backup=bool(args.backup),
    )

    events = event_log_entries(Path(report["learning_events_path"]))
    if args.traces_out:
        harvest_traces = [
            RouteTrace(
                query_id=f"harvest:{idx}",
                ts=float(event.get("ts", 0.0) or 0.0),
                query_text=str(event.get("content", "") or ""),
                chat_id=None,
                seeds=[],
                fired_nodes=[],
                traversal_config={},
                route_policy={"route_mode": "off"},
                query_vector=None,
                decision_points=[],
            )
            for idx, event in enumerate(events)
            if isinstance(event, dict)
        ]
        _write_route_traces_jsonl(str(args.traces_out), harvest_traces)

    if args.labels_out:
        labels: list[LabelRecord] = []
        for idx, event in enumerate(events):
            if not isinstance(event, dict):
                continue
            event_type = str(event.get("type", "")).upper()
            query_id = str(event.get("session_pointer", event.get("session", f"harvest:{idx}")))
            node_id = event.get("node_id")
            if isinstance(node_id, str) and node_id:
                base_score = -1.0 if event_type == "CORRECTION" else 1.0
                labels.append(
                    LabelRecord(
                        query_id=query_id,
                        decision_point_idx=0,
                        candidate_scores={node_id: base_score},
                        reward_source=RewardSource.HARVESTER,
                        weight=1.0,
                        ts=float(event.get("ts", 0.0) or 0.0),
                        metadata={"event_type": event_type},
                    )
                )
                continue
            fallback = from_self_learning_event(
                {
                    "query_id": query_id,
                    "decision_point_idx": 0,
                    "fired_ids": event.get("fired_ids"),
                    "outcome": event.get("outcome", -1.0 if event_type == "CORRECTION" else 1.0),
                    "ts": float(event.get("ts", 0.0) or 0.0),
                    "metadata": {"event_type": event_type},
                }
            )
            if fallback is not None:
                labels.append(fallback)
                continue
            if event_type in {"TEACHING", "REINFORCEMENT"}:
                labels.append(
                    from_teacher_output(
                        query_id=query_id,
                        decision_point_idx=0,
                        teacher_scores={},
                        ts=float(event.get("ts", 0.0) or 0.0),
                        metadata={"event_type": event_type},
                    )
                )
        write_labels_jsonl(str(args.labels_out), labels)

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    maintenance = report["maintenance"]
    print(
        "\n".join(
            [
                "Harvest report:",
                f"  events_seen: {report['events_seen']}",
                f"  correction_nodes: {report['correction_nodes']}",
                f"  damped_edges: {report['damped_edges']}",
                f"  tasks: {', '.join(maintenance['tasks_run']) if maintenance['tasks_run'] else '(none)'}",
                f"  edges: {maintenance['edges_before']} -> {maintenance['edges_after']}",
            ]
        )
    )
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """cmd status."""
    state_path = _resolve_state_path(args.state, allow_default=False)
    if state_path is None:
        raise SystemExit("--state is required")

    graph, index, meta = load_state(state_path)
    payload = _status_payload(state_path, meta, graph, index)
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(
        "\n".join(
            [
                f"OpenClawBrain v{payload['version']}",
                f"State: {payload['state']}",
                f"Nodes: {payload['nodes']} | Edges: {payload['edges']}",
                f"Reflex: {payload['reflex_pct']:.1f}% | Habitual: {payload['habitual_pct']:.1f}% | Dormant: {payload['dormant_pct']:.1f}%",
                f"Inhibitory edges: {payload['inhibitory_edges']}",
                f"Constitutional: {payload['constitutional_nodes']} | Canonical: {payload['canonical_nodes']}",
                f"Decay half-life: {payload['decay_half_life']} (adaptive)",
                f"Last replayed: {payload['last_replayed']}",
                f"Embedder: {payload['embedder_name']} ({payload['embedder_dim']}-dim)",
                f"Daemon: {'running' if payload['daemon_running'] else 'not running'} {payload['daemon_socket']}",
            ]
        )
    )
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    """cmd doctor."""
    state_path = _resolve_state_path(args.state, allow_default=True)
    if state_path is None:
        raise SystemExit("--state is required")

    checks_passed = 0
    checks_total = 0
    failed = False

    checks_total += 1
    python_ok = sys.version_info >= (3, 10)
    checks_passed += _check_result(python_ok, "python_version", f"{sys.version.split()[0]}")
    failed = failed or (not python_ok)

    resolved_state_path = Path(state_path).expanduser()
    checks_total += 1
    state_exists = resolved_state_path.exists()
    checks_passed += _check_result(state_exists, "state_file_exists", str(resolved_state_path))
    failed = failed or (not state_exists)
    if not state_exists:
        print(f"Summary: {checks_passed}/{checks_total} checks passed")
        return 1

    checks_total += 1
    try:
        payload = _load_json(str(resolved_state_path))
        checks_passed += _check_result(True, "state_json_valid", str(resolved_state_path))
    except SystemExit as exc:
        checks_passed += _check_result(False, "state_json_valid", str(exc))
        print(f"Summary: {checks_passed}/{checks_total} checks passed")
        return 1
    graph_payload = payload.get("graph", {})
    checks_total += 1
    if not isinstance(graph_payload, dict):
        checks_passed += _check_result(False, "state_json_valid", "graph payload must be object")
        print(f"Summary: {checks_passed}/{checks_total} checks passed")
        return 1

    graph = _load_graph(str(resolved_state_path))
    index_payload = payload.get("index", {})
    checks_total += 1
    checks_passed += _check_result(
        bool(graph_payload.get("nodes")),
        "graph_has_nodes",
        f"nodes={graph.node_count()}",
    )
    failed = failed or (not graph_payload.get("nodes"))

    checks_total += 1
    checks_passed += _check_result(
        graph.edge_count() > 0,
        "graph_has_edges",
        f"edges={graph.edge_count()}",
    )
    failed = failed or (graph.edge_count() == 0)

    checks_total += 1
    embedder_name = payload.get("meta", {}).get("embedder_name")
    embedder_dim = payload.get("meta", {}).get("embedder_dim")
    checks_passed += _check_result(
        isinstance(embedder_name, str) and isinstance(embedder_dim, int),
        "embedder_metadata_present",
        f"name={embedder_name}, dim={embedder_dim}",
    )
    failed = failed or (not isinstance(embedder_name, str) or not isinstance(embedder_dim, int))

    checks_total += 1
    if not isinstance(embedder_dim, int):
        checks_passed += _check_result(False, "index_dimension_matches_embedder", "missing embedder_dim")
        failed = True
    else:
        if not isinstance(index_payload, dict):
            checks_passed += _check_result(False, "index_dimension_matches_embedder", "index payload must be an object")
            failed = True
        elif not index_payload:
            checks_passed += _check_result(False, "index_dimension_matches_embedder", "missing index payload")
            failed = True
        else:
            index_dims: set[int] = set()
            for node_id, vector in index_payload.items():
                if not isinstance(vector, list):
                    checks_passed += _check_result(False, "index_dimension_matches_embedder", f"{node_id}: not a list")
                    failed = True
                    break
                index_dims.add(len(vector))
            else:
                dim_ok = len(index_dims) == 1 and next(iter(index_dims)) == embedder_dim
                dim_value = next(iter(index_dims), None)
                checks_passed += _check_result(dim_ok, "index_dimension_matches_embedder", f"index_dim={dim_value}")
                failed = failed or (not dim_ok)

    checks_total += 1
    journal_path = _resolve_journal_path(args, allow_default_state=True)
    if journal_path is not None and Path(journal_path).exists():
        try:
            Path(journal_path).open("a", encoding="utf-8").close()
            journal_ok = True
        except OSError:
            journal_ok = False
        checks_passed += _check_result(journal_ok, "journal_writable", str(journal_path))
        failed = failed or (not journal_ok)
    else:
        checks_passed += _check_result(True, "journal_writable", "not present")

    print(f"Summary: {checks_passed}/{checks_total} checks passed")
    return 1 if failed else 0


def cmd_info(args: argparse.Namespace) -> int:
    """cmd info."""
    allow_default_state = getattr(args, "graph", None) is None
    state_path = _resolve_state_path(args.state, allow_default=allow_default_state)
    if state_path is None and args.graph is not None:
        graph = _load_graph(args.graph)
        state_bytes = Path(args.graph).expanduser().stat().st_size
        embedder_name = "n/a"
        embedder_dim = "n/a"
    elif state_path is not None:
        state_file = Path(state_path).expanduser()
        if not state_file.exists():
            raise SystemExit(f"state file not found: {state_file}")
        payload, _, _, graph = _state_payload(state_path)
        meta = payload.get("meta", {})
        state_bytes = state_file.stat().st_size
        embedder_name = meta.get("embedder_name", "n/a")
        embedder_dim = meta.get("embedder_dim", "n/a")
    else:
        raise SystemExit("--state or --graph is required")

    health = measure_health(graph)
    payload = {
        "version": __version__,
        "node_count": graph.node_count(),
        "edge_count": graph.edge_count(),
        "embedder_name": embedder_name,
        "embedder_dim": embedder_dim,
        "dormant_pct": health.dormant_pct * 100,
        "habitual_pct": health.habitual_pct * 100,
        "reflex_pct": health.reflex_pct * 100,
        "journal_entry_count": _journal_entry_count(_resolve_journal_path(args, allow_default_state=allow_default_state)),
        "state_file_size": state_bytes,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print("\n".join(
        [
            f"version: {payload['version']}",
            f"node_count: {payload['node_count']}",
            f"edge_count: {payload['edge_count']}",
            f"embedder_name: {payload['embedder_name']}",
            f"embedder_dim: {payload['embedder_dim']}",
            f"dormant_pct: {payload['dormant_pct']:.2f}",
            f"habitual_pct: {payload['habitual_pct']:.2f}",
            f"reflex_pct: {payload['reflex_pct']:.2f}",
            f"journal_entry_count: {payload['journal_entry_count'] if payload['journal_entry_count'] is not None else 'n/a'}",
            f"state_file_size: {payload['state_file_size']}",
        ]
    ))
    return 0


def _parse_authority_map(raw: str | None) -> dict[str, str]:
    """Parse authority map json string."""
    if raw is None:
        return dict(DEFAULT_AUTHORITY_MAP)
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise SystemExit("--authority-map must be a JSON object")

    parsed: dict[str, str] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise SystemExit("--authority-map must map string keys to string values")
        parsed[key] = value
    return parsed


def _sum_sync_reports(reports: dict[str, SyncReport]) -> SyncReport:
    """Aggregate per-workspace sync reports."""
    totals = SyncReport(
        nodes_added=0,
        nodes_updated=0,
        nodes_removed=0,
        nodes_unchanged=0,
        nodes_metadata_updated=0,
        embeddings_computed=0,
        authority_set={},
    )
    for report in reports.values():
        totals.nodes_added += report.nodes_added
        totals.nodes_updated += report.nodes_updated
        totals.nodes_removed += report.nodes_removed
        totals.nodes_unchanged += report.nodes_unchanged
        totals.nodes_metadata_updated += report.nodes_metadata_updated
        totals.embeddings_computed += report.embeddings_computed
        for key, value in report.authority_set.items():
            totals.authority_set[key] = totals.authority_set.get(key, 0) + value
    return totals


def cmd_sync(args: argparse.Namespace) -> int:
    """cmd sync."""
    state_path = _resolve_state_path(args.state, allow_default=True)
    if state_path is None:
        raise SystemExit("--state is required for sync")
    raw_workspaces: list[str] = []
    if args.workspace:
        raw_workspaces.extend(args.workspace)
    if args.workspaces:
        raw_workspaces.extend([item.strip() for item in args.workspaces.split(",") if item.strip()])
    if not raw_workspaces:
        raise SystemExit("--workspace or --workspaces is required for sync")

    workspaces: list[str] = []
    seen: set[str] = set()
    for item in raw_workspaces:
        resolved = str(Path(item).expanduser())
        if resolved in seen:
            continue
        seen.add(resolved)
        workspaces.append(resolved)

    authority_map = _parse_authority_map(getattr(args, "authority_map", None))
    graph, index, meta = _resolve_graph_index(args, allow_default_state=True)

    embed_fn, embed_batch_fn, _, _, _ = _resolve_embedder(args, meta)

    if args.dry_run:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_state = Path(tmp_dir) / "state.json"
            shutil.copy2(state_path, tmp_state)
            reports: dict[str, SyncReport] = {}
            for workspace in workspaces:
                workspace_id = _resolve_workspace_id(workspace)
                reports[workspace_id] = sync_workspace(
                    state_path=str(tmp_state),
                    workspace_dir=workspace,
                    workspace_id=workspace_id,
                    embed_fn=embed_fn,
                    embed_batch_fn=embed_batch_fn,
                    journal_path=None,
                    authority_map=authority_map,
                )
            if args.json:
                payload = {"workspaces": {key: asdict(value) for key, value in reports.items()}}
                print(json.dumps(payload, indent=2))
            else:
                if len(reports) == 1:
                    report = next(iter(reports.values()))
                    print(
                        f"sync report: +{report.nodes_added}/~{report.nodes_updated} "
                        f"-{report.nodes_removed} ={report.nodes_unchanged} unchanged | "
                        f"{report.embeddings_computed} embeddings"
                    )
                else:
                    for workspace_id, report in reports.items():
                        print(
                            f"sync[{workspace_id}]: +{report.nodes_added}/~{report.nodes_updated} "
                            f"-{report.nodes_removed} ={report.nodes_unchanged} unchanged | "
                            f"{report.embeddings_computed} embeddings"
                        )
                    totals = _sum_sync_reports(reports)
                    print(
                        f"sync total: +{totals.nodes_added}/~{totals.nodes_updated} "
                        f"-{totals.nodes_removed} ={totals.nodes_unchanged} unchanged | "
                        f"{totals.embeddings_computed} embeddings"
                    )
                graph, _, _ = load_state(str(tmp_state))
                health = measure_health(graph)
                print(
                    "health: "
                    f"nodes={graph.node_count()} edges={graph.edge_count()} orphans={health.orphan_nodes} "
                    f"reflex={health.reflex_pct:.1%} habitual={health.habitual_pct:.1%} dormant={health.dormant_pct:.1%}"
                )
            return 0

    reports = {}
    for workspace in workspaces:
        workspace_id = _resolve_workspace_id(workspace)
        reports[workspace_id] = sync_workspace(
            state_path=state_path,
            workspace_dir=workspace,
            workspace_id=workspace_id,
            embed_fn=embed_fn,
            embed_batch_fn=embed_batch_fn,
            journal_path=_resolve_journal_path(args, allow_default_state=True),
            authority_map=authority_map,
        )

    _write_json_atomic(
        _sync_report_path(state_path),
        {
            "workspaces": {key: asdict(value) for key, value in reports.items()},
            "state_path": str(Path(state_path).expanduser()),
            "ts": time.time(),
            "iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )

    if args.json:
        print(json.dumps({"workspaces": {key: asdict(value) for key, value in reports.items()}}, indent=2))
    else:
        if len(reports) == 1:
            report = next(iter(reports.values()))
            print(
                f"sync report: +{report.nodes_added}/~{report.nodes_updated} "
                f"-{report.nodes_removed} ={report.nodes_unchanged} unchanged | "
                f"{report.embeddings_computed} embeddings"
            )
        else:
            for workspace_id, report in reports.items():
                print(
                    f"sync[{workspace_id}]: +{report.nodes_added}/~{report.nodes_updated} "
                    f"-{report.nodes_removed} ={report.nodes_unchanged} unchanged | "
                    f"{report.embeddings_computed} embeddings"
                )
            totals = _sum_sync_reports(reports)
            print(
                f"sync total: +{totals.nodes_added}/~{totals.nodes_updated} "
                f"-{totals.nodes_removed} ={totals.nodes_unchanged} unchanged | "
                f"{totals.embeddings_computed} embeddings"
            )
        graph, _, _ = load_state(state_path)
        health = measure_health(graph)
        print(
            "health: "
            f"nodes={graph.node_count()} edges={graph.edge_count()} orphans={health.orphan_nodes} "
            f"reflex={health.reflex_pct:.1%} habitual={health.habitual_pct:.1%} dormant={health.dormant_pct:.1%}"
        )
    return 0


def cmd_health(args: argparse.Namespace) -> int:
    """cmd health."""
    graph, _, _ = _resolve_graph_index(args, allow_default_state=(getattr(args, "graph", None) is None))
    payload = measure_health(graph).__dict__
    payload["nodes"] = graph.node_count()
    payload["edges"] = graph.edge_count()
    log_health(payload, journal_path=_resolve_journal_path(args, allow_default_state=(getattr(args, "graph", None) is None)))
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0
    print(
        "\n".join(
            [
                "Brain health:",
                f"  Nodes: {payload['nodes']}",
                f"  Edges: {payload['edges']}",
                f"  Reflex: {payload['reflex_pct']:.1%}  Habitual: {payload['habitual_pct']:.1%}  Dormant: {payload['dormant_pct']:.1%}",
                f"  Orphans: {payload['orphan_nodes']}",
                f"  Cross-file edges: {payload['cross_file_edge_pct']:.1%}",
            ]
        )
    )
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Print consolidated daily summary report."""
    state_path = _resolve_state_path(args.state, allow_default=True)
    if state_path is None:
        raise SystemExit("--state is required for report")
    resolved_state = str(Path(state_path).expanduser())
    graph, _, _ = load_state(resolved_state)
    local_health = measure_health(graph)

    socket_path = _default_daemon_socket_path(resolved_state)
    socket_exists = Path(socket_path).exists()
    pid_path = Path(socket_path).expanduser().parent / "daemon.pid"
    pid = None
    if pid_path.exists():
        raw_pid = pid_path.read_text(encoding="utf-8").strip()
        if raw_pid.isdigit():
            pid = int(raw_pid)
    health_payload = None
    health_warning = None
    if socket_exists:
        ping_ok, daemon_health, _error = _socket_health_status(socket_path)
        if ping_ok and isinstance(daemon_health, dict):
            health_payload = daemon_health
        else:
            health_warning = "health unavailable (daemon timeout)"

    sync_payload = _read_json_optional(_sync_report_path(resolved_state))
    maintain_payload = _read_json_optional(_maintain_report_path(resolved_state))

    journal_path = _resolve_journal_path(args, allow_default_state=True)
    entries = read_journal(journal_path) if journal_path else []
    last_report_ts = None
    last_report_iso = None
    for entry in reversed(entries):
        if entry.get("type") == "report":
            ts = entry.get("ts")
            if isinstance(ts, (int, float)):
                last_report_ts = float(ts)
                last_report_iso = entry.get("iso") if isinstance(entry.get("iso"), str) else None
                break

    def _include_entry(entry: dict) -> bool:
        ts = entry.get("ts")
        if last_report_ts is None:
            return True
        if isinstance(ts, (int, float)):
            return float(ts) > last_report_ts
        return False

    learn_positive = 0
    learn_negative = 0
    splits_suggested = 0
    splits_applied = 0
    merges_suggested = 0
    merges_applied = 0
    pruned_edges = 0
    pruned_nodes = 0
    replay_edges_reinforced = 0
    replay_cross_file_created = 0

    for entry in entries:
        if not _include_entry(entry):
            continue
        if entry.get("type") == "learn":
            outcome = entry.get("outcome", 0)
            try:
                outcome_val = float(outcome)
            except (TypeError, ValueError):
                outcome_val = 0.0
            if outcome_val > 0:
                learn_positive += 1
            elif outcome_val < 0:
                learn_negative += 1
        elif entry.get("type") == "maintenance":
            task = entry.get("task")
            if task == "split":
                splits_suggested += int(entry.get("suggested", 0) or 0)
                splits_applied += int(entry.get("applied", 0) or 0)
            elif task == "merge":
                merges_suggested += int(entry.get("suggested", 0) or 0)
                merges_applied += int(entry.get("applied", 0) or 0)
            elif task == "prune":
                pruned_edges += int(entry.get("edges_removed", 0) or 0)
                pruned_nodes += int(entry.get("nodes_removed", 0) or 0)
        elif entry.get("type") == "replay":
            replay_edges_reinforced += int(entry.get("edges_reinforced", 0) or 0)
            replay_cross_file_created += int(entry.get("cross_file_created", 0) or 0)

    since_label = f"since {last_report_iso}" if last_report_iso else "all time"

    def _get_int(payload: dict[str, object], key: str) -> int:
        try:
            return int(payload.get(key, 0) or 0)
        except (TypeError, ValueError):
            return 0

    def _get_nested_int(payload: dict[str, object], key: str, subkey: str) -> int:
        value = payload.get(key)
        if isinstance(value, dict):
            return _get_int(value, subkey)
        return 0

    lines = [
        "Daily brain update summary",
        f"  state: {resolved_state}",
        f"  daemon.sock: {'present' if socket_exists else 'missing'}" + (f" (pid {pid})" if pid is not None else ""),
        f"  nodes: {graph.node_count()}  edges: {graph.edge_count()}  orphans: {local_health.orphan_nodes}",
        f"  reflex: {local_health.reflex_pct:.1%}  habitual: {local_health.habitual_pct:.1%}  dormant: {local_health.dormant_pct:.1%}",
    ]
    if health_warning is not None:
        lines.append(f"  warning: {health_warning}")

    route_model_present = None
    route_mode_configured = None
    route_mode_effective = None
    route_model_error = None
    route_model_path = None
    if health_payload is not None:
        route_model_present = health_payload.get("route_model_present")
        route_mode_configured = health_payload.get("route_mode_configured")
        route_mode_effective = health_payload.get("route_mode_effective")
        route_model_error = health_payload.get("route_model_error")
        route_model_path = health_payload.get("route_model_path")
    if route_model_present is None:
        route_model_present = (Path(resolved_state).expanduser().parent / "route_model.npz").exists()
    if route_mode_configured is None:
        route_mode_configured = "unknown"
    if route_mode_effective is None:
        route_mode_effective = "learned" if route_model_present else "edge+sim"

    lines.append(f"  route_model: {'present' if route_model_present else 'missing'}")
    lines.append(f"  route_mode_configured: {route_mode_configured}")
    lines.append(f"  route_mode_effective: {route_mode_effective}")
    if route_model_path:
        lines.append(f"  route_model_path: {route_model_path}")
    if route_model_error:
        lines.append(f"  route_model_error: {route_model_error}")
    if route_mode_configured == "learned" and route_mode_effective != "learned":
        lines.append("  warning: learned routing configured but effective mode is degraded")
    elif not route_model_present and route_mode_effective != "learned":
        lines.append("  warning: route_model missing; learned routing will fall back to edge+sim")

    sync_totals = None
    sync_workspace_count = 0
    if isinstance(sync_payload, dict) and isinstance(sync_payload.get("workspaces"), dict):
        workspaces_payload = sync_payload["workspaces"]
        if workspaces_payload:
            sync_workspace_count = len(workspaces_payload)
            sync_totals = {
                "nodes_added": 0,
                "nodes_updated": 0,
                "nodes_removed": 0,
                "nodes_unchanged": 0,
                "embeddings_computed": 0,
            }
            for report in workspaces_payload.values():
                if not isinstance(report, dict):
                    continue
                for key in sync_totals:
                    sync_totals[key] += _get_int(report, key)
    elif sync_payload:
        sync_totals = sync_payload

    if sync_totals:
        suffix = ""
        if sync_workspace_count > 1:
            suffix = f" ({sync_workspace_count} workspaces)"
        lines.append(
            "  last sync: "
            f"+{_get_int(sync_totals, 'nodes_added')}/~{_get_int(sync_totals, 'nodes_updated')} "
            f"-{_get_int(sync_totals, 'nodes_removed')} ={_get_int(sync_totals, 'nodes_unchanged')} unchanged | "
            f"{_get_int(sync_totals, 'embeddings_computed')} embeddings{suffix}"
        )
    else:
        lines.append("  last sync: (none)")

    if maintain_payload:
        lines.append(
            "  last maintenance: "
            f"nodes {_get_nested_int(maintain_payload, 'health_before', 'nodes')}"
            f" -> {_get_nested_int(maintain_payload, 'health_after', 'nodes')} "
            f"edges {_get_int(maintain_payload, 'edges_before')}"
            f" -> {_get_int(maintain_payload, 'edges_after')}"
        )
        lines.append(
            "  maintenance deltas: "
            f"merges {_get_int(maintain_payload, 'merges_applied')}/{_get_int(maintain_payload, 'merges_proposed')} "
            f"splits {_get_int(maintain_payload, 'splits_applied')}/{_get_int(maintain_payload, 'splits_proposed')} "
            f"pruned edges={_get_int(maintain_payload, 'pruned_edges')} "
            f"nodes={_get_int(maintain_payload, 'pruned_nodes')} "
            f"decay_applied={bool(maintain_payload.get('decay_applied', False))}"
        )
    else:
        lines.append("  last maintenance: (none)")

    lines.extend(
        [
            f"  label usage ({since_label}):",
            f"  learn outcomes: +{learn_positive} -{learn_negative}",
            f"  maintenance: splits {splits_applied}/{splits_suggested} merges {merges_applied}/{merges_suggested} "
            f"pruned edges={pruned_edges} nodes={pruned_nodes}",
            f"  replay: edges_reinforced={replay_edges_reinforced} cross_file_created={replay_cross_file_created}",
        ]
    )

    print("\n".join(lines))

    if journal_path is not None:
        log_event(
            {
                "type": "report",
                "state_path": resolved_state,
                "since_ts": last_report_ts,
                "sync_report": bool(sync_payload),
                "maintenance_report": bool(maintain_payload),
            },
            journal_path=journal_path,
        )

    return 0


def cmd_route_audit(args: argparse.Namespace) -> int:
    """Print learned routing audit snapshot."""
    state_path = _resolve_state_path(args.state, allow_default=True)
    if state_path is None:
        raise SystemExit("--state is required for route-audit")
    resolved_state = str(Path(state_path).expanduser())
    state_dir = Path(resolved_state).expanduser().parent
    route_model_path = state_dir / "route_model.npz"
    labels_path = _default_labels_path(resolved_state)

    model_present = route_model_path.exists()
    model_loaded = False
    model_error = None
    if model_present:
        try:
            RouteModel.load_npz(route_model_path)
            model_loaded = True
        except Exception as exc:  # noqa: BLE001
            model_error = str(exc)

    labels_count = 0
    if labels_path.exists():
        labels_count = sum(1 for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip())

    last_train_ts = None
    last_train_iso = None
    if model_present:
        last_train_ts = route_model_path.stat().st_mtime
        last_train_iso = datetime.fromtimestamp(last_train_ts, tz=timezone.utc).isoformat()

    socket_path = _default_daemon_socket_path(resolved_state)
    ping_ok, health_payload, health_error = _socket_health_status(socket_path)
    route_mode_configured = None
    route_mode_effective = None
    route_model_present = None
    route_enable_stop = None
    route_stop_margin = None
    route_model_error = None
    if isinstance(health_payload, dict):
        route_mode_configured = health_payload.get("route_mode_configured")
        route_mode_effective = health_payload.get("route_mode_effective")
        route_model_present = health_payload.get("route_model_present")
        route_enable_stop = health_payload.get("route_enable_stop")
        route_stop_margin = health_payload.get("route_stop_margin")
        route_model_error = health_payload.get("route_model_error")

    if route_mode_effective is None:
        route_mode_effective = "learned" if model_present else "edge+sim"
    if route_mode_configured is None:
        route_mode_configured = "unknown"
    if route_model_present is None:
        route_model_present = model_present

    payload = {
        "state_path": resolved_state,
        "daemon_running": bool(ping_ok),
        "daemon_socket": socket_path,
        "daemon_health_error": health_error,
        "route_mode_configured": route_mode_configured,
        "route_mode_effective": route_mode_effective,
        "route_model_present": route_model_present,
        "route_model_loaded": model_loaded if model_present else False,
        "route_model_path": str(route_model_path),
        "route_model_error": route_model_error or model_error,
        "last_train_ts": last_train_ts,
        "last_train_iso": last_train_iso,
        "labels_path": str(labels_path),
        "labels_count": labels_count,
        "route_enable_stop": route_enable_stop,
        "route_stop_margin": route_stop_margin,
    }

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    lines = [
        "Route audit",
        f"  state: {resolved_state}",
        f"  daemon_running: {bool(ping_ok)}",
        f"  route_mode_configured: {route_mode_configured}",
        f"  route_mode_effective: {route_mode_effective}",
        f"  route_model_present: {route_model_present}",
        f"  route_model_loaded: {model_loaded if model_present else False}",
        f"  route_model_path: {route_model_path}",
        f"  last_train_iso: {last_train_iso or 'n/a'}",
        f"  labels_count: {labels_count}",
        f"  labels_path: {labels_path}",
        f"  stop_enabled: {route_enable_stop if route_enable_stop is not None else 'unknown'}",
        f"  stop_margin: {route_stop_margin if route_stop_margin is not None else 'unknown'}",
    ]
    if payload.get("route_model_error"):
        lines.append(f"  route_model_error: {payload['route_model_error']}")
    if not ping_ok and health_error:
        lines.append(f"  warning: daemon health unavailable ({health_error})")
    if route_mode_configured == "learned" and route_mode_effective != "learned":
        lines.append("  warning: learned routing configured but effective mode is degraded")
    print("\n".join(lines))
    return 0


def cmd_maintain(args: argparse.Namespace) -> int:
    """cmd maintain."""
    state_path = _resolve_state_path(args.state, allow_default=True)
    if state_path is None:
        raise SystemExit("--state is required for maintain")
    requested_tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    _, _, meta = _resolve_graph_index(args, allow_default_state=True)
    embed_fn, _, _, _, _ = _resolve_embedder(args, meta)
    llm_fn, _ = _resolve_llm(args)
    report = run_maintenance(
        state_path=state_path,
        tasks=requested_tasks,
        embed_fn=embed_fn,
        llm_fn=llm_fn,
        journal_path=_resolve_journal_path(args, allow_default_state=True),
        dry_run=args.dry_run,
        max_merges=args.max_merges,
        prune_below=args.prune_below,
    )
    if not args.dry_run:
        _write_json_atomic(
            _maintain_report_path(state_path),
            {
                **asdict(report),
                "state_path": str(Path(state_path).expanduser()),
                "ts": time.time(),
                "iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
        )

    if args.json:
        print(json.dumps(asdict(report), indent=2))
        return 0

    print("\n".join([
        "Maintenance report:",
        f"  tasks: {', '.join(report.tasks_run) if report.tasks_run else '(none)'}",
        f"  nodes: {report.health_before['nodes']} -> {report.health_after['nodes']}",
        f"  edges: {report.edges_before} -> {report.edges_after}",
        f"  merges: {report.merges_applied}/{report.merges_proposed}",
        f"  splits: {report.splits_applied}/{report.splits_proposed}",
        f"  pruned: edges={report.pruned_edges} nodes={report.pruned_nodes}",
        f"  soft_pruned_edges: {report.soft_pruned_edges}",
        f"  nodes_scaled: {report.nodes_scaled}",
        f"  decay_applied: {report.decay_applied}",
        f"  dry_run: {args.dry_run}",
    ]))
    if report.notes:
        print(f"  notes: {', '.join(report.notes)}")
    return 0


def cmd_daemon(args: argparse.Namespace) -> int:
    """cmd daemon."""
    profile = _load_profile(args.profile)
    state_input = args.state if args.state is not None else (profile.paths.state_path if profile is not None else None)
    state_path = _resolve_state_path(state_input, allow_default=True)
    if state_path is None:
        raise SystemExit("--state is required for daemon")

    embed_model = _coalesce(args.embed_model, profile.embedder.embed_model if profile is not None else None, "auto")
    max_prompt_context_chars = _coalesce_int(
        args.max_prompt_context_chars,
        profile.policy.max_prompt_context_chars if profile is not None else None,
        30000,
    )
    max_fired_nodes = _coalesce_int(
        args.max_fired_nodes,
        profile.policy.max_fired_nodes if profile is not None else None,
        30,
    )
    route_mode = _coalesce(args.route_mode, profile.policy.route_mode if profile is not None else None, "learned")
    route_top_k = _coalesce_int(
        args.route_top_k,
        profile.policy.route_top_k if profile is not None else None,
        5,
    )
    route_alpha_sim = _coalesce_float(
        args.route_alpha_sim,
        profile.policy.route_alpha_sim if profile is not None else None,
        0.5,
    )
    route_use_relevance = _coalesce_route_use_relevance(
        args.route_use_relevance,
        profile.policy.route_use_relevance if profile is not None else None,
        True,
    )
    route_enable_stop = _coalesce_route_enable_stop(
        args.route_enable_stop,
        profile.policy.route_enable_stop if profile is not None else None,
        False,
    )
    route_stop_margin = _coalesce_float(
        args.route_stop_margin,
        profile.policy.route_stop_margin if profile is not None else None,
        0.1,
    )
    assert_learned = _coalesce_assert_learned(
        args.assert_learned,
        profile.policy.assert_learned if profile is not None else None,
        False,
    )
    if route_stop_margin < 0.0:
        raise SystemExit("--route-stop-margin must be >= 0.0")
    route_model = args.route_model
    from .daemon import main as daemon_main

    return daemon_main(
        [
            "--state",
            str(state_path),
            "--embed-model",
            embed_model,
            "--max-prompt-context-chars",
            str(max_prompt_context_chars),
            "--max-fired-nodes",
            str(max_fired_nodes),
            "--route-mode",
            route_mode,
            "--route-top-k",
            str(route_top_k),
            "--route-alpha-sim",
            str(route_alpha_sim),
            "--route-use-relevance",
            "true" if route_use_relevance else "false",
            "--route-enable-stop",
            "true" if route_enable_stop else "false",
            "--route-stop-margin",
            str(route_stop_margin),
            "--auto-save-interval",
            str(args.auto_save_interval),
        ]
        + (["--assert-learned"] if assert_learned else [])
        + (["--route-model", str(route_model)] if route_model else [])
    )


def cmd_serve(args: argparse.Namespace) -> int:
    """cmd serve."""
    profile = _load_profile(args.profile)
    state_input = args.state if args.state is not None else (profile.paths.state_path if profile is not None else None)
    state_path = _resolve_state_path(state_input, allow_default=False)
    if state_path is None:
        raise SystemExit("--state is required for serve")
    state_path = str(Path(state_path).expanduser())

    embed_model = _coalesce(args.embed_model, profile.embedder.embed_model if profile is not None else None, "auto")
    max_prompt_context_chars = _coalesce_int(
        args.max_prompt_context_chars,
        profile.policy.max_prompt_context_chars if profile is not None else None,
        30000,
    )
    max_fired_nodes = _coalesce_int(
        args.max_fired_nodes,
        profile.policy.max_fired_nodes if profile is not None else None,
        30,
    )
    route_mode = _coalesce(args.route_mode, profile.policy.route_mode if profile is not None else None, "learned")
    route_top_k = _coalesce_int(
        args.route_top_k,
        profile.policy.route_top_k if profile is not None else None,
        5,
    )
    route_alpha_sim = _coalesce_float(
        args.route_alpha_sim,
        profile.policy.route_alpha_sim if profile is not None else None,
        0.5,
    )
    route_use_relevance = _coalesce_route_use_relevance(
        args.route_use_relevance,
        profile.policy.route_use_relevance if profile is not None else None,
        True,
    )
    route_enable_stop = _coalesce_route_enable_stop(
        args.route_enable_stop,
        profile.policy.route_enable_stop if profile is not None else None,
        False,
    )
    route_stop_margin = _coalesce_float(
        args.route_stop_margin,
        profile.policy.route_stop_margin if profile is not None else None,
        0.1,
    )
    assert_learned = _coalesce_assert_learned(
        args.assert_learned,
        profile.policy.assert_learned if profile is not None else None,
        False,
    )
    if route_stop_margin < 0.0:
        raise SystemExit("--route-stop-margin must be >= 0.0")
    route_model = args.route_model
    from .socket_server import main as socket_server_main

    socket_path = (
        str(Path(args.socket_path).expanduser()) if args.socket_path else _default_daemon_socket_path(state_path)
    )
    explicit_socket_path = str(Path(args.socket_path).expanduser()) if args.socket_path else None
    action = getattr(args, "serve_action", "start")
    start_argv = _serve_start_arguments(
        state_path=state_path,
        socket_path=explicit_socket_path,
        embed_model=embed_model,
        max_prompt_context_chars=max_prompt_context_chars,
        max_fired_nodes=max_fired_nodes,
        route_mode=route_mode,
        route_top_k=route_top_k,
        route_alpha_sim=route_alpha_sim,
        route_use_relevance=route_use_relevance,
        route_enable_stop=route_enable_stop,
        route_stop_margin=route_stop_margin,
        assert_learned=assert_learned,
        route_model=route_model,
    )
    launchd_program_arguments = _resolve_serve_launchd_program_arguments(start_argv)

    if action == "install":
        if sys.platform != "darwin":
            print(
                "launchd lifecycle commands are supported on macOS only. "
                "Use `openclawbrain serve --systemd` on Linux/systemd hosts.",
                file=sys.stderr,
            )
            return 1

        if not args.dry_run:
            _ensure_route_model_exists(state_path)

        label = args.label or _derive_serve_label(state_path)
        plist_path = Path(args.plist_path).expanduser() if args.plist_path else _derive_launchd_plist_path(label)
        env_vars = _parse_env_file(args.env_file) if args.env_file else None
        program = _render_launchd_plist(
            label=label,
            state_path=state_path,
            program_arguments=launchd_program_arguments,
            env_vars=env_vars,
        )
        uid = os.getuid()
        bootout_cmd = ["launchctl", "bootout", f"gui/{uid}", str(plist_path)]
        bootstrap_cmd = ["launchctl", "bootstrap", f"gui/{uid}", str(plist_path)]

        print(program)
        print("Planned launchctl commands:")
        print(f"  {' '.join(bootout_cmd)}")
        print(f"  {' '.join(bootstrap_cmd)}")

        if args.dry_run:
            return 0

        _run_launchctl(bootout_cmd, ignore_errors=True)
        plist_path.parent.mkdir(parents=True, exist_ok=True)
        plist_path.write_text(program, encoding="utf-8")
        if env_vars:
            plist_path.chmod(0o600)
        _run_launchctl(bootstrap_cmd)

        print(f"wrote launchd plist: {plist_path}")
        return 0

    if action == "uninstall":
        if sys.platform != "darwin":
            print(
                "launchd lifecycle commands are supported on macOS only. "
                "Use `openclawbrain serve --systemd` on Linux/systemd hosts.",
                file=sys.stderr,
            )
            return 1

        label = args.label or _derive_serve_label(state_path)
        plist_path = Path(args.plist_path).expanduser() if args.plist_path else _derive_launchd_plist_path(label)
        uid = os.getuid()
        bootout_cmd = ["launchctl", "bootout", f"gui/{uid}", str(plist_path)]
        print("Planned launchctl commands:")
        print(f"  {' '.join(bootout_cmd)}")

        if args.dry_run:
            return 0

        _run_launchctl(bootout_cmd, ignore_errors=True)
        try:
            plist_path.unlink(missing_ok=True)
            print(f"removed launchd plist: {plist_path}")
        except OSError as exc:
            print(f"could not remove plist {plist_path}: {exc}", file=sys.stderr)
            return 1
        return 0

    if action == "status":
        payload = _serve_status_payload(state_path, socket_path)
        if payload["daemon_running"]:
            health = payload["health"] if isinstance(payload.get("health"), dict) else {}
            print(
                "\n".join(
                    [
                        "OpenClawBrain serve status: running",
                        f"State: {payload['state_path']}",
                        f"Socket: {payload['socket_path']}",
                        (
                            "Health: "
                            f"nodes={health.get('nodes', 'n/a')} "
                            f"edges={health.get('edges', 'n/a')} "
                            f"dormant_pct={health.get('dormant_pct', 'n/a')}"
                        ),
                    ]
                )
            )
            return 0
        print(
            "\n".join(
                [
                    "OpenClawBrain serve status: not running",
                    f"State: {payload['state_path']}",
                    f"Socket: {payload['socket_path']}",
                    f"Error: {payload['error']}",
                ]
            )
        )
        return 1

    if action == "stop":
        ping_ok, _, error = _socket_health_status(socket_path)
        if ping_ok:
            from .socket_client import OCBClient

            try:
                with OCBClient(socket_path, timeout=2.0) as client:
                    client.request("shutdown", {})
                print(f"Shutdown request sent to {socket_path}")
                return 0
            except Exception as exc:  # noqa: BLE001
                error = str(exc)

        print(
            "\n".join(
                [
                    f"Could not stop via socket: {error}",
                    "If service is managed by launchd:",
                    "  launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.openclawbrain.daemon.plist",
                    "If service is managed by systemd:",
                    "  sudo systemctl stop openclawbrain-daemon.service",
                ]
            ),
            file=sys.stderr,
        )
        return 1

    if args.launchd:
        label = args.label or _derive_serve_label(state_path)
        print(
            _render_launchd_plist(
                label=label,
                state_path=state_path,
                program_arguments=launchd_program_arguments,
            )
        )
        return 0
    if args.systemd:
        print(_render_systemd_unit(state_path=state_path, socket_path=socket_path))
        return 0

    print("OpenClawBrain socket service (foreground)", file=sys.stderr)
    print(f"  socket path: {socket_path}", file=sys.stderr)
    print(f"  state path: {state_path}", file=sys.stderr)
    print(f"  query status: openclawbrain serve status --state {state_path}", file=sys.stderr)
    print("  stop: Ctrl-C", file=sys.stderr)

    return socket_server_main(start_argv)


def cmd_journal(args: argparse.Namespace) -> int:
    """cmd journal."""
    journal_path = _resolve_journal_path(args, allow_default_state=True)
    if args.stats:
        print(
            json.dumps(journal_stats(journal_path=journal_path), indent=2)
            if args.json
            else "\n".join(
                f"{k}: {v}"
                for k, v in journal_stats(journal_path=journal_path).items()
                if k != "avg_fired_per_query"
            )
        )
        return 0
    entries = read_journal(last_n=args.last, journal_path=journal_path)
    print(
        json.dumps(entries, indent=2)
        if args.json
        else "\n".join(f"{idx+1:>2}. {entry.get('type')} @ {entry.get('iso', entry.get('ts', ''))}: {entry}" for idx, entry in enumerate(entries))
        or "No entries."
    )
    return 0


def cmd_async_route_pg(args: argparse.Namespace) -> int:
    """Run teacher-shadow async routing updates with PG."""
    state_path = _resolve_state_path(args.state, allow_default=False)
    if state_path is None:
        raise SystemExit("--state is required")
    journal_path = _resolve_journal_path(args, allow_default_state=False)
    if journal_path is None:
        raise SystemExit("unable to resolve journal path")

    try:
        parsed_weights = RewardWeights.from_string(args.reward_weights) if args.reward_weights else RewardWeights.from_env()
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    labels_out = str(Path(args.labels_out).expanduser()) if args.labels_out else str(_default_labels_path(state_path))
    summary = run_async_route_pg(
        state_path=state_path,
        journal_path=journal_path,
        since_hours=float(args.since_hours),
        max_queries=max(1, int(args.max_queries)),
        sample_rate=max(0.0, min(1.0, float(args.sample_rate))),
        max_candidates_per_node=max(1, int(args.max_candidates_per_node)),
        max_decision_points=max(1, int(args.max_decision_points)),
        teacher=str(args.teacher),
        teacher_model=str(args.teacher_model),
        apply=bool(args.apply),
        write_relevance_metadata=bool(args.write_relevance_metadata),
        score_scale=float(args.score_scale),
        traces_out=str(args.traces_out) if args.traces_out else None,
        traces_in=str(args.traces_in) if args.traces_in else None,
        labels_out=labels_out,
        include_query_vector=bool(args.include_query_vector),
        reward_source=RewardSource.parse(args.reward_source),
        reward_weights=parsed_weights,
    )
    payload = summary.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(
            f"async-route-pg: sampled_queries={payload['sampled_queries']} "
            f"decision_points={payload['decision_points_total']} "
            f"labeled={payload['decision_points_labeled']} "
            f"updates={payload['updates_applied']} "
            f"total_abs_delta={payload['total_abs_weight_delta']:.6f} "
            f"max_abs_delta={payload['max_abs_weight_delta']:.6f} "
            f"dry_run={payload['dry_run']}"
        )
        if payload["errors"]:
            print("errors:")
            for item in payload["errors"]:
                print(f"- {item}")
    return 0


def cmd_dream(args: argparse.Namespace) -> int:
    """Run background dreaming loop (teacher-shadow async routing updates)."""
    state_path = _resolve_state_path(args.state, allow_default=False)
    if state_path is None:
        raise SystemExit("--state is required")
    journal_path = _resolve_journal_path(args, allow_default_state=False)
    if journal_path is None:
        raise SystemExit("unable to resolve journal path")

    try:
        parsed_weights = RewardWeights.from_string(args.reward_weights) if args.reward_weights else RewardWeights.from_env()
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    traces_dir = Path(args.traces_dir).expanduser() if args.traces_dir else _default_dream_traces_dir(state_path)
    labels_out = str(Path(args.labels_out).expanduser()) if args.labels_out else str(_default_labels_path(state_path))
    interval_seconds = max(1, int(args.interval_seconds))
    cycle = 0

    while True:
        cycle += 1
        cycle_start = datetime.now(timezone.utc)
        traces_out = traces_dir / f"dream_traces_{cycle_start.strftime('%Y%m%dT%H%M%S')}{cycle_start.microsecond // 1000:03d}Z.jsonl"
        payload: dict[str, object] = {
            "cycle": cycle,
            "ts": cycle_start.isoformat(),
            "state_path": state_path,
            "journal_path": journal_path,
            "apply": bool(args.apply),
            "traces_out": str(traces_out),
        }

        if args.apply:
            try:
                with state_write_lock(
                    state_path,
                    force=False,
                    command_hint="openclawbrain dream",
                ):
                    summary = run_async_route_pg(
                        state_path=state_path,
                        journal_path=journal_path,
                        since_hours=float(args.since_hours),
                        max_queries=max(1, int(args.max_queries)),
                        sample_rate=max(0.0, min(1.0, float(args.sample_rate))),
                        max_candidates_per_node=max(1, int(args.max_candidates_per_node)),
                        max_decision_points=max(1, int(args.max_decision_points)),
                        teacher=str(args.teacher),
                        teacher_model=str(args.teacher_model),
                        apply=True,
                        write_relevance_metadata=True,
                        score_scale=float(args.score_scale),
                        traces_out=str(traces_out),
                        traces_in=None,
                        labels_out=labels_out,
                        include_query_vector=False,
                        reward_source=RewardSource.parse(args.reward_source),
                        reward_weights=parsed_weights,
                    )
                    payload["status"] = "ok"
                    payload["summary"] = summary.to_dict()
            except StateLockError as exc:
                if args.skip_if_locked:
                    payload["status"] = "skipped"
                    payload["skip_reason"] = "state_lock_held"
                    payload["message"] = str(exc)
                else:
                    raise SystemExit(str(exc)) from None
        else:
            summary = run_async_route_pg(
                state_path=state_path,
                journal_path=journal_path,
                since_hours=float(args.since_hours),
                max_queries=max(1, int(args.max_queries)),
                sample_rate=max(0.0, min(1.0, float(args.sample_rate))),
                max_candidates_per_node=max(1, int(args.max_candidates_per_node)),
                max_decision_points=max(1, int(args.max_decision_points)),
                teacher=str(args.teacher),
                teacher_model=str(args.teacher_model),
                apply=False,
                write_relevance_metadata=True,
                score_scale=float(args.score_scale),
                traces_out=str(traces_out),
                traces_in=None,
                labels_out=labels_out,
                include_query_vector=False,
                reward_source=RewardSource.parse(args.reward_source),
                reward_weights=parsed_weights,
            )
            payload["status"] = "ok"
            payload["summary"] = summary.to_dict()

        if args.json:
            print(json.dumps(payload, separators=(",", ":")))
        else:
            status = payload.get("status")
            if status == "skipped":
                print(f"dream: skipped (state lock held) traces_out={traces_out}")
            else:
                summary_payload = payload.get("summary", {})
                if isinstance(summary_payload, dict):
                    print(
                        "dream: "
                        f"sampled_queries={summary_payload.get('sampled_queries')} "
                        f"decision_points={summary_payload.get('decision_points_total')} "
                        f"labeled={summary_payload.get('decision_points_labeled')} "
                        f"updates={summary_payload.get('updates_applied')} "
                        f"total_abs_delta={summary_payload.get('total_abs_weight_delta')} "
                        f"max_abs_delta={summary_payload.get('max_abs_weight_delta')} "
                        f"dry_run={summary_payload.get('dry_run')} "
                        f"traces_out={traces_out}"
                    )
                else:
                    print(f"dream: completed traces_out={traces_out}")

        if args.once:
            break
        time.sleep(interval_seconds)
    return 0


def cmd_loop(args: argparse.Namespace) -> int:
    """Run/install always-learning loop (replay + maintenance + optional teacher)."""
    if args.state:
        state_path = _resolve_state_path(args.state, allow_default=False)
    elif args.agent:
        state_path = _state_path_for_agent(args.agent)
    else:
        state_path = _resolve_state_path(None, allow_default=True)
    if state_path is None:
        raise SystemExit("--state or --agent is required for loop")
    state_path = str(Path(state_path).expanduser())
    labels_path = _default_labels_path(state_path)
    if args.advance_offsets_on_skip is None:
        args.advance_offsets_on_skip = bool(str(args.mode) != "edges-only")
    if args.include_tool_results and getattr(args, "tool_result_max_chars", None) is None:
        args.tool_result_max_chars = int(DEFAULT_TOOL_RESULT_MAX_CHARS)

    if getattr(args, "fast", False):
        _apply_fast_loop_defaults(args)
    if getattr(args, "skip_maintain", False):
        args.maintain = False

    agent_root = _loop_state_root(state_path)
    scratch_root = agent_root / "scratch"
    scratch_root.mkdir(parents=True, exist_ok=True)
    agent_id = agent_root.name or "main"
    action = getattr(args, "loop_action", "run")
    log_path = _loop_log_path(state_path)
    events_path = _loop_events_path(state_path)
    lock_path = _loop_lock_path(state_path)
    checkpoint_path = _loop_checkpoint_path(state_path)
    manifest_path = _loop_manifest_path(state_path)
    stdout_path = _loop_stdout_path(state_path)
    stderr_path = _loop_stderr_path(state_path)

    sessions_path = (
        Path(args.sessions).expanduser()
        if args.sessions
        else Path.home() / ".openclaw" / "agents" / agent_id / "sessions"
    )

    ocb_prefix = [_resolve_loop_python(), "-m", "openclawbrain.cli"]
    ocb_subprocess_prefix = _resolve_subprocess_prefix()

    def _loop_run_program_arguments() -> list[str]:
        argv = ocb_prefix + [
            "loop",
            "run",
            "--state",
            state_path,
            "--sessions",
            str(sessions_path),
            "--mode",
            str(args.mode),
            "--llm",
            str(args.llm),
            "--checkpoint-every-seconds",
            str(args.checkpoint_every_seconds),
            "--replay-progress-interval-seconds",
            str(args.replay_progress_interval_seconds),
            "--tool-result-max-chars",
            str(args.tool_result_max_chars),
            "--maintain-tasks",
            str(args.maintain_tasks),
            "--maintain-llm",
            str(args.maintain_llm),
            "--maintain-embedder",
            str(args.maintain_embedder),
            "--harvest-labels" if args.harvest_labels else "--no-harvest-labels",
            "--since-hours",
            str(args.since_hours),
            "--max-queries",
            str(args.max_queries),
            "--sample-rate",
            str(args.sample_rate),
            "--max-candidates-per-node",
            str(args.max_candidates_per_node),
            "--max-decision-points",
            str(args.max_decision_points),
            "--teacher",
            str(args.teacher),
            "--teacher-model",
            str(args.teacher_model),
            "--score-scale",
            str(args.score_scale),
            "--enable-async-route-pg" if args.enable_async_route_pg else "--no-enable-async-route-pg",
            "--enable-dreaming" if args.enable_dreaming else "--no-enable-dreaming",
            "--dream-since-hours",
            str(args.dream_since_hours),
            "--dream-max-queries",
            str(args.dream_max_queries),
            "--dream-sample-rate",
            str(args.dream_sample_rate),
            "--dream-max-candidates-per-node",
            str(args.dream_max_candidates_per_node),
            "--dream-max-decision-points",
            str(args.dream_max_decision_points),
            "--reward-source",
            str(args.reward_source),
            "--hourly-interval-seconds",
            str(args.hourly_interval_seconds),
            "--nightly-hour",
            str(args.nightly_hour),
            "--nightly-minute",
            str(args.nightly_minute),
            "--replay-stall-timeout-seconds",
            str(args.replay_stall_timeout_seconds),
            "--replay-stall-max-restarts",
            str(args.replay_stall_max_restarts),
            "--replay-stall-fallback-mode",
            str(args.replay_stall_fallback_mode),
            "--dream-stall-timeout-seconds",
            str(args.dream_stall_timeout_seconds),
            "--dream-stall-max-restarts",
            str(args.dream_stall_max_restarts),
        ]
        if args.llm_model:
            argv.extend(["--llm-model", str(args.llm_model)])
        if args.workers is not None:
            argv.extend(["--workers", str(int(args.workers))])
        if args.replay_max_interactions is not None:
            argv.extend(["--replay-max-interactions", str(args.replay_max_interactions)])
        if args.replay_priority != "all":
            argv.extend(["--replay-priority", str(args.replay_priority)])
        if args.advance_offsets_on_skip:
            argv.append("--advance-offsets-on-skip")
        if args.resume:
            argv.append("--resume")
        else:
            argv.append("--no-resume")
        if args.include_tool_results:
            argv.append("--include-tool-results")
            if args.tool_result_max_chars is not None:
                argv.extend(["--tool-result-max-chars", str(int(args.tool_result_max_chars))])
        else:
            argv.append("--no-include-tool-results")
        argv.append("--maintain" if args.maintain else "--no-maintain")
        argv.append("--harvest-labels" if args.harvest_labels else "--no-harvest-labels")
        argv.append("--enable-teacher" if args.enable_teacher else "--no-enable-teacher")
        argv.append("--enable-async-route-pg" if args.enable_async_route_pg else "--no-enable-async-route-pg")
        argv.append("--enable-train-route-model" if args.enable_train_route_model else "--no-enable-train-route-model")
        argv.append("--enable-dreaming" if args.enable_dreaming else "--no-enable-dreaming")
        if args.train_route_model_out:
            argv.extend(["--train-route-model-out", str(args.train_route_model_out)])
        if args.reward_weights:
            argv.extend(["--reward-weights", str(args.reward_weights)])
        if args.write_relevance_metadata:
            argv.append("--write-relevance-metadata")
        else:
            argv.append("--no-write-relevance-metadata")
        if args.skip_if_locked:
            argv.append("--skip-if-locked")
        else:
            argv.append("--no-skip-if-locked")
        if args.pause_serve_when_locked:
            argv.append("--pause-serve-when-locked")
        else:
            argv.append("--no-pause-serve-when-locked")
        argv.extend(["--pause-serve-timeout-seconds", str(int(args.pause_serve_timeout_seconds))])
        return argv

    if args.launchd:
        label = _derive_loop_label(state_path)
        program = _render_loop_launchd_plist(
            label=label,
            program_arguments=_loop_run_program_arguments(),
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            schedule={},
            env_vars=_parse_env_file(args.env_file) if args.env_file else None,
        )
        print(program)
        return 0

    if args.systemd:
        print(
            _render_loop_systemd_templates(
                state_path=state_path,
                sessions_path=str(sessions_path),
                loop_cmd=" ".join(shlex.quote(part) for part in _loop_run_program_arguments()),
            )
        )
        return 0

    if action == "install":
        if sys.platform != "darwin":
            print(
                "launchd lifecycle commands are supported on macOS only. "
                "Use `openclawbrain loop --systemd` on Linux/systemd hosts.",
                file=sys.stderr,
            )
            return 1
        label = _derive_loop_label(state_path)
        plist_path = _derive_launchd_plist_path(label)
        env_vars = _parse_env_file(args.env_file) if args.env_file else None
        program = _render_loop_launchd_plist(
            label=label,
            program_arguments=_loop_run_program_arguments(),
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            schedule={},
            env_vars=env_vars,
        )
        uid = os.getuid()
        bootout_cmd = ["launchctl", "bootout", f"gui/{uid}", str(plist_path)]
        bootstrap_cmd = ["launchctl", "bootstrap", f"gui/{uid}", str(plist_path)]

        print("Planned launchctl commands:")
        print(f"  {' '.join(bootout_cmd)}")
        print(f"  {' '.join(bootstrap_cmd)}")

        if args.dry_run:
            print(program)
            return 0

        _run_launchctl(bootout_cmd, ignore_errors=True)
        plist_path.parent.mkdir(parents=True, exist_ok=True)
        plist_path.write_text(program, encoding="utf-8")
        if env_vars:
            plist_path.chmod(0o600)
        _run_launchctl(bootstrap_cmd)
        print(f"wrote launchd plist: {plist_path}")
        print(f"Loop log: {log_path}")
        print(f"Loop events: {events_path}")
        print(f"Loop manifest: {manifest_path}")
        print(f"Loop stdout: {stdout_path}")
        print(f"Loop stderr: {stderr_path}")
        return 0

    if action == "uninstall":
        if sys.platform != "darwin":
            print(
                "launchd lifecycle commands are supported on macOS only. "
                "Use `openclawbrain loop --systemd` on Linux/systemd hosts.",
                file=sys.stderr,
            )
            return 1
        label = _derive_loop_label(state_path)
        plist_path = _derive_launchd_plist_path(label)
        uid = os.getuid()
        bootout_cmd = ["launchctl", "bootout", f"gui/{uid}", str(plist_path)]
        print("Planned launchctl commands:")
        print(f"  {' '.join(bootout_cmd)}")

        if args.dry_run:
            return 0

        _run_launchctl(bootout_cmd, ignore_errors=True)
        try:
            plist_path.unlink(missing_ok=True)
            print(f"removed launchd plist: {plist_path}")
        except OSError as exc:
            print(f"could not remove plist: {exc}", file=sys.stderr)
            return 1
        return 0

    if action == "status":
        lock_held = _loop_lock_held(lock_path)
        checkpoint: dict[str, object] | None = None
        if checkpoint_path.exists():
            try:
                checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                checkpoint = None
        manifest: dict[str, object] | None = None
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                manifest = None
        label = _derive_loop_label(state_path)
        plist_path = _derive_launchd_plist_path(label)
        print("OpenClawBrain loop status")
        print(f"State: {state_path}")
        print(f"Sessions: {sessions_path}")
        print(f"Log: {log_path}")
        print(f"Stdout: {stdout_path}")
        print(f"Stderr: {stderr_path}")
        print(f"Events: {events_path}")
        print(f"Manifest: {manifest_path}")
        print(f"Lock: {lock_path} (held={lock_held})")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Launchd plist: {plist_path} (exists={plist_path.exists()})")
        if isinstance(checkpoint, dict):
            status = checkpoint.get("status", "unknown")
            step = checkpoint.get("step", "unknown")
            job = checkpoint.get("job", "unknown")
            started_at = checkpoint.get("started_at")
            completed_at = checkpoint.get("completed_at")
            last_exit = checkpoint.get("exit_code")
            print(f"Last run: job={job} status={status} step={step} exit={last_exit}")
            if started_at:
                print(f"  started_at={started_at}")
            if completed_at:
                print(f"  completed_at={completed_at}")
            reason = checkpoint.get("reason")
            if status in {"failed", "skipped"} and reason:
                print(f"  reason={reason}")
        if isinstance(manifest, dict):
            for job_name in ("hourly", "nightly"):
                payload = manifest.get(job_name)
                if not isinstance(payload, dict):
                    continue
                last_status = payload.get("last_status")
                if last_status is None:
                    continue
                last_reason = payload.get("last_reason")
                last_exit = payload.get("last_exit_code")
                last_completed = payload.get("last_completed_at") or payload.get("last_started_at")
                print(
                    f"{job_name.capitalize()}: status={last_status} exit={last_exit} reason={last_reason} completed_at={last_completed}"
                )
        return 0

    if action != "run":
        raise SystemExit(f"Unknown loop action: {action}")

    def log_line(message: str) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).isoformat()
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{ts}] {message}\n")

    def emit_event(event: dict[str, object]) -> None:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "agent_id": agent_id,
            "state_path": state_path,
            "sessions_path": str(sessions_path),
            **event,
        }
        _append_jsonl(events_path, payload)

    def load_manifest() -> dict[str, object]:
        if not manifest_path.exists():
            return {}
        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def write_manifest(job: str, update: dict[str, object]) -> None:
        manifest = load_manifest()
        manifest.update(
            {
                "agent_id": agent_id,
                "state_path": state_path,
                "sessions_path": str(sessions_path),
                "log_path": str(log_path),
                "events_path": str(events_path),
                "checkpoint_path": str(checkpoint_path),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        job_payload = manifest.get(job)
        if not isinstance(job_payload, dict):
            job_payload = {}
        job_payload.update(update)
        manifest[job] = job_payload
        _write_json_atomic(manifest_path, manifest)

    def write_checkpoint(
        *,
        status: str,
        job: str,
        step: str,
        exit_code: int | None = None,
        reason: str | None = None,
        started_at: str | None = None,
        completed_at: str | None = None,
    ) -> None:
        payload: dict[str, object] = {
            "status": status,
            "job": job,
            "step": step,
            "state_path": state_path,
            "sessions_path": str(sessions_path),
            "mode": str(args.mode),
            "maintain_tasks": str(args.maintain_tasks),
            "teacher_enabled": bool(args.enable_teacher),
            "train_route_model_enabled": bool(args.enable_train_route_model),
        }
        if started_at is not None:
            payload["started_at"] = started_at
        if completed_at is not None:
            payload["completed_at"] = completed_at
        if exit_code is not None:
            payload["exit_code"] = int(exit_code)
        if reason:
            payload["reason"] = reason
        _write_json_atomic(checkpoint_path, payload)

    if not Path(state_path).exists():
        log_line(f"missing state: {state_path}")
        write_checkpoint(status="failed", job="preflight", step="preflight", exit_code=2, reason="missing_state")
        return 2
    if not sessions_path.exists():
        log_line(f"missing sessions: {sessions_path}")
        write_checkpoint(status="failed", job="preflight", step="preflight", exit_code=2, reason="missing_sessions")
        return 2

    if args.dry_run:
        log_line("loop run dry-run: exiting without executing jobs")
        return 0

    def run_replay(
        *,
        job: str,
        mode: str,
        llm: str,
    ) -> int:
        replay_cmd = [
            *ocb_subprocess_prefix,
            "replay",
            "--state",
            state_path,
            "--sessions",
            str(sessions_path),
            "--mode",
            str(mode),
            "--llm",
            str(llm),
            "--checkpoint-every-seconds",
            str(args.checkpoint_every_seconds),
        ]
        if args.llm_model:
            replay_cmd.extend(["--llm-model", str(args.llm_model)])
        if args.workers is not None:
            replay_cmd.extend(["--workers", str(int(args.workers))])
        if args.resume:
            replay_cmd.append("--resume")
        if args.include_tool_results:
            replay_cmd.append("--include-tool-results")
            tool_max = (
                args.tool_result_max_chars
                if getattr(args, "tool_result_max_chars", None) is not None
                else DEFAULT_TOOL_RESULT_MAX_CHARS
            )
            replay_cmd.extend(["--tool-result-max-chars", str(int(tool_max))])
        else:
            replay_cmd.append("--no-include-tool-results")
        if getattr(args, "replay_max_interactions", None) is not None:
            replay_cmd.extend(["--replay-max-interactions", str(int(args.replay_max_interactions))])
        if getattr(args, "replay_priority", "all") != "all":
            replay_cmd.extend(["--replay-priority", str(args.replay_priority)])
        if getattr(args, "advance_offsets_on_skip", False):
            replay_cmd.append("--advance-offsets-on-skip")

        write_checkpoint(status="running", job=job, step="replay")
        emit_event({"type": "step_start", "job": job, "step": "replay"})
        replay_env = dict(os.environ)
        replay_env["PYTHONUNBUFFERED"] = "1"
        run_id = f"loop.{job}.{datetime.now(timezone.utc).isoformat()}"
        watchdog_path = scratch_root / "replay_watchdog.jsonl"
        code = _run_logged_replay_command_with_watchdog(
            replay_cmd,
            log_path=log_path,
            step_name=f"{job}.replay",
            checkpoint_path=agent_root / REPLAY_CHECKPOINT_FILENAME,
            agent_id=agent_id,
            run_id=run_id,
            watchdog_path=watchdog_path,
            progress_interval_seconds=max(0, int(args.replay_progress_interval_seconds)),
            stall_timeout_seconds=max(0, int(args.replay_stall_timeout_seconds)),
            stall_max_restarts=max(0, int(args.replay_stall_max_restarts)),
            stall_fallback_mode=str(args.replay_stall_fallback_mode),
            env=replay_env,
        )
        emit_event({"type": "step_end", "job": job, "step": "replay", "exit_code": code})
        return code

    def run_harvest(*, job: str) -> int:
        harvest_cmd = [
            *ocb_subprocess_prefix,
            "harvest",
            "--state",
            state_path,
            "--json",
        ]
        write_checkpoint(status="running", job=job, step="harvest")
        emit_event({"type": "step_start", "job": job, "step": "harvest"})
        code = _run_logged_command(harvest_cmd, log_path=log_path, step_name=f"{job}.harvest")
        emit_event({"type": "step_end", "job": job, "step": "harvest", "exit_code": code})
        return code

    def run_harvest_labels(*, job: str) -> int:
        temp_path = labels_path.with_suffix(".harvest.tmp")
        harvest_cmd = [
            *ocb_subprocess_prefix,
            "harvest",
            "--state",
            state_path,
            "--dry-run",
            "--labels-out",
            str(temp_path),
            "--json",
        ]
        write_checkpoint(status="running", job=job, step="harvest_labels")
        emit_event({"type": "step_start", "job": job, "step": "harvest_labels"})
        code = _run_logged_command(harvest_cmd, log_path=log_path, step_name=f"{job}.harvest_labels")
        emit_event({"type": "step_end", "job": job, "step": "harvest_labels", "exit_code": code})
        if code != 0:
            return code
        try:
            _refresh_labels_from_harvest(labels_path, temp_path)
        except OSError as exc:
            log_line(f"{job}: harvest labels refresh failed to replace labels file: {exc}")
            return 1
        return 0

    def run_maintain(*, job: str) -> int:
        maintain_cmd = [
            *ocb_subprocess_prefix,
            "maintain",
            "--state",
            state_path,
            "--tasks",
            str(args.maintain_tasks),
            "--llm",
            str(args.maintain_llm),
            "--embedder",
            str(args.maintain_embedder),
            "--json",
        ]
        write_checkpoint(status="running", job=job, step="maintain")
        emit_event({"type": "step_start", "job": job, "step": "maintain"})
        code = _run_logged_command(maintain_cmd, log_path=log_path, step_name=f"{job}.maintain")
        emit_event({"type": "step_end", "job": job, "step": "maintain", "exit_code": code})
        return code

    def run_async_route(
        *,
        job: str,
        traces_out: Path | None,
    ) -> int:
        async_cmd = [
            *ocb_subprocess_prefix,
            "async-route-pg",
            "--state",
            state_path,
            "--teacher",
            str(args.teacher),
            "--teacher-model",
            str(args.teacher_model),
            "--since-hours",
            str(args.since_hours),
            "--max-queries",
            str(args.max_queries),
            "--sample-rate",
            str(args.sample_rate),
            "--max-candidates-per-node",
            str(args.max_candidates_per_node),
            "--max-decision-points",
            str(args.max_decision_points),
            "--score-scale",
            str(args.score_scale),
            "--reward-source",
            str(args.reward_source),
            "--apply",
            "--json",
            "--labels-out",
            str(labels_path),
        ]
        if args.reward_weights:
            async_cmd.extend(["--reward-weights", str(args.reward_weights)])
        if args.write_relevance_metadata:
            async_cmd.append("--write-relevance-metadata")
        else:
            async_cmd.append("--no-write-relevance-metadata")
        if traces_out is not None:
            async_cmd.extend(["--traces-out", str(traces_out)])

        write_checkpoint(status="running", job=job, step="async_route_pg")
        emit_event({"type": "step_start", "job": job, "step": "async_route_pg"})
        code = _run_logged_command(async_cmd, log_path=log_path, step_name=f"{job}.async_route_pg")
        emit_event({"type": "step_end", "job": job, "step": "async_route_pg", "exit_code": code})
        return code

    def run_train_route(
        *,
        job: str,
        traces_in: Path,
        out_path: Path,
    ) -> int:
        train_cmd = [
            *ocb_subprocess_prefix,
            "train-route-model",
            "--state",
            state_path,
            "--traces-in",
            str(traces_in),
            "--labels-in",
            str(labels_path),
            "--out",
            str(out_path),
            "--json",
        ]
        if args.reward_weights:
            train_cmd.extend(["--reward-weights", str(args.reward_weights)])
        write_checkpoint(status="running", job=job, step="train_route_model")
        emit_event({"type": "step_start", "job": job, "step": "train_route_model"})
        code = _run_logged_command(train_cmd, log_path=log_path, step_name=f"{job}.train_route_model")
        emit_event({"type": "step_end", "job": job, "step": "train_route_model", "exit_code": code})
        return code

    def run_dream(*, job: str, traces_dir: Path) -> int:
        dream_cmd = [
            *ocb_prefix,
            "dream",
            "--state",
            state_path,
            "--once",
            "--teacher",
            str(args.teacher),
            "--teacher-model",
            str(args.teacher_model),
            "--since-hours",
            str(args.dream_since_hours),
            "--max-queries",
            str(args.dream_max_queries),
            "--sample-rate",
            str(args.dream_sample_rate),
            "--max-candidates-per-node",
            str(args.dream_max_candidates_per_node),
            "--max-decision-points",
            str(args.dream_max_decision_points),
            "--score-scale",
            str(args.score_scale),
            "--reward-source",
            str(args.reward_source),
            "--labels-out",
            str(labels_path),
            "--traces-dir",
            str(traces_dir),
            "--json",
        ]
        if args.reward_weights:
            dream_cmd.extend(["--reward-weights", str(args.reward_weights)])

        write_checkpoint(status="running", job=job, step="dream")
        emit_event({"type": "step_start", "job": job, "step": "dream"})
        run_id = f"loop.{job}.dream.{datetime.now(timezone.utc).isoformat()}"
        watchdog_path = scratch_root / "dream_watchdog.jsonl"
        progress_paths = [labels_path, traces_dir]
        code = _run_logged_command_with_watchdog(
            dream_cmd,
            log_path=log_path,
            step_name=f"{job}.dream",
            watchdog_path=watchdog_path,
            run_id=run_id,
            progress_paths=progress_paths,
            stall_timeout_seconds=max(0, int(args.dream_stall_timeout_seconds)),
            stall_max_restarts=max(0, int(args.dream_stall_max_restarts)),
        )
        emit_event({"type": "step_end", "job": job, "step": "dream", "exit_code": code})
        return code

    def run_job(
        *,
        job: str,
        mode: str,
        llm: str,
        include_maintain: bool,
        include_teacher: bool,
    ) -> int:
        lock_fd, acquired = _try_acquire_loop_lock(lock_path)
        if not acquired:
            msg = f"{job}: loop lock held; skipping"
            log_line(msg)
            write_checkpoint(status="skipped", job=job, step="lock", exit_code=0, reason="loop_lock_held")
            emit_event({"type": "job_skipped", "job": job, "reason": "loop_lock_held"})
            write_manifest(job, {"last_status": "skipped", "last_reason": "loop_lock_held"})
            if args.skip_if_locked:
                return 0
            raise SystemExit("loop lock held")

        state_lock_ready, resume_cmd, state_lock_reason = _maybe_pause_serve_for_state_lock(
            state_path=state_path,
            pause_when_locked=bool(args.pause_serve_when_locked),
            timeout_seconds=max(0, int(args.pause_serve_timeout_seconds)),
        )
        if not state_lock_ready:
            reason = state_lock_reason or "state_lock_held"
            msg = f"{job}: state lock held; skipping (reason={reason})"
            log_line(msg)
            write_checkpoint(status="skipped", job=job, step="lock", exit_code=0, reason=reason)
            emit_event({"type": "job_skipped", "job": job, "reason": reason})
            write_manifest(job, {"last_status": "skipped", "last_reason": reason})
            return 0

        started_at = datetime.now(timezone.utc).isoformat()
        try:
            write_checkpoint(status="running", job=job, step="preflight", started_at=started_at)
            write_manifest(job, {"last_status": "running", "last_started_at": started_at})
            emit_event({"type": "job_start", "job": job, "started_at": started_at})
            log_line(f"{job}: run start mode={mode} llm={llm} sessions={sessions_path}")

            replay_code = run_replay(job=job, mode=mode, llm=llm)
            if replay_code != 0:
                log_line(f"{job}: replay failed exit={replay_code}")
                completed_at = datetime.now(timezone.utc).isoformat()
                write_checkpoint(
                    status="failed",
                    job=job,
                    step="replay",
                    exit_code=replay_code,
                    reason="replay_failed",
                    started_at=started_at,
                    completed_at=completed_at,
                )
                write_manifest(
                    job,
                    {
                        "last_status": "failed",
                        "last_step": "replay",
                        "last_exit_code": replay_code,
                        "last_reason": "replay_failed",
                        "last_completed_at": completed_at,
                    },
                )
                return replay_code

            if job == "hourly":
                harvest_code = run_harvest(job=job)
                if harvest_code != 0:
                    log_line(f"{job}: harvest failed exit={harvest_code}")
                    completed_at = datetime.now(timezone.utc).isoformat()
                    write_checkpoint(
                        status="failed",
                        job=job,
                        step="harvest",
                        exit_code=harvest_code,
                        reason="harvest_failed",
                        started_at=started_at,
                        completed_at=completed_at,
                    )
                    write_manifest(
                        job,
                        {
                            "last_status": "failed",
                            "last_step": "harvest",
                            "last_exit_code": harvest_code,
                            "last_reason": "harvest_failed",
                            "last_completed_at": completed_at,
                        },
                    )
                    return harvest_code
                if include_maintain:
                    maintain_code = run_maintain(job=job)
                    if maintain_code != 0:
                        log_line(f"{job}: maintenance failed exit={maintain_code}")
                        completed_at = datetime.now(timezone.utc).isoformat()
                        write_checkpoint(
                            status="failed",
                            job=job,
                            step="maintain",
                            exit_code=maintain_code,
                            reason="maintenance_failed",
                            started_at=started_at,
                            completed_at=completed_at,
                        )
                        write_manifest(
                            job,
                            {
                                "last_status": "failed",
                                "last_step": "maintain",
                                "last_exit_code": maintain_code,
                                "last_reason": "maintenance_failed",
                                "last_completed_at": completed_at,
                            },
                        )
                        return maintain_code
                else:
                    log_line(f"{job}: maintenance skipped")
                if args.harvest_labels:
                    labels_refresh_code = run_harvest_labels(job=job)
                    if labels_refresh_code != 0:
                        log_line(f"{job}: harvest labels refresh failed exit={labels_refresh_code}")
            else:
                if include_maintain:
                    maintain_code = run_maintain(job=job)
                    if maintain_code != 0:
                        log_line(f"{job}: maintenance failed exit={maintain_code}")
                        completed_at = datetime.now(timezone.utc).isoformat()
                        write_checkpoint(
                            status="failed",
                            job=job,
                            step="maintain",
                            exit_code=maintain_code,
                            reason="maintenance_failed",
                            started_at=started_at,
                            completed_at=completed_at,
                        )
                        write_manifest(
                            job,
                            {
                                "last_status": "failed",
                                "last_step": "maintain",
                                "last_exit_code": maintain_code,
                                "last_reason": "maintenance_failed",
                                "last_completed_at": completed_at,
                            },
                        )
                        return maintain_code
                else:
                    log_line(f"{job}: maintenance skipped")
                if args.harvest_labels:
                    labels_refresh_code = run_harvest_labels(job=job)
                    if labels_refresh_code != 0:
                        log_line(f"{job}: harvest labels refresh failed exit={labels_refresh_code}")

                traces_out: Path | None = None
                if include_teacher and args.enable_async_route_pg:
                    if args.enable_train_route_model:
                        trace_ts = datetime.now(timezone.utc)
                        traces_out = scratch_root / (
                            f"loop.{job}.{trace_ts.strftime('%Y%m%dT%H%M%S')}"
                            f"{trace_ts.microsecond // 1000:03d}Z.route_traces.jsonl"
                        )
                    async_code = run_async_route(job=job, traces_out=traces_out)
                    if async_code != 0:
                        log_line(f"{job}: async-route-pg failed exit={async_code}")
                        completed_at = datetime.now(timezone.utc).isoformat()
                        write_checkpoint(
                            status="failed",
                            job=job,
                            step="async_route_pg",
                            exit_code=async_code,
                            reason="async_route_pg_failed",
                            started_at=started_at,
                            completed_at=completed_at,
                        )
                        write_manifest(
                            job,
                            {
                                "last_status": "failed",
                                "last_step": "async_route_pg",
                                "last_exit_code": async_code,
                                "last_reason": "async_route_pg_failed",
                                "last_completed_at": completed_at,
                            },
                        )
                        return async_code
                else:
                    if not include_teacher:
                        log_line(f"{job}: async-route-pg skipped (teacher disabled)")
                    else:
                        log_line(f"{job}: async-route-pg skipped (disabled)")

                if args.enable_train_route_model:
                    if not include_teacher:
                        log_line(f"{job}: train-route-model skipped (teacher disabled)")
                    elif traces_out is None:
                        log_line(f"{job}: train-route-model skipped (no traces)")
                    else:
                        out_path = (
                            Path(args.train_route_model_out).expanduser()
                            if args.train_route_model_out
                            else Path(state_path).expanduser().parent / "route_model.npz"
                        )
                        train_code = run_train_route(job=job, traces_in=traces_out, out_path=out_path)
                        if train_code != 0:
                            log_line(f"{job}: train-route-model failed exit={train_code}")
                            completed_at = datetime.now(timezone.utc).isoformat()
                            write_checkpoint(
                                status="failed",
                                job=job,
                                step="train_route_model",
                                exit_code=train_code,
                                reason="train_route_model_failed",
                                started_at=started_at,
                                completed_at=completed_at,
                            )
                            write_manifest(
                                job,
                                {
                                    "last_status": "failed",
                                    "last_step": "train_route_model",
                                    "last_exit_code": train_code,
                                    "last_reason": "train_route_model_failed",
                                    "last_completed_at": completed_at,
                                },
                            )
                            return train_code

                if include_teacher and args.enable_dreaming:
                    dream_dir = scratch_root / "dream"
                    dream_dir.mkdir(parents=True, exist_ok=True)
                    dream_code = run_dream(job=job, traces_dir=dream_dir)
                    if dream_code != 0:
                        log_line(f"{job}: dream failed exit={dream_code}")
                        completed_at = datetime.now(timezone.utc).isoformat()
                        write_checkpoint(
                            status="failed",
                            job=job,
                            step="dream",
                            exit_code=dream_code,
                            reason="dream_failed",
                            started_at=started_at,
                            completed_at=completed_at,
                        )
                        write_manifest(
                            job,
                            {
                                "last_status": "failed",
                                "last_step": "dream",
                                "last_exit_code": dream_code,
                                "last_reason": "dream_failed",
                                "last_completed_at": completed_at,
                            },
                        )
                        return dream_code
                else:
                    if not include_teacher:
                        log_line(f"{job}: dream skipped (teacher disabled)")
                    else:
                        log_line(f"{job}: dream skipped (disabled)")

            completed_at = datetime.now(timezone.utc).isoformat()
            write_checkpoint(status="ok", job=job, step="complete", exit_code=0, started_at=started_at, completed_at=completed_at)
            write_manifest(
                job,
                {
                    "last_status": "ok",
                    "last_step": "complete",
                    "last_exit_code": 0,
                    "last_completed_at": completed_at,
                },
            )
            emit_event({"type": "job_end", "job": job, "status": "ok", "completed_at": completed_at})
            log_line(f"{job}: run complete")
            return 0
        finally:
            if resume_cmd is not None:
                _run_launchctl_returncode(resume_cmd)
            if lock_fd is not None:
                try:
                    import fcntl  # type: ignore

                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                except OSError:
                    pass
                os.close(lock_fd)

    def parse_manifest_time(value: object) -> datetime | None:
        if not isinstance(value, str):
            return None
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
        return parsed

    def should_run_hourly(now_utc: datetime, manifest: dict[str, object]) -> bool:
        hourly_payload = manifest.get("hourly") if isinstance(manifest.get("hourly"), dict) else {}
        last_run = parse_manifest_time(hourly_payload.get("last_completed_at")) or parse_manifest_time(hourly_payload.get("last_started_at"))
        if last_run is None:
            return True
        return (now_utc - last_run).total_seconds() >= max(60, int(args.hourly_interval_seconds))

    def should_run_nightly(now_local: datetime, manifest: dict[str, object]) -> bool:
        nightly_payload = manifest.get("nightly") if isinstance(manifest.get("nightly"), dict) else {}
        last_run = parse_manifest_time(nightly_payload.get("last_completed_at")) or parse_manifest_time(nightly_payload.get("last_started_at"))
        if last_run is not None:
            last_local = last_run.astimezone(now_local.tzinfo)
            if last_local.date() == now_local.date():
                return False
        scheduled = now_local.replace(hour=int(args.nightly_hour), minute=int(args.nightly_minute), second=0, microsecond=0)
        return now_local >= scheduled

    def next_sleep_seconds(now_local: datetime, now_utc: datetime, manifest: dict[str, object]) -> float:
        hourly_payload = manifest.get("hourly") if isinstance(manifest.get("hourly"), dict) else {}
        last_run = parse_manifest_time(hourly_payload.get("last_completed_at")) or parse_manifest_time(hourly_payload.get("last_started_at"))
        interval = max(60, int(args.hourly_interval_seconds))
        if last_run is None:
            next_hourly = now_utc
        else:
            next_hourly = last_run + timedelta(seconds=interval)

        scheduled_today = now_local.replace(hour=int(args.nightly_hour), minute=int(args.nightly_minute), second=0, microsecond=0)
        if now_local >= scheduled_today:
            scheduled_today = scheduled_today + timedelta(days=1)
        next_nightly = scheduled_today.astimezone(now_utc.tzinfo)

        next_due = min(next_hourly, next_nightly)
        return max(5.0, (next_due - now_utc).total_seconds())

    log_line("loop scheduler started")
    emit_event({"type": "loop_start"})
    while True:
        now_utc = datetime.now(timezone.utc)
        now_local = now_utc.astimezone()
        manifest = load_manifest()

        if should_run_hourly(now_utc, manifest):
            run_job(job="hourly", mode="edges-only", llm="none", include_maintain=bool(args.maintain), include_teacher=False)

        if should_run_nightly(now_local, manifest):
            run_job(job="nightly", mode=str(args.mode), llm=str(args.llm), include_maintain=bool(args.maintain), include_teacher=bool(args.enable_teacher))

        sleep_seconds = next_sleep_seconds(now_local, now_utc, load_manifest())
        time.sleep(sleep_seconds)


def cmd_train_route_model(args: argparse.Namespace) -> int:
    """Train learned route model from traces and optional labels."""
    try:
        parsed_weights = RewardWeights.from_string(args.reward_weights) if args.reward_weights else RewardWeights.from_env()
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    summary = train_route_model(
        state_path=str(args.state),
        traces_in=str(args.traces_in),
        labels_in=str(args.labels_in) if args.labels_in else None,
        out_path=str(args.out),
        rank=max(1, int(args.rank)),
        epochs=max(1, int(args.epochs)),
        lr=float(args.lr),
        label_temp=float(args.label_temp),
        reward_weights=parsed_weights,
    )

    if args.json:
        print(write_summary_json(summary))
    else:
        payload = summary.to_dict()
        print(
            f"train-route-model: points_used={payload['points_used']} "
            f"initial_ce={payload['initial_ce_loss']:.6f} "
            f"final_ce={payload['final_ce_loss']:.6f} "
            f"out={payload['out_path']}"
        )
    return 0


def cmd_build_all(args: argparse.Namespace) -> int:
    """Run unattended brain-building pipeline for all configured agents."""
    _maybe_warn_long_running()
    agent_ids = _resolve_agent_ids(args)
    parallel = max(1, int(args.parallel_agents))
    if args.advance_offsets_on_skip is None:
        args.advance_offsets_on_skip = bool(str(args.mode) != "edges-only")
    if args.include_tool_results and getattr(args, "tool_result_max_chars", None) is None:
        args.tool_result_max_chars = int(DEFAULT_TOOL_RESULT_MAX_CHARS)
    ocb_prefix = _resolve_subprocess_prefix()
    run_ts = datetime.now(timezone.utc)
    ts_label = run_ts.strftime("%Y%m%dT%H%M%S") + f"{run_ts.microsecond // 1000:03d}Z"
    run_started_at = time.perf_counter()

    root = Path.home() / ".openclawbrain" / "scratch"
    root.mkdir(parents=True, exist_ok=True)
    root_manifest_path = root / f"build-all.{ts_label}.manifest.json"
    stall_audit_path = root / f"build-all.{ts_label}.stall_audit.jsonl"
    events_jsonl_path = (
        Path(args.events_jsonl)
        if getattr(args, "events_jsonl", None) is not None
        else root / f"build-all.{ts_label}.events.jsonl"
    )
    events_jsonl_path = events_jsonl_path.expanduser()

    event_lock = threading.Lock()

    def emit_event(event: dict[str, object]) -> None:
        with event_lock:
            payload = {
                "run_id": ts_label,
                **event,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            _append_jsonl(events_jsonl_path, payload)

    manifest: dict[str, object]
    emit_event(
        {
            "type": "run_start",
            "agent_id": None,
            "status": "running",
            "artifact_paths": {
                "manifest_path": str(root_manifest_path),
                "events_jsonl": str(events_jsonl_path),
                "stall_audit_jsonl": str(stall_audit_path),
            },
        }
    )

    manifest = _build_all_root_manifest_payload(
        run_id=ts_label,
        run_ts=run_ts,
        args=args,
        ocb_prefix=ocb_prefix,
        agent_ids=agent_ids,
        parallel_agents=parallel,
        events_jsonl=events_jsonl_path,
        stall_audit_jsonl=stall_audit_path,
        status="running",
        agents=[],
    )
    _write_json_atomic(root_manifest_path, manifest)

    results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=parallel) as executor:
        future_map = {
            executor.submit(
                _build_all_agent_pipeline,
                agent_id=agent_id,
                args=args,
                ocb_prefix=ocb_prefix,
                ts_label=ts_label,
                run_ts=run_ts,
                stall_audit_path=stall_audit_path,
                emit_event=emit_event,
            ): agent_id
            for agent_id in agent_ids
        }
        for future in as_completed(future_map):
            agent_id = future_map[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001
                result = {
                    "agent_id": agent_id,
                    "exit_code": 1,
                    "error": str(exc),
                }
            results.append(result)
            manifest = _build_all_root_manifest_payload(
                run_id=ts_label,
                run_ts=run_ts,
                args=args,
                ocb_prefix=ocb_prefix,
                agent_ids=agent_ids,
                parallel_agents=parallel,
                events_jsonl=events_jsonl_path,
                stall_audit_jsonl=stall_audit_path,
                status="running",
                agents=results,
            )
            _write_json_atomic(root_manifest_path, manifest)

    manifest = _build_all_root_manifest_payload(
        run_id=ts_label,
        run_ts=run_ts,
        args=args,
        ocb_prefix=ocb_prefix,
        agent_ids=agent_ids,
        parallel_agents=parallel,
        events_jsonl=events_jsonl_path,
        stall_audit_jsonl=stall_audit_path,
        status="complete",
        agents=results,
    )
    _write_json_atomic(root_manifest_path, manifest)
    emit_event(
        {
            "type": "run_end",
            "agent_id": None,
            "status": "ok" if all(int(item.get("exit_code", 1)) == 0 for item in results) else "failed",
            "duration_seconds": max(0.0, time.perf_counter() - run_started_at),
            "artifact_paths": {
                "manifest_path": str(root_manifest_path),
                "events_jsonl": str(events_jsonl_path),
                "stall_audit_jsonl": str(stall_audit_path),
            },
        }
    )
    print(f"[build-all] manifest={root_manifest_path}")
    return 0 if all(int(item.get("exit_code", 1)) == 0 for item in results) else 1


def cmd_bootstrap(args: argparse.Namespace) -> int:
    """Fast-boot a brain and install serve+loop services."""
    agent_id = str(args.agent).strip()
    if not agent_id:
        raise SystemExit("--agent is required")

    state_path = str(Path(_state_path_for_agent(agent_id)).expanduser())
    workspace = Path(args.workspace).expanduser() if args.workspace else _resolve_agent_workspace(agent_id)
    workspace = workspace.expanduser()
    if not workspace.exists():
        raise SystemExit(f"workspace not found: {workspace}")

    ocb_bin = _resolve_openclawbrain_bin()

    def run_step(label: str, argv: list[str]) -> int:
        print(f"$ {' '.join(shlex.quote(part) for part in argv)}")
        result = _run_subprocess_command(argv)
        if result.returncode != 0:
            print(f"{label} failed (exit={result.returncode})", file=sys.stderr)
        return int(result.returncode)

    codes: list[int] = []
    state_file = Path(state_path)
    if not state_file.exists():
        output_dir = state_file.parent
        init_cmd = [
            ocb_bin,
            "init",
            "--workspace",
            str(workspace),
            "--output",
            str(output_dir),
        ]
        if args.fast:
            init_cmd.extend(["--embedder", "local", "--llm", "none", "--llm-split-mode", "off"])
        codes.append(run_step("init", init_cmd))
        if codes[-1] != 0:
            return codes[-1]
    else:
        print(f"state exists: {state_path}")

    serve_cmd = [ocb_bin, "serve", "install", "--state", state_path]
    if args.env_file:
        serve_cmd.extend(["--env-file", str(Path(args.env_file).expanduser())])
    codes.append(run_step("serve install", serve_cmd))

    loop_cmd = [ocb_bin, "loop", "install", "--state", state_path]
    if args.fast:
        loop_cmd.append("--fast")
    if args.env_file:
        loop_cmd.extend(["--env-file", str(Path(args.env_file).expanduser())])
    codes.append(run_step("loop install", loop_cmd))

    if all(code == 0 for code in codes):
        print("Verify:")
        print(f"  openclawbrain route-audit --state {state_path} && openclawbrain serve status --state {state_path}")
        return 0
    return 1


def _resolve_hooks_path(hooks_path: str | None) -> Path:
    if hooks_path:
        path = Path(hooks_path).expanduser()
    else:
        path = Path(__file__).resolve().parents[1] / "integrations" / "openclaw" / "hooks" / "openclawbrain-context-injector"
    if not path.exists():
        raise SystemExit(
            "openclaw hook path not found; pass --hooks-path pointing to integrations/openclaw/hooks/openclawbrain-context-injector"
        )
    return path


def _run_subprocess_command(argv: list[str]) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(argv, check=False, env=_subprocess_env())
    except FileNotFoundError:
        return subprocess.CompletedProcess(argv, 127)


def cmd_openclaw(args: argparse.Namespace) -> int:
    """Install/uninstall/status helper for OpenClaw integration."""
    if args.state:
        state_path = _resolve_state_path(args.state, allow_default=False)
    else:
        state_path = _state_path_for_agent(args.agent)
    if state_path is None:
        raise SystemExit("--state or --agent is required")
    state_path = str(Path(state_path).expanduser())

    def confirm(action: str) -> bool:
        if args.yes:
            return True
        response = input(f"{action} OpenClaw integration for {state_path}? [y/N]: ").strip().lower()
        return response in {"y", "yes"}

    def run_step(label: str, argv: list[str]) -> int:
        print(f"$ {' '.join(shlex.quote(part) for part in argv)}")
        result = _run_subprocess_command(argv)
        if result.returncode != 0:
            print(f"{label} failed (exit={result.returncode})", file=sys.stderr)
        return int(result.returncode)

    ocb_prefix = [sys.executable, "-m", "openclawbrain.cli"]

    if args.action == "install":
        if not confirm("Install"):
            print("Aborted.")
            return 1
        hooks_path = _resolve_hooks_path(args.hooks_path)
        labels_path = _default_labels_path(state_path)
        scratch_dir = Path(state_path).expanduser().parent / "scratch"
        scratch_dir.mkdir(parents=True, exist_ok=True)
        traces_out = scratch_dir / INIT_ROUTE_TRACES_FILENAME
        codes = []
        serve_cmd = [*ocb_prefix, "serve", "install", "--state", state_path]
        if args.env_file:
            serve_cmd.extend(["--env-file", str(Path(args.env_file).expanduser())])
        codes.append(run_step("serve install", serve_cmd))
        codes.append(run_step("hooks install", ["openclaw", "hooks", "install", str(hooks_path)]))
        codes.append(run_step("hooks enable", ["openclaw", "hooks", "enable", "openclawbrain-context-injector"]))
        loop_cmd = [*ocb_prefix, "loop", "install", "--state", state_path]
        if args.env_file:
            loop_cmd.extend(["--env-file", str(Path(args.env_file).expanduser())])
        codes.append(run_step("loop install", loop_cmd))
        if not args.skip_init_route_model:
            preserved = [record for record in read_labels_jsonl(labels_path) if record.reward_source != RewardSource.HARVESTER]
            harvest_cmd = [
                *ocb_prefix,
                "harvest",
                "--state",
                state_path,
                "--dry-run",
                "--labels-out",
                str(labels_path),
            ]
            harvest_code = run_step("init harvest labels", harvest_cmd)
            codes.append(harvest_code)
            if harvest_code == 0 and preserved:
                append_labels_jsonl(labels_path, preserved)
            if harvest_code == 0:
                async_cmd = [
                    *ocb_prefix,
                    "async-route-pg",
                    "--state",
                    state_path,
                    "--since-hours",
                    str(INIT_ROUTE_SINCE_HOURS),
                    "--max-queries",
                    str(INIT_ROUTE_MAX_QUERIES),
                    "--sample-rate",
                    str(INIT_ROUTE_SAMPLE_RATE),
                    "--labels-out",
                    str(labels_path),
                    "--traces-out",
                    str(traces_out),
                    "--include-query-vector",
                    "--apply",
                ]
                async_code = run_step("init async-route-pg", async_cmd)
                codes.append(async_code)
                if async_code == 0 and traces_out.exists() and traces_out.stat().st_size > 0:
                    train_cmd = [
                        *ocb_prefix,
                        "train-route-model",
                        "--state",
                        state_path,
                        "--traces-in",
                        str(traces_out),
                        "--labels-in",
                        str(labels_path),
                        "--out",
                        str(Path(state_path).expanduser().parent / "route_model.npz"),
                    ]
                    codes.append(run_step("init train-route-model", train_cmd))
        codes.append(run_step("gateway restart", ["openclaw", "gateway", "restart"]))
        return 0 if all(code == 0 for code in codes) else 1

    if args.action == "uninstall":
        if not confirm("Uninstall"):
            print("Aborted.")
            return 1
        codes = []
        codes.append(run_step("hooks disable", ["openclaw", "hooks", "disable", "openclawbrain-context-injector"]))
        codes.append(run_step("gateway restart", ["openclaw", "gateway", "restart"]))
        codes.append(run_step("loop uninstall", [*ocb_prefix, "loop", "uninstall", "--state", state_path]))
        codes.append(run_step("serve uninstall", [*ocb_prefix, "serve", "uninstall", "--state", state_path]))
        return 0 if all(code == 0 for code in codes) else 1

    if args.action == "status":
        codes = []
        codes.append(run_step("serve status", [*ocb_prefix, "serve", "status", "--state", state_path]))
        codes.append(run_step("loop status", [*ocb_prefix, "loop", "status", "--state", state_path]))
        codes.append(run_step("hooks info", ["openclaw", "hooks", "info", "openclawbrain-context-injector"]))
        return 0 if all(code == 0 for code in codes) else 1

    raise SystemExit(f"unknown action: {args.action}")


def main(argv: list[str] | None = None) -> int:
    """main."""
    args = _build_parser().parse_args(argv)
    handlers = {
        "bootstrap": cmd_bootstrap,
        "init": cmd_init,
        "query": cmd_query,
        "learn": cmd_learn,
        "merge": cmd_merge,
        "maintain": cmd_maintain,
        "anchor": cmd_anchor,
        "connect": cmd_connect,
        "compact": cmd_compact,
        "reembed": cmd_reembed,
        "daemon": cmd_daemon,
        "serve": cmd_serve,
        "inject": cmd_inject,
        "replay": cmd_replay,
        "status": cmd_status,
        "self-learn": cmd_self_correct,
        "self-correct": cmd_self_correct,
        "harvest": cmd_harvest,
        "health": cmd_health,
        "report": cmd_report,
        "route-audit": cmd_route_audit,
        "async-route-pg": cmd_async_route_pg,
        "dream": cmd_dream,
        "dreaming": cmd_dream,
        "loop": cmd_loop,
        "train-route-model": cmd_train_route_model,
        "build-all": cmd_build_all,
        "openclaw": cmd_openclaw,
        "journal": cmd_journal,
        "doctor": cmd_doctor,
        "info": cmd_info,
        "sync": cmd_sync,
    }
    handler = handlers[args.command]
    with _state_lock_context_for_command(args):
        return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
