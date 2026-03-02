"""Tool/action provenance helpers for OpenClawBrain replay."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

from .graph import Edge, Graph, Node


TOOL_ACTION_PREFIX = "tool_action::"
TOOL_EVIDENCE_PREFIX = "tool_evidence::"

TIER1_STORE_EVIDENCE = frozenset(
    {
        "web_search",
        "web_fetch",
        "read",
        "image",
        "pdf",
        "openai-whisper",
        "openai-whisper-api",
        "summarize",
    }
)

TIER2_ACTION_ONLY = frozenset(
    {
        "exec",
        "process",
        "write",
        "edit",
        "browser",
        "canvas",
        "nodes",
        "message",
        "tts",
    }
)

TIER3_SKIP = frozenset({"subagents"})

SENSITIVE_KEY_FRAGMENTS = frozenset(
    {
        "api_key",
        "apikey",
        "access_key",
        "access_token",
        "refresh_token",
        "secret",
        "password",
        "passwd",
        "authorization",
        "bearer",
        "cookie",
        "set-cookie",
        "client_secret",
        "private_key",
        "session",
        "token",
    }
)

REDACTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{8,}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"AIza[0-9A-Za-z\-_]{20,}"),
    re.compile(r"(?is)-----BEGIN [A-Z ]+PRIVATE KEY-----.*?-----END [A-Z ]+PRIVATE KEY-----"),
    re.compile(r"(?i)(secret|token|password)\s*[:=]\s*[^\s\"']{6,}"),
)


@dataclass(frozen=True)
class ToolProvenanceConfig:
    """Configuration for tool-action and evidence nodes."""

    store_evidence_tools: frozenset[str] = field(default_factory=lambda: TIER1_STORE_EVIDENCE)
    action_only_tools: frozenset[str] = field(default_factory=lambda: TIER2_ACTION_ONLY)
    skip_tools: frozenset[str] = field(default_factory=lambda: TIER3_SKIP)
    max_evidence_chars: int = 4000
    max_argument_chars: int = 800


def _normalize_tool_name(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip().lower()
    return cleaned or None


def _safe_tool_slug(tool_name: str | None) -> str:
    if not tool_name:
        return "tool"
    cleaned = re.sub(r"[^a-z0-9_.-]+", "-", tool_name.lower()).strip("-")
    return cleaned or "tool"


def _hash_call_id(call_id: str) -> str:
    return hashlib.sha256(call_id.encode("utf-8")).hexdigest()[:12]


def _redact_string(value: str) -> str:
    redacted = value
    for pattern in REDACTION_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def _should_redact_key(key: str) -> bool:
    lowered = key.lower()
    return any(fragment in lowered for fragment in SENSITIVE_KEY_FRAGMENTS)


def _redact_value(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(key, str) and _should_redact_key(key):
                redacted[key] = "[REDACTED]"
            else:
                redacted[key] = _redact_value(item)
        return redacted
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    if isinstance(value, tuple):
        return [_redact_value(item) for item in value]
    if isinstance(value, str):
        return _redact_string(value)
    return value


def _maybe_parse_json(value: str) -> Any | None:
    stripped = value.strip()
    if not stripped:
        return None
    if stripped[0] not in "[{":
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return None


def _stringify(value: Any, *, max_chars: int) -> str:
    if isinstance(value, str):
        text = _redact_string(value)
    else:
        redacted = _redact_value(value)
        text = json.dumps(redacted, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    if max_chars <= 0:
        return ""
    if len(text) > max_chars:
        return text[:max_chars]
    return text


def _extract_arguments(raw: object) -> Any:
    if raw is None:
        return None
    if isinstance(raw, str):
        parsed = _maybe_parse_json(raw)
        return parsed if parsed is not None else raw
    return raw


def _ensure_node(graph: Graph, node_id: str, *, content: str, summary: str, metadata: dict[str, Any]) -> bool:
    if graph.get_node(node_id) is not None:
        return False
    graph.add_node(Node(id=node_id, content=content, summary=summary, metadata=dict(metadata)))
    return True


def _ensure_edge(
    graph: Graph,
    source: str,
    target: str,
    *,
    weight: float,
    kind: str,
    metadata: dict[str, Any] | None = None,
) -> bool:
    existing = graph._edges.get(source, {}).get(target)
    if existing is not None:
        return False
    graph.add_edge(
        Edge(
            source=source,
            target=target,
            weight=weight,
            kind=kind,
            metadata=dict(metadata or {}),
        )
    )
    return True


def build_tool_provenance(
    *,
    graph: Graph,
    fired_nodes: list[str],
    tool_calls: list[dict[str, object]],
    tool_results: list[dict[str, object]] | None = None,
    session: str | None = None,
    session_path: str | None = None,
    line_no_start: int | None = None,
    line_no_end: int | None = None,
    config: ToolProvenanceConfig | None = None,
) -> dict[str, int]:
    """Create tool_action/tool_evidence nodes and edges for tool calls."""
    if not tool_calls:
        return {"actions": 0, "evidence": 0, "edges": 0}

    cfg = config or ToolProvenanceConfig()
    results_by_id: dict[str, dict[str, object]] = {}
    if tool_results:
        for result in tool_results:
            if not isinstance(result, dict):
                continue
            call_id = result.get("tool_call_id") or result.get("toolCallId") or result.get("tool_call_id")
            if isinstance(call_id, str) and call_id:
                results_by_id[call_id] = result

    created_actions = 0
    created_evidence = 0
    created_edges = 0

    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        call_id = call.get("id") or call.get("toolCallId") or call.get("tool_call_id")
        call_id = call_id if isinstance(call_id, str) and call_id.strip() else None
        tool_name = _normalize_tool_name(call.get("name") or call.get("toolName") or call.get("tool_name"))
        tool_slug = _safe_tool_slug(tool_name)

        if tool_name in cfg.skip_tools:
            continue

        if call_id is None:
            fallback_basis = f"{tool_slug}:{session_path or session or ''}:{line_no_start or ''}:{call.get('arguments')}"
            call_id = f"fallback:{_hash_call_id(fallback_basis)}"

        action_hash = _hash_call_id(call_id)
        action_node_id = f"{TOOL_ACTION_PREFIX}{tool_slug}::{action_hash}"

        raw_args = _extract_arguments(call.get("arguments"))
        args_text = ""
        if raw_args is not None:
            args_text = _stringify(raw_args, max_chars=cfg.max_argument_chars)

        action_content = f"Tool action: {tool_slug}"
        if args_text:
            action_content = f"{action_content}\nArgs: {args_text}"

        action_metadata: dict[str, Any] = {
            "type": "tool_action",
            "tool_name": tool_name,
            "tool_call_id": call_id,
            "session": session,
            "session_path": session_path,
            "line_no_start": line_no_start,
            "line_no_end": line_no_end,
        }
        if raw_args is not None:
            action_metadata["arguments"] = _redact_value(raw_args)

        if _ensure_node(
            graph,
            action_node_id,
            content=action_content,
            summary=f"Tool action: {tool_slug}",
            metadata=action_metadata,
        ):
            created_actions += 1

        for fired_node in fired_nodes:
            if _ensure_edge(
                graph,
                fired_node,
                action_node_id,
                weight=0.20,
                kind="tool_action",
                metadata={"source": "replay"},
            ):
                created_edges += 1

        if tool_name in cfg.action_only_tools:
            continue

        if tool_name not in cfg.store_evidence_tools:
            continue

        result = results_by_id.get(call_id) or call.get("result")
        if isinstance(result, dict):
            evidence_text = result.get("content") or result.get("text") or result.get("result")
            result_line = result.get("line_no")
        else:
            evidence_text = result
            result_line = None

        if evidence_text is None:
            continue

        evidence_payload = _extract_arguments(evidence_text)
        if evidence_payload is None:
            continue
        evidence_text_out = _stringify(evidence_payload, max_chars=cfg.max_evidence_chars)
        if not evidence_text_out:
            continue

        evidence_node_id = f"{TOOL_EVIDENCE_PREFIX}{tool_slug}::{action_hash}"
        evidence_content = f"Tool evidence: {tool_slug}\n{evidence_text_out}"
        evidence_metadata: dict[str, Any] = {
            "type": "tool_evidence",
            "tool_name": tool_name,
            "tool_call_id": call_id,
            "session": session,
            "session_path": session_path,
            "line_no": int(result_line) if isinstance(result_line, (int, float)) else line_no_end,
        }

        if _ensure_node(
            graph,
            evidence_node_id,
            content=evidence_content,
            summary=f"Tool evidence: {tool_slug}",
            metadata=evidence_metadata,
        ):
            created_evidence += 1

        if _ensure_edge(
            graph,
            action_node_id,
            evidence_node_id,
            weight=0.30,
            kind="action",
            metadata={"source": "replay"},
        ):
            created_edges += 1

    return {
        "actions": created_actions,
        "evidence": created_evidence,
        "edges": created_edges,
    }
