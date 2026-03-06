"""Phase 1 contract validators and fixture loaders for the canonical v1 boundary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from openclawbrain.feedback_events import FeedbackEvent as LegacyFeedbackEvent

RUNTIME_COMPILE_VERSION = "runtime_compile.v1"
INTERACTION_EVENTS_VERSION = "interaction_events.v1"
FEEDBACK_EVENT_VERSION = "feedback_events.v1"
ARTIFACT_MANIFEST_VERSION = "artifact_manifest.v1"

ROUTE_MODES = {"off", "edge", "edge+sim", "learned"}
INTERACTION_EVENT_KINDS = {
    "turn_started",
    "memory_compiled",
    "assistant_completed",
    "tool_called",
    "tool_result",
    "feedback_recorded",
    "session_closed",
}
FEEDBACK_KINDS = {"correction", "teaching", "preference", "approval", "operator_override"}
FEEDBACK_SOURCE_KINDS = {"human", "system", "evaluation"}

LEGACY_FEEDBACK_KIND_MAP = {
    "CORRECTION": "correction",
    "TEACHING": "teaching",
    "DIRECTIVE": "operator_override",
    "REINFORCEMENT": "approval",
}
LEGACY_SOURCE_KIND_MAP = {
    "human": "human",
    "self": "system",
    "scanner": "system",
    "teacher": "evaluation",
    "harvester": "system",
}

CONTRACTS_ROOT = Path(__file__).resolve().parents[2] / "contracts"


class ContractValidationError(ValueError):
    """Raised when a payload fails canonical Phase 1 contract validation."""


def load_contract_json(*relative_path: str) -> Any:
    """Load one JSON contract artifact from the repo-root contracts tree."""
    path = CONTRACTS_ROOT.joinpath(*relative_path)
    return json.loads(path.read_text(encoding="utf-8"))


def _expect_mapping(payload: object, label: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ContractValidationError(f"{label} must be an object")
    return dict(payload)


def _require_string(payload: dict[str, Any], key: str, label: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError(f"{label}.{key} must be a non-empty string")
    return value.strip()


def _optional_string(payload: dict[str, Any], key: str, label: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ContractValidationError(f"{label}.{key} must be a string or null")
    normalized = value.strip()
    return normalized or None


def _require_bool(payload: dict[str, Any], key: str, label: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise ContractValidationError(f"{label}.{key} must be a boolean")
    return value


def _require_int(payload: dict[str, Any], key: str, label: str, *, minimum: int | None = None) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractValidationError(f"{label}.{key} must be an integer")
    if minimum is not None and value < minimum:
        raise ContractValidationError(f"{label}.{key} must be >= {minimum}")
    return value


def _optional_int(payload: dict[str, Any], key: str, label: str, *, minimum: int | None = None) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractValidationError(f"{label}.{key} must be an integer or null")
    if minimum is not None and value < minimum:
        raise ContractValidationError(f"{label}.{key} must be >= {minimum}")
    return value


def _require_number(payload: dict[str, Any], key: str, label: str, *, minimum: float | None = None) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractValidationError(f"{label}.{key} must be a number")
    number = float(value)
    if minimum is not None and number < minimum:
        raise ContractValidationError(f"{label}.{key} must be >= {minimum}")
    return number


def _require_enum(payload: dict[str, Any], key: str, label: str, allowed: set[str]) -> str:
    value = _require_string(payload, key, label)
    if value not in allowed:
        options = ", ".join(sorted(allowed))
        raise ContractValidationError(f"{label}.{key} must be one of: {options}")
    return value


def _require_string_list(
    payload: dict[str, Any],
    key: str,
    label: str,
    *,
    allow_empty: bool = True,
) -> list[str]:
    raw = payload.get(key)
    if not isinstance(raw, list):
        raise ContractValidationError(f"{label}.{key} must be an array")
    items: list[str] = []
    for index, item in enumerate(raw):
        if not isinstance(item, str) or not item.strip():
            raise ContractValidationError(f"{label}.{key}[{index}] must be a non-empty string")
        items.append(item.strip())
    if not allow_empty and not items:
        raise ContractValidationError(f"{label}.{key} must not be empty")
    return items


def _require_mapping_list(payload: dict[str, Any], key: str, label: str) -> list[dict[str, Any]]:
    raw = payload.get(key)
    if not isinstance(raw, list):
        raise ContractValidationError(f"{label}.{key} must be an array")
    items: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ContractValidationError(f"{label}.{key}[{index}] must be an object")
        items.append(dict(item))
    return items


def _validate_routing_request(payload: object, label: str) -> dict[str, Any]:
    routing = _expect_mapping(payload, label)
    return {
        "mode_requested": _require_enum(routing, "mode_requested", label, ROUTE_MODES),
    }


def _validate_runtime_routing(payload: object, label: str) -> dict[str, Any]:
    routing = _expect_mapping(payload, label)
    normalized = {
        "mode_requested": _require_enum(routing, "mode_requested", label, ROUTE_MODES),
        "mode_effective": _require_enum(routing, "mode_effective", label, ROUTE_MODES),
        "used_learned_route_fn": _require_bool(routing, "used_learned_route_fn", label),
        "route_model_id": _optional_string(routing, "route_model_id", label),
        "decision_trace_ref": _optional_string(routing, "decision_trace_ref", label),
    }
    if normalized["used_learned_route_fn"] and normalized["mode_effective"] != "learned":
        raise ContractValidationError(
            f"{label}.used_learned_route_fn cannot be true when {label}.mode_effective is not learned"
        )
    if normalized["mode_effective"] == "learned" and not normalized["decision_trace_ref"]:
        raise ContractValidationError(f"{label}.decision_trace_ref must be explicit for learned runtime routing")
    return normalized


def _validate_context_source(payload: object, label: str) -> dict[str, Any]:
    source = _expect_mapping(payload, label)
    return {
        "kind": _require_string(source, "kind", label),
        "ref": _require_string(source, "ref", label),
    }


def _validate_context_block(payload: object, label: str) -> dict[str, Any]:
    block = _expect_mapping(payload, label)
    sources = [_validate_context_source(item, f"{label}.sources[{index}]") for index, item in enumerate(block.get("sources", []))]
    if "sources" not in block or not sources:
        raise ContractValidationError(f"{label}.sources must contain at least one source")
    return {
        "id": _require_string(block, "id", label),
        "kind": _require_string(block, "kind", label),
        "title": _require_string(block, "title", label),
        "content": _require_string(block, "content", label),
        "score": _require_number(block, "score", label),
        "sources": sources,
    }


def validate_runtime_compile_request(payload: object) -> dict[str, Any]:
    request = _expect_mapping(payload, "runtime_compile_request")
    contract_version = _require_string(request, "contract_version", "runtime_compile_request")
    if contract_version != RUNTIME_COMPILE_VERSION:
        raise ContractValidationError("runtime_compile_request.contract_version must be runtime_compile.v1")
    recent_turns: list[dict[str, Any]] = []
    for index, turn in enumerate(_require_mapping_list(request, "recent_turns", "runtime_compile_request")):
        recent_turns.append(
            {
                "role": _require_enum(
                    turn,
                    "role",
                    f"runtime_compile_request.recent_turns[{index}]",
                    {"user", "assistant", "system", "tool"},
                ),
                "content": _require_string(turn, "content", f"runtime_compile_request.recent_turns[{index}]"),
                "turn_id": _optional_string(turn, "turn_id", f"runtime_compile_request.recent_turns[{index}]"),
            }
        )
    tool_context: list[dict[str, Any]] = []
    for index, item in enumerate(_require_mapping_list(request, "tool_context", "runtime_compile_request")):
        tool_context.append(
            {
                "tool_name": _require_string(item, "tool_name", f"runtime_compile_request.tool_context[{index}]"),
                "summary": _require_string(item, "summary", f"runtime_compile_request.tool_context[{index}]"),
                "output_ref": _optional_string(item, "output_ref", f"runtime_compile_request.tool_context[{index}]"),
            }
        )
    budget = _expect_mapping(request.get("budget"), "runtime_compile_request.budget")
    hints = _expect_mapping(request.get("hints"), "runtime_compile_request.hints")
    return {
        "contract_version": contract_version,
        "request_id": _require_string(request, "request_id", "runtime_compile_request"),
        "agent_id": _require_string(request, "agent_id", "runtime_compile_request"),
        "session_id": _require_string(request, "session_id", "runtime_compile_request"),
        "turn_id": _require_string(request, "turn_id", "runtime_compile_request"),
        "pack_id": _require_string(request, "pack_id", "runtime_compile_request"),
        "user_message": _require_string(request, "user_message", "runtime_compile_request"),
        "recent_turns": recent_turns,
        "tool_context": tool_context,
        "budget": {
            "max_chars": _require_int(budget, "max_chars", "runtime_compile_request.budget", minimum=1),
            "max_blocks": _require_int(budget, "max_blocks", "runtime_compile_request.budget", minimum=1),
        },
        "hints": {
            "recall": _require_bool(hints, "recall", "runtime_compile_request.hints"),
            "correction_sensitive": _require_bool(
                hints,
                "correction_sensitive",
                "runtime_compile_request.hints",
            ),
        },
        "routing": _validate_routing_request(request.get("routing"), "runtime_compile_request.routing"),
    }


def validate_runtime_compile_response(payload: object) -> dict[str, Any]:
    response = _expect_mapping(payload, "runtime_compile_response")
    contract_version = _require_string(response, "contract_version", "runtime_compile_response")
    if contract_version != RUNTIME_COMPILE_VERSION:
        raise ContractValidationError("runtime_compile_response.contract_version must be runtime_compile.v1")
    blocks = [
        _validate_context_block(item, f"runtime_compile_response.context_blocks[{index}]")
        for index, item in enumerate(_require_mapping_list(response, "context_blocks", "runtime_compile_response"))
    ]
    diagnostics = _expect_mapping(response.get("diagnostics"), "runtime_compile_response.diagnostics")
    normalized = {
        "contract_version": contract_version,
        "request_id": _require_string(response, "request_id", "runtime_compile_response"),
        "pack_id": _require_string(response, "pack_id", "runtime_compile_response"),
        "context_blocks": blocks,
        "suppressed_sources": _require_string_list(
            response,
            "suppressed_sources",
            "runtime_compile_response",
            allow_empty=True,
        ),
        "routing": _validate_runtime_routing(response.get("routing"), "runtime_compile_response.routing"),
        "diagnostics": {
            "candidate_count": _require_int(diagnostics, "candidate_count", "runtime_compile_response.diagnostics", minimum=0),
            "selected_count": _require_int(diagnostics, "selected_count", "runtime_compile_response.diagnostics", minimum=0),
            "compile_ms": _require_number(diagnostics, "compile_ms", "runtime_compile_response.diagnostics", minimum=0.0),
            "router": _optional_string(diagnostics, "router", "runtime_compile_response.diagnostics"),
        },
    }
    return normalized


def _validate_interaction_payload(event_kind: str, payload: object, label: str) -> dict[str, Any]:
    data = _expect_mapping(payload, label)
    if event_kind == "turn_started":
        return {
            "user_message_ref": _require_string(data, "user_message_ref", label),
        }
    if event_kind == "memory_compiled":
        normalized = {
            "selected_context_ids": _require_string_list(data, "selected_context_ids", label, allow_empty=True),
            "suppressed_source_refs": _require_string_list(data, "suppressed_source_refs", label, allow_empty=True),
            "routing": _validate_runtime_routing(data.get("routing"), f"{label}.routing"),
            "compile_latency_ms": _require_number(data, "compile_latency_ms", label, minimum=0.0),
            "candidate_count": _optional_int(data, "candidate_count", label, minimum=0),
            "selected_count": _optional_int(data, "selected_count", label, minimum=0),
        }
        return normalized
    if event_kind == "assistant_completed":
        return {
            "assistant_message_ref": _require_string(data, "assistant_message_ref", label),
            "finish_reason": _require_string(data, "finish_reason", label),
        }
    if event_kind == "tool_called":
        return {
            "tool_name": _require_string(data, "tool_name", label),
            "tool_call_id": _require_string(data, "tool_call_id", label),
        }
    if event_kind == "tool_result":
        return {
            "tool_name": _require_string(data, "tool_name", label),
            "tool_call_id": _require_string(data, "tool_call_id", label),
            "status": _require_string(data, "status", label),
        }
    if event_kind == "feedback_recorded":
        return {
            "feedback_event_id": _require_string(data, "feedback_event_id", label),
            "feedback_kind": _require_enum(data, "feedback_kind", label, FEEDBACK_KINDS),
        }
    if event_kind == "session_closed":
        return {
            "close_reason": _require_string(data, "close_reason", label),
        }
    raise ContractValidationError(f"{label} has unsupported event_kind {event_kind}")


def validate_interaction_event(payload: object) -> dict[str, Any]:
    event = _expect_mapping(payload, "interaction_event")
    contract_version = _require_string(event, "contract_version", "interaction_event")
    if contract_version != INTERACTION_EVENTS_VERSION:
        raise ContractValidationError("interaction_event.contract_version must be interaction_events.v1")
    schema_version = _require_int(event, "schema_version", "interaction_event", minimum=1)
    if schema_version != 1:
        raise ContractValidationError("interaction_event.schema_version must be 1")
    event_kind = _require_enum(event, "event_kind", "interaction_event", INTERACTION_EVENT_KINDS)
    return {
        "contract_version": contract_version,
        "schema_version": schema_version,
        "event_id": _require_string(event, "event_id", "interaction_event"),
        "event_kind": event_kind,
        "event_ts": _require_string(event, "event_ts", "interaction_event"),
        "agent_id": _require_string(event, "agent_id", "interaction_event"),
        "session_id": _require_string(event, "session_id", "interaction_event"),
        "turn_id": _require_string(event, "turn_id", "interaction_event"),
        "request_id": _require_string(event, "request_id", "interaction_event"),
        "pack_id": _require_string(event, "pack_id", "interaction_event"),
        "payload": _validate_interaction_payload(event_kind, event.get("payload"), "interaction_event.payload"),
    }


def validate_feedback_event(payload: object) -> dict[str, Any]:
    event = _expect_mapping(payload, "feedback_event")
    contract_version = _require_string(event, "contract_version", "feedback_event")
    if contract_version != FEEDBACK_EVENT_VERSION:
        raise ContractValidationError("feedback_event.contract_version must be feedback_events.v1")
    return {
        "contract_version": contract_version,
        "event_id": _require_string(event, "event_id", "feedback_event"),
        "event_ts": _require_string(event, "event_ts", "feedback_event"),
        "agent_id": _require_string(event, "agent_id", "feedback_event"),
        "session_id": _require_string(event, "session_id", "feedback_event"),
        "turn_id": _require_string(event, "turn_id", "feedback_event"),
        "request_id": _require_string(event, "request_id", "feedback_event"),
        "pack_id": _require_string(event, "pack_id", "feedback_event"),
        "feedback_kind": _require_enum(event, "feedback_kind", "feedback_event", FEEDBACK_KINDS),
        "source_kind": _require_enum(event, "source_kind", "feedback_event", FEEDBACK_SOURCE_KINDS),
        "message_ref": _require_string(event, "message_ref", "feedback_event"),
        "affected_turn_id": _require_string(event, "affected_turn_id", "feedback_event"),
        "affected_context_ids": _require_string_list(event, "affected_context_ids", "feedback_event", allow_empty=True),
        "dedup_key": _require_string(event, "dedup_key", "feedback_event"),
        "content": _optional_string(event, "content", "feedback_event"),
        "metadata": dict(event.get("metadata", {})) if isinstance(event.get("metadata"), dict) else {},
    }


def validate_artifact_manifest(payload: object) -> dict[str, Any]:
    manifest = _expect_mapping(payload, "artifact_manifest")
    contract_version = _require_string(manifest, "contract_version", "artifact_manifest")
    if contract_version != ARTIFACT_MANIFEST_VERSION:
        raise ContractValidationError("artifact_manifest.contract_version must be artifact_manifest.v1")
    contract_versions = _expect_mapping(manifest.get("contract_versions"), "artifact_manifest.contract_versions")
    checksums = _expect_mapping(manifest.get("checksums"), "artifact_manifest.checksums")
    if not checksums:
        raise ContractValidationError("artifact_manifest.checksums must not be empty")
    compiler_fingerprint = _expect_mapping(
        manifest.get("compiler_fingerprint"),
        "artifact_manifest.compiler_fingerprint",
    )
    model_fingerprint = _expect_mapping(
        manifest.get("model_fingerprint"),
        "artifact_manifest.model_fingerprint",
    )
    runtime_compat = _expect_mapping(manifest.get("runtime_compat"), "artifact_manifest.runtime_compat")
    route_policy = _expect_mapping(manifest.get("route_policy"), "artifact_manifest.route_policy")
    serve_requirements = _expect_mapping(
        manifest.get("serve_requirements"),
        "artifact_manifest.serve_requirements",
    )
    event_range = _expect_mapping(manifest.get("event_range"), "artifact_manifest.event_range")
    eval_summary = _expect_mapping(manifest.get("eval_summary"), "artifact_manifest.eval_summary")
    return {
        "contract_version": contract_version,
        "pack_id": _require_string(manifest, "pack_id", "artifact_manifest"),
        "agent_id": _require_string(manifest, "agent_id", "artifact_manifest"),
        "build_id": _require_string(manifest, "build_id", "artifact_manifest"),
        "parent_pack_id": _optional_string(manifest, "parent_pack_id", "artifact_manifest"),
        "contract_versions": {
            "runtime_compile": _require_string(contract_versions, "runtime_compile", "artifact_manifest.contract_versions"),
            "interaction_events": _require_string(
                contract_versions,
                "interaction_events",
                "artifact_manifest.contract_versions",
            ),
            "feedback_events": _require_string(
                contract_versions,
                "feedback_events",
                "artifact_manifest.contract_versions",
            ),
            "artifact_manifest": _require_string(
                contract_versions,
                "artifact_manifest",
                "artifact_manifest.contract_versions",
            ),
        },
        "pack_checksum": _require_string(manifest, "pack_checksum", "artifact_manifest"),
        "compiler_fingerprint": {
            "compiler": _require_string(compiler_fingerprint, "compiler", "artifact_manifest.compiler_fingerprint"),
            "version": _require_string(compiler_fingerprint, "version", "artifact_manifest.compiler_fingerprint"),
            "build_hash": _require_string(compiler_fingerprint, "build_hash", "artifact_manifest.compiler_fingerprint"),
        },
        "model_fingerprint": {
            "embedding_model": _require_string(model_fingerprint, "embedding_model", "artifact_manifest.model_fingerprint"),
            "embedding_dimension": _require_int(
                model_fingerprint,
                "embedding_dimension",
                "artifact_manifest.model_fingerprint",
                minimum=1,
            ),
            "route_model_id": _optional_string(
                model_fingerprint,
                "route_model_id",
                "artifact_manifest.model_fingerprint",
            ),
        },
        "runtime_compat": {
            "openclaw_min": _require_string(runtime_compat, "openclaw_min", "artifact_manifest.runtime_compat"),
            "openclawbrain_min": _require_string(
                runtime_compat,
                "openclawbrain_min",
                "artifact_manifest.runtime_compat",
            ),
        },
        "route_policy": {
            "mode_requested": _require_enum(route_policy, "mode_requested", "artifact_manifest.route_policy", ROUTE_MODES),
            "route_model_id": _optional_string(route_policy, "route_model_id", "artifact_manifest.route_policy"),
            "decision_trace_refs_emitted": _require_bool(
                route_policy,
                "decision_trace_refs_emitted",
                "artifact_manifest.route_policy",
            ),
        },
        "serve_requirements": {
            "requires_learned_routing": _require_bool(
                serve_requirements,
                "requires_learned_routing",
                "artifact_manifest.serve_requirements",
            ),
            "required_route_assets": _require_string_list(
                serve_requirements,
                "required_route_assets",
                "artifact_manifest.serve_requirements",
                allow_empty=True,
            ),
            "embedding_model": _require_string(
                serve_requirements,
                "embedding_model",
                "artifact_manifest.serve_requirements",
            ),
            "embedding_dimension": _require_int(
                serve_requirements,
                "embedding_dimension",
                "artifact_manifest.serve_requirements",
                minimum=1,
            ),
        },
        "workspace_snapshot_id": _require_string(manifest, "workspace_snapshot_id", "artifact_manifest"),
        "event_range": {
            "interaction_event_start_id": _require_string(
                event_range,
                "interaction_event_start_id",
                "artifact_manifest.event_range",
            ),
            "interaction_event_end_id": _require_string(
                event_range,
                "interaction_event_end_id",
                "artifact_manifest.event_range",
            ),
            "feedback_event_start_id": _require_string(
                event_range,
                "feedback_event_start_id",
                "artifact_manifest.event_range",
            ),
            "feedback_event_end_id": _require_string(
                event_range,
                "feedback_event_end_id",
                "artifact_manifest.event_range",
            ),
        },
        "checksums": {str(key): _require_string(checksums, str(key), "artifact_manifest.checksums") for key in checksums},
        "eval_summary": {
            "baseline_pack_id": _require_string(eval_summary, "baseline_pack_id", "artifact_manifest.eval_summary"),
            "score_delta": _require_number(eval_summary, "score_delta", "artifact_manifest.eval_summary"),
            "regressions": _require_int(eval_summary, "regressions", "artifact_manifest.eval_summary", minimum=0),
            "status": _require_string(eval_summary, "status", "artifact_manifest.eval_summary"),
        },
        "created_at": _require_string(manifest, "created_at", "artifact_manifest"),
    }


def feedback_event_v1_from_legacy(
    event: LegacyFeedbackEvent,
    *,
    event_id: str,
    agent_id: str,
    session_id: str,
    turn_id: str,
    request_id: str,
    pack_id: str,
    message_ref: str | None = None,
    affected_turn_id: str | None = None,
    affected_context_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Bridge the existing feedback event model into feedback_events.v1 without cutover."""
    payload = event.to_dict()
    feedback_kind = LEGACY_FEEDBACK_KIND_MAP.get(str(payload.get("feedback_kind")), "teaching")
    source_kind = LEGACY_SOURCE_KIND_MAP.get(str(payload.get("source_kind")), "system")
    dedup_key = str(
        payload.get("dedup_key")
        or payload.get("message_id")
        or payload.get("event_hash")
        or event_id
    )
    normalized = {
        "contract_version": FEEDBACK_EVENT_VERSION,
        "event_id": event_id,
        "event_ts": str(payload.get("ts")),
        "agent_id": agent_id,
        "session_id": session_id,
        "turn_id": turn_id,
        "request_id": request_id,
        "pack_id": pack_id,
        "feedback_kind": feedback_kind,
        "source_kind": source_kind,
        "message_ref": message_ref or str(payload.get("message_id") or event_id),
        "affected_turn_id": affected_turn_id or turn_id,
        "affected_context_ids": affected_context_ids if affected_context_ids is not None else list(payload.get("fired_ids", [])),
        "dedup_key": dedup_key,
        "content": str(payload.get("content", "")).strip() or None,
        "metadata": {
            "legacy_feedback_kind": str(payload.get("feedback_kind", "")),
            "legacy_source_kind": str(payload.get("source_kind", "")),
            "legacy_event_hash": str(payload.get("event_hash", "")),
        },
    }
    return validate_feedback_event(normalized)
