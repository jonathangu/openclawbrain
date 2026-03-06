from __future__ import annotations

import pytest

from openclawbrain.contracts.v1 import (
    ARTIFACT_MANIFEST_VERSION,
    FEEDBACK_EVENT_VERSION,
    INTERACTION_EVENT_KINDS,
    INTERACTION_EVENTS_VERSION,
    RUNTIME_COMPILE_VERSION,
    ContractValidationError,
    feedback_event_v1_from_legacy,
    load_contract_json,
    validate_artifact_manifest,
    validate_feedback_event,
    validate_interaction_event,
    validate_runtime_compile_request,
    validate_runtime_compile_response,
)
from openclawbrain.feedback_events import FeedbackEvent


def test_contract_schema_files_have_stable_ids() -> None:
    schema_expectations = [
        ("runtime_compile", "v1", "request.schema.json", "https://openclawbrain.dev/contracts/runtime_compile/v1/request.schema.json"),
        ("runtime_compile", "v1", "response.schema.json", "https://openclawbrain.dev/contracts/runtime_compile/v1/response.schema.json"),
        ("interaction_events", "v1", "event.schema.json", "https://openclawbrain.dev/contracts/interaction_events/v1/event.schema.json"),
        ("feedback_events", "v1", "event.schema.json", "https://openclawbrain.dev/contracts/feedback_events/v1/event.schema.json"),
        ("artifact_manifest", "v1", "manifest.schema.json", "https://openclawbrain.dev/contracts/artifact_manifest/v1/manifest.schema.json"),
    ]
    for contract_name, version, filename, expected_id in schema_expectations:
        payload = load_contract_json(contract_name, version, filename)
        assert payload["$id"] == expected_id


def test_runtime_compile_golden_fixtures_validate() -> None:
    request = load_contract_json("runtime_compile", "v1", "golden-request.json")
    response = load_contract_json("runtime_compile", "v1", "golden-response.json")

    normalized_request = validate_runtime_compile_request(request)
    normalized_response = validate_runtime_compile_response(response)

    assert normalized_request["contract_version"] == RUNTIME_COMPILE_VERSION
    assert normalized_request["routing"]["mode_requested"] == "learned"
    assert normalized_response["contract_version"] == RUNTIME_COMPILE_VERSION
    assert normalized_response["routing"]["mode_effective"] == "learned"
    assert normalized_response["routing"]["used_learned_route_fn"] is True
    assert normalized_response["routing"]["decision_trace_ref"] == "route_decision_req_123"


def test_runtime_compile_response_rejects_incoherent_learned_routing_flags() -> None:
    response = load_contract_json("runtime_compile", "v1", "golden-response.json")
    response["routing"]["mode_effective"] = "edge+sim"

    with pytest.raises(ContractValidationError, match="used_learned_route_fn"):
        validate_runtime_compile_response(response)


def test_interaction_event_golden_stream_validates_required_event_kinds() -> None:
    stream = load_contract_json("interaction_events", "v1", "golden-stream.json")

    seen_event_kinds: set[str] = set()
    memory_event = None
    for raw_event in stream:
        event = validate_interaction_event(raw_event)
        assert event["contract_version"] == INTERACTION_EVENTS_VERSION
        seen_event_kinds.add(event["event_kind"])
        if event["event_kind"] == "memory_compiled":
            memory_event = event

    assert INTERACTION_EVENT_KINDS.issubset(seen_event_kinds)
    assert memory_event is not None
    assert memory_event["payload"]["routing"]["mode_requested"] == "learned"
    assert memory_event["payload"]["routing"]["mode_effective"] == "learned"
    assert memory_event["payload"]["routing"]["used_learned_route_fn"] is True
    assert memory_event["payload"]["routing"]["decision_trace_ref"] == "route_decision_req_123"


def test_feedback_event_golden_fixture_validates() -> None:
    event = load_contract_json("feedback_events", "v1", "golden-event.json")

    normalized = validate_feedback_event(event)

    assert normalized["contract_version"] == FEEDBACK_EVENT_VERSION
    assert normalized["feedback_kind"] == "correction"
    assert normalized["source_kind"] == "human"
    assert normalized["affected_context_ids"] == ["ctx_1"]


def test_legacy_feedback_event_can_bridge_to_feedback_events_v1() -> None:
    legacy_event = FeedbackEvent(
        source_kind="human",
        feedback_kind="CORRECTION",
        content="Rollback before restarting workers.",
        ts=1709851200.0,
        message_id="msg_123",
        dedup_key="feedback-msg-123",
        fired_ids=["ctx_1"],
    )

    normalized = feedback_event_v1_from_legacy(
        legacy_event,
        event_id="feedback_evt_legacy_1",
        agent_id="main",
        session_id="sess_456",
        turn_id="turn_789",
        request_id="req_123",
        pack_id="brainpack_2026_03_06_001",
        message_ref="message:msg_123",
        affected_turn_id="turn_789",
        affected_context_ids=["ctx_1"],
    )

    assert normalized["feedback_kind"] == "correction"
    assert normalized["source_kind"] == "human"
    assert normalized["dedup_key"] == "feedback-msg-123"
    assert normalized["affected_context_ids"] == ["ctx_1"]
    assert normalized["metadata"]["legacy_feedback_kind"] == "CORRECTION"


def test_artifact_manifest_golden_fixture_validates() -> None:
    manifest = load_contract_json("artifact_manifest", "v1", "golden-manifest.json")

    normalized = validate_artifact_manifest(manifest)

    assert normalized["contract_version"] == ARTIFACT_MANIFEST_VERSION
    assert normalized["route_policy"]["mode_requested"] == "learned"
    assert normalized["serve_requirements"]["requires_learned_routing"] is True
    assert normalized["contract_versions"]["runtime_compile"] == RUNTIME_COMPILE_VERSION
    assert normalized["contract_versions"]["interaction_events"] == INTERACTION_EVENTS_VERSION
    assert normalized["contract_versions"]["feedback_events"] == FEEDBACK_EVENT_VERSION
