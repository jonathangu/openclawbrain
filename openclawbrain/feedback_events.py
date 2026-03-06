"""Canonical feedback event contract shared across scanner, realtime capture, self-learn, and harvest."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any

SOURCE_KINDS = {"human", "self", "scanner", "teacher", "harvester"}
FEEDBACK_KINDS = {"CORRECTION", "TEACHING", "DIRECTIVE", "REINFORCEMENT"}
SEVERITIES = {"low", "medium", "high"}


@dataclass(frozen=True)
class FeedbackEvent:
    source_kind: str
    feedback_kind: str
    content: str
    confidence: float = 1.0
    ts: float = field(default_factory=time.time)
    chat_id: str | None = None
    message_id: str | None = None
    dedup_key: str | None = None
    fired_ids: list[str] = field(default_factory=list)
    outcome: float | None = None
    session_pointer: str | None = None
    session: str | None = None
    context: str | None = None
    severity: str | None = None
    node_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "source_kind": self.source_kind,
            "feedback_kind": self.feedback_kind,
            "type": self.feedback_kind,
            "content": self.content,
            "confidence": float(self.confidence),
            "ts": float(self.ts),
            "metadata": dict(self.metadata),
        }
        if self.chat_id:
            payload["chat_id"] = self.chat_id
        if self.message_id:
            payload["message_id"] = self.message_id
        if self.dedup_key:
            payload["dedup_key"] = self.dedup_key
        if self.fired_ids:
            payload["fired_ids"] = list(self.fired_ids)
        if self.outcome is not None:
            payload["outcome"] = float(self.outcome)
        if self.session_pointer:
            payload["session_pointer"] = self.session_pointer
        if self.session:
            payload["session"] = self.session
        if self.context:
            payload["context"] = self.context
        if self.severity:
            payload["severity"] = self.severity
        if self.node_id:
            payload["node_id"] = self.node_id
        content_hash = hashlib.sha256(self.content.encode("utf-8")).hexdigest()
        payload["content_hash"] = content_hash
        payload["event_hash"] = event_hash(payload)
        return payload


def normalize_feedback_kind(value: object, *, default: str = "TEACHING") -> str:
    raw = str(value or default).strip().upper()
    return raw if raw in FEEDBACK_KINDS else default


def normalize_source_kind(value: object, *, default: str = "scanner") -> str:
    raw = str(value or default).strip().lower()
    return raw if raw in SOURCE_KINDS else default


def normalize_severity(value: object) -> str | None:
    if value is None:
        return None
    raw = str(value).strip().lower()
    return raw if raw in SEVERITIES else None


def confidence_from_severity(severity: str | None) -> float:
    return {"low": 0.4, "medium": 0.7, "high": 0.95}.get(severity or "", 1.0)


def event_hash(payload: dict[str, Any]) -> str:
    feedback_kind = normalize_feedback_kind(payload.get("feedback_kind") or payload.get("type"))
    content_hash = str(payload.get("content_hash") or hashlib.sha256(str(payload.get("content", "")).encode("utf-8")).hexdigest())
    pointer = str(payload.get("session_pointer") or "")
    dedup_key = str(payload.get("dedup_key") or payload.get("message_id") or "")
    if dedup_key:
        source_kind = normalize_source_kind(payload.get("source_kind"))
        return "|".join((source_kind, feedback_kind, content_hash, pointer, dedup_key))
    return "|".join((feedback_kind, content_hash, pointer))


def from_dict(payload: dict[str, Any]) -> FeedbackEvent:
    severity = normalize_severity(payload.get("severity"))
    confidence = payload.get("confidence")
    try:
        confidence_value = float(confidence) if confidence is not None else confidence_from_severity(severity)
    except (TypeError, ValueError):
        confidence_value = confidence_from_severity(severity)
    fired_ids = payload.get("fired_ids") or payload.get("fired") or []
    if not isinstance(fired_ids, list):
        fired_ids = []
    outcome = payload.get("outcome")
    try:
        outcome_value = float(outcome) if outcome is not None else None
    except (TypeError, ValueError):
        outcome_value = None
    ts = payload.get("ts")
    try:
        ts_value = float(ts) if ts is not None else time.time()
    except (TypeError, ValueError):
        ts_value = time.time()
    return FeedbackEvent(
        source_kind=normalize_source_kind(payload.get("source_kind") or payload.get("source") or ("self" if payload.get("source") == "self" else "scanner")),
        feedback_kind=normalize_feedback_kind(payload.get("feedback_kind") or payload.get("type")),
        content=str(payload.get("content", "")).strip(),
        confidence=confidence_value,
        ts=ts_value,
        chat_id=str(payload.get("chat_id")) if payload.get("chat_id") is not None else None,
        message_id=str(payload.get("message_id")) if payload.get("message_id") is not None else None,
        dedup_key=str(payload.get("dedup_key")) if payload.get("dedup_key") is not None else None,
        fired_ids=[str(v) for v in fired_ids if isinstance(v, str) and v],
        outcome=outcome_value,
        session_pointer=str(payload.get("session_pointer")) if payload.get("session_pointer") is not None else None,
        session=str(payload.get("session")) if payload.get("session") is not None else None,
        context=str(payload.get("context")) if payload.get("context") is not None else None,
        severity=severity,
        node_id=str(payload.get("node_id")) if payload.get("node_id") is not None else None,
        metadata=dict(payload.get("metadata", {})) if isinstance(payload.get("metadata"), dict) else {},
    )
