"""Analyze OpenClaw session JSONL for LLM call and token usage metrics."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TokenUsage:
    prompt: int = 0
    completion: int = 0
    total: int = 0

    def add(self, other: "TokenUsage") -> None:
        self.prompt += other.prompt
        self.completion += other.completion
        self.total += other.total


def _coerce_int(value: object) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
    return 0


def _extract_usage_from_dict(payload: dict[str, Any]) -> TokenUsage:
    prompt = _coerce_int(
        payload.get("prompt_tokens")
        or payload.get("input_tokens")
        or payload.get("promptTokens")
        or payload.get("inputTokens")
    )
    completion = _coerce_int(
        payload.get("completion_tokens")
        or payload.get("output_tokens")
        or payload.get("completionTokens")
        or payload.get("outputTokens")
    )
    total = _coerce_int(payload.get("total_tokens") or payload.get("totalTokens"))
    if total == 0:
        total = prompt + completion
    return TokenUsage(prompt=prompt, completion=completion, total=total)


def _extract_usage(record: dict[str, Any]) -> TokenUsage:
    for key in ("usage", "token_usage", "tokens"):
        value = record.get(key)
        if isinstance(value, dict):
            return _extract_usage_from_dict(value)

    if isinstance(record.get("message"), dict):
        message = record["message"]
        for key in ("usage", "token_usage", "tokens"):
            value = message.get(key)
            if isinstance(value, dict):
                return _extract_usage_from_dict(value)

    return _extract_usage_from_dict(record)


def _extract_message_payload(record: dict[str, Any]) -> dict[str, Any] | None:
    if record.get("type") == "message" and isinstance(record.get("message"), dict):
        return record["message"]
    return record


def _role(record: dict[str, Any]) -> str | None:
    payload = _extract_message_payload(record)
    if not isinstance(payload, dict):
        return None
    role = payload.get("role")
    if isinstance(role, str):
        return role.strip().lower()
    return None


def _is_user_message(record: dict[str, Any]) -> bool:
    return _role(record) == "user"


def _is_assistant_message(record: dict[str, Any]) -> bool:
    return _role(record) == "assistant"


def _is_tool_message(record: dict[str, Any]) -> bool:
    return _role(record) in {"tool", "toolresult", "tool_result"}


def _is_llm_call(record: dict[str, Any]) -> bool:
    if _is_user_message(record) or _is_tool_message(record):
        return False
    if _is_assistant_message(record):
        return True
    type_value = record.get("type")
    if isinstance(type_value, str) and type_value.strip().lower() in {
        "llm_call",
        "llm",
        "model",
        "completion",
        "response",
        "assistant_response",
    }:
        return True
    if any(key in record for key in ("usage", "token_usage", "tokens", "prompt_tokens", "completion_tokens")):
        return True
    if isinstance(record.get("model"), str):
        return True
    return False


def _load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
    return records


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    total_usage = TokenUsage()
    llm_calls = 0
    user_messages = 0
    exchanges: list[dict[str, Any]] = []
    current_exchange: dict[str, Any] | None = None

    for record in records:
        if _is_user_message(record):
            user_messages += 1
            current_exchange = {"index": user_messages, "llm_calls": 0}
            exchanges.append(current_exchange)

        if _is_llm_call(record):
            llm_calls += 1
            usage = _extract_usage(record)
            total_usage.add(usage)
            if current_exchange is not None:
                current_exchange["llm_calls"] += 1

    per_exchange = [entry["llm_calls"] for entry in exchanges]
    per_exchange_mean = sum(per_exchange) / len(per_exchange) if per_exchange else 0.0
    per_exchange_min = min(per_exchange) if per_exchange else 0
    per_exchange_max = max(per_exchange) if per_exchange else 0

    return {
        "records": len(records),
        "user_messages": user_messages,
        "llm_calls": llm_calls,
        "tokens": {
            "prompt": total_usage.prompt,
            "completion": total_usage.completion,
            "total": total_usage.total,
        },
        "llm_calls_per_exchange": {
            "mean": per_exchange_mean,
            "min": per_exchange_min,
            "max": per_exchange_max,
        },
        "exchanges": exchanges,
    }


def _render_table(summary: dict[str, Any]) -> str:
    rows = [
        ("records", str(summary["records"])),
        ("user messages", str(summary["user_messages"])),
        ("llm calls (turns)", str(summary["llm_calls"])),
        ("tokens in", str(summary["tokens"]["prompt"])),
        ("tokens out", str(summary["tokens"]["completion"])),
        ("tokens total", str(summary["tokens"]["total"])),
        (
            "llm calls per exchange",
            f"mean={summary['llm_calls_per_exchange']['mean']:.2f} "
            f"min={summary['llm_calls_per_exchange']['min']} "
            f"max={summary['llm_calls_per_exchange']['max']}",
        ),
    ]

    left_width = max(len(row[0]) for row in rows)
    lines = ["metric".ljust(left_width) + "  value", "-" * (left_width + 2 + 20)]
    for key, value in rows:
        lines.append(key.ljust(left_width) + "  " + value)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze OpenClaw session JSONL for LLM call metrics.")
    parser.add_argument("session", help="Path to an OpenClaw session .jsonl file")
    parser.add_argument("--json-out", help="Optional path to write JSON summary")
    parser.add_argument("--no-exchanges", action="store_true", help="Omit per-exchange breakdown in JSON output")
    args = parser.parse_args()

    session_path = Path(args.session).expanduser()
    if not session_path.exists():
        raise SystemExit(f"missing session file: {session_path}")

    records = _load_records(session_path)
    summary = _summarize(records)
    summary["session_path"] = str(session_path)
    summary["notes"] = "LLM calls are inferred from assistant-role messages or records with usage/model metadata."
    if args.no_exchanges:
        summary = dict(summary)
        summary.pop("exchanges", None)

    print(_render_table(summary))
    print()
    print(json.dumps(summary, indent=2))

    if args.json_out:
        output_path = Path(args.json_out).expanduser()
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
