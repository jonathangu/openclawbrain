from __future__ import annotations

import re
from pathlib import Path


HANDLER_PATH = Path("integrations/openclaw/hooks/openclawbrain-context-injector/handler.ts")


def _load_handler() -> str:
    return HANDLER_PATH.read_text(encoding="utf-8")


def _extract_feedback_prefixes(text: str) -> set[str]:
    lines = text.splitlines()
    start_idx = next(
        (idx for idx, line in enumerate(lines) if "const FEEDBACK_PREFIXES" in line),
        None,
    )
    if start_idx is None:
        raise AssertionError("FEEDBACK_PREFIXES block not found in handler.ts")
    keys: set[str] = set()
    for line in lines[start_idx + 1 :]:
        if "};" in line:
            break
        if "kind:" not in line:
            continue
        if ":" not in line:
            continue
        key = line.strip().split(":", 1)[0].strip()
        if key.isalpha():
            keys.add(key.lower())
    return keys


def _extract_feedback_regex_prefixes(text: str) -> set[str]:
    for line in text.splitlines():
        if "message.match" in line and "Correction" in line:
            prefixes = re.findall(r"(Correction|Fix|Teaching|Note)", line)
            if prefixes:
                return {item.lower() for item in prefixes}
    raise AssertionError("feedback prefix regex not found in handler.ts")


def test_feedback_prefix_regex_includes_mapping_keys() -> None:
    text = _load_handler()
    mapping = _extract_feedback_prefixes(text)
    regex_prefixes = _extract_feedback_regex_prefixes(text)
    assert mapping
    assert mapping.issubset(regex_prefixes)


def test_feedback_prefix_regex_matches_examples() -> None:
    text = _load_handler()
    regex_prefixes = _extract_feedback_regex_prefixes(text)
    pattern = re.compile(r"^\s*(" + "|".join(sorted(regex_prefixes)) + r"):\s*(.*)$", re.IGNORECASE)
    for prefix in regex_prefixes:
        match = pattern.match(f"{prefix}: fix the edge case")
        assert match is not None
        assert match.group(1).lower() == prefix
