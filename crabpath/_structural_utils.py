"""Internal structural helpers shared across modules."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, replace
from typing import Any, TypeVar


@dataclass
class ConfigBase:
    """Small helper base for config-style dataclasses."""

    def with_updates(self: "T", **updates: Any) -> "T":
        """Return a copy with updated fields."""
        return replace(self, **updates)

    def as_dict(self) -> dict[str, Any]:
        """Serialize dataclass fields as a plain dictionary."""
        return asdict(self)


class JSONStateMixin:
    """Small persistence helpers for JSON-backed state."""

    @staticmethod
    def _load_json_file(path: str, default: Any) -> Any:
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return default
        except (TypeError, OSError):
            return default

    @staticmethod
    def _write_json_file(path: str, payload: Any, *, sort_keys: bool = False) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=sort_keys)


def parse_markdown_json(raw: str, *, require_object: bool = False) -> Any:
    """Parse JSON payloads, stripping markdown JSON fences when present."""
    if not isinstance(raw, str):
        raise TypeError("raw model output must be a string")

    cleaned = raw.strip()
    if not cleaned:
        raise ValueError("empty model output")

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 2 and lines[0].startswith("```"):
            if lines[-1].strip().endswith("```"):
                lines = lines[1:-1]
            cleaned = "\n".join(lines).strip()

    payload = json.loads(cleaned)
    if require_object and not isinstance(payload, dict):
        raise TypeError("parsed JSON payload must be an object")
    return payload


def split_fallback_sections(
    content: str,
    *,
    min_header_chars: int = 1,
    min_paragraph_chars: int = 1,
    merge_short_paragraphs: int = 0,
) -> list[str]:
    """Structural fallback split with heading-first decomposition."""
    parts = [p.strip() for p in re.split(r"\n(?=## )", content) if p.strip()]
    parts = [p for p in parts if len(p) >= max(1, min_header_chars)]
    if len(parts) >= 2:
        return parts

    parts = [p.strip() for p in content.split("\n\n") if p.strip()]
    parts = [p for p in parts if len(p) >= max(1, min_paragraph_chars)]
    if len(parts) >= 2:
        if merge_short_paragraphs > 0:
            merged: list[str] = [parts[0]]
            for p in parts[1:]:
                if len(merged[-1]) < merge_short_paragraphs:
                    merged[-1] = f"{merged[-1]}\n\n{p}"
                else:
                    merged.append(p)
            if len(merged) >= 2:
                return merged
        return parts

    return [content]


def node_file_id(node_id: Any) -> str:
    """Normalize node ids to a coarse file-level identifier."""
    return str(node_id).split("::", 1)[0]


def count_cross_file_edges(graph: Any) -> int:
    """Count edges that cross file boundaries in node ids."""
    if getattr(graph, "node_count", 0) <= 1:
        return 0

    return sum(
        1
        for edge in graph.edges()
        if node_file_id(edge.source) != node_file_id(edge.target)
    )


def classify_edge_tier(
    weight: float,
    *,
    dormant_threshold: float = 0.3,
    reflex_threshold: float = 0.8,
) -> str:
    """Map an edge weight to a routing tier."""
    if weight >= reflex_threshold:
        return "reflex"
    if weight >= dormant_threshold:
        return "habitual"
    return "dormant"


T = TypeVar("T", bound=ConfigBase)

