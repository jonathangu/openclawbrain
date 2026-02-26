#!/usr/bin/env python3
"""Build a CrabPath graph from workspace markdown files."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crabpath.graph import Edge, Graph, Node  # noqa: E402

SECTION_RE = re.compile(r"^(#{2,3})\s+(.*)$")
TOOL_PATTERNS = (
    "run:",
    "bash ",
    "python ",
    "python3 ",
    "npm ",
    "pip ",
    "gh ",
    "ssh ",
    "docker ",
    "kubectl ",
    "make ",
    "curl ",
    "git ",
)
GUARDRAIL_PATTERNS = ("never", "always", "must", "do not")
PROCEDURE_PATTERNS = (
    r"^\s*\d+[.)]\s+",
    r"^\s*-\s+\[.\]\s+",
    r"^step\s+\d+",
)
TOOL_CALL_RE = re.compile("|".join(re.escape(x) for x in TOOL_PATTERNS), re.IGNORECASE)
GUARDRAIL_RE = re.compile("|".join(re.escape(x) for x in GUARDRAIL_PATTERNS), re.IGNORECASE)
PROCEDURE_RE = re.compile("|".join(PROCEDURE_PATTERNS), re.IGNORECASE | re.MULTILINE)


@dataclass
class Section:
    file_path: Path
    heading: str
    content: str
    node_id: str
    line_start: int
    line_end: int
    section_type: str
    metadata: dict[str, Any]


def _normalize_heading(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return normalized.strip()


def _slugify(text: str, fallback: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    if not slug:
        return fallback
    if len(slug) > 100:
        return slug[:100]
    return slug


def _classify_node_type(content: str) -> str:
    lowered = content.lower()
    if "run:" in lowered or TOOL_CALL_RE.search(content):
        return "tool_call"
    if GUARDRAIL_RE.search(lowered):
        return "guardrail"
    if PROCEDURE_RE.search(content):
        return "procedure"
    return "fact"


def _visible_files(root: Path) -> list[Path]:
    result = []
    for path in root.rglob("*.md"):
        if any(part.startswith(".") for part in path.relative_to(root).parts):
            continue
        result.append(path)
    result.sort()
    return result


def _parse_sections(file_path: Path, file_sections: list[str]) -> list[tuple[str, int, int, str]]:
    sections: list[tuple[str, int, int, str]] = []
    start_idx: int | None = None
    heading: str | None = None
    lines: list[str] = []

    for index, line in enumerate(file_sections, start=1):
        match = SECTION_RE.match(line)
        if not match:
            if heading is not None:
                lines.append(line)
            continue

        if heading is not None and start_idx is not None:
            sections.append((heading, start_idx, index - 1, "\n".join(lines).strip()))

        level = len(match.group(1))
        heading = match.group(2).strip()
        start_idx = index
        lines = [f"{'#' * level} {heading}"]

    if heading is not None and start_idx is not None:
        sections.append((heading, start_idx, len(file_sections), "\n".join(lines).strip()))

    return sections


def _collect_sections(workspace: Path) -> list[Section]:
    sections: list[Section] = []
    slug_counters: dict[str, dict[str, int]] = defaultdict(dict)

    for file_path in _visible_files(workspace):
        text = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
        parsed = _parse_sections(file_path, text)
        for heading, line_start, line_end, body in parsed:
            heading_key = _normalize_heading(heading)
            file_slug = file_path.stem
            base_id = f"{file_slug}::{_slugify(heading, 'section')}"
            count = slug_counters[file_slug].get(heading_key, 0) + 1
            slug_counters[file_slug][heading_key] = count
            node_id = base_id if count == 1 else f"{base_id}#{count}"

            section_type = _classify_node_type(body)
            metadata: dict[str, Any] = {
                "file": str(file_path.relative_to(workspace)),
                "heading": heading,
                "line_start": line_start,
                "line_end": line_end,
                "bootstrap_weight": 0.5,
            }

            sections.append(
                Section(
                    file_path=file_path,
                    heading=heading,
                    content=body,
                    node_id=node_id,
                    line_start=line_start,
                    line_end=line_end,
                    section_type=section_type,
                    metadata=metadata,
                )
            )

    return sections


def _by_file_sections(sections: list[Section]) -> dict[Path, list[Section]]:
    grouped: dict[Path, list[Section]] = defaultdict(list)
    for section in sections:
        grouped[section.file_path].append(section)
    return grouped


def _build_heading_index(sections: list[Section]) -> dict[str, list[Section]]:
    index: dict[str, list[Section]] = defaultdict(list)
    for section in sections:
        index[_normalize_heading(section.heading)].append(section)
    return index


def bootstrap_workspace(workspace: str | Path, output_path: str | Path) -> dict[str, Any]:
    workspace_path = Path(workspace)
    output_file = Path(output_path)

    sections = _collect_sections(workspace_path)
    sections_by_file = _by_file_sections(sections)
    heading_index = _build_heading_index(sections)

    graph = Graph()

    for section in sections:
        node = Node(
            id=section.node_id,
            content=section.content,
            type=section.section_type,
            metadata=section.metadata,
            threshold=0.5,
        )
        graph.add_node(node)

    intra_edges = 0
    for file_sections in sections_by_file.values():
        ordered = file_sections
        for left, right in zip(ordered, ordered[1:]):
            graph.add_edge(Edge(source=left.node_id, target=right.node_id, weight=0.6))
            intra_edges += 1

    inbound_ref_counts: dict[str, int] = defaultdict(int)
    cross_edges = 0
    for source in sections:
        source_text = _normalize_heading(source.content)
        for heading_key, targets in heading_index.items():
            if not heading_key:
                continue
            if heading_key not in source_text:
                continue
            for target in targets:
                if target.file_path == source.file_path and target.node_id == source.node_id:
                    continue
                if target.file_path == source.file_path:
                    # We only create cross-file reference edges here.
                    continue
                graph.add_edge(Edge(source=source.node_id, target=target.node_id, weight=0.4))
                inbound_ref_counts[target.node_id] += 1
                cross_edges += 1

    # Raise bootstrap weights for frequently referenced sections.
    for section in sections:
        node = graph.get_node(section.node_id)
        if node is None:
            continue
        hits = inbound_ref_counts.get(section.node_id, 0)
        node.metadata["reference_count"] = hits
        node.metadata["bootstrap_weight"] = round(0.5 + min(hits * 0.1, 0.5), 3)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    graph.save(str(output_file))

    node_types: dict[str, int] = {}
    for node in graph.nodes():
        node_types[node.type] = node_types.get(node.type, 0) + 1

    stats = {
        "workspace_path": str(workspace_path),
        "output_path": str(output_file),
        "files_seen": len(sections_by_file),
        "nodes": graph.node_count,
        "edges": graph.edge_count,
        "intra_edges": intra_edges,
        "cross_edges": cross_edges,
        "node_types": node_types,
    }
    print(json.dumps(stats, indent=2))
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bootstrap a CrabPath graph from workspace markdown sections."
    )
    parser.add_argument("workspace", help="Workspace directory")
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write graph JSON",
    )
    args = parser.parse_args()
    bootstrap_workspace(args.workspace, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
