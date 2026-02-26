"""Workspace splitter utilities for constructing an initial graph."""

from __future__ import annotations

from pathlib import Path
import hashlib

from .graph import Edge, Graph, Node


def _chunk_markdown(content: str) -> list[str]:
    """Split markdown content by level-2 headers.

    If there are no ``##`` headings, split on blank lines.
    """
    lines = content.splitlines()
    has_headers = any(line.startswith("## ") for line in lines)

    if not has_headers:
        parts = [part.strip() for part in content.split("\n\n") if part.strip()]
        return parts or [content]

    chunks: list[str] = []
    current: list[str] = []
    for line in lines:
        if line.startswith("## ") and current:
            chunk = "\n".join(current).strip()
            if chunk:
                chunks.append(chunk)
            current = [line]
        else:
            current.append(line)

    final = "\n".join(current).strip()
    if final:
        chunks.append(final)
    return chunks or [content]


def _sibling_weight(file_id: str, idx: int) -> float:
    """Return deterministic sibling baseline weight around ``0.5`` with tiny jitter."""
    digest = hashlib.sha256(f"{file_id}:{idx}".encode("utf-8")).hexdigest()[:8]
    jitter = (int(digest, 16) % 2001 - 1000) / 100000.0
    return max(0.4, min(0.6, 0.5 + jitter))


def split_workspace(
    workspace_dir: str | Path,
) -> tuple[Graph, dict[str, str]]:
    """Read markdown files from workspace and convert them into a graph.

    Args:
        workspace_dir: Directory containing markdown source files.

    Returns:
        ``(graph, texts)`` where each ``texts[node_id]`` is the chunk content for
        caller-provided embeddings.
    """
    workspace = Path(workspace_dir).expanduser()
    if not workspace.exists():
        raise FileNotFoundError(f"workspace not found: {workspace}")

    graph = Graph()
    texts: dict[str, str] = {}

    for file_path in sorted(
        path for path in workspace.rglob("*") if path.is_file() and path.suffix.lower() == ".md"
    ):
        rel = file_path.relative_to(workspace).as_posix()
        text = file_path.read_text(encoding="utf-8")
        chunks = _chunk_markdown(text)

        node_ids: list[str] = []
        for idx, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            node_id = f"{rel}::{idx}"
            summary = chunk.splitlines()[0] if chunk.splitlines() else ""
            node = Node(
                id=node_id,
                content=chunk,
                summary=summary,
                metadata={"file": rel, "chunk": idx, "kind": "markdown"},
            )
            graph.add_node(node)
            texts[node_id] = chunk
            node_ids.append(node_id)

        for source_offset, (source_id, target_id) in enumerate(zip(node_ids, node_ids[1:])):
            weight = _sibling_weight(rel, source_offset)
            graph.add_edge(Edge(source=source_id, target=target_id, weight=weight, kind="sibling"))
            graph.add_edge(Edge(source=target_id, target=source_id, weight=weight, kind="sibling"))

    return graph, texts
