"""Workspace splitter utilities for constructing an initial graph."""

from __future__ import annotations

import hashlib
import fnmatch
import os
from collections.abc import Iterable
from pathlib import Path

from .graph import Edge, Graph, Node


DEFAULT_EXCLUDES = {
    "node_modules",
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    "dist",
    "build",
    ".next",
    ".cache",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "vendor",
    "target",
}


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


def _load_gitignore_patterns(workspace: Path) -> list[str]:
    """Load non-empty, non-comment patterns from ``.gitignore``.

    This is intentionally minimal but useful for production directories where default
    ignores are not enough.
    """
    gitignore = workspace / ".gitignore"
    if not gitignore.exists():
        return []
    raw_lines = gitignore.read_text(encoding="utf-8").splitlines()
    return [line.strip().replace("\\", "/") for line in raw_lines if line.strip() and not line.strip().startswith("#")]


def _match_gitignore(path_posix: str, patterns: list[str]) -> bool:
    for pattern in patterns:
        if pattern.startswith("!"):
            continue
        normalized = pattern.lstrip("./").replace("\\", "/")
        is_dir_pattern = normalized.endswith("/")
        if is_dir_pattern:
            normalized = normalized[:-1]
            if path_posix == normalized or path_posix.startswith(normalized + "/"):
                return True
            continue
        if pattern.startswith("/"):
            normalized = normalized[1:]
            if fnmatch.fnmatch(path_posix, normalized):
                return True
            continue
        if "/" in normalized:
            if fnmatch.fnmatch(path_posix, normalized):
                return True
            continue
        if fnmatch.fnmatch(Path(path_posix).name, normalized) or fnmatch.fnmatch(path_posix, f"*/{normalized}"):
            return True
    return False


def _normalize_excludes(exclude: Iterable[str] | None) -> set[str]:
    excludes = set(DEFAULT_EXCLUDES)
    if exclude is None:
        return excludes
    for item in exclude:
        value = item.strip()
        if value:
            excludes.add(value)
    return excludes


def _should_skip_path(relative_path: str, excludes: set[str], gitignore_patterns: list[str]) -> bool:
    if not relative_path:
        return False
    path = Path(relative_path)
    if any(part.startswith(".") for part in path.parts):
        return True
    normalized = str(path).replace("\\", "/").lstrip("./")
    for pattern in excludes:
        if path.name == pattern:
            return True
        if pattern.endswith("/"):
            if normalized == pattern[:-1] or normalized.startswith(pattern):
                return True
            continue
        if "*" in pattern or "/" in pattern:
            if fnmatch.fnmatch(normalized, pattern) or fnmatch.fnmatch(path.name, pattern):
                return True
            continue
        if fnmatch.fnmatch(path.name, pattern) or normalized == pattern:
            return True
    normalized = str(path).replace("\\", "/").lstrip("./")
    if _match_gitignore(normalized, gitignore_patterns):
        return True
    return False


def split_workspace(
    workspace_dir: str | Path,
    *,
    max_depth: int = 3,
    exclude: Iterable[str] | None = None,
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
    if max_depth < 0:
        raise ValueError("max_depth must be >= 0")

    excludes = _normalize_excludes(exclude)
    gitignore_patterns = _load_gitignore_patterns(workspace)

    graph = Graph()
    texts: dict[str, str] = {}

    for dir_path, dir_names, file_names in os.walk(workspace):
        rel_dir = Path(dir_path).resolve().relative_to(workspace.resolve())
        depth = len(rel_dir.parts)
        if depth > max_depth:
            dir_names[:] = []
            continue
        if depth == max_depth:
            dir_names[:] = []

        for dir_name in sorted(list(dir_names)):
            rel = (rel_dir / dir_name).as_posix() if rel_dir.parts else dir_name
            if _should_skip_path(rel, excludes, gitignore_patterns):
                dir_names.remove(dir_name)

        for filename in sorted(file_names):
            rel = (rel_dir / filename).as_posix() if rel_dir.parts else filename
            file_path = Path(dir_path) / filename
            if not file_path.is_file():
                continue
            if not file_path.suffix.lower() == ".md":
                continue
            if _should_skip_path(rel, excludes, gitignore_patterns):
                continue

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
