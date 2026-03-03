#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable

from openclawbrain import HashEmbedder, TraversalConfig, load_state, traverse
from openclawbrain.local_embedder import LocalEmbedder, resolve_local_model
from openclawbrain.prompt_context import build_prompt_context_ranked_with_stats
from openclawbrain.socket_client import OCBClient


EMBED_MODEL = "text-embedding-3-small"
FIRED_LOG_TTL_SECONDS = 7 * 24 * 60 * 60
DEFAULT_MAX_PROMPT_CONTEXT_CHARS = 12000
BOOTSTRAP_EXCLUDE_FILES = ["AGENTS.md", "SOUL.md", "USER.md", "MEMORY.md", "active-tasks.md"]
PATH_EXCLUDED_NODE_STAT = "prompt_context_excluded_paths_count"
OPENAI_API_KEY_PATTERN = re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")
OPENAI_PROJECT_KEY_PATTERN = re.compile(r"\bsk-proj-[A-Za-z0-9_-]{20,}\b")
GITHUB_PAT_PATTERNS = (
    re.compile(r"\bghp_[A-Za-z0-9]{35,}\b"),
    re.compile(r"\bgho_[A-Za-z0-9]{35,}\b"),
    re.compile(r"\bghu_[A-Za-z0-9]{35,}\b"),
    re.compile(r"\bghs_[A-Za-z0-9]{35,}\b"),
    re.compile(r"\bghr_[A-Za-z0-9]{35,}\b"),
    re.compile(r"\bghl_[A-Za-z0-9]{35,}\b"),
)
BEARER_TOKEN_PATTERN = re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{20,}\b", re.IGNORECASE)


def _fired_log_path(state_path: Path) -> Path:
    return state_path.parent / "fired_log.jsonl"


def _read_fired_log(log_path: Path) -> list[dict[str, object]]:
    if not log_path.exists():
        return []

    entries = []
    for raw_line in log_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        entries.append(record)
    return entries


def _write_fired_log(log_path: Path, entries: list[dict[str, object]]) -> None:
    tmp = log_path.with_name(f"{log_path.name}.tmp")
    payload = "\n".join(json.dumps(entry) for entry in entries)
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(log_path)


def _prune_fired_log(entries: list[dict[str, object]], now: float) -> list[dict[str, object]]:
    cutoff = now - FIRED_LOG_TTL_SECONDS
    output = []
    for entry in entries:
        ts = entry.get("ts")
        if isinstance(ts, (int, float)) and ts >= cutoff:
            output.append(entry)
    return output


def _append_fired_log(log_path: Path, entry: dict[str, object], now: float | None = None) -> None:
    entries = _read_fired_log(log_path)
    entries.append(entry)
    entries = _prune_fired_log(entries, now if now is not None else time.time())
    _write_fired_log(log_path, entries)


def _load_query_via_socket(
    socket_path: str | None,
    query_text: str,
    chat_id: str | None,
    top: int,
    route_mode: str,
    route_top_k: int,
    route_alpha_sim: float,
    route_use_relevance: bool,
    max_prompt_context_chars: int,
    exclude_files: list[str],
    exclude_file_prefixes: list[str],
    exclude_paths: list[str],
    prompt_context_include_node_ids: bool,
    include_provenance: bool,
) -> dict[str, object] | None:
    if socket_path is None:
        return None

    with OCBClient(socket_path) as client:
        params: dict[str, Any] = {
            "query": query_text,
            "top_k": top,
            "route_mode": route_mode,
            "route_top_k": route_top_k,
            "route_alpha_sim": route_alpha_sim,
            "route_use_relevance": route_use_relevance,
            "max_prompt_context_chars": max_prompt_context_chars,
            "include_provenance": include_provenance,
        }
        if chat_id is not None:
            params["chat_id"] = chat_id
        if exclude_files:
            params["exclude_files"] = exclude_files
        if exclude_file_prefixes:
            params["exclude_file_prefixes"] = exclude_file_prefixes
        if exclude_paths:
            params["exclude_paths"] = exclude_paths
        params["prompt_context_include_node_ids"] = prompt_context_include_node_ids
        payload = client.request("query", params)
        if not isinstance(payload, dict):
            raise RuntimeError("invalid socket response payload")
        return payload


def _collect_prompt_context_stats(payload: dict[str, object]) -> dict[str, object]:
    stats: dict[str, object] = {}
    for key, value in payload.items():
        if isinstance(key, str) and key.startswith("prompt_context_"):
            stats[key] = value
    return stats


def _expand_recent_memory_files(values: list[str]) -> list[str]:
    expanded: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = raw.strip()
        if not value:
            continue
        candidates = [value]
        if "/" not in value:
            candidates.append(f"memory/{value}")
        for candidate in candidates:
            if candidate not in seen:
                seen.add(candidate)
                expanded.append(candidate)
    return expanded


def _compact_stats_subset(
    prompt_context_stats: dict[str, object],
    timings: dict[str, object] | None,
) -> dict[str, object]:
    allowed_prompt_keys = (
        "prompt_context_len",
        "prompt_context_max_chars",
        "prompt_context_trimmed",
        "prompt_context_dropped_count",
        "prompt_context_dropped_authority_counts",
        "prompt_context_excluded_files_count",
        "prompt_context_excluded_node_ids_count",
        PATH_EXCLUDED_NODE_STAT,
    )
    subset = {key: prompt_context_stats[key] for key in allowed_prompt_keys if key in prompt_context_stats}
    if timings:
        for key in ("embed_query_ms", "traverse_ms", "total_ms"):
            if key in timings:
                subset[key] = timings[key]
    return subset


def _normalize_exclude_values(values: list[str] | None) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        item = value.strip()
        if not item:
            continue
        if item not in seen:
            seen.add(item)
            normalized.append(item)
    return normalized


def _node_source_path(node_metadata: object) -> str | None:
    if not isinstance(node_metadata, dict):
        return None
    for key in ("source", "path", "file"):
        raw_value = node_metadata.get(key)
        if isinstance(raw_value, str):
            value = raw_value.strip()
            if value:
                return value
    return None


def _node_matches_excluded_path(graph, node_id: str, exclude_paths: list[str]) -> bool:
    if not exclude_paths:
        return False
    node = graph.get_node(node_id)
    source = _node_source_path(node.metadata if node is not None else None)
    if not source:
        return False
    lower_source = source.lower()
    return any(exclude.lower() in lower_source for exclude in exclude_paths)


def _apply_prompt_path_filters(
    graph,
    node_ids: list[str],
    *,
    exclude_paths: list[str],
) -> tuple[list[str], int]:
    if not exclude_paths:
        return node_ids, 0

    kept: list[str] = []
    filtered_count = 0
    for node_id in node_ids:
        if _node_matches_excluded_path(graph, node_id, exclude_paths):
            filtered_count += 1
            continue
        kept.append(node_id)

    return kept, filtered_count


def _redact_prompt_context_text(prompt_context: str) -> str:
    redacted = OPENAI_API_KEY_PATTERN.sub("<REDACTED_OPENAI_API_KEY>", prompt_context)
    redacted = OPENAI_PROJECT_KEY_PATTERN.sub("<REDACTED_OPENAI_PROJECT_KEY>", redacted)
    for pattern in GITHUB_PAT_PATTERNS:
        redacted = pattern.sub("<REDACTED_GITHUB_TOKEN>", redacted)
    redacted = BEARER_TOKEN_PATTERN.sub("<REDACTED_BEARER_TOKEN>", redacted)
    return redacted


def _dump_json(payload: dict[str, object], *, pretty: bool) -> str:
    if pretty:
        return json.dumps(payload, indent=2)
    return json.dumps(payload, separators=(",", ":"))


def _build_json_output(
    *,
    state_path: Path,
    query_text: str,
    fired_nodes: list[str],
    prompt_context: str,
    prompt_context_stats: dict[str, object],
    seeds: object | None = None,
    context: object | None = None,
    timings: dict[str, object] | None = None,
    compact: bool,
    include_stats: bool,
) -> dict[str, object]:
    if compact:
        output: dict[str, object] = {
            "state": str(state_path),
            "query": query_text,
            "fired_nodes": fired_nodes,
            "prompt_context": prompt_context,
        }
        if include_stats:
            output.update(_compact_stats_subset(prompt_context_stats, timings))
        return output

    output = {
        "state": str(state_path),
        "query": query_text,
        "seeds": seeds if seeds is not None else [],
        "fired_nodes": fired_nodes,
        "context": context,
        "prompt_context": prompt_context,
    }
    output.update(prompt_context_stats)
    if timings:
        output.update(timings)
    return output


def require_api_key() -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "This script must run inside the agent framework exec environment where OPENAI_API_KEY is injected."
        )
    return api_key


def embed_query(client, query: str) -> list[float]:
    response = client.embeddings.create(model=EMBED_MODEL, input=[query])
    return response.data[0].embedding


def _embed_fn_from_state(meta: dict[str, object]) -> tuple[Callable[[str], list[float]], str]:
    embedder_name = str(meta.get("embedder_name", "hash-v1"))
    hash_dim = meta.get("embedder_dim")

    if embedder_name.startswith("local:"):
        embedder = LocalEmbedder(model_name=resolve_local_model(meta))
        return embedder.embed, embedder.name
    if embedder_name == "hash-v1" and hash_dim == HashEmbedder().dim:
        return HashEmbedder().embed, embedder_name
    if embedder_name == "hash-v1":
        # Legacy fallback: some historical states used hash-v1 with non-hash dims.
        from openai import OpenAI

        api_key = require_api_key()
        client = OpenAI(api_key=api_key)
        return lambda text: embed_query(client, text), "text-embedding-3-small"
    if embedder_name in {"text-embedding-3-small", "openai-text-embedding-3-small"}:
        from openai import OpenAI
        api_key = require_api_key()
        client = OpenAI(api_key=api_key)
        return lambda text: embed_query(client, text), embedder_name
    return HashEmbedder().embed, "hash-v1"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Query a OpenClawBrain state.json with OpenAI embeddings"
    )
    parser.add_argument("state_path", help="Path to state.json")
    parser.add_argument("query", nargs="+", help="Query text")
    parser.add_argument("--chat-id", help="Conversation id to persist fired nodes")
    parser.add_argument("--socket", help="Unix socket path for daemon mode")
    parser.add_argument("--top", type=int, default=4, help="Top-k vector matches")
    parser.add_argument(
        "--route-mode",
        choices=["off", "edge", "edge+sim"],
        default="off",
        help="Runtime route policy for daemon query mode (local fallback ignores this)",
    )
    parser.add_argument(
        "--route-top-k",
        type=int,
        default=5,
        help="Top-k candidates for runtime route policy (daemon mode only)",
    )
    parser.add_argument(
        "--route-alpha-sim",
        type=float,
        default=0.5,
        help="Similarity blend weight for route_mode=edge+sim (daemon mode only)",
    )
    parser.add_argument(
        "--route-use-relevance",
        dest="route_use_relevance",
        action="store_true",
        default=True,
        help="Apply relevance priors in runtime route policy (daemon mode only, default)",
    )
    parser.add_argument(
        "--no-route-use-relevance",
        dest="route_use_relevance",
        action="store_false",
        help="Disable relevance priors in runtime route policy (daemon mode only)",
    )
    parser.add_argument(
        "--max-prompt-context-chars",
        type=int,
        default=DEFAULT_MAX_PROMPT_CONTEXT_CHARS,
        help="Max chars for deterministic prompt_context",
    )
    parser.add_argument(
        "--exclude-bootstrap",
        dest="exclude_bootstrap",
        action="store_true",
        default=True,
        help="Exclude OpenClaw bootstrap files from prompt_context (default: on)",
    )
    parser.add_argument(
        "--no-exclude-bootstrap",
        dest="exclude_bootstrap",
        action="store_false",
        help="Keep bootstrap files in prompt_context",
    )
    parser.add_argument(
        "--exclude-recent-memory",
        nargs="+",
        default=None,
        metavar="FILE",
        help="Exclude specific recent memory note files from prompt_context",
    )
    parser.add_argument(
        "--exclude-paths",
        action="append",
        default=None,
        metavar="SUBSTRING",
        help="Drop fired nodes whose source path contains this substring (repeatable)",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json", "prompt"],
        default="text",
        help="Output format (json includes both context and prompt_context)",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    parser.add_argument(
        "--redact",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Redact common secret patterns in emitted prompt_context",
    )
    parser.add_argument(
        "--compact",
        dest="compact",
        action="store_true",
        default=None,
        help="Compact JSON output (omit non-deterministic context)",
    )
    parser.add_argument(
        "--no-compact",
        dest="compact",
        action="store_false",
        help="Full JSON output (includes context/seeds)",
    )
    parser.add_argument(
        "--include-node-ids",
        dest="include_node_ids",
        action="store_true",
        default=None,
        help="Include '- node: <id>' lines in prompt_context",
    )
    parser.add_argument("--provenance", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--no-include-node-ids",
        dest="include_node_ids",
        action="store_false",
        help="Omit '- node: <id>' lines in prompt_context",
    )
    parser.add_argument(
        "--include-stats",
        action="store_true",
        help="In compact JSON, include a small scalar stats/timings subset",
    )
    parser.add_argument(
        "--pretty-json",
        action="store_true",
        help="Pretty-print JSON with indentation",
    )
    args = parser.parse_args(argv)
    output_format = "json" if args.json else args.format
    compact = args.compact if args.compact is not None else bool(args.json)
    include_node_ids = args.include_node_ids if args.include_node_ids is not None else (not compact)
    redact = bool(args.redact) if args.redact is not None else output_format == "prompt"
    exclude_paths = _normalize_exclude_values(args.exclude_paths)

    query_text = " ".join(args.query).strip()
    state_path = Path(args.state_path)
    if args.top <= 0:
        raise SystemExit("--top must be >= 1")
    if args.route_top_k <= 0:
        raise SystemExit("--route-top-k must be >= 1")
    if not 0.0 <= args.route_alpha_sim <= 1.0:
        raise SystemExit("--route-alpha-sim must be between 0 and 1")
    if args.max_prompt_context_chars <= 0:
        raise SystemExit("--max-prompt-context-chars must be >= 1")

    resolved_socket = args.socket
    if resolved_socket is None:
        resolved_socket = OCBClient.default_socket_path(state_path.expanduser().parent.name)

    exclude_files: list[str] = []
    if args.exclude_bootstrap:
        exclude_files.extend(BOOTSTRAP_EXCLUDE_FILES)
    if args.exclude_recent_memory:
        exclude_files.extend(_expand_recent_memory_files(args.exclude_recent_memory))
    # Preserve insertion order and remove duplicates.
    exclude_files = list(dict.fromkeys(exclude_files))
    # Normalize repeatable path filters.
    exclude_paths = _normalize_exclude_values(exclude_paths)

    if args.socket is not None or Path(resolved_socket).expanduser().exists():
        try:
            result = _load_query_via_socket(
                resolved_socket,
                query_text,
                args.chat_id,
                args.top,
                args.route_mode,
                args.route_top_k,
                args.route_alpha_sim,
                args.route_use_relevance,
                args.max_prompt_context_chars,
                exclude_files,
                [],
                exclude_paths,
                include_node_ids,
                bool(args.provenance),
            )
            if result is not None:
                fired_nodes = [str(node_id) for node_id in result.get("fired_nodes", [])]
                raw_fired_scores = result.get("fired_scores")
                fired_scores = (
                    {str(node_id): float(score) for node_id, score in raw_fired_scores.items()}
                    if isinstance(raw_fired_scores, dict)
                    else None
                )
                prompt_context = str(result.get("prompt_context") or "")
                prompt_context_stats = _collect_prompt_context_stats(result)
                if (not prompt_context or exclude_paths) and state_path.exists() and output_format in {"json", "prompt"}:
                    graph, _index, _meta = load_state(str(state_path))
                    prompt_node_ids = fired_nodes
                    path_excluded_count = 0
                    if exclude_paths:
                        prompt_node_ids, path_excluded_count = _apply_prompt_path_filters(
                            graph,
                            fired_nodes,
                            exclude_paths=exclude_paths,
                        )
                    if prompt_node_ids != fired_nodes or not prompt_context:
                        prompt_context, prompt_context_stats = build_prompt_context_ranked_with_stats(
                            graph=graph,
                            node_ids=prompt_node_ids,
                            node_scores=fired_scores,
                            max_chars=args.max_prompt_context_chars,
                            include_node_ids=include_node_ids,
                        )
                    if path_excluded_count:
                        prompt_context_stats["prompt_context_excluded_node_ids_count"] = (
                            int(prompt_context_stats.get("prompt_context_excluded_node_ids_count", 0))
                            + path_excluded_count
                        )
                        prompt_context_stats[PATH_EXCLUDED_NODE_STAT] = path_excluded_count
                if redact:
                    prompt_context = _redact_prompt_context_text(prompt_context)
                timings = {
                    key: result[key]
                    for key in ("embed_query_ms", "traverse_ms", "total_ms")
                    if key in result
                }
                if output_format == "json":
                    output = _build_json_output(
                        state_path=state_path,
                        query_text=query_text,
                        fired_nodes=fired_nodes,
                        prompt_context=prompt_context,
                        prompt_context_stats=prompt_context_stats,
                        seeds=result.get("seeds", []),
                        context=result.get("context"),
                        timings=timings,
                        compact=compact,
                        include_stats=bool(args.include_stats),
                    )
                    print(_dump_json(output, pretty=bool(args.pretty_json or not compact)))
                    return
                if output_format == "prompt":
                    print(prompt_context)
                    return
                print("Fired nodes:")
                for node_id in fired_nodes:
                    print(f"- {node_id}")
                print("Context:")
                print(result.get("context") or "(no context)")
                return
        except Exception as exc:
            print(f"socket unavailable, falling back to local state: {exc}", file=sys.stderr)

    if not state_path.exists():
        raise SystemExit(f"state file not found: {state_path}")

    graph, index, meta = load_state(str(state_path))
    query_embed_fn, embedder_name = _embed_fn_from_state(meta)
    query_vector = query_embed_fn(query_text)

    expected_dim = meta.get("embedder_dim")
    if isinstance(expected_dim, int) and len(query_vector) != expected_dim:
        raise SystemExit(
            "Embedding dimension mismatch: "
            f"query={len(query_vector)} index={expected_dim}. "
            f"Rebuild state.json with {embedder_name}."
        )

    seeds = index.search(query_vector, top_k=args.top)
    result = traverse(
        graph=graph,
        seeds=seeds,
        query_text=query_text,
        config=TraversalConfig(max_context_chars=20000, include_provenance=bool(args.provenance)),
    )

    excluded_node_ids: set[str] = set()
    excluded_files: set[str] = set()
    path_excluded_count = 0
    if exclude_files:
        for node_id in result.fired:
            node = graph.get_node(node_id)
            metadata = node.metadata if node is not None and isinstance(node.metadata, dict) else {}
            file_path = metadata.get("file")
            if not isinstance(file_path, str) or not file_path:
                continue
            if file_path in exclude_files:
                excluded_node_ids.add(node_id)
                excluded_files.add(file_path)
    if exclude_paths:
        for node_id in result.fired:
            if node_id in excluded_node_ids:
                continue
            if _node_matches_excluded_path(graph, node_id, exclude_paths):
                excluded_node_ids.add(node_id)
                path_excluded_count += 1

    prompt_node_ids = [node_id for node_id in result.fired if node_id not in excluded_node_ids]
    prompt_context, prompt_context_stats = build_prompt_context_ranked_with_stats(
        graph=graph,
        node_ids=prompt_node_ids,
        node_scores={str(node_id): float(score) for node_id, score in result.fired_scores.items()},
        max_chars=args.max_prompt_context_chars,
        include_node_ids=include_node_ids,
    )
    prompt_context_stats["prompt_context_excluded_files_count"] = len(excluded_files)
    prompt_context_stats["prompt_context_excluded_node_ids_count"] = len(excluded_node_ids)
    if exclude_paths:
        prompt_context_stats[PATH_EXCLUDED_NODE_STAT] = path_excluded_count
    if redact:
        prompt_context = _redact_prompt_context_text(prompt_context)

    if args.chat_id:
        log_entry = {
            "chat_id": args.chat_id,
            "query": query_text,
            "fired_nodes": result.fired,
            "ts": time.time(),
        }
        _append_fired_log(_fired_log_path(state_path), log_entry)

    if output_format == "json":
        output = _build_json_output(
            state_path=state_path,
            query_text=query_text,
            fired_nodes=result.fired,
            prompt_context=prompt_context,
            prompt_context_stats=prompt_context_stats,
            seeds=seeds,
            context=result.context,
            compact=compact,
            include_stats=bool(args.include_stats),
        )
        print(_dump_json(output, pretty=bool(args.pretty_json or not compact)))
        return
    if output_format == "prompt":
        print(prompt_context)
        return

    print("Fired nodes:")
    for node_id in result.fired:
        print(f"- {node_id}")
    print("Context:")
    print(result.context or "(no context)")


if __name__ == "__main__":
    main()
