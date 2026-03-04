from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from openclawbrain import Graph, HashEmbedder, TraversalConfig, VectorIndex, traverse
from openclawbrain.graph import Edge, Node
from openclawbrain.local_embedder import LocalEmbedder, resolve_local_model
from openclawbrain.prompt_context import build_prompt_context_ranked_with_stats


@dataclass
class EmbedderConfig:
    embed_fn: Callable[[str], list[float]]
    name: str
    dim: int
    mode: str


def resolve_embedder(mode: str | None = None) -> EmbedderConfig:
    if mode is not None and mode not in {"local", "hash"}:
        raise ValueError(f"Unsupported embedder mode: {mode}")

    if mode == "hash":
        fallback = HashEmbedder()
        return EmbedderConfig(
            embed_fn=fallback.embed,
            name=fallback.name,
            dim=fallback.dim,
            mode="hash",
        )

    try:
        embedder = LocalEmbedder(model_name=resolve_local_model())
        # Force initialization to detect missing fastembed.
        _ = embedder.embed("")
        return EmbedderConfig(
            embed_fn=embedder.embed,
            name=embedder.name,
            dim=embedder.dim,
            mode="local",
        )
    except Exception:
        fallback = HashEmbedder()
        return EmbedderConfig(
            embed_fn=fallback.embed,
            name=fallback.name,
            dim=fallback.dim,
            mode="hash",
        )


def normalize_messages(messages: Iterable[object]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for item in messages:
        if isinstance(item, dict):
            role = str(item.get("role") or item.get("speaker") or item.get("from") or "user")
            content = str(item.get("content") or item.get("text") or item.get("utterance") or "")
        else:
            role = "user"
            content = str(item)
        content = content.strip()
        if not content:
            continue
        normalized.append({"role": role, "content": content})
    return normalized


def build_state_from_messages(
    messages: Iterable[object],
    embedder: EmbedderConfig,
) -> tuple[Graph, VectorIndex]:
    graph = Graph()
    index = VectorIndex()
    normalized = normalize_messages(messages)
    prev_id: str | None = None
    for idx, msg in enumerate(normalized):
        node_id = f"msg::{idx}"
        content = f"{msg['role']}: {msg['content']}"
        graph.add_node(Node(node_id, content=content, summary=msg["content"][:120]))
        index.upsert(node_id, embedder.embed_fn(content))
        if prev_id is not None:
            graph.add_edge(Edge(source=prev_id, target=node_id, weight=0.35, kind="sibling"))
            graph.add_edge(Edge(source=node_id, target=prev_id, weight=0.35, kind="sibling"))
        prev_id = node_id
    return graph, index


def retrieve_prompt_context(
    graph: Graph,
    index: VectorIndex,
    embedder: EmbedderConfig,
    query: str,
    *,
    top_k: int = 6,
    max_prompt_context_chars: int = 20000,
) -> tuple[str, dict[str, object]]:
    query_vec = embedder.embed_fn(query)
    seeds = index.search(query_vec, top_k=top_k)
    if not seeds:
        return "", {
            "prompt_context_len": 0,
            "prompt_context_max_chars": max_prompt_context_chars,
            "prompt_context_trimmed": False,
        }
    node_ids = [node_id for node_id, _score in seeds]
    node_scores = {node_id: score for node_id, score in seeds}

    # Traverse is used to respect max_context_chars fairness cap if desired.
    traversal = traverse(
        graph,
        seeds=seeds,
        config=TraversalConfig(max_hops=1, beam_width=len(seeds), max_context_chars=None),
        query_text=query,
    )
    fired_ids = traversal.fired or node_ids
    prompt_context, stats = build_prompt_context_ranked_with_stats(
        graph,
        fired_ids,
        node_scores=node_scores,
        max_chars=max_prompt_context_chars,
        include_node_ids=False,
    )
    stats["retrieved_nodes"] = fired_ids
    stats["retrieved_raw_chars"] = len(traversal.context)
    return prompt_context, stats
