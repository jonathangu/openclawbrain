"""OpenClaw adapter for CrabPath.

This module wraps graph loading, activation, seeding, learning, and snapshot
persistence into one session-oriented helper.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Optional

from .activation import Firing, activate as _activate, learn as _learn
from .embeddings import EmbeddingIndex
from .feedback import snapshot_path
from .graph import Graph

EmbeddingFn = Callable[[list[str]], list[list[float]]]

MEMORY_SEARCH_ENERGY = 0.25


class OpenClawCrabPathAdapter:
    """Adapter used by OpenClaw session runtime.

    The adapter keeps the graph, embedding index, and snapshot path in one object.
    It intentionally stays lightweight and dependency free.
    """

    def __init__(
        self,
        graph_path: str,
        index_path: str,
        embed_fn: Optional[EmbeddingFn] = None,
    ) -> None:
        """Create an adapter bound to graph/index paths.

        Args:
            graph_path: JSON path for the graph artifact.
            index_path: JSON path for the embedding index.
            embed_fn: Optional embedding function to use with EmbeddingIndex.
        """
        self.graph_path = graph_path
        self.index_path = index_path
        self.embed_fn = embed_fn

        self.graph = Graph()
        self.index = EmbeddingIndex()
        self.snapshot_path = str(snapshot_path(graph_path))

    # -- Lifecycle ---------------------------------------------------------

    def load(self) -> tuple[Graph, EmbeddingIndex]:
        """Load graph + index from disk, or start empty if missing."""
        graph_file = Path(self.graph_path)
        if graph_file.exists():
            self.graph = Graph.load(self.graph_path)
        else:
            self.graph = Graph()

        index_file = Path(self.index_path)
        if index_file.exists():
            self.index = EmbeddingIndex.load(self.index_path)
        else:
            self.index = EmbeddingIndex()
        return self.graph, self.index

    # -- Seeding -----------------------------------------------------------

    def seed(
        self,
        query_text: str,
        memory_search_ids: Optional[list[str]] = None,
        top_k: int = 8,
    ) -> dict[str, float]:
        """Build the seed map used for activation.

        The seed map combines:
          1) semantic seeds from the EmbeddingIndex (if available)
          2) `memory_search_ids` as weak symbolic seeds (default 0.25 each)
        """
        seeds: dict[str, float] = {}

        if self.embed_fn is not None and self.index.vectors:
            embed_seeds = self.index.seed(
                query_text,
                embed_fn=self.embed_fn,
                top_k=top_k,
            )
            seeds.update(embed_seeds)

        if memory_search_ids:
            for node_id in memory_search_ids:
                if self.graph.get_node(node_id) is None:
                    continue
                seeds[node_id] = max(seeds.get(node_id, 0.0), MEMORY_SEARCH_ENERGY)

        return seeds

    # -- Activation --------------------------------------------------------

    def activate(
        self,
        seeds: dict[str, float],
        max_steps: int = 3,
        decay: float = 0.1,
        top_k: int = 12,
    ) -> Firing:
        """Run one activation pass over the graph.

        Uses `reset=False` to retain warm state between turns by default.
        """
        return _activate(
            self.graph,
            seeds,
            max_steps=max_steps,
            decay=decay,
            top_k=top_k,
            reset=False,
        )

    # -- Context -----------------------------------------------------------

    def context(self, firing_result: Firing) -> dict[str, Any]:
        """Create context payload from a firing result.

        Returns:
            {
                "contents": content strings ordered by firing energy desc,
                "guardrails": inhibited node ids,
                "fired_ids": ids ordered by firing energy desc,
                "fired_scores": firing energies ordered by same order,
            }
        """
        ranked = sorted(firing_result.fired, key=lambda item: item[1], reverse=True)
        return {
            "contents": [node.content for node, _ in ranked],
            "guardrails": list(firing_result.inhibited),
            "fired_ids": [node.id for node, _ in ranked],
            "fired_scores": [score for _, score in ranked],
        }

    # -- Learning ----------------------------------------------------------

    def learn(self, firing_result: Firing, outcome: float) -> None:
        """Apply STDP-style learning for the firing outcome."""
        _learn(self.graph, firing_result, outcome=outcome)

    # -- Snapshotting ------------------------------------------------------

    def snapshot(self, session_id: str, turn_id: int | str, firing_result: Firing) -> dict[str, Any]:
        """Persist metadata about one assistant turn for delayed feedback."""
        record = {
            "session_id": session_id,
            "turn_id": turn_id,
            "timestamp": time.time(),
            "fired_ids": [node.id for node, _ in firing_result.fired],
            "fired_scores": [score for _, score in firing_result.fired],
            "fired_at": firing_result.fired_at,
            "inhibited": list(firing_result.inhibited),
            "attributed": False,
        }

        path = Path(self.snapshot_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        return record

    # -- Save ----------------------------------------------------------------

    def save(self) -> None:
        """Persist graph and index to their paths."""
        self.graph.save(self.graph_path)
        self.index.save(self.index_path)
