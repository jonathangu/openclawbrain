"""Edge weight decay mechanics."""

from __future__ import annotations

from dataclasses import dataclass

from .graph import Graph


@dataclass
class DecayConfig:
    """Decay control for weights in long-running graphs."""

    half_life: int = 80
    min_weight: float = 0.01


def apply_decay(graph: Graph, config: DecayConfig | None = None) -> int:
    """Apply exponential decay to all edges.

    Each edge weight is multiplied by ``0.5 ** (1/half_life)``. Edges with absolute
    weight below ``min_weight`` are set to ``0.0``.
    """
    cfg = config or DecayConfig()
    if cfg.half_life <= 0:
        return 0

    factor = 0.5 ** (1.0 / cfg.half_life)
    changed = 0

    for source_edges in graph._edges.values():
        for edge in source_edges.values():
            old = edge.weight
            new = old * factor
            if abs(new) < cfg.min_weight:
                new = 0.0
            if new != old:
                edge.weight = new
                changed += 1

    return changed
