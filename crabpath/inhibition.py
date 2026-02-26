"""
Inhibition mechanics for learned suppression in CrabPath routing.
"""

from __future__ import annotations

from dataclasses import dataclass

from .graph import Edge, Graph


@dataclass
class InhibitionConfig:
    """Configuration knobs for inhibitory updates and routing suppression."""

    correction_decay: float = 0.5
    negative_cap: float = -1.0
    inhibition_strength: float = 0.15
    suppression_lambda: float = 1.0


def apply_correction(
    graph: Graph,
    trajectory: list[str],
    reward: float,
    config: InhibitionConfig,
) -> list[dict[str, float | str | None]]:
    """Apply inhibitory correction along a traversed trajectory."""
    if reward >= 0.0:
        return []

    modified: list[dict[str, float | str | None]] = []

    if len(trajectory) < 2:
        return modified

    for source, target in zip(trajectory, trajectory[1:]):
        existing = graph.get_edge(source, target)
        if existing is not None:
            before = existing.weight
            if existing.weight > 0:
                existing.weight = existing.weight * config.correction_decay
            else:
                existing.weight = max(existing.weight - 0.1, config.negative_cap)
            after = existing.weight
            if after != before:
                modified.append(
                    {
                        "source": source,
                        "target": target,
                        "before": before,
                        "after": after,
                    }
                )
            continue

        before = None
        new_edge = Edge(
            source=source,
            target=target,
            weight=-config.inhibition_strength,
            created_by="auto",
        )
        if not _add_edge_with_competition(graph, new_edge, config):
            continue

        added = graph.get_edge(source, target)
        if added is not None:
            modified.append(
                {
                    "source": source,
                    "target": target,
                    "before": before,
                    "after": added.weight,
                }
            )

    return modified


def score_with_inhibition(
    candidates: list[tuple[str, float]],
    graph: Graph,
    source_node: str,
    config: InhibitionConfig,
) -> list[tuple[str, float]]:
    """Apply inhibitory suppression to candidate scores and return sorted results."""
    scored = []
    for target_node, base_score in candidates:
        suppressed = base_score + (
            _get_inhibitory_weight(graph, source_node, target_node)
            * config.suppression_lambda
        )
        scored.append((target_node, suppressed))

    return sorted(scored, key=lambda c: c[1], reverse=True)


def is_inhibited(graph: Graph, source: str, target: str, threshold: float = -0.1) -> bool:
    """Check whether an inhibitory edge exists from source to target."""
    edge = graph.get_edge(source, target)
    if edge is None:
        return False
    return edge.weight <= threshold


def get_inhibitory_edges(graph: Graph, node_id: str) -> list[tuple[str, float]]:
    """Return all outgoing inhibitory edges from node_id."""
    return [
        (target.id, edge.weight)
        for target, edge in graph.outgoing(node_id)
        if edge.weight < 0
    ]


def inhibition_stats(graph: Graph) -> dict[str, int | float]:
    """Summarize inhibitory edges in a graph."""
    inhibitory = [edge.weight for edge in graph.edges() if edge.weight < 0]

    if not inhibitory:
        return {
            "total_inhibitory_edges": 0,
            "strongest_inhibition": 0.0,
            "average_inhibitory_weight": 0.0,
        }

    return {
        "total_inhibitory_edges": len(inhibitory),
        "strongest_inhibition": min(inhibitory),
        "average_inhibitory_weight": sum(inhibitory) / len(inhibitory),
    }


def _get_inhibitory_weight(graph: Graph, source: str, target: str) -> float:
    edge = graph.get_edge(source, target)
    if edge is None or edge.weight >= 0:
        return 0.0
    return edge.weight


def _add_edge_with_competition(
    graph: Graph,
    new_edge: Edge,
    config: InhibitionConfig,
) -> bool:
    """Add edge with outgoing competition cap."""
    outgoing = graph.outgoing(new_edge.source)
    max_outgoing = getattr(config, "max_outgoing", 20)

    if len(outgoing) >= max_outgoing:
        weakest = None
        weakest_weight = float("inf")
        for _, edge in outgoing:
            if graph.is_node_protected(edge.target):
                continue
            current_weight = abs(edge.weight)
            if current_weight < weakest_weight:
                weakest = edge
                weakest_weight = current_weight

        if weakest is not None and abs(new_edge.weight) > weakest_weight:
            graph._remove_edge(weakest.source, weakest.target)
        else:
            return False

    graph.add_edge(new_edge)
    return True
