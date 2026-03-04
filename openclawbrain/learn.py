"""Policy updates for graph edges and Hebbian co-firing."""

from __future__ import annotations

import math
import hashlib
from collections import defaultdict
from dataclasses import dataclass
from collections.abc import Callable

from .graph import Edge, Graph, Node
from ._util import _extract_json, _first_line
from .decay import apply_decay


def _unique_node_id(content: str, graph: Graph) -> str:
    """ unique node id."""
    base = f"auto:{hashlib.sha1(content.encode('utf-8')).hexdigest()[:10]}"
    candidate = base
    suffix = 0
    while graph.get_node(candidate) is not None:
        suffix += 1
        candidate = f"{base}:{suffix}"
    return candidate


def maybe_create_node(
    graph: Graph,
    query_text: str,
    fired_nodes: list[str],
    llm_fn: Callable[[str, str], str] | None = None,
) -> str | None:
    """If no good nodes were found for this query, ask LLM to create a new node."""
    if llm_fn is None:
        return None

    system = (
        "A query found no relevant documents. Should a new memory node be created from this "
        'query? Return JSON: {"should_create": true/false, "content": "...", "reason": "..."}'
    )
    nearby = ", ".join(
        f"{node_id}: {node.content[:80]}"
        for node_id in fired_nodes
        if (node := graph.get_node(node_id)) is not None
    )
    if not nearby:
        nearby = "<no recent fired nodes>"
    user = f"Query: {query_text}\n\nRecent context: {nearby}"

    try:
        payload = _extract_json(llm_fn(system, user)) or {}
        if not bool(payload.get("should_create", False)):
            return None

        content = str(payload.get("content", query_text)).strip() or query_text
        reason = str(payload.get("reason", "")).strip()
        summary = str(payload.get("summary", _first_line(content))).strip() or _first_line(content)
        node_id = _unique_node_id(content, graph)
        graph.add_node(
            Node(
                id=node_id,
                content=content,
                summary=_first_line(summary),
                metadata={"source": "llm-create", "query": query_text, "reason": reason},
            )
        )
        for source_id in fired_nodes[:5]:
            if source_id == node_id or graph.get_node(source_id) is None:
                continue
            graph.add_edge(Edge(source=source_id, target=node_id, weight=0.15))
            graph.add_edge(Edge(source=node_id, target=source_id, weight=0.15))
        return node_id
    except (Exception, SystemExit):
        return None


@dataclass
class LearningConfig:
    """Hyperparameters for policy-like updates.

    learning_rate controls outcome update magnitude. Default is ``0.1``. For
    small graphs (<100 nodes) use ``0.05``-``0.1``. For large graphs (>1000
    nodes) use ``0.01``-``0.05``. Higher values are faster but more volatile.

    promotion_threshold is the minimum co-firing count before Hebbian creates an
    edge. Default is ``2``. Lower values create more edges, higher values are
    more selective.
    """

    learning_rate: float = 0.1
    discount: float = 0.95
    hebbian_increment: float = 0.06
    promotion_threshold: int = 2
    weight_bounds: tuple[float, float] = (-1.0, 1.0)
    temperature: float = 1.0
    baseline: float = 0.0


_apply_outcome_call_count = 0


def _clip_weight(weight: float, bounds: tuple[float, float]) -> float:
    """ clip weight."""
    return max(bounds[0], min(bounds[1], weight))


def _node_stop_values(node: Node | None) -> tuple[float, float]:
    """Extract stop relevance/weight from node metadata with safe fallbacks."""
    if node is None or not isinstance(node.metadata, dict):
        return 0.0, 0.0
    raw_relevance = node.metadata.get("stop_relevance", 0.0)
    raw_weight = node.metadata.get("stop_weight", 0.0)
    stop_relevance = float(raw_relevance) if isinstance(raw_relevance, (int, float)) else 0.0
    stop_weight = float(raw_weight) if isinstance(raw_weight, (int, float)) else 0.0
    return stop_relevance, stop_weight


def _project_conserving_updates(
    weights: list[float],
    deltas: list[float],
    bounds: tuple[float, float],
    *,
    tol: float = 1e-12,
) -> list[float]:
    """Project proposed updates onto box constraints while conserving total weight."""
    if len(weights) != len(deltas):
        raise ValueError("weights and deltas must be the same length")
    if not weights:
        return []

    lo, hi = bounds
    target_sum = sum(weights)
    proposed = [weight + delta for weight, delta in zip(weights, deltas)]
    new_weights = [min(hi, max(lo, value)) for value in proposed]
    diff = sum(new_weights) - target_sum
    if abs(diff) <= tol:
        return new_weights

    max_iters = max(1, len(weights) * 5)
    for _ in range(max_iters):
        if abs(diff) <= tol:
            break
        if diff > 0:
            free = [idx for idx, value in enumerate(new_weights) if value > lo + tol]
            if not free:
                break
            share = diff / len(free)
            for idx in free:
                new_weights[idx] = max(lo, new_weights[idx] - share)
        else:
            free = [idx for idx, value in enumerate(new_weights) if value < hi - tol]
            if not free:
                break
            share = (-diff) / len(free)
            for idx in free:
                new_weights[idx] = min(hi, new_weights[idx] + share)
        diff = sum(new_weights) - target_sum

    return new_weights


def hebbian_update(
    graph: Graph,
    fired_nodes: list[str],
    config: LearningConfig | None = None,
    max_hebbian_pairs: int = 20,
) -> None:
    """Apply co-firing updates between observed node pairs.

    By default, every observed pair receives ``hebbian_increment``. When
    ``max_hebbian_pairs`` is set and the fire trajectory is long, only the
    closest firing-order pairs are updated, capped to ``max_hebbian_pairs`` total
    updates.
    """
    cfg = config or LearningConfig()
    if len(fired_nodes) < 2:
        return

    if max_hebbian_pairs <= 0:
        return

    max_pairs = max_hebbian_pairs
    if len(fired_nodes) <= math.sqrt(2 * max_pairs):
        pair_indexes = [(i, j) for i in range(len(fired_nodes)) for j in range(i + 1, len(fired_nodes))]
    else:
        candidates: list[tuple[int, int, int]] = []
        for source_idx in range(len(fired_nodes)):
            for target_idx in range(source_idx + 1, len(fired_nodes)):
                distance = target_idx - source_idx
                candidates.append((distance, source_idx, target_idx))

        candidates.sort(key=lambda item: (item[0], item[1]))
        pair_indexes = [(source_idx, target_idx) for _, source_idx, target_idx in candidates[:max_pairs]]

    for source_idx, target_idx in pair_indexes:
        co_firing_count = target_idx - source_idx + 1
        if co_firing_count < cfg.promotion_threshold:
            continue

        source_id = fired_nodes[source_idx]
        target_id = fired_nodes[target_idx]
        edge = graph._edges.get(source_id, {}).get(target_id)
        if edge is None:
            graph.add_edge(
                Edge(
                    source=source_id,
                    target=target_id,
                    weight=cfg.hebbian_increment,
                    kind="sibling",
                )
            )
        else:
            edge.weight = _clip_weight(
                edge.weight + cfg.hebbian_increment,
                cfg.weight_bounds,
            )
            graph._edges[source_id][target_id] = edge


def _softmax_with_stop(graph: Graph, node_id: str, temperature: float) -> tuple[dict[str, float], float]:
    """Compute numerically stable softmax over outgoing actions plus STOP."""
    outgoing = graph.outgoing(node_id)
    logits_by_target: dict[str, float] = {}
    max_logit = 0.0
    for _target_node, edge in outgoing:
        relevance = float(edge.metadata.get("relevance", 0.0)) if isinstance(edge.metadata, dict) else 0.0
        logit = (relevance + edge.weight) / temperature
        logits_by_target[edge.target] = logit
        if logit > max_logit:
            max_logit = logit

    stop_relevance, stop_weight = _node_stop_values(graph.get_node(node_id))
    stop_logit = (stop_relevance + stop_weight) / temperature
    exp_stop = math.exp(stop_logit - max_logit)
    exp_sum = exp_stop
    probs_by_target: dict[str, float] = {"__STOP__": exp_stop}
    for target_id, logit in logits_by_target.items():
        exp_logit = math.exp(logit - max_logit)
        probs_by_target[target_id] = exp_logit
        exp_sum += exp_logit

    return {action: prob / exp_sum for action, prob in probs_by_target.items()}, max_logit


def apply_outcome_pg(
    graph: Graph,
    fired_nodes: list[str],
    outcome: float,
    config: LearningConfig | None = None,
    auto_decay: bool = False,
    decay_interval: int = 10,
    per_node_outcomes: dict[str, float] | None = None,
    baseline: float = 0.0,
    temperature: float = 1.0,
) -> dict:
    """Apply REINFORCE policy-gradient updates over the full fired trajectory."""
    global _apply_outcome_call_count
    cfg = config or LearningConfig()
    updates: dict[str, float] = defaultdict(float)

    effective_temperature = temperature if temperature > 0 else cfg.temperature

    if len(fired_nodes) < 2:
        hebbian_update(graph, fired_nodes, cfg)
        _apply_outcome_call_count += 1
        if auto_decay:
            if decay_interval <= 0:
                decay_interval = 1
            if _apply_outcome_call_count % decay_interval == 0:
                apply_decay(graph)
        return dict(updates)

    for idx in range(len(fired_nodes) - 1):
        source_id = fired_nodes[idx]
        target_id = fired_nodes[idx + 1]

        outgoing = graph.outgoing(source_id)
        if not outgoing:
            continue

        node_outcome = outcome
        if per_node_outcomes is not None and source_id in per_node_outcomes:
            node_outcome = per_node_outcomes[source_id]

        probs, _ = _softmax_with_stop(graph, source_id, effective_temperature)
        scalar = cfg.learning_rate * (node_outcome - baseline) * (cfg.discount ** idx) / effective_temperature

        node = graph.get_node(source_id)
        _stop_relevance, stop_weight = _node_stop_values(node)

        weights = [edge.weight for _target_node, edge in outgoing]
        deltas: list[float] = []
        for _target_node, edge in outgoing:
            pi_j = probs.get(edge.target, 0.0)
            if edge.target == target_id:
                delta = scalar * (1.0 - pi_j)
            else:
                delta = -scalar * pi_j
            deltas.append(delta)

        stop_delta = -scalar * probs.get("__STOP__", 0.0)
        weights.append(stop_weight)
        deltas.append(stop_delta)

        new_weights = _project_conserving_updates(weights, deltas, cfg.weight_bounds)

        for (edge_target_node, edge), new_weight in zip(outgoing, new_weights[:-1]):
            delta_actual = new_weight - edge.weight
            edge.weight = new_weight
            if edge.weight < -0.01:
                edge.kind = "inhibitory"
            else:
                edge.kind = "sibling"
            graph._edges[source_id][edge.target] = edge
            updates[f"{source_id}->{edge.target}"] += delta_actual

        if node is not None:
            new_stop_weight = new_weights[-1]
            delta_actual = new_stop_weight - stop_weight
            if not isinstance(node.metadata, dict):
                node.metadata = {}
            node.metadata["stop_weight"] = new_stop_weight
            updates[f"{source_id}->__STOP__"] += delta_actual

    hebbian_update(graph, fired_nodes, cfg)
    _apply_outcome_call_count += 1
    if auto_decay:
        if decay_interval <= 0:
            decay_interval = 1
        if _apply_outcome_call_count % decay_interval == 0:
            apply_decay(graph)

    return dict(updates)


def apply_outcome(
    graph: Graph,
    fired_nodes: list[str],
    outcome: float,
    config: LearningConfig | None = None,
    auto_decay: bool = False,
    decay_interval: int = 10,
    per_node_outcomes: dict[str, float] | None = None,
) -> dict:
    """Apply outcome-based policy updates over the full fired trajectory.

    Positive outcome strengthens traversed edges; negative outcome weakens them.
    Negative outcomes may create inhibitory edges when missing.
    Returns a mapping of ``"source->target"`` to weight delta.
    ``per_node_outcomes`` can supply a different reward/penalty for each source
    node in ``fired_nodes``; when omitted, ``outcome`` is used for all edges.

    When ``auto_decay`` is true, the graph is decayed after every
    ``decay_interval`` successful calls to this function.
    """
    global _apply_outcome_call_count
    cfg = config or LearningConfig()
    updates: dict[str, float] = defaultdict(float)

    if len(fired_nodes) < 2:
        hebbian_update(graph, fired_nodes, cfg)
        _apply_outcome_call_count += 1
        if auto_decay:
            if decay_interval <= 0:
                decay_interval = 1
            if _apply_outcome_call_count % decay_interval == 0:
                apply_decay(graph)
        return dict(updates)

    for idx in range(len(fired_nodes) - 1):
        source_id = fired_nodes[idx]
        target_id = fired_nodes[idx + 1]
        edge = graph._edges.get(source_id, {}).get(target_id)
        node_outcome = outcome
        if per_node_outcomes is not None and source_id in per_node_outcomes:
            node_outcome = per_node_outcomes[source_id]

        delta = cfg.learning_rate * (cfg.discount ** (idx + 1)) * node_outcome

        if edge is None:
            graph.add_edge(
                Edge(
                    source=source_id,
                    target=target_id,
                    weight=_clip_weight(delta, cfg.weight_bounds),
                    kind="inhibitory" if node_outcome < 0 else "sibling",
                )
            )
            updates[f"{source_id}->{target_id}"] = delta
            continue

        edge.weight = _clip_weight(edge.weight + delta, cfg.weight_bounds)
        if node_outcome < 0:
            edge.kind = "inhibitory"
        graph._edges[source_id][target_id] = edge
        updates[f"{source_id}->{target_id}"] = delta

    hebbian_update(graph, fired_nodes, cfg)

    _apply_outcome_call_count += 1
    if auto_decay:
        if decay_interval <= 0:
            decay_interval = 1
        if _apply_outcome_call_count % decay_interval == 0:
            apply_decay(graph)

    return dict(updates)
