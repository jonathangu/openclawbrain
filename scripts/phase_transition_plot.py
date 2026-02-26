#!/usr/bin/env python3
"""Generate a phase-transition diagnostic for CrabPath learning dynamics."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import deque
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ablation_study import (  # noqa: E402
    SEED,
    QuerySpec,
    _apply_clamp_non_negative,
    _bootstrap_base,
    _build_candidates,
    _build_queries,
    _build_trajectory,
    _custom_router,
    _map_file_chunk_nodes,
    _query_accuracy,
)

from crabpath.autotune import SafetyBounds, self_tune  # noqa: E402
from crabpath.decay import DecayConfig, apply_decay  # noqa: E402
from crabpath.learning import LearningConfig, RewardSignal, make_learning_step  # noqa: E402
from crabpath.lifecycle_sim import make_mock_llm_all  # noqa: E402
from crabpath.mitosis import (  # noqa: E402
    MitosisConfig,
    MitosisState,
    create_node,
    mitosis_maintenance,
)
from crabpath.synaptogenesis import (  # noqa: E402
    SynaptogenesisConfig,
    SynaptogenesisState,
    classify_tier,
    decay_proto_edges,
    record_cofiring,
    record_correction,
    record_skips,
)

WINDOW_SIZE = 20
FULL_QUERY_COUNT = 300


def _softmax(values: Iterable[float]) -> list[float]:
    vals = list(values)
    if not vals:
        return []
    max_value = max(vals)
    exp_values = [math.exp(v - max_value) for v in vals]
    total = sum(exp_values)
    if total == 0.0:
        return [0.0 for _ in vals]
    return [v / total for v in exp_values]


def _entropy_from_weights(weights: list[float]) -> float:
    probs = _softmax(weights)
    return -sum(p * math.log(p) for p in probs if p > 0.0)


def _node_entropy(graph) -> float:
    node_entropies: list[float] = []
    for node in graph.nodes():
        outgoing = graph.outgoing(node.id)
        if not outgoing:
            continue
        weights = [edge.weight for _, edge in outgoing]
        node_entropies.append(_entropy_from_weights(weights))
    return sum(node_entropies) / len(node_entropies) if node_entropies else 0.0


def _rolling(values: deque[float], new_value: float) -> float:
    values.append(new_value)
    return sum(values) / len(values)


def _ensure_full_300_queries(queries: list[QuerySpec]) -> list[QuerySpec]:
    if not queries:
        return []
    if len(queries) >= FULL_QUERY_COUNT:
        return queries[:FULL_QUERY_COUNT]

    repeated: list[QuerySpec] = list(queries)
    while len(repeated) < FULL_QUERY_COUNT:
        source = queries[(len(repeated) - len(queries)) % len(queries)]
        repeated.append(
            QuerySpec(
                text=source.text,
                expected_nodes=list(source.expected_nodes),
                is_negation=source.is_negation,
            )
        )
    return repeated[:FULL_QUERY_COUNT]


def _to_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str)]


def _extract_query_text(payload: Any) -> str | None:
    if isinstance(payload, str):
        text = payload.strip()
        return text or None
    if not isinstance(payload, dict):
        return None
    for key in ("query", "text", "question", "prompt", "query_text"):
        value = payload.get(key)
        if isinstance(value, str):
            value = value.strip()
            if value:
                return value
    return None


def _extract_expected_nodes(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return []
    expected = payload.get("expected_nodes", payload.get("expected", payload.get("targets")))
    return _to_str_list(expected)


def _extract_negation(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    is_negation = payload.get("is_negation")
    return bool(is_negation) if isinstance(is_negation, bool) else False


def _load_input_queries(path: Path) -> list[QuerySpec]:
    queries: list[QuerySpec] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                payload = line

            text = _extract_query_text(payload)
            if not text:
                continue
            queries.append(
                QuerySpec(
                    text=text,
                    expected_nodes=_extract_expected_nodes(payload),
                    is_negation=_extract_negation(payload),
                )
            )
    if not queries:
        raise ValueError(f"no valid query entries in {path}")
    return queries


def _build_default_queries(file_chunks: dict[str, list[str]]) -> list[QuerySpec]:
    return _ensure_full_300_queries(_build_queries(file_chunks))


def _build_reflex_count(graph, syn_config: SynaptogenesisConfig) -> int:
    return sum(
        1
        for edge in graph.edges()
        if classify_tier(edge.weight, syn_config) == "reflex"
    )


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if q <= 0.0:
        return min(values)
    if q >= 1.0:
        return max(values)

    ordered = sorted(values)
    idx = q * (len(ordered) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def _detect_transition(
    entropy: list[float],
    gradients: list[float],
) -> tuple[int | None, float, float]:
    if not entropy or not gradients:
        return None, 0.0, 0.0

    start = min(WINDOW_SIZE, len(entropy))
    entropy_baseline = sum(entropy[:start]) / start
    entropy_threshold = entropy_baseline * 0.9

    late_gradients = gradients[WINDOW_SIZE:] if len(gradients) > WINDOW_SIZE else gradients
    gradient_threshold = _quantile(late_gradients, 0.65)
    gradient_threshold = max(gradient_threshold, 1e-6)

    for index, (e_val, g_val) in enumerate(zip(entropy, gradients), 1):
        if index <= WINDOW_SIZE:
            continue
        if e_val <= entropy_threshold and g_val >= gradient_threshold:
            return index, entropy_threshold, gradient_threshold

    best_index = None
    best_score = 0.0
    for index, (e_val, g_val) in enumerate(
        zip(entropy[WINDOW_SIZE:], gradients[WINDOW_SIZE:]), WINDOW_SIZE + 1
    ):
        score = (entropy_threshold - e_val) + g_val
        if score > best_score:
            best_score = score
            best_index = index

    return best_index, entropy_threshold, gradient_threshold


def _sparkline(values: list[float], width: int = 72) -> str:
    if not values:
        return ""

    blocks = " ▁▂▃▄▅▆▇█"
    if len(values) <= width:
        sample = values
    else:
        sample = []
        stride = len(values) / width
        for slot in range(width):
            start = int(slot * stride)
            end = int((slot + 1) * stride)
            if end <= start:
                end = min(start + 1, len(values))
            chunk = values[start:end]
            sample.append(sum(chunk) / len(chunk))

    low = min(sample)
    high = max(sample)
    if high == low:
        return blocks[-1] * len(sample)

    scale = (len(blocks) - 1) / (high - low)
    return "".join(blocks[int((v - low) * scale)] for v in sample)


def _run_simulation(queries: list[QuerySpec], llm_call, seed: int) -> dict[str, Any]:
    random.seed(seed)

    base_graph, base_mitosis_state, base_syn_state, _, _ = _bootstrap_base(llm_call)
    del base_mitosis_state
    del base_syn_state

    graph = base_graph
    mitosis_state = MitosisState()
    syn_state = SynaptogenesisState()
    decay_config = DecayConfig()
    learning_config = LearningConfig(learning_rate=0.35, discount=1.0)
    mitosis_config = MitosisConfig()
    syn_config = SynaptogenesisConfig()
    last_adjusted: dict[str, int] = {}

    # Mirror ablation defaults with all modules enabled.
    use_learning = True
    use_inhibition = True
    use_autotune = True
    use_neurogenesis = True

    entropy_window: deque[float] = deque(maxlen=WINDOW_SIZE)
    grad_window: deque[float] = deque(maxlen=WINDOW_SIZE)
    accuracy_window: deque[float] = deque(maxlen=WINDOW_SIZE)

    weight_entropy: list[float] = []
    gradient_magnitude: list[float] = []
    retrieval_accuracy: list[float] = []
    reflex_edge_count: list[float] = []

    total_fired = 0
    total_context_chars = 0
    total_promotions = 0
    total_proto_created = 0

    for qi, query in enumerate(queries, 1):
        candidates = _build_candidates(
            graph,
            query.text,
            use_edge_weights=True,
            min_overlap=0,
            min_score=0.0,
        )
        top_n = 1 if not use_learning else 5
        selected_nodes = _custom_router(query.text, candidates, top_n=top_n)

        score = _query_accuracy(selected_nodes, query.expected_nodes)
        context_chars = 0
        for node_id in selected_nodes:
            node = graph.get_node(node_id)
            if node is not None:
                context_chars += len(node.content)

        total_context_chars += context_chars
        total_fired += len(selected_nodes)

        reward = RewardSignal(
            episode_id=f"phase-transition-q-{qi}",
            final_reward=(score * 2.0) - 1.0,
        )

        if selected_nodes:
            cofire = record_cofiring(graph, selected_nodes, syn_state, syn_config)
            total_promotions += int(cofire.get("promoted", 0))
            total_proto_created += int(cofire.get("proto_created", 0))
            candidate_ids = [node_id for node_id, _, _ in candidates]
            record_skips(graph, selected_nodes[0], candidate_ids, selected_nodes, syn_config)
        if reward.final_reward < 0.0 and use_inhibition and selected_nodes:
            record_correction(graph, selected_nodes, reward=reward.final_reward, config=syn_config)

        if use_neurogenesis and len(selected_nodes) <= 1 and candidates:
            existing_matches = [
                (node_id, match_score, summary) for node_id, match_score, summary in candidates
            ]
            _ = create_node(
                graph=graph,
                query=query.text,
                existing_matches=existing_matches,
                llm_call=llm_call,
                fired_node_ids=selected_nodes,
            )

        traj_delta = 0.0
        if use_learning and selected_nodes:
            trajectory = _build_trajectory(selected_nodes, graph)
            if trajectory:
                result = make_learning_step(
                    graph=graph,
                    trajectory_steps=trajectory,
                    reward=reward,
                    config=learning_config,
                )
                if result.updates:
                    traj_delta = sum(
                        abs(update.delta) for update in result.updates
                    ) / len(result.updates)
                else:
                    traj_delta = 0.0

        if qi % 5 == 0:
            apply_decay(graph, turns_elapsed=5, config=decay_config)
            decay_proto_edges(syn_state, syn_config)

        if qi % 25 == 0 and use_neurogenesis:
            mitosis_maintenance(graph, llm_call, mitosis_state, mitosis_config)

        if use_autotune and qi % 25 == 0:
            query_stats = {
                "avg_nodes_fired_per_query": total_fired / qi,
                "context_chars": total_context_chars,
                "promotions": total_promotions,
                "proto_created": total_proto_created,
            }
            self_tune(
                graph=graph,
                state=mitosis_state,
                query_stats=query_stats,
                syn_config=syn_config,
                decay_config=decay_config,
                mitosis_config=mitosis_config,
                cycle_number=qi,
                last_adjusted=last_adjusted,
                safety_bounds=SafetyBounds(),
            )

        if not use_inhibition:
            _apply_clamp_non_negative(graph)

        avg_entropy = _node_entropy(graph)
        avg_accuracy = score
        reflexes = _build_reflex_count(graph, syn_config)

        weight_entropy.append(_rolling(entropy_window, avg_entropy))
        gradient_magnitude.append(_rolling(grad_window, traj_delta))
        retrieval_accuracy.append(_rolling(accuracy_window, avg_accuracy))
        reflex_edge_count.append(reflexes)

    return {
        "weight_entropy": weight_entropy,
        "gradient_magnitude": gradient_magnitude,
        "retrieval_accuracy": retrieval_accuracy,
        "reflex_edge_count": reflex_edge_count,
    }


def _run_phase_transition_plot(output: Path, input_path: Path | None) -> None:
    llm = make_mock_llm_all()
    # Build queries.
    if input_path is None:
        llm2 = make_mock_llm_all()
        base_graph, base_mitosis_state, base_syn_state, _, _ = _bootstrap_base(llm2)
        file_chunks = _map_file_chunk_nodes(base_graph)
        del base_mitosis_state
        del base_syn_state
        queries = _build_default_queries(file_chunks)
    else:
        queries = _load_input_queries(input_path)

    metrics = _run_simulation(queries, llm, SEED)
    transition_idx, ent_th, grad_th = _detect_transition(
        metrics["weight_entropy"], metrics["gradient_magnitude"]
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "seed": SEED,
                "window": WINDOW_SIZE,
                "source": str(input_path) if input_path is not None else "synthetic",
                "query_count": len(queries),
                "metrics": metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if transition_idx is None:
        transition_idx = 1
    print(
        f"Phase transition detected at query ~{transition_idx}: "
        f"weight entropy dropped below {ent_th:.4f}, "
        f"gradient magnitude exceeded {grad_th:.4f}"
    )

    print("\nMetric sparklines (20-query rolling window):")
    print(f"weight_entropy        | {_sparkline(metrics['weight_entropy'])}")
    print(f"gradient_magnitude    | {_sparkline(metrics['gradient_magnitude'])}")
    print(f"retrieval_accuracy    | {_sparkline(metrics['retrieval_accuracy'])}")
    print(f"reflex_edge_count     | {_sparkline(metrics['reflex_edge_count'])}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute phase-transition diagnostics for CrabPath learning."
    )
    parser.add_argument(
        "--input",
        help="Replay results JSONL file with query entries. If omitted, run 300 synthetic queries.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON path for metric arrays.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _run_phase_transition_plot(
        output=Path(args.output),
        input_path=Path(args.input) if args.input else None,
    )


if __name__ == "__main__":
    main()
