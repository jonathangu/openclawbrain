"""Sim: repeated mistakes should be suppressed after negative outcomes."""

from __future__ import annotations

import json
from pathlib import Path

from openclawbrain import Edge, Graph, LearningConfig, Node, TraversalConfig, apply_outcome, traverse


RESULT_PATH = Path(__file__).with_name("mistake_recurrence_results.json")


def _build_graph() -> Graph:
    graph = Graph()

    source = "hotfix_query"
    wrong_target = "skip_tests_path"
    correct_target = "run_tests_path"

    graph.add_node(Node(source, "hotfix query"))
    graph.add_node(Node(wrong_target, "skip tests for hotfix"))
    graph.add_node(Node(correct_target, "run tests before hotfix"))

    graph.add_edge(Edge(source, wrong_target, 0.8))
    graph.add_edge(Edge(source, correct_target, 0.3))
    return graph


def _edge_weight(graph: Graph, source: str, target: str) -> float:
    for _, edge in graph.outgoing(source):
        if edge.target == target:
            return edge.weight
    return 0.0


def _window_rate(values: list[bool]) -> float:
    if not values:
        return 0.0
    return sum(1.0 for value in values if value) / len(values)


def _run() -> dict:
    graph = _build_graph()
    config = TraversalConfig(max_hops=1, beam_width=1)
    learning_config = LearningConfig(learning_rate=0.15)

    history: list[dict] = []
    wrong_fired_flags: list[bool] = []

    for query_index in range(1, 26):
        result = traverse(graph, [("hotfix_query", 1.0)], config=config)
        wrong_fired = "skip_tests_path" in result.fired
        outcome = -1.0 if wrong_fired else 1.0
        apply_outcome(graph, result.fired, outcome=outcome, config=learning_config)

        wrong_fired_flags.append(wrong_fired)
        history.append(
            {
                "query": query_index,
                "wrong_fired": wrong_fired,
                "wrong_weight": _edge_weight(graph, "hotfix_query", "skip_tests_path"),
                "correct_weight": _edge_weight(graph, "hotfix_query", "run_tests_path"),
            }
        )

    early_window = wrong_fired_flags[:5]
    late_window = wrong_fired_flags[-5:]
    early_rate = _window_rate(early_window)
    late_rate = _window_rate(late_window)

    return {
        "simulation": "mistake_recurrence",
        "query_count": len(history),
        "wrong_path": "skip_tests_path",
        "correct_path": "run_tests_path",
        "window": 5,
        "wrong_fire_history": history,
        "wrong_fire_rate": {
            "early": early_rate,
            "late": late_rate,
        },
        "final_weights": {
            "wrong": history[-1]["wrong_weight"],
            "correct": history[-1]["correct_weight"],
        },
        "claim": {
            "wrong_path_fire_rate_drops": {
                "early_rate": early_rate,
                "late_rate": late_rate,
                "met": early_rate >= 0.6 and late_rate <= 0.2,
            }
        },
    }


def main() -> None:
    RESULT_PATH.write_text(json.dumps(_run(), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
