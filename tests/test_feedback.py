from __future__ import annotations

from crabpath import Edge, Graph, Node, SynaptogenesisConfig, SynaptogenesisState
from crabpath.feedback import auto_feedback, detect_correction, score_retrieval


def test_detect_correction_strong_and_mild_patterns() -> None:
    assert detect_correction("No, that's wrong", "answer") == -1.0
    assert detect_correction("actually, should be B", "answer") == -1.0
    assert detect_correction("don't do that", "answer") == -0.5


def test_score_retrieval_parses_llm_json() -> None:
    nodes = [
        ("node-a", "Alpha node content"),
        ("node-b", "Beta node content"),
    ]

    def llm(prompt: str, system: str) -> str:
        return '[{"node_id": "node-a", "score": 1.0}, {"node_id": "node-b", "score": -0.5}]'

    assert score_retrieval(
        "What is alpha?",
        nodes,
        "Here is the answer.",
        llm,
    ) == [("node-a", 1.0), ("node-b", -0.5)]


def test_score_retrieval_default_on_invalid_response() -> None:
    def llm(prompt: str, system: str) -> str:
        return "not json"

    assert score_retrieval(
        "query",
        [("x", "x"), ("y", "y")],
        "answer",
        llm,
    ) == [("x", 0.0), ("y", 0.0)]


def test_auto_feedback_applies_negative_correction() -> None:
    graph = Graph()
    graph.add_node(Node(id="a", content="a"))
    graph.add_node(Node(id="b", content="b"))
    graph.add_edge(Edge(source="a", target="b", weight=1.0))

    result = auto_feedback(
        query="query",
        user_followup="No, that's not right",
        trajectory=["a", "b"],
        graph=graph,
        syn_state=SynaptogenesisState(),
        config=SynaptogenesisConfig(correction_decay=0.5),
    )

    assert result["action"] == "record_correction"
    assert graph.get_edge("a", "b").weight == 0.5


def test_auto_feedback_applies_implicit_positive() -> None:
    graph = Graph()
    graph.add_node(Node(id="a", content="a"))
    graph.add_node(Node(id="b", content="b"))
    graph.add_edge(Edge(source="a", target="b", weight=1.0))

    result = auto_feedback(
        query="query",
        user_followup="looks good to me",
        trajectory=["a", "b"],
        graph=graph,
        syn_state=SynaptogenesisState(),
    )

    assert result["action"] == "record_cofiring"
    assert graph.get_edge("a", "b").weight > 1.0
