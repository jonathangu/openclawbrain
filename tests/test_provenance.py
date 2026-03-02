from __future__ import annotations

from openclawbrain.graph import Graph, Node
from openclawbrain.provenance import build_tool_provenance


def test_provenance_redacts_sensitive_args_and_results() -> None:
    graph = Graph()
    graph.add_node(Node("n1", "alpha", metadata={"file": "a.md"}))

    tool_calls = [
        {
            "id": "call-1",
            "name": "web_search",
            "arguments": {"api_key": "sk-THISSHOULDNOTAPPEAR", "q": "cats"},
        }
    ]
    tool_results = [
        {
            "tool_call_id": "call-1",
            "tool_name": "web_search",
            "content": "Bearer SECRET_TOKEN_12345",
        }
    ]

    build_tool_provenance(
        graph=graph,
        fired_nodes=["n1"],
        tool_calls=tool_calls,
        tool_results=tool_results,
        session="session.jsonl",
    )

    tool_nodes = [node for node in graph.nodes() if node.id.startswith("tool_action::")]
    assert tool_nodes
    action = tool_nodes[0]
    assert "sk-THISSHOULDNOTAPPEAR" not in action.content
    assert "[REDACTED]" in action.content

    evidence_nodes = [node for node in graph.nodes() if node.id.startswith("tool_evidence::")]
    assert evidence_nodes
    evidence = evidence_nodes[0]
    assert "SECRET_TOKEN_12345" not in evidence.content
    assert "[REDACTED]" in evidence.content
