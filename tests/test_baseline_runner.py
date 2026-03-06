from __future__ import annotations

import socket
from pathlib import Path

from openclawbrain.graph import Edge, Graph, Node
from openclawbrain.hasher import HashEmbedder
from openclawbrain.index import VectorIndex
from openclawbrain.store import save_state

from openclawbrain.eval.runner import run_baseline_suite


def _write_state(tmp_path: Path) -> Path:
    graph = Graph()
    graph.add_node(Node("n1", "Alpha specs and build steps."))
    graph.add_node(Node("n2", "Beta workflow guide and release notes."))
    graph.add_node(Node("n3", "Gamma pointer to beta; follow the link."))
    graph.add_edge(Edge("n3", "n2", weight=0.7, kind="pointer"))
    graph.add_edge(Edge("n1", "n3", weight=0.4, kind="pointer"))

    index = VectorIndex()
    embedder = HashEmbedder()
    for node in graph.nodes():
        index.upsert(node.id, embedder.embed(node.content))

    state_path = tmp_path / "state.json"
    save_state(graph=graph, index=index, path=str(state_path))
    return state_path


def _write_queries(tmp_path: Path) -> Path:
    path = tmp_path / "queries.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"id":"q1","query":"alpha build steps","category":"ops"}',
                '{"id":"q2","query":"beta release notes","category":"pointer"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_baseline_suite_runs_offline(tmp_path: Path, monkeypatch) -> None:
    state_path = _write_state(tmp_path)
    queries_path = _write_queries(tmp_path)
    output_dir = tmp_path / "out"

    def _block_socket(*_args, **_kwargs):
        raise RuntimeError("network disabled")

    monkeypatch.setattr(socket, "socket", _block_socket)

    summary = run_baseline_suite(
        state_path=state_path,
        queries_path=queries_path,
        modes=["vector_topk", "vector_topk_rerank", "pointer_chase", "edge_sim_legacy"],
        embed_model="hash",
        route_model_path=None,
        top_k=2,
        route_top_k=2,
        max_fired_nodes=5,
        max_prompt_context_chars=2000,
        output_dir=output_dir,
        include_per_query=False,
    )

    assert output_dir.joinpath("summary.json").exists()
    assert output_dir.joinpath("summary.csv").exists()
    assert output_dir.joinpath("report.md").exists()
    mode_status = summary.get("mode_status", {})
    assert mode_status["vector_topk"]["status"] == "ok"
    assert mode_status["pointer_chase"]["status"] == "ok"
    assert mode_status["edge_sim_legacy"]["status"] == "ok"
    assert mode_status["vector_topk_rerank"]["status"] in {"ok", "skipped"}
