from __future__ import annotations

from pathlib import Path

from crabpath.split import split_workspace


def test_split_workspace_creates_chunks(tmp_path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "a.md").write_text(
        "# Title\n\nIntro paragraph.\n\n## Section One\nBody one.\n\n## Section Two\nBody two.",
        encoding="utf-8",
    )
    (workspace / "b.md").write_text("single paragraph", encoding="utf-8")

    graph, texts = split_workspace(workspace)

    assert graph.node_count() == 4
    assert len(texts) == 4
    assert any("a.md::0" in node.id for node in graph.nodes())
    # sibling edges should exist between sections in same file
    assert graph.edge_count() >= 2
