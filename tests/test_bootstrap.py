from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from crabpath.graph import Graph


def _write(file_path: Path, text: str) -> None:
    file_path.write_text(text.strip() + "\n", encoding="utf-8")


def _bootstrap_script() -> Path:
    return Path(__file__).resolve().parents[1] / "scripts" / "bootstrap_from_workspace.py"


def test_bootstrap_from_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    _write(
        workspace / "AGENTS.md",
        """
## AGENTS Core
Run: `gh run list --limit 1`.
Never deploy without review.
See Tooling Standards.

## Safety Rules
Do not skip required checks.
1. Read the runbook.
2. Follow the gate policy.
""",
    )

    _write(
        workspace / "TOOLS.md",
        """
## Tooling Standards
Use git, pytest, and ssh where required.
""",
    )

    output_graph = tmp_path / "bootstrapped.json"
    result = subprocess.run(
        [sys.executable, str(_bootstrap_script()), str(workspace), "--output", str(output_graph)],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0
    assert "cross_edges" in result.stdout
    assert output_graph.exists()

    graph = Graph.load(str(output_graph))
    assert graph.node_count == 3
    assert graph.edge_count >= 2

    types = {node.type for node in graph.nodes()}
    assert "tool_call" in types
    assert "guardrail" in types

    assert any(edge.weight == 0.6 for edge in graph.edges())
    assert any(edge.weight == 0.4 for edge in graph.edges())

    weights = {node.id: float(node.metadata.get("bootstrap_weight", 0.0)) for node in graph.nodes()}
    assert any(weight > 0.5 for weight in weights.values())
