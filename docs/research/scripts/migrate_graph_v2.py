#!/usr/bin/env python3
"""Migrate a v0.6 graph JSON snapshot to the v2 graph schema."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crabpath.graph import Graph  # noqa: E402


def migrate_graph_v2(input_path: str, output_path: str | None = None) -> dict[str, Any]:
    source = Path(input_path)
    target = (
        Path(output_path)
        if output_path
        else source.with_name(f"{source.stem}_v2{source.suffix}")
    )

    with open(source) as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid graph JSON: expected object in {source}")

    legacy_nodes = data.get("nodes", [])
    legacy_edges = data.get("edges", [])
    if not isinstance(legacy_nodes, list):
        legacy_nodes = []
    if not isinstance(legacy_edges, list):
        legacy_edges = []

    legacy_summary_count = sum(
        1 for node in legacy_nodes if isinstance(node, dict) and "summary" in node
    )
    legacy_type_count = sum(
        1 for node in legacy_nodes if isinstance(node, dict) and "type" in node
    )
    legacy_created_by_count = sum(
        1 for edge in legacy_edges if isinstance(edge, dict) and "created_by" in edge
    )
    legacy_decay_count = sum(
        1 for edge in legacy_edges if isinstance(edge, dict) and "decay_rate" in edge
    )

    graph = Graph.load(str(source))
    out_data = graph.to_v2_dict()
    with open(target, "w") as f:
        json.dump(out_data, f, indent=2)

    return {
        "input_path": str(source),
        "output_path": str(target),
        "input_nodes": len(legacy_nodes),
        "input_edges": len(legacy_edges),
        "backfilled_node_summary": len(legacy_nodes) - legacy_summary_count,
        "backfilled_node_type": len(legacy_nodes) - legacy_type_count,
        "backfilled_created_by": len(legacy_edges) - legacy_created_by_count,
        "backfilled_decay_rate": len(legacy_edges) - legacy_decay_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate v0.6 graph JSON to v2 schema.")
    parser.add_argument("input_path", help="Path to source graph JSON.")
    parser.add_argument(
        "output_path",
        nargs="?",
        default=None,
        help="Optional destination path. Defaults to <input>_v2.json",
    )
    args = parser.parse_args()

    stats = migrate_graph_v2(args.input_path, args.output_path)

    print(f"Migrated {stats['input_nodes']} nodes and {stats['input_edges']} edges.")
    print(
        "Backfilled fields: "
        f"{stats['backfilled_node_summary']} node summaries, "
        f"{stats['backfilled_node_type']} node types, "
        f"{stats['backfilled_created_by']} edge created_by, "
        f"{stats['backfilled_decay_rate']} edge decay_rate."
    )
    print(f"Wrote v2 graph to: {stats['output_path']}")


if __name__ == "__main__":
    main()
