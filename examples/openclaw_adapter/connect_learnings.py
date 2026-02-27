#!/usr/bin/env python3
"""Connect learning nodes to workspace nodes via embedding similarity.

Adds bidirectional "cross-file" links from each learning node to similar workspace
nodes so corrections become reachable during traversal.
"""
from __future__ import annotations
import argparse

import time
from pathlib import Path

from crabpath.store import load_state, save_state
from crabpath.graph import Edge
from crabpath.autotune import measure_health

AGENT_STATES = {
    "main": Path.home() / ".crabpath" / "main" / "state.json",
    "pelican": Path.home() / ".crabpath" / "pelican" / "state.json",
    "bountiful": Path.home() / ".crabpath" / "bountiful" / "state.json",
}


def cosine_sim(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def connect_learnings(state_path: str, top_k: int = 3, min_sim: float = 0.3):
    graph, index, meta = load_state(state_path)
    
    # Find learning nodes
    learning_nodes = [n for n in graph.nodes() if n.id.startswith("learning::")]
    workspace_nodes = [n for n in graph.nodes() if not n.id.startswith("learning::")]
    
    if not learning_nodes:
        print("  No learning nodes found")
        return 0
    
    # Get vectors
    learning_vecs = {n.id: index._vectors.get(n.id) for n in learning_nodes if n.id in index._vectors}
    workspace_vecs = {n.id: index._vectors.get(n.id) for n in workspace_nodes if n.id in index._vectors}
    
    edges_added = 0
    for lid, lvec in learning_vecs.items():
        if lvec is None:
            continue
        
        # Find top-k most similar workspace nodes
        sims = []
        for wid, wvec in workspace_vecs.items():
            if wvec is None:
                continue
            sim = cosine_sim(lvec, wvec)
            sims.append((wid, sim))
        
        sims.sort(key=lambda x: x[1], reverse=True)
        
        for wid, sim in sims[:top_k]:
            if sim < min_sim:
                continue
            
            # Bidirectional edges with similarity as weight (clamped to habitual range)
            weight = min(0.7, max(0.35, sim))
            
            # Learning -> Workspace (when traversing from correction, reach relevant workspace)
            if not graph._edges.get(lid, {}).get(wid):
                graph.add_edge(Edge(source=lid, target=wid, weight=weight, kind="cross_file"))
                edges_added += 1
            
            # Workspace -> Learning (when traversing from workspace, reach relevant corrections)
            if not graph._edges.get(wid, {}).get(lid):
                graph.add_edge(Edge(source=wid, target=lid, weight=weight, kind="cross_file"))
                edges_added += 1
    
    # Save
    save_state(graph, index, state_path,
               embedder_name=meta.get("embedder_name", "openai-text-embedding-3-small"),
               embedder_dim=meta.get("embedder_dim", 1536),
               meta=meta)
    
    h = measure_health(graph)
    return edges_added


def main():
    parser = argparse.ArgumentParser(
        description="Connect learning nodes into workspace neighborhoods for one or more adapter states"
    )
    parser.add_argument("--agent", choices=sorted(AGENT_STATES.keys()), help="Connect learning nodes for one agent state")
    parser.add_argument("--state", help="Connect learning nodes for this explicit state.json path")
    args = parser.parse_args()

    if args.agent and args.state:
        raise SystemExit("--agent and --state are mutually exclusive")

    if args.state:
        state_paths = [Path(args.state)]
    elif args.agent:
        state_paths = [AGENT_STATES[args.agent]]
    else:
        state_paths = list(AGENT_STATES.values())

    for state_path in state_paths:
        label = str(state_path)
        if state_path.is_absolute():
            label = state_path.name
            for agent, path in AGENT_STATES.items():
                if path == state_path:
                    label = agent.upper()
                    break
        print(f"\n=== {label} ===")

        graph, index, meta = load_state(str(state_path))
        learning_count = sum(1 for n in graph.nodes() if n.id.startswith("learning::"))
        orphans_before = sum(1 for n in graph.nodes() 
                           if not any(n.id in edges for edges in graph._edges.values())
                           and n.id not in graph._edges)
        print(f"  Before: {graph.node_count()} nodes, {graph.edge_count()} edges, {learning_count} learning nodes, {orphans_before} orphans")
        
        t0 = time.time()
        edges_added = connect_learnings(str(state_path))
        elapsed = time.time() - t0
        
        # Reload and check
        graph2, _, _ = load_state(str(state_path))
        orphans_after = sum(1 for n in graph2.nodes() 
                          if not any(n.id in edges for edges in graph2._edges.values())
                          and n.id not in graph2._edges)
        h = measure_health(graph2)
        print(f"  After: {graph2.node_count()} nodes, {graph2.edge_count()} edges, {orphans_after} orphans")
        print(f"  Added: {edges_added} edges in {elapsed:.1f}s")
        print(f"  Health: dormant={h.dormant_pct:.0%} habitual={h.habitual_pct:.0%} reflex={h.reflex_pct:.0%} cross-file={h.cross_file_edge_pct:.0%}")


if __name__ == "__main__":
    main()
