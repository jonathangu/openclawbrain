"""Thin, stdlib-only CLI wrapper for CrabPath workflows."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from .graph import Edge, Graph, Node
from .index import VectorIndex
from .replay import extract_queries, extract_queries_from_dir, replay_queries
from .split import split_workspace
from .traverse import TraversalConfig, traverse
from .learn import apply_outcome
from .autotune import measure_health


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="crabpath")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="split workspace and output node text payload")
    init.add_argument("--workspace", required=True)
    init.add_argument("--output", required=True)
    init.add_argument("--sessions", required=False)
    init.add_argument("--json", action="store_true")

    query = sub.add_parser("query", help="seed from index and traverse graph")
    query.add_argument("text")
    query.add_argument("--graph", required=True)
    query.add_argument("--index", required=False)
    query.add_argument("--top", type=int, default=10)
    query.add_argument("--json", action="store_true")
    query.add_argument("--query-vector", nargs="+", required=False)
    query.add_argument("--query-vector-stdin", action="store_true")

    learn = sub.add_parser("learn", help="apply outcome update")
    learn.add_argument("--graph", required=True)
    learn.add_argument("--outcome", type=float, required=True)
    learn.add_argument("--fired-ids", required=True)
    learn.add_argument("--json", action="store_true")

    replay = sub.add_parser("replay", help="warm up graph from historical sessions")
    replay.add_argument("--graph", required=True)
    replay.add_argument("--sessions", nargs="+", required=True)
    replay.add_argument("--max-queries", type=int, default=None)
    replay.add_argument("--json", action="store_true")

    health = sub.add_parser("health", help="compute graph health")
    health.add_argument("--graph", required=True)
    health.add_argument("--json", action="store_true")

    return parser


def _load_payload(path: str) -> dict:
    payload_path = Path(path)
    if payload_path.is_dir():
        payload_path = payload_path / "graph.json"
    if not payload_path.exists():
        raise SystemExit(f"missing graph file: {path}")
    return json.loads(payload_path.read_text(encoding="utf-8"))


def _load_graph(path: str) -> Graph:
    payload = _load_payload(path)
    graph_payload = payload["graph"] if "graph" in payload else payload
    graph = Graph()
    for node_data in graph_payload.get("nodes", []):
        graph.add_node(
            Node(
                id=node_data["id"],
                content=node_data["content"],
                summary=node_data.get("summary", ""),
                metadata=node_data.get("metadata", {}),
            )
        )
    for edge_data in graph_payload.get("edges", []):
        graph.add_edge(
            Edge(
                source=edge_data["source"],
                target=edge_data["target"],
                weight=edge_data.get("weight", 0.5),
                kind=edge_data.get("kind", "sibling"),
                metadata=edge_data.get("metadata", {}),
            )
        )
    return graph


def _parse_vector(values: list[str] | None) -> list[float] | None:
    if values is None:
        return None
    vector: list[float] = []
    for value in values:
        for chunk in value.split(","):
            if chunk:
                vector.append(float(chunk))
    return vector


def _load_session_queries(session_paths: list[str] | str) -> list[str]:
    if isinstance(session_paths, str):
        session_paths = [session_paths]

    queries: list[str] = []
    for session_path in session_paths:
        path = Path(session_path).expanduser()
        if not path.exists():
            raise SystemExit(f"missing sessions path: {path}")
        if path.is_dir():
            queries.extend(extract_queries_from_dir(path))
        elif path.is_file():
            queries.extend(extract_queries(path))
        else:
            raise SystemExit(f"invalid sessions path: {path}")
    return queries


_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def _tokenize_text(text: str) -> set[str]:
    return {match.group(0).lower() for match in _WORD_RE.finditer(text)}


def _load_query_vector_from_stdin() -> list[float]:
    raw = sys.stdin.read().strip()
    if not raw:
        raise SystemExit("query vector JSON required on stdin")
    data = json.loads(raw)
    if not isinstance(data, list):
        raise SystemExit("query vector stdin payload must be a JSON array")
    vector: list[float] = []
    for value in data:
        vector.append(float(value))
    return vector


def _keyword_seeds(graph: Graph, text: str, top_k: int) -> list[tuple[str, float]]:
    query_tokens = _tokenize_text(text)
    if not query_tokens or top_k <= 0:
        return []

    scores: list[tuple[str, float]] = []
    for node in graph.nodes():
        node_tokens = _tokenize_text(node.content)
        overlap = len(query_tokens & node_tokens)
        scores.append((node.id, overlap / len(query_tokens)))

    if not scores:
        return []
    scores.sort(key=lambda item: (item[1], item[0]), reverse=True)
    return scores[:top_k]


def cmd_init(args: argparse.Namespace) -> int:
    output_dir = Path(args.output).expanduser()
    if output_dir.suffix == ".json" and not output_dir.is_dir():
        output_dir = output_dir.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    graph, texts = split_workspace(args.workspace)
    if args.sessions is not None:
        queries = _load_session_queries(args.sessions)
        replay_queries(graph=graph, queries=queries)

    graph_path = output_dir / "graph.json"
    texts_path = output_dir / "texts.json"
    payload = {
        "nodes": [
            {
                "id": node.id,
                "content": node.content,
                "summary": node.summary,
                "metadata": node.metadata,
            }
            for node in graph.nodes()
        ],
        "edges": [
            {
                "source": edge.source,
                "target": edge.target,
                "weight": edge.weight,
                "kind": edge.kind,
                "metadata": edge.metadata,
            }
            for source_edges in graph._edges.values()
            for edge in source_edges.values()
        ],
    }
    graph_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    texts_path.write_text(json.dumps(texts, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps({"graph": str(graph_path), "texts": str(texts_path)}))
    else:
        print(f"graph_path: {graph_path}")
        print(f"texts_path: {texts_path}")
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    graph = _load_graph(args.graph)
    query_vec = _parse_vector(args.query_vector)
    using_stdin_vector = bool(args.query_vector_stdin)

    if args.query_vector_stdin and args.query_vector:
        raise SystemExit("use only one of --query-vector or --query-vector-stdin")

    if query_vec is not None:
        index_path = args.index
    elif using_stdin_vector:
        if not args.index:
            raise SystemExit("query-vector-stdin requires --index")
        query_vec = _load_query_vector_from_stdin()
        index_path = args.index
    else:
        query_vec = None
        index_path = args.index

    if query_vec is not None:
        if not index_path:
            raise SystemExit("query vector mode requires --index")
        if not Path(index_path).exists():
            raise SystemExit(f"missing index file: {index_path}")
        index_payload = json.loads(Path(index_path).read_text(encoding="utf-8"))
        index = VectorIndex()
        for node_id, vector in index_payload.items():
            index.upsert(node_id, vector)
        seeds = index.search(query_vec, top_k=args.top)
    else:
        seeds = _keyword_seeds(graph=graph, text=args.text, top_k=args.top)

    result = traverse(graph=graph, seeds=seeds, config=TraversalConfig(max_hops=15), route_fn=None)

    if args.json:
        print(
            json.dumps(
                {
                    "fired": result.fired,
                    "steps": [step.__dict__ for step in result.steps],
                    "context": result.context,
                }
            )
        )
    else:
        print(result.context)
        print()
        print("\"fired\":", result.fired)
    return 0


def cmd_learn(args: argparse.Namespace) -> int:
    graph = _load_graph(args.graph)
    fired_ids = [value.strip() for value in args.fired_ids.split(",") if value.strip()]
    apply_outcome(graph, fired_nodes=fired_ids, outcome=args.outcome)
    payload = {
        "graph": {
            "nodes": [
                {
                    "id": node.id,
                    "content": node.content,
                    "summary": node.summary,
                    "metadata": node.metadata,
                }
                for node in graph.nodes()
            ],
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "weight": edge.weight,
                    "kind": edge.kind,
                    "metadata": edge.metadata,
                }
            for source_edges in graph._edges.values()
                for edge in source_edges.values()
            ],
        },
    }
    Path(args.graph).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(payload["graph"], indent=2))
    else:
        print(f"updated {args.graph}")
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    graph = _load_graph(args.graph)
    queries = _load_session_queries(args.sessions)
    if args.max_queries is not None:
        if args.max_queries <= 0:
            queries = []
        else:
            queries = queries[: args.max_queries]

    stats = replay_queries(graph=graph, queries=queries, verbose=not args.json)

    graph_path = Path(args.graph).expanduser()
    if graph_path.is_dir():
        graph_path = graph_path / "graph.json"
    payload = {
        "nodes": [
            {
                "id": node.id,
                "content": node.content,
                "summary": node.summary,
                "metadata": node.metadata,
            }
            for node in graph.nodes()
        ],
        "edges": [
            {
                "source": edge.source,
                "target": edge.target,
                "weight": edge.weight,
                "kind": edge.kind,
                "metadata": edge.metadata,
            }
            for source_edges in graph._edges.values()
            for edge in source_edges.values()
        ],
    }
    graph_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(stats, indent=2))
    else:
        print(
            f"Replayed {stats['queries_replayed']}/{len(queries)} queries, "
            f"{stats['cross_file_edges_created']} cross-file edges created"
        )
    return 0


def cmd_health(args: argparse.Namespace) -> int:
    graph = _load_graph(args.graph)
    health = measure_health(graph)
    payload = health.__dict__
    payload["nodes"] = graph.node_count()
    payload["edges"] = graph.edge_count()
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(
            "nodes: {nodes}\nedges: {edges}\ndormant_pct: {dormant_pct:.2f}\nhabitual_pct: {habitual_pct:.2f}\n"
            "reflex_pct: {reflex_pct:.2f}\ncross_file_edge_pct: {cross_file_edge_pct:.2f}\norphan_nodes: {orphan_nodes}".format(
                **payload
            )
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "init":
        return cmd_init(args)
    if args.command == "query":
        return cmd_query(args)
    if args.command == "learn":
        return cmd_learn(args)
    if args.command == "replay":
        return cmd_replay(args)
    if args.command == "health":
        return cmd_health(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
