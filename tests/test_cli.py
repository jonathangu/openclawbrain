from __future__ import annotations

import json

from crabpath.cli import main


def _write_graph(path) -> None:
    payload = {
        "graph": {
            "nodes": [
                {"id": "a", "content": "alpha", "summary": "", "metadata": {"file": "a.md"}},
                {"id": "b", "content": "beta", "summary": "", "metadata": {"file": "a.md"}},
            ],
            "edges": [
                {"source": "a", "target": "b", "weight": 0.7, "kind": "sibling", "metadata": {}}
            ],
        }
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def test_init_command(tmp_path, capsys):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "a.md").write_text("# Title\n\n## One\nA")

    output = tmp_path / "init.json"
    code = main(["init", "--workspace", str(ws), "--output", str(output)])
    assert code == 0
    data = json.loads(output.read_text(encoding="utf-8"))
    assert "node_texts" in data
    assert "graph" in data


def test_query_and_health_commands(tmp_path, capsys):
    graph_path = tmp_path / "g.json"
    index_path = tmp_path / "i.json"

    _write_graph(graph_path)
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump({"a": [1.0, 0.0, 0.0], "b": [0.0, 1.0, 0.0]}, f)

    code = main(["query", "hello", "--graph", str(graph_path), "--index", str(index_path), "--top", "1", "--json", "--query-vector", "1,0,0"])
    assert code == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert "fired" in payload

    code2 = main(["health", "--graph", str(graph_path)])
    assert code2 == 0
    health_out = capsys.readouterr().out
    data = json.loads(health_out)
    assert "orphan_nodes" in data


def test_learn_command_updates_graph(tmp_path):
    graph_path = tmp_path / "g.json"
    _write_graph(graph_path)
    code = main(["learn", "--graph", str(graph_path), "--outcome", "1.0", "--fired-ids", "a,b"])
    assert code == 0
    text = graph_path.read_text(encoding="utf-8")
    assert "updated" not in text
    loaded = json.loads(text)
    assert loaded["graph"]["edges"][0]["weight"] != 0.7
