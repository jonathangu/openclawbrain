# CrabPath

CrabPath is a memory graph for AI agents with migration, query, and graph maintenance operations.

## Installation

```bash
# Local package installation
python -m pip install -e .

# Run commands using Python module path
PYTHONPATH=. python -m crabpath.cli --help
```

The project defines a `crabpath` console script entry point in `pyproject.toml`, so running `pip install -e .` also enables:

```bash
crabpath --help
```

## CLI usage

- Query a graph:
  - `crabpath query "how do I use codex" --graph crabpath_graph.json`
- Learn from outcome:
  - `crabpath learn --graph crabpath_graph.json --outcome 1.0 --fired-ids deploy-config,cache-refresh`
- Add and remove nodes:
  - `crabpath add --id my-node --content "..." --graph crabpath_graph.json`
  - `crabpath remove --id my-node --graph crabpath_graph.json`
- Inspect graph stats:
  - `crabpath stats --graph crabpath_graph.json`
- Split nodes:
  - `crabpath split --graph crabpath_graph.json --node-id my-node --save`
- Run lifecycle simulation:
  - `crabpath sim --queries 20 --decay-interval 5 --decay-half-life 80`
- Migrate a workspace:
  - `crabpath migrate --workspace ~/.openclaw/workspace --output-graph crabpath_graph.json --output-embeddings crabpath_embeddings.json --include-memory --verbose`

All successful commands emit JSON to `stdout`.

## Migration

`crabpath migrate` runs `crabpath.migrate.migrate` with options:
- `--workspace` workspace directory
- `--session-logs` optional replay files
- `--include-memory`, `--include-docs`
- `--output-graph` destination graph file
- `--output-embeddings` destination embedding index file
- `--verbose`

The command returns an object including `ok`, `graph_path`, optional `embeddings_path`, and detailed migration `info`.

## Python API

Import common operations directly from package modules:

```python
from crabpath import migrate, Graph
from crabpath.mcp_server import main as mcp_main
from crabpath.lifecycle_sim import run_simulation
from crabpath.mitosis import split_node
```

Use `run_simulation` for deterministic self-organization experiments and `migrate` for bootstrap + optional replay.

The MCP server entrypoint lives at `crabpath/mcp_server.py` and exposes the same core operations as a JSON-RPC tool surface.
