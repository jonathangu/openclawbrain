---
name: crabpath
description: Memory graph engine with learned routing. Pure callbacks — caller provides embed/LLM functions.
metadata:
  openclaw:
    emoji: "🦀"
    requires:
      python: ">=3.10"
---

# CrabPath

Memory graph engine. Zero deps. Zero network calls. Caller provides everything.

## Install

```bash
pip install crabpath
```

## Integration

```python
from crabpath import split_workspace, traverse, apply_outcome, VectorIndex
from crabpath._batch import batch_or_single_embed
from crabpath.store import save_state, load_state

# --- Build (once) ---
graph, texts = split_workspace("~/.openclaw/workspace")
index = VectorIndex()
vecs = batch_or_single_embed(list(texts.items()), embed_batch_fn=your_embed_batch)
for nid, vec in vecs.items():
    index.upsert(nid, vec)
save_state(graph, index, "~/.crabpath/state.json")

# --- Query (every turn) ---
graph, index = load_state("~/.crabpath/state.json")
seeds = index.search(your_embed("user question"), top_k=8)
result = traverse(graph, seeds)
context = result.context  # assembled text from fired nodes

# --- Learn (after response) ---
apply_outcome(graph, result.fired, outcome=1.0)  # +1 good, -1 bad
save_state(graph, index, "~/.crabpath/state.json")
```

## Callbacks

CrabPath never calls any API. The caller provides:

```python
embed_fn(text: str) -> list[float]                              # single
embed_batch_fn(texts: list[tuple[str, str]]) -> dict[str, list[float]]  # batch
llm_fn(system: str, user: str) -> str                           # single
llm_batch_fn(requests: list[dict]) -> list[dict]                # batch
```

## Session Replay (warm start)

```python
from crabpath.replay import extract_queries_from_dir, replay_queries
queries = extract_queries_from_dir("~/.openclaw/agents/main/sessions/")
replay_queries(graph=graph, queries=queries)
```

Or: `crabpath init --workspace W --output O --sessions ~/.openclaw/agents/main/sessions/`

## CLI

Pure graph operations only. No network calls.

```
crabpath init --workspace W --output O [--sessions S]
crabpath query TEXT --graph G [--index I] [--query-vector-stdin]
crabpath learn --graph G --outcome N --fired-ids a,b,c
crabpath replay --graph G --sessions S
crabpath health --graph G
crabpath merge --graph G
crabpath connect --graph G
crabpath journal [--stats]
```

## Local Embeddings (optional)

```bash
pip install crabpath[embeddings]
```

```python
from crabpath.embeddings import local_embed_fn, local_embed_batch_fn
# all-MiniLM-L6-v2 — 80MB, CPU, no API key
```

## Default Embeddings

By default (including CLI `init`/`query`), CrabPath uses `HashEmbedder`:

- class: `crabpath.HashEmbedder`
- embedder name: `hash-v1`
- deterministic, zero dependencies, no network calls
- dimension: 1024

## Paper

https://jonathangu.com/crabpath/
