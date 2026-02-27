# CrabPath

Pure graph engine for retrieval routing. Zero deps. Zero network calls. Caller provides embeddings and LLM callbacks.

## Install

```bash
pip install crabpath             # pure graph engine
pip install crabpath[embeddings] # + local embeddings (no API key)
```

## Python API

```python
from crabpath import split_workspace, traverse, apply_outcome, VectorIndex

# 1. Split workspace into graph + texts
graph, texts = split_workspace("./workspace")

# 2. Caller embeds (use whatever you have)
index = VectorIndex()
for nid, content in texts.items():
    index.upsert(nid, your_embed_fn(content))

# 3. Query
seeds = index.search(your_embed_fn("how do I deploy"), top_k=8)
result = traverse(graph, seeds)

# 4. Learn
apply_outcome(graph, result.fired, outcome=1.0)
```

## Batch Callbacks

```python
from crabpath._batch import batch_or_single, batch_or_single_embed

# One call for all embeddings
vecs = batch_or_single_embed(
    list(texts.items()),
    embed_batch_fn=your_batch_embed
)

# One call for all LLM work
results = batch_or_single(
    [{"id": "n0", "system": "summarize", "user": content}],
    llm_batch_fn=your_batch_llm
)
```

## Local Embeddings

```bash
pip install crabpath[embeddings]
```

```python
from crabpath.embeddings import local_embed_fn, local_embed_batch_fn
# all-MiniLM-L6-v2, 80MB, CPU, no API key
```

## Default Hash Embeddings

If you don’t install `crabpath[embeddings]`, CrabPath uses a built-in zero-dependency
`HashEmbedder` (`hash-v1`, 1024 dimensions) for indexing and query vectors.

```python
from crabpath import HashEmbedder

embedder = HashEmbedder()  # default in CLI and default_embed
vec = embedder.embed("deploy to production")
vec2 = embedder.embed("deploy to production")
assert vec == vec2  # deterministic
```

The `HashEmbedder` is deterministic and dependency-free: it tokenizes text into
words and char n-grams, hashes features into a fixed vector size, and normalizes
to unit length.

## Session Replay (warm start)

A fresh graph is 100% habitual — every edge requires deliberation. Replay warms it up by feeding historical session logs through the graph:

```python
from crabpath import replay_queries, split_workspace
from crabpath.replay import extract_queries_from_dir

graph, texts = split_workspace("./workspace")
queries = extract_queries_from_dir("./sessions/")
replay_queries(graph=graph, queries=queries)
# Graph now has learned edges from real usage patterns
```

Or via CLI:
```bash
crabpath init --workspace ./ws --output ./data --sessions ./sessions/
# or separately:
crabpath replay --graph ./data/graph.json --sessions ./sessions/
```

On a 1,012-node graph, replaying 120 real queries created 39% cross-file edges and 5% reflex edges in seconds.

## CLI (pure graph ops)

```
crabpath init --workspace W --output O [--sessions S]
crabpath query TEXT --graph G [--index I] [--query-vector-stdin] [--top N] [--json]
crabpath learn --graph G --outcome N --fired-ids a,b,c
crabpath replay --graph G --sessions S
crabpath health --graph G
crabpath merge --graph G
crabpath connect --graph G
crabpath journal [--stats]
```

## Reproduce Results

```bash
git clone https://github.com/jonathangu/crabpath.git && cd crabpath
pip install -e . && python sims/run_all.py
```

8 deterministic sims. No API keys. See [REPRODUCE.md](REPRODUCE.md).

## Paper

https://jonathangu.com/crabpath/
