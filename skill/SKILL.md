---
name: crabpath
description: "Memory graph for AI agents — learned document routing with semantic embeddings. Use when: agent needs better context retrieval, memory recall is missing relevant docs, or you want to reduce context bloat. Replaces static file loading with a learned graph that knows what to retrieve (and what NOT to)."
homepage: https://github.com/jonathangu/crabpath
metadata:
  {
    "openclaw":
      {
        "emoji": "🦀",
        "requires": { "bins": ["crabpath"] },
        "install":
          [
            {
              "id": "uv",
              "kind": "uv",
              "package": "crabpath",
              "bins": ["crabpath"],
              "label": "Install CrabPath (uv)",
            },
          ],
      },
  }
---

# CrabPath 🦀

Memory graph for AI agents. Learns which documents help and which to suppress.

## Setup (one time)

### 1. Install

```bash
pip install crabpath
```

### 2. Initialize graph + embeddings

Requires one of: `OPENAI_API_KEY`, `GEMINI_API_KEY`, or local Ollama.

```bash
crabpath init \
  --workspace ~/.openclaw/workspace \
  --sessions ~/.openclaw/agents/main/sessions/
```

Without an embedding provider: `crabpath init --workspace ~/.openclaw/workspace --no-embeddings`

### 3. Wire into agent

```bash
crabpath install-hook --agent-workspace ~/.openclaw/workspace
```

This writes an integration block to AGENTS.md. Idempotent — safe to run twice.

## Usage

### Query the memory graph

Before answering questions about prior work, context, or decisions:

```bash
crabpath query '<summary of user message>' \
  --graph ~/.crabpath/graph.json \
  --index ~/.crabpath/embed.json \
  --top 8 --json
```

Returns JSON with `fired` nodes (relevant context) and `inhibited` nodes (suppressed).

Use the returned context to supplement memory files. CrabPath suggests context — it never modifies responses directly.

### Learn from outcomes

After a successful response (no correction):
```bash
crabpath learn --graph ~/.crabpath/graph.json --outcome 1.0 --fired-ids node1,node2
```

After a correction:
```bash
crabpath learn --graph ~/.crabpath/graph.json --outcome -1.0 --fired-ids node1,node2
```

### Health check

```bash
crabpath health --graph ~/.crabpath/graph.json --query-stats ~/.crabpath/graph.stats.json
```

Shows 8 metrics with ✅/❌ status. Key targets:
- `avg_nodes_fired_per_query`: 3–10
- `dormant_pct`: 60–97% (the graph learns what NOT to load)
- `reflex_pct`: 0.5–5% (compiled procedures, zero deliberation)

### Track evolution

```bash
crabpath evolve --graph ~/.crabpath/graph.json \
  --snapshots ~/.crabpath/evolution.jsonl --report
```

### Rebuild graph (after workspace changes)

```bash
crabpath init \
  --workspace ~/.openclaw/workspace \
  --sessions ~/.openclaw/agents/main/sessions/
```

Re-running `init` rebuilds from scratch. Your old graph is overwritten.

## When to Use

✅ **USE this skill when:**
- Agent needs better recall of prior work, decisions, or context
- `memory_search` returns low-confidence or irrelevant results
- Context is bloated (loading too many tokens per turn)
- Agent keeps forgetting procedures or repeating mistakes

❌ **DON'T USE when:**
- Simple factual lookups (use web_search)
- Fresh workspace with < 10 files (not enough structure to learn from)
- One-shot questions with no recurring patterns

## Key Concepts

- **Nodes** = document chunks (split from workspace files by heading/paragraph)
- **Edges** = weighted pointers (positive = excitatory, negative = inhibitory)
- **Reflex** (weight > 0.8) = auto-follow, no deliberation
- **Habitual** (0.3–0.8) = router decides
- **Dormant** (< 0.3) = suppressed, the graph learned this isn't useful
- **Edge damping** = synaptic fatigue within an episode (prevents loops)

## Uninstall

```bash
pip uninstall crabpath
rm -rf ~/.crabpath
```

Remove the "CrabPath Memory Graph" section from AGENTS.md.

## Links

- PyPI: https://pypi.org/project/crabpath/
- GitHub: https://github.com/jonathangu/crabpath
- Paper: https://jonathangu.com/crabpath/
