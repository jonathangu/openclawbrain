# Configuration guide

This guide covers the practical operator setup for **OpenClawBrain v2**.

For repo truth, read:
- `README.md`
- `docs/RELEASE_CONTRACT.md`
- `docs/END_STATE.md`

## Quick start

Install the plugin with OpenClaw's plugin installer:

```bash
openclaw plugins install @jonathangu/openclawbrain
```

If you're running from a local OpenClaw checkout:

```bash
pnpm openclaw plugins install @jonathangu/openclawbrain
```

For local development, link your working copy:

```bash
openclaw plugins install --link /path/to/openclawbrain
```

`openclaw plugins install` handles plugin registration, enabling, and compatible slot selection automatically.

## Context engine slot

On current OpenClaw hosts, **do not manually write** `plugins.slots.contextEngine` for OpenClawBrain. The linked/package installer is the supported path, and OpenClawBrain now falls back to hook-based integration on hosts where the older `registerContextEngine` seam is gone.

If you are debugging an older host build, treat any manual slot override as version-specific surgery rather than a stable setup step.

## Recommended starting configuration

```json
{
  "plugins": {
    "entries": {
      "openclawbrain": {
        "enabled": true,
        "config": {
          "freshTailCount": 32,
          "contextThreshold": 0.75,
          "incrementalMaxDepth": -1,
          "brainRoot": "~/.openclaw/openclawbrain",
          "brainEmbeddingProvider": "ollama",
          "brainEmbeddingModel": "bge-large:latest",
          "brainWorkerMode": "child"
        }
      }
    }
  }
}
```

Why these defaults:
- `freshTailCount=32` keeps recent turns raw for continuity
- `contextThreshold=0.75` leaves response headroom
- `incrementalMaxDepth=-1` lets compaction keep cascading when needed
- `brainWorkerMode=child` is the practical serving boundary

## Initialization

The transcript layer works immediately after install. Learned retrieval needs an explicit init:

```bash
openclawbrain init /path/to/workspace
```

That creates the initial graph, writes `state.db`, snapshots pack `v000001`, and promotes it.

## Embeddings

OpenClawBrain currently targets tested OpenAI-compatible embeddings APIs.

### Local Ollama

```json
{
  "plugins": {
    "entries": {
      "openclawbrain": {
        "config": {
          "brainEmbeddingProvider": "ollama",
          "brainEmbeddingModel": "bge-large:latest"
        }
      }
    }
  }
}
```

This defaults to `http://127.0.0.1:11434/v1`.

### Remote OpenAI-compatible endpoint

```json
{
  "plugins": {
    "entries": {
      "openclawbrain": {
        "config": {
          "brainEmbeddingProvider": "openai",
          "brainEmbeddingModel": "text-embedding-3-large",
          "brainEmbeddingBaseUrl": "https://your-endpoint.example/v1"
        }
      }
    }
  }
}
```

If the endpoint requires auth, provide `OPENCLAWBRAIN_EMBEDDING_API_KEY`.

## Important environment variables

| Variable | Description |
|---|---|
| `LCM_DATABASE_PATH` | SQLite path for transcript / summary storage |
| `LCM_CONTEXT_THRESHOLD` | Fraction of context window that triggers compaction |
| `LCM_FRESH_TAIL_COUNT` | Recent raw messages protected from compaction |
| `LCM_INCREMENTAL_MAX_DEPTH` | Automatic condensation depth (`-1` = unlimited) |
| `OPENCLAWBRAIN_ROOT` | Root for `state.db` and immutable packs |
| `OPENCLAWBRAIN_EMBEDDING_PROVIDER` | Embedding provider (`openai`, `openai-resp`, `ollama`) |
| `OPENCLAWBRAIN_EMBEDDING_MODEL` | Embedding model used for init/retrieval/teach |
| `OPENCLAWBRAIN_EMBEDDING_BASE_URL` | Optional embeddings API base URL |
| `OPENCLAWBRAIN_EMBEDDING_API_KEY` | Optional explicit auth for remote embedding endpoints |
| `OPENCLAWBRAIN_SHADOW_MODE` | Record routing without injecting learned context |

## Operator commands

```bash
openclawbrain init [workspace]
openclawbrain status
openclawbrain trace [traceId]
openclawbrain replay
openclawbrain promote
openclawbrain rollback [version]
openclawbrain disable
openclawbrain enable
openclawbrain doctor
```

## Validation commands

Deterministic runtime proof harness:

```bash
node scripts/validate-brain-runtime-behavior.ts
```

Disposable host-surface harness:

```bash
node scripts/validate-openclaw-install.mjs --setup-only

OPENCLAWBRAIN_VALIDATION_EMBEDDING_PROVIDER=ollama \
OPENCLAWBRAIN_VALIDATION_EMBEDDING_MODEL=bge-large:latest \
OPENCLAWBRAIN_VALIDATION_MODEL=ollama/qwen2.5:7b-instruct \
node scripts/validate-openclaw-install.mjs
```

## Session reset note

OpenClawBrain preserves history through compaction, but it does not override OpenClaw's core session reset policy. If sessions reset sooner than you want, increase OpenClaw's `session.reset.idleMinutes`.
