# Configuration guide

This guide covers the practical operator setup for **OpenClawBrain v2**.

If you want the repo's exact truth contract first, read:
- `README.md`
- `docs/RELEASE_CONTRACT.md`
- `docs/EVIDENCE.md`

## Happy-path operator flow

1. install the plugin with the OpenClaw plugin installer
2. configure embeddings and choose a `brainRoot`
3. prefer `brainWorkerMode=child`
4. run `openclawbrain init <workspace>`
5. check `openclawbrain status`
6. run the validation harnesses appropriate to the claim you want to make

## Install

Published package:

```bash
openclaw plugins install @jonathangu/openclawbrain
```

From a local OpenClaw checkout:

```bash
pnpm openclaw plugins install @jonathangu/openclawbrain
```

For local development, link your working copy:

```bash
openclaw plugins install --link /path/to/openclawbrain
```

The plugin installer is the supported install boundary.

## Important host-seam truth

On current OpenClaw hosts, **do not manually write** `plugins.slots.contextEngine` for OpenClawBrain.

That older seam is not the stable install story anymore. OpenClawBrain now uses a hook-based compatibility bridge on hosts where `api.registerContextEngine` is gone.

If you are debugging an older host build, treat any manual slot override as version-specific surgery rather than normal operator setup.

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

The transcript-memory layer works immediately after install. Learned retrieval needs an explicit init pass:

```bash
openclawbrain init /path/to/workspace
```

That creates the initial graph, writes `state.db`, snapshots pack `v000001`, and promotes it.

## Embeddings

OpenClawBrain currently targets tested OpenAI-compatible `/v1/embeddings` APIs.

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

Default base URL:

```text
http://127.0.0.1:11434/v1
```

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

If the remote endpoint requires auth, set `OPENCLAWBRAIN_EMBEDDING_API_KEY`.

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
| `OPENCLAWBRAIN_EMBEDDING_API_KEY` | Optional auth for remote embedding endpoints |
| `OPENCLAWBRAIN_SHADOW_MODE` | Record routing without injecting learned context |
| `OPENCLAWBRAIN_WORKER_MODE` | Worker mode; prefer `child` for the real operator boundary |

## Child-worker boundary

`brainWorkerMode=child` is the recommended serving boundary.

Why:
- serving reads immutable promoted packs
- the learner runs in a separate supervised child process
- status/doctor surfaces can report PID, heartbeat, restart, and exit truth clearly
- `in_process` remains useful for development or debugging, but it should not be treated as the production operator boundary

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

### Deterministic runtime proof harness

```bash
pnpm exec tsx scripts/validate-brain-runtime-behavior.ts
```

Use this when you want a real signal on:
- immediate post-teach retrieval
- serve-from-last-promoted-pack after worker failure

### Disposable host-surface harness

```bash
node scripts/validate-openclaw-install.mjs --setup-only

OPENCLAWBRAIN_VALIDATION_EMBEDDING_PROVIDER=ollama \
OPENCLAWBRAIN_VALIDATION_EMBEDDING_MODEL=bge-large:latest \
OPENCLAWBRAIN_VALIDATION_MODEL=ollama/qwen2.5:7b-instruct \
node scripts/validate-openclaw-install.mjs
```

Current honest status:
- sterile preflight/config seam repairs are real
- deterministic runtime proof is real
- sterile host harness passes 7/7 assertions (teachRetrieval, workerDownFailOpen, recurrentQuery, shortLookup, shadowMode, noEmbedding, uninitialized)
- full end-to-end host-surface proof bundle capture is pending host-seam adaptation

So treat the host harness as active proof work, not a closed release gate.

## Fallback behavior

- if the brain has not been initialized, OpenClawBrain serves transcript-memory context only
- if embeddings are not configured, learned retrieval and `brain_teach` stay disabled
- local loopback embedding endpoints do not require a bearer token by default
- if the worker is unavailable, serving still uses the last promoted pack
- `openclawbrain status` and `openclawbrain doctor` expose embedding and worker truth so operator state stays visible

## Session reset note

OpenClawBrain preserves history through compaction, but it does not override OpenClaw's core session reset policy. If sessions reset sooner than you want, increase OpenClaw's `session.reset.idleMinutes`.
