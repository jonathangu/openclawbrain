# Configuration guide

This guide covers the practical operator setup for **OpenClawBrain v2**.

If you want the repo's exact truth contract first, read:
- `README.md`
- `docs/lifecycle.md`
- `docs/RELEASE_CONTRACT.md`
- `docs/EVIDENCE.md`

## Happy-path operator flow

These examples intentionally use the exact public-registry flow that already passed on `redogfood`.

1. install the published plugin/runtime payload into OpenClaw: `openclaw plugins install @openclawbrain/openclaw@0.4.0`
2. attach it to one OpenClaw home with `npx @openclawbrain/cli@0.4.3 install --openclaw-home ~/.openclaw`
3. run `openclaw gateway restart`
4. verify with `npx @openclawbrain/cli@0.4.3 status --openclaw-home ~/.openclaw --detailed`
5. configure embeddings and choose a `brainRoot` if you need to tune defaults
6. run the validation harnesses appropriate to the claim you want to make

## Canonical install, upgrade, remove, and verify path

Front door packages:

```bash
openclaw plugins install @openclawbrain/openclaw@0.4.0
```

Install or attach:

```bash
npx @openclawbrain/cli@0.4.3 install --openclaw-home ~/.openclaw
openclaw gateway restart
npx @openclawbrain/cli@0.4.3 status --openclaw-home ~/.openclaw --detailed
```

Upgrade uses the same lane:

```bash
openclaw plugins install @openclawbrain/openclaw@0.4.0
npx @openclawbrain/cli@0.4.3 install --openclaw-home ~/.openclaw
openclaw gateway restart
npx @openclawbrain/cli@0.4.3 status --openclaw-home ~/.openclaw --detailed
```

Verify the target install at any time:

```bash
npx @openclawbrain/cli@0.4.3 status --openclaw-home ~/.openclaw --detailed
npx @openclawbrain/cli@0.4.3 status --openclaw-home ~/.openclaw --json
```

Remove only the profile hook and keep OpenClawBrain data:

```bash
npx @openclawbrain/cli@0.4.3 detach --openclaw-home ~/.openclaw
openclaw gateway restart
```

Remove the profile hook and keep data explicitly:

```bash
npx @openclawbrain/cli@0.4.3 uninstall --openclaw-home ~/.openclaw --keep-data
openclaw gateway restart
```

Remove the profile hook and purge OpenClawBrain data for that install:

```bash
npx @openclawbrain/cli@0.4.3 uninstall --openclaw-home ~/.openclaw --purge-data
openclaw gateway restart
```

If you want to remove the hook but keep the data, use `detach` or `npx @openclawbrain/cli@0.4.3 uninstall --openclaw-home ~/.openclaw --keep-data`. `detach` is the simpler keep-data path. The plugin payload itself lives under OpenClaw's plugin manager, so remove `@openclawbrain/openclaw` there separately only if you want the package files gone too.

## Compatibility path

`@jonathangu/openclawbrain@0.3.5` remains published for older plugin or wrapper installs:

```bash
openclaw plugins install @jonathangu/openclawbrain@0.3.5
```

Keep that as compatibility guidance only. It is no longer the main operator story.

Decision and migration note: [`docs/lifecycle.md`](lifecycle.md)

## Important host-seam truth

On current OpenClaw hosts, **do not manually write** `plugins.slots.contextEngine` for OpenClawBrain.

That older seam is not the stable install story anymore. OpenClawBrain now uses a hook-based compatibility bridge on hosts where `api.registerContextEngine` is gone.

Current install caveat:
- some hosts still warn about a plugin id mismatch because the plugin manifest uses `openclawbrain` while the package/entry hint uses `openclaw`
- the install still works; treat that warning as currently cosmetic rather than a failed attach

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

## Optional one-shot learning pass

The lifecycle path above is the canonical install lane. If you want to force one explicit local learning pass after attach, run:

```bash
npx @openclawbrain/cli@0.4.3 learn --openclaw-home ~/.openclaw --json
```

That gives you a machine-readable snapshot of what the learner scanned, whether anything materialized, and whether a promotion occurred.

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
| `OPENCLAWBRAIN_AUTO_USER_CORRECTIONS_ENABLED` | Enable async user-correction proposals grounded in explicit user turns |
| `OPENCLAWBRAIN_AUTO_USER_CORRECTIONS_PROVIDER` | Provider override for async user-correction proposals |
| `OPENCLAWBRAIN_AUTO_USER_CORRECTIONS_MODEL` | Model override for async user-correction proposals |
| `OPENCLAWBRAIN_AUTO_USER_CORRECTIONS_MIN_CONFIDENCE` | Minimum confidence required before auto-committing a correction |

## Summary-aware correction lanes

OpenClawBrain now has two complementary user-correction paths:

- **fast deterministic lane** — catches obvious explicit corrections immediately on ingest
- **async proposal lane** — lets a model read recent messages plus recent LCM summaries and propose a typed correction write off the hot path

Both lanes follow the same authority rule:

- **LCM summaries are context, not truth**
- **the explicit user quote is the authority**
- **the committed typed correction memory becomes the durable current-truth layer**

At serve time, LCM summaries also act as a routing prior:

- broad recap questions may stay at summary level
- precision/conflict-sensitive questions should expand back toward source
- explicit correction cards outrank summary recap when they conflict

## Child-worker boundary

`brainWorkerMode=child` is the recommended serving boundary.

Why:
- serving reads immutable promoted packs
- the learner runs in a separate supervised child process
- status/doctor surfaces can report PID, heartbeat, restart, and exit truth clearly
- `in_process` remains useful for development or debugging, but it should not be treated as the production operator boundary

## Operator commands

```bash
npx @openclawbrain/cli@0.4.3 install --openclaw-home ~/.openclaw
npx @openclawbrain/cli@0.4.3 attach --openclaw-home ~/.openclaw --activation-root /path/to/activation
npx @openclawbrain/cli@0.4.3 status --openclaw-home ~/.openclaw --detailed
npx @openclawbrain/cli@0.4.3 status --openclaw-home ~/.openclaw --json
npx @openclawbrain/cli@0.4.3 learn --openclaw-home ~/.openclaw --json
npx @openclawbrain/cli@0.4.3 history --openclaw-home ~/.openclaw --limit 20 --json
npx @openclawbrain/cli@0.4.3 context "How should I answer this?" --openclaw-home ~/.openclaw
npx @openclawbrain/cli@0.4.3 rollback --openclaw-home ~/.openclaw --dry-run
npx @openclawbrain/cli@0.4.3 detach --openclaw-home ~/.openclaw
npx @openclawbrain/cli@0.4.3 uninstall --openclaw-home ~/.openclaw --keep-data|--purge-data
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
- `npx @openclawbrain/cli@0.4.3 status --openclaw-home ~/.openclaw --detailed` keeps embedding, worker, and hook truth visible for one installed target

## Session reset note

OpenClawBrain preserves history through compaction, but it does not override OpenClaw's core session reset policy. If sessions reset sooner than you want, increase OpenClaw's `session.reset.idleMinutes`.
