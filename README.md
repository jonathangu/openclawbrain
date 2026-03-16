# OpenClawBrain v2

OpenClawBrain is an OpenClaw plugin that keeps the inherited lossless transcript-memory substrate and adds a learned routing layer on top.

Front doors:
- Project site: https://openclawbrain.ai
- GitHub repo: https://github.com/jonathangu/openclawbrain
- Jonathan Gu's 2016 reinforcement-learning paper: https://openclawbrain.ai/jonathan-gu-2016-reinforcement-learning-paper.pdf

This repo is the active v2 trunk. The earlier spike lives at [jonathangu/openclawbrain-v1-spike-archive](https://github.com/jonathangu/openclawbrain-v1-spike-archive).

## Release truth in 30 seconds

| Public label | Status | What it means right now |
| --- | --- | --- |
| **paper-faithful core** | yes | finite-horizon traversal, stochastic policy, terminal reward, full-trajectory REINFORCE updates, learned seed routing, and immutable promoted packs are implemented in the current repo |
| **live-path implemented** | yes | OpenClaw runtime decisioning, shadow mode, correction-first assembly, immediate `brain_teach` retrieval, and replay-gated promotion are wired into the live path |
| **operationally validated** | not yet | deterministic runtime proof is real, but the full sterile host-surface harness is still not frozen end to end; bundle-level mutation evaluation, CI-enforced proof gates, and packaging/type hardening remain open |

If you want the exact contract rather than the pitch, read [docs/RELEASE_CONTRACT.md](docs/RELEASE_CONTRACT.md).

## What OpenClawBrain does

OpenClawBrain has two layers:

1. **LCM / transcript memory**
   - persists conversation history in SQLite
   - compacts older turns into a summary DAG instead of dropping them
   - assembles summaries plus fresh raw turns back into model context
   - exposes recall tools like `lcm_grep`, `lcm_describe`, and `lcm_expand_query`

2. **Learned routing layer**
   - decides whether to use learned retrieval, shadow the route, or skip with an explicit reason
   - retrieves from immutable promoted packs only
   - supports immediate `brain_teach` correction retrieval
   - trains in the background from human/self/scanner/teacher evidence
   - gates promotion with replay checks before serving new packs

Nothing in the transcript-memory substrate is supposed to be thrown away casually. The point is to keep lossless recall while adding a learned context-routing layer that can improve over time.

## Current reality

### True in the repo now
- paper-faithful traversal/update path exists
- child-worker mode is a real runtime boundary, with supervisor/protocol/restart truth
- shadow mode is a real runtime decision rather than a fake `use_brain` variant
- deterministic session-bound `brain_teach` proof exists
- deterministic runtime proof exists for immediate teach retrieval and serve-from-last-promoted-pack after worker failure
- structured raw evidence and worker-side trust-ordered resolution are real

### Implemented but not frozen
- the real OpenClaw host-surface validation lane
- mutation evaluation at the intended bundle level
- CI-enforced proof gates
- clean npm/package boundary for outside operators

### Honest current blocker
The current docs and runtime story should stay aligned with this exact state:
- sterile preflight/config seam has been repaired
- deterministic runtime proof is repaired and repeatably passing on fresh isolated roots
- the full sterile host harness is **still not frozen end to end** because it currently stalls during `openclawbrain init` before the host-turn proof bundle completes

That means the remaining pain is mainly host/operator/release-boundary work, not another learning-architecture rewrite.

## Quick start

### Prerequisites
- OpenClaw
- Node.js 22+
- an LLM provider for transcript summarization
- an embeddings provider for `openclawbrain init`, learned retrieval, and `brain_teach`

### Install

Published package:

```bash
openclaw plugins install @jonathangu/openclawbrain
```

From a local OpenClaw checkout:

```bash
pnpm openclaw plugins install @jonathangu/openclawbrain
```

For local development, link your working tree instead of copying files:

```bash
openclaw plugins install --link /path/to/openclawbrain
# or from a local OpenClaw checkout:
# pnpm openclaw plugins install --link /path/to/openclawbrain
```

### Important host-seam truth

On current OpenClaw hosts, **do not manually write** `plugins.slots.contextEngine` for OpenClawBrain.

That older seam is no longer the stable installation boundary. OpenClawBrain now includes a hook-based compatibility bridge for hosts where `api.registerContextEngine` is gone, and the plugin installer is the supported path.

If you are debugging an older host build, treat any manual slot/config surgery as version-specific debugging rather than the normal install story.

### Recommended starting config

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
- `freshTailCount=32` keeps recent turns raw
- `contextThreshold=0.75` leaves response headroom
- `incrementalMaxDepth=-1` lets compaction keep cascading when needed
- `brainWorkerMode=child` is the practical operator boundary

### Initialize the graph

The transcript-memory layer works immediately after install. The learned layer needs an explicit init pass:

```bash
openclawbrain init /path/to/workspace
```

That creates the initial graph, writes `state.db`, creates pack `v000001`, and promotes it.

## Embeddings

OpenClawBrain currently targets tested OpenAI-compatible `/v1/embeddings` APIs. That includes local Ollama endpoints and remote OpenAI-compatible services.

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

If the remote endpoint needs auth, set `OPENCLAWBRAIN_EMBEDDING_API_KEY`.

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
pnpm exec tsx scripts/validate-brain-runtime-behavior.ts
```

Disposable host-surface harness:

```bash
node scripts/validate-openclaw-install.mjs --setup-only

OPENCLAWBRAIN_VALIDATION_EMBEDDING_PROVIDER=ollama \
OPENCLAWBRAIN_VALIDATION_EMBEDDING_MODEL=bge-large:latest \
OPENCLAWBRAIN_VALIDATION_MODEL=ollama/qwen2.5:7b-instruct \
node scripts/validate-openclaw-install.mjs
```

Current honest boundary: the runtime proof harness is a real release signal today; the full sterile host harness is still not a frozen end-to-end release gate.

## Fallback behavior

- if the brain has not been initialized, the plugin serves transcript-memory context only
- if embeddings are not configured, learned retrieval and `brain_teach` stay disabled
- local loopback embedding endpoints do not require a bearer token by default
- if the background worker is unavailable, serving still uses the last promoted pack
- `openclawbrain status` and `openclawbrain doctor` surface resolved embedding and worker truth so operator state is visible

## What is still open

1. freeze the host-surface proof boundary honestly
2. move mutation evaluation from proposal-level gating to bundle-level replay decisions
3. turn the evidence ladder into a real CI/release gate
4. clean the npm/package boundary and type surface
5. keep pushing evidence sourcing away from heuristics toward structured signals

## Documentation map

- [docs/RELEASE_CONTRACT.md](docs/RELEASE_CONTRACT.md) — exact truth contract: true now vs not frozen vs not done
- [docs/EVIDENCE.md](docs/EVIDENCE.md) — proof ladder and artifact contract
- [docs/configuration.md](docs/configuration.md) — practical operator setup
- [docs/END_STATE.md](docs/END_STATE.md) — maintainer execution guide
- [docs/architecture.md](docs/architecture.md) — inherited LCM substrate plus product architecture context
- [docs/agent-tools.md](docs/agent-tools.md) — recall tools vs live brain tools
- [docs/tui.md](docs/tui.md) — TUI reference

## Development

```bash
npm test
npm pack --dry-run
npx tsc --noEmit
```

Current repo truth: `npm test` and targeted runtime validation are ahead of full-repo `npx tsc --noEmit`, which still has known drift outside the latest runtime slices.

## License

MIT
