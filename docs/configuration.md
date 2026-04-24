# Configuration guide

OpenClawBrain configuration is centered on one selected OpenClaw home, a bounded live path, and honest proof surfaces. The current product job is selective intervention: preserve a current correction or other small relevant slice when it helps, stay out when it does not, and keep the operator truth inspectable.

## Default operator flow

Keep the same `--openclaw-home` value through the whole operator flow. The public lane stays pinned to one OpenClaw home.

That home can be `~/.openclaw`, a profile-specific path like `~/.openclaw-example`, or a repo-local/custom path like `./openclaw-cormorantai`.

```bash
openclawbrain install --openclaw-home ./openclaw-cormorantai
openclaw gateway restart
openclawbrain status --openclaw-home ./openclaw-cormorantai --detailed
```

`status --detailed` is the quick verify surface for that home.

When you need durable operator evidence today, run:

```bash
openclawbrain proof --openclaw-home ./openclaw-cormorantai
```

The intended canonical lane is the same install command with optional `--proof`. Until that flag lands cleanly across the operator surfaces, proof stays a separate follow-up command. `proof` writes `summary.md`, `steps.json`, `verdict.json`, raw step logs, and proof pointers under one bundle directory.

That bundle proves install / runtime / reporting truth for the selected home. It does not, by itself, prove broader decision-quality gains.

If you only need the minimal happy path, stop there and use [Quick start](getting-started/quick-start.md).

## Runtime posture

The live path is intentionally narrow:

- serve only promoted packs
- fail open if runtime context cannot be prepared safely
- keep learning off the response path; `brainWorkerMode=child` is the supported serving boundary
- use `OPENCLAWBRAIN_SHADOW_MODE` when you want to inspect decisions without injecting brain context

The near-term decision lanes to care about are current-choice fidelity and restraint / specificity, not broad "memory coverage."

## Canonical serve-time model

OpenClawBrain does not have a separate operator-only "tier" model for context. The real serve-time model is:

- **Hot context** = the ordered summary spine plus the protected fresh tail of raw messages. `LCM_FRESH_TAIL_COUNT` controls how many latest raw messages stay protected from compaction and truncation.
- **Summary freshness** = lineage state on each summary. `fresh` can start recall. `stale_source`, `stale_branch`, `stale_pack`, `superseded`, and `tombstoned` mean the summary is a locator map, not proof, so exact or current-truth claims should expand back to source.
- **Prefetch** = an opportunistic traversal cache, not a second serve-time tier. It is keyed by query digest, active pack version, budget class, summary-routing mode, and kind. It can be reused on a hit, but pack, routing, or budget changes can make it stale or invalidated before serve time.
- **Budget controls** = `LCM_FRESH_TAIL_COUNT` protects the raw tail, `OPENCLAWBRAIN_BUDGET_FRACTION` splits learned-query budget from the turn budget, and `LCM_MAX_EXPAND_TOKENS` caps expand-to-source work. Per-turn status and trace surfaces report `queryBudgetChars`, `maxContextChars`, `injectedChars`, and `droppedChars` when serve-time clipping happens.

In the runtime and internal CLI JSON status surfaces, this model is grouped under `contextManagement`. Use that as the canonical status vocabulary.

## Recommended starting configuration

This is a practical starting point for local embeddings and the supervised child-worker boundary:

```json
{
  "plugins": {
    "entries": {
      "openclawbrain": {
        "enabled": true,
        "config": {
          "brainEmbeddingProvider": "ollama",
          "brainEmbeddingModel": "bge-large:latest",
          "brainWorkerMode": "child"
        }
      }
    }
  }
}
```

These defaults optimize bounded local serving and operator inspectability, not max-recall behavior.

Why these defaults:

- `brainEmbeddingProvider=ollama` keeps embeddings local
- `brainEmbeddingModel=bge-large:latest` matches the tested local default in this repo
- `brainWorkerMode=child` keeps learning off the serving process

## Optional teacher wiring

Teacher wiring is separate from brain activation. Making a model available in Ollama is not the same thing as telling OpenClawBrain to use it as the teacher. `BRAIN LOADED` proves the runtime hook is attached; teacher configuration is a separate status-tracked surface.

Teacher wiring is optional and not part of the current frozen public proof boundary.

A conceptual teacher configuration looks like this:

```json
{
  "plugins": {
    "entries": {
      "openclawbrain": {
        "enabled": true,
        "config": {
          "brainTeacherEnabled": true,
          "brainTeacherProvider": "ollama",
          "brainTeacherModel": "unsloth/Qwen3.5-27B-GGUF"
        }
      }
    }
  }
}
```

After setting a teacher, restart the gateway and verify the same home with:

```bash
openclawbrain status --openclaw-home ~/.openclaw --detailed
```

A correct teacher wiring should report `teacherConfigured=true`, the expected provider/model, and `teacherConfigError=null`.

## Embeddings

### Local Ollama

```json
{
  "brainEmbeddingProvider": "ollama",
  "brainEmbeddingModel": "bge-large:latest"
}
```

Default base URL:

```text
http://127.0.0.1:11434/v1
```

### Remote OpenAI-compatible endpoint

```json
{
  "brainEmbeddingProvider": "openai",
  "brainEmbeddingModel": "text-embedding-3-large",
  "brainEmbeddingBaseUrl": "https://your-endpoint.example/v1"
}
```

If the remote endpoint requires authentication, set `OPENCLAWBRAIN_EMBEDDING_API_KEY`.

## Important environment variables

| Variable | Description |
| --- | --- |
| `OPENCLAWBRAIN_ROOT` | Root for `state.db` and immutable packs |
| `OPENCLAWBRAIN_EMBEDDING_PROVIDER` | Embedding provider (`openai`, `openai-resp`, `ollama`) |
| `OPENCLAWBRAIN_EMBEDDING_MODEL` | Embedding model used for init, retrieval, and teach |
| `OPENCLAWBRAIN_EMBEDDING_BASE_URL` | Optional embeddings API base URL |
| `OPENCLAWBRAIN_EMBEDDING_API_KEY` | Optional auth for remote embedding endpoints |
| `OPENCLAWBRAIN_WORKER_MODE` | Worker mode; `child` is the supported serving boundary |
| `OPENCLAWBRAIN_SHADOW_MODE` | Record routing without injecting learned context |
| `OPENCLAWBRAIN_TEACHER_ENABLED` | Enable or disable the optional teacher lane |
| `OPENCLAWBRAIN_TEACHER_PROVIDER` | Teacher provider (`ollama`, `openai`, etc.) |
| `OPENCLAWBRAIN_TEACHER_MODEL` | Teacher model name selected for supervision |

## Optional diagnostics

Use the pinned `status --detailed` command above as the canonical public check. When you need to explain why proof has not advanced yet, these commands are also available:

```bash
openclawbrain status --openclaw-home ~/.openclaw --json
openclawbrain learn --openclaw-home ~/.openclaw --json
openclawbrain rollback --openclaw-home ~/.openclaw --dry-run
```

Use `learn --json` when you want a one-shot snapshot of what the learner scanned, whether it materialized a candidate pack, and whether a promotion occurred.

## Advanced operating surfaces

The CLI also exposes foreground and daemonized learner controls. These are optional operator tools, not part of the default public install and proof lane:

```bash
openclawbrain daemon status --activation-root ~/.openclawbrain/activation
openclawbrain history --openclaw-home ~/.openclaw --limit 20 --json
lcm-tui --db ~/.openclaw/lcm.db
```

This repo does not implement a standalone `openclawbrain context` command. Use `status` for runtime truth and `lcm-tui` Context View when you need to inspect the assembled hot context directly.

If you hit an operator seam, start with [Troubleshooting](operating/troubleshooting.md).

Use `--activation-root` only when you already know the exact boundary you want to inspect. For public install and proof, keep using `--openclaw-home`.
