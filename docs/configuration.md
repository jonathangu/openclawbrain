# Configuration guide

OpenClawBrain works with its default install path. Most operators only need `openclawbrain install --openclaw-home <path>`, a gateway restart, a status check for that same home, and proof only when they need durable operator evidence. The live path serves promoted packs so useful context stays bounded while learning stays off the response path.

## Default operator flow

Keep the same `--openclaw-home` value through the whole operator flow. The public lane stays pinned to one OpenClaw home.

```bash
openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
openclawbrain status --openclaw-home ~/.openclaw --detailed
```

`status --detailed` is the quick verify surface.

When you need durable operator evidence today, run:

```bash
openclawbrain proof --openclaw-home ~/.openclaw
```

The intended canonical lane is the same install command with optional `--proof`. Until that flag lands cleanly across the operator surfaces, proof stays a separate follow-up command. `proof` writes `summary.md`, `steps.json`, `verdict.json`, raw step logs, and proof pointers under one bundle directory.

If you only need the minimal happy path, stop there and use [Quick start](getting-started/quick-start.md).

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

Why these defaults:

- `brainEmbeddingProvider=ollama` keeps embeddings local
- `brainEmbeddingModel=bge-large:latest` matches the tested local default in this repo
- `brainWorkerMode=child` keeps learning off the serving process

## Optional teacher wiring

Teacher wiring is separate from brain activation. Making a model available in Ollama is not the same thing as telling OpenClawBrain to use it as the teacher. `BRAIN LOADED` proves the runtime hook is attached; teacher configuration is a separate status-tracked surface.

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
openclawbrain context "How should I answer this?" --openclaw-home ~/.openclaw
```

If you hit an operator seam, start with [Troubleshooting](operating/troubleshooting.md).

Use `--activation-root` only when you already know the exact boundary you want to inspect. For public install and proof, keep using `--openclaw-home`.
