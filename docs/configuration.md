# Configuration guide

OpenClawBrain works with its default install path. Most operators only need to install it, restart the gateway, verify the selected OpenClaw home, and capture a proof bundle when they need durable operator evidence.

## Default operator flow

Keep the same `--openclaw-home` value through the whole operator flow. The public lane stays pinned to one OpenClaw home.

```bash
openclaw plugins install @openclawbrain/openclaw
npx @openclawbrain/cli install --openclaw-home ~/.openclaw
openclaw gateway restart
npx @openclawbrain/cli status --openclaw-home ~/.openclaw --detailed
npx @openclawbrain/cli proof --openclaw-home ~/.openclaw --skip-install --skip-restart
```

`status --detailed` is the quick verify surface. `proof` writes `summary.md`, `steps.json`, `verdict.json`, raw step logs, and proof pointers under one bundle directory.

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

## Optional diagnostics

Use the pinned `status --detailed` command above as the canonical public check. When you need to explain why proof has not advanced yet, these commands are also available:

```bash
npx @openclawbrain/cli status --openclaw-home ~/.openclaw --json
npx @openclawbrain/cli learn --openclaw-home ~/.openclaw --json
npx @openclawbrain/cli rollback --openclaw-home ~/.openclaw --dry-run
```

Use `learn --json` when you want a one-shot snapshot of what the learner scanned, whether it materialized a candidate pack, and whether a promotion occurred.

## Advanced operating surfaces

The CLI also exposes foreground and daemonized learner controls. These are optional operator tools, not part of the default public install and proof lane:

```bash
npx @openclawbrain/cli daemon status --activation-root ~/.openclawbrain/activation
npx @openclawbrain/cli history --openclaw-home ~/.openclaw --limit 20 --json
npx @openclawbrain/cli context "How should I answer this?" --openclaw-home ~/.openclaw
```

If you hit an operator seam, start with [Troubleshooting](operating/troubleshooting.md).

Use `--activation-root` only when you already know the exact boundary you want to inspect. For public install and proof, keep using `--openclaw-home`.
