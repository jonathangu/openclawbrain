# Getting Started with OpenClawBrain

This is the fastest honest path to a working OpenClawBrain install.

## What you are installing

OpenClawBrain is a native OpenClaw plugin that helps an agent remember useful corrections, preferences, and prior work without bloating every prompt.

## Before you start

You need:

- OpenClaw `2026.4.29` or later
- a running OpenClaw gateway
- local Ollama or another local OpenAI-compatible endpoint **only if** you want automatic learning

## 1) Install the plugin

```bash
openclaw plugins install clawhub:openclawbrain
openclaw plugins enable openclawbrain
```

## 2) Turn on the main runtime path

```bash
openclaw config set plugins.entries.openclawbrain.config.enabled true --strict-json
openclaw config set plugins.entries.openclawbrain.config.mode '"balanced"' --strict-json
openclaw config set plugins.entries.openclawbrain.config.hooks.allowPromptContext true --strict-json
openclaw config set plugins.entries.openclawbrain.config.hooks.allowConversationAccess true --strict-json
openclaw config validate
openclaw gateway restart
```

`balanced` is the recommended default. It keeps the common path cheap and only does extra work when the turn looks like it needs help.

## 3) Optional: turn on automatic learning

If you want OpenClawBrain to turn corrections into memory automatically, point it at a local OpenAI-compatible endpoint. Local Ollama is the standard path.

```bash
ollama list

openclaw config set plugins.entries.openclawbrain.config.llm '{
  "enabled": true,
  "baseUrl": "http://127.0.0.1:11434/v1",
  "routeModel": "qwen3.5:9b",
  "plannerModel": "qwen3.5:9b",
  "feedbackModel": "qwen3.5:9b",
  "learningModel": "qwen3.5:9b"
}' --strict-json
openclaw config validate
openclaw gateway restart
```

If you skip this step, the plugin still loads and exposes its local proof, graph, health, and search surfaces. It just will not auto-distill fresh corrections.

## 4) Check that it is live

```bash
openclaw plugins inspect openclawbrain --json
curl http://127.0.0.1:18789/plugins/openclawbrain/status
curl http://127.0.0.1:18789/plugins/openclawbrain/doctor
```

What you want to see:

- the plugin is `enabled`
- the plugin is `activated`
- the plugin `status` is `loaded`
- the doctor route succeeds

## 5) Try it

Teach the agent something small and practical, for example:

```text
Use pnpm instead of npm in this repo.
```

Then inspect the plugin:

```bash
curl 'http://127.0.0.1:18789/plugins/openclawbrain/proof?limit=10'
curl 'http://127.0.0.1:18789/plugins/openclawbrain/search?query=pnpm&limit=10'
curl 'http://127.0.0.1:18789/plugins/openclawbrain/graph?limit=10'
```

## Compatibility note

Older file-backed modes still exist:

- `proof-only`
- `conservative`
- `active`

They are still supported, but they are not the main product story anymore. The main path is the local memory graph in `balanced` or `aggressive` mode.
