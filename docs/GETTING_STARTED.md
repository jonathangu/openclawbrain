# Getting Started with OpenClawBrain

This is the fastest honest path to a working OpenClawBrain install.

## What you are installing

OpenClawBrain is a native OpenClaw plugin that helps an agent remember useful corrections, preferences, and prior work without bloating every prompt.

## Before you start

You need:

- OpenClaw `2026.4.29` or later
- a running OpenClaw gateway
- local Ollama or another local OpenAI-compatible endpoint for the full default learning path

## 1) Install the plugin

```bash
openclaw plugins install clawhub:openclawbrain
openclaw plugins enable openclawbrain
```

## 2) Use the default runtime path

The default runtime path is already the full local setup: balanced mode, prompt/conversation/tool hooks on, and local Ollama at `127.0.0.1:11434/v1`.

```bash
openclaw config validate
openclaw gateway restart
```

`balanced` is the recommended default. It keeps the common path cheap and only does extra work when the turn looks like it needs help.

## 3) Default automatic learning models

If you want to set the local model block explicitly, use a local OpenAI-compatible endpoint. Local Ollama is the standard path.

```bash
ollama list

openclaw config set plugins.entries.openclawbrain.config.llm '{
  "enabled": true,
  "baseUrl": "http://127.0.0.1:11434/v1",
  "routeModel": "qwen2.5:32b-instruct",
  "plannerModel": "qwen2.5:32b-instruct",
  "feedbackModel": "qwen2.5:32b-instruct",
  "learningModel": "qwen2.5:32b-instruct"
}' --strict-json
openclaw config validate
openclaw gateway restart
```

If you deliberately disable this step, the plugin still loads and exposes its local proof, graph, health, and search surfaces. It just will not auto-distill fresh corrections.

## 4) Check that it is live

If you have multiple agents, add each configured agent id to `scopes.agents`; each agent gets its own activation root and memory graph.

```bash
openclaw config set plugins.entries.openclawbrain.config.scopes.agents '["main","pelican","bountiful"]' --strict-json
openclaw gateway restart
```

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
curl 'http://127.0.0.1:18789/plugins/openclawbrain/graph?agentId=main&limit=10'
```

## Compatibility note

Older file-backed modes still exist:

- `proof-only`
- `conservative`
- `active`

They are still supported, but they are not the main product story anymore. The main path is the local memory graph in `balanced` or `aggressive` mode.
