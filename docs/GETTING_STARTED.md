# Getting Started

This guide gets the **v0.2 memory-graph path** running.

## Prerequisites

- OpenClaw `2026.4.29` or later
- A local or remote OpenAI-compatible JSON model endpoint if you want automatic semantic capture

## 1) Install the plugin

```bash
openclaw plugins install clawhub:openclawbrain
openclaw plugins enable openclawbrain
```

## 2) Enable the v0.2 runtime path

```bash
openclaw config set plugins.entries.openclawbrain.config.enabled true --strict-json
openclaw config set plugins.entries.openclawbrain.config.mode '"balanced"' --strict-json
openclaw config set plugins.entries.openclawbrain.hooks.allowPromptInjection true --strict-json
openclaw config set plugins.entries.openclawbrain.hooks.allowConversationAccess true --strict-json
```

`balanced` is the recommended v0.2 default. It keeps most turns on the no-extra-LLM path and escalates only when the turn is ambiguous or high-signal.

## 3) Configure a structured JSON model endpoint

OpenClawBrain's automatic capture/learning path currently uses a local or remote **OpenAI-compatible** endpoint. The intended privacy-first setup is a local server:

```bash
openclaw config set plugins.entries.openclawbrain.config.llm '{
  "enabled": true,
  "provider": "local",
  "baseUrl": "http://127.0.0.1:11434/v1",
  "plannerModel": "your-local-model",
  "feedbackModel": "your-local-model",
  "learningModel": "your-local-model"
}' --strict-json
openclaw config validate
openclaw gateway restart
```

If you skip this step, OpenClawBrain can still run its proof/search/status surfaces and legacy compatibility modes, but it will not auto-distill new corrections.

For a remote OpenAI-compatible endpoint, set the gateway environment variable first:

```bash
export OPENCLAWBRAIN_LLM_API_KEY=your_api_key_here
```

## 4) Verify that the plugin is live

```bash
openclaw plugins inspect openclawbrain --json
curl http://127.0.0.1:18789/plugins/openclawbrain/status
```

You should see:

- `pluginVersion: "0.2.2"`
- `mode: "balanced"`
- routing / memory / latency sections in the status payload

## 5) Exercise the memory path

Try a correction like:

```text
Actually, use pnpm instead of npm for this repo.
```

Then inspect:

```bash
curl 'http://127.0.0.1:18789/plugins/openclawbrain/proof?limit=10'
curl 'http://127.0.0.1:18789/plugins/openclawbrain/search?query=pnpm&limit=10'
curl 'http://127.0.0.1:18789/plugins/openclawbrain/graph?limit=10'
```

## Compatibility note

The package still supports the older file-backed modes:

- `proof-only`
- `conservative`
- `active`

Those modes read `context.md`, `corrections.md`, and `tool-guidance.md` under the activation root. They remain available for compatibility, but they are **not** the v0.2 product vision.
