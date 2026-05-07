# Getting Started with OpenClawBrain

This is the fastest honest path to a working OpenClawBrain `0.2.21` install.

## What you are installing

OpenClawBrain is a native OpenClaw plugin that gives agents local, inspectable memory. It remembers durable corrections, preferences, workflows, and context, then learns when that memory should affect a future turn.

The production route brain is `route-policy-v3`: a compact learned route function backed by redacted route frames, SQLite evidence, shadow decisions, replay/eval cases, calibration, gated promotion, and rollback lineage.

## Before you start

You need:

- OpenClaw `2026.5.2` or later
- a running OpenClaw gateway
- local Ollama or another local OpenAI-compatible endpoint for the full default learning path

## 1) Install or upgrade the plugin

Use the same command for a fresh install or an upgrade. `--force` replaces an older local copy.

```bash
openclaw plugins install clawhub:openclawbrain@0.2.21 --force
openclaw plugins enable openclawbrain
```

If ClawHub metadata is still propagating, install the GitHub release archive:

```bash
curl -L -o /tmp/openclawbrain-0.2.21.tgz \
  https://github.com/jonathangu/openclawbrain/releases/download/v0.2.21/openclawbrain-0.2.21.tgz
openclaw plugins install /tmp/openclawbrain-0.2.21.tgz --force
openclaw plugins enable openclawbrain
```

## 2) Use the recommended runtime path

`balanced` is the recommended default. It keeps prompt-time retrieval bounded and lets heavier distillation/replay/policy work happen after the turn.

```bash
openclaw config set plugins.entries.openclawbrain.config.enabled true --strict-json
openclaw config set plugins.entries.openclawbrain.config.mode '"balanced"' --strict-json
openclaw config set plugins.entries.openclawbrain.config.hooks.allowPromptContext true --strict-json
openclaw config set plugins.entries.openclawbrain.config.hooks.allowConversationAccess true --strict-json
openclaw config set plugins.entries.openclawbrain.config.hooks.allowToolObservation true --strict-json
openclaw config validate
openclaw gateway restart
```

## 3) Configure local learning models

OpenClawBrain uses the local LLM path for semantic updates: feedback distillation, route examples, route-policy snapshots, and learning. The LLM proposes structured JSON. Code validates, redacts, scopes, dedupes, thresholds, stores, replays, calibrates, promotes, and rolls back.

Default local path is Ollama through an OpenAI-compatible endpoint:

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

If local model calls are unavailable, the main agent keeps working. Known memories can still be searched and injected; background learning gets quieter or retries later.

## 4) Scope multiple agents

If you run multiple named agents/profiles, scope OpenClawBrain to all of them so each gets its own local graph:

```bash
openclaw config set plugins.entries.openclawbrain.config.scopes.agents '["main","pelican","bountiful"]' --strict-json
openclaw gateway restart
```

Use your real agent ids. Single-agent installs can skip this.

## 5) Verify it is live

Use runtime inspection, not just package metadata.

```bash
openclaw plugins inspect openclawbrain --runtime
openclaw doctor
```

What you want to see:

- the plugin is enabled, activated, and loaded
- hooks and HTTP routes are registered
- SQLite + FTS health is clean
- no `No active memory plugin` warning from `openclaw doctor`

HTTP plugin routes are authenticated on normal OpenClaw installs. Use the authenticated dashboard/client, or pass your gateway auth header when using curl.

## 6) Try it

Teach the agent something small and practical, for example:

```text
Use pnpm instead of npm in this repo.
```

Then inspect the plugin:

```text
/plugins/openclawbrain/proof?limit=10
/plugins/openclawbrain/search?query=pnpm&limit=10
/plugins/openclawbrain/graph?agentId=main&limit=10
/plugins/openclawbrain/route-policy
```

A later test/build turn should receive a small bounded context block, not a transcript dump. If route-policy-v3 does not have enough support, it should abstain and keep the prompt clean.

## Compatibility note

Older file-backed modes still exist:

- `proof-only`
- `conservative`
- `active`

They are supported for compatibility, but the product center is the local SQLite memory graph in `balanced` or `aggressive` mode plus route-policy-v3 learned routing.
