# Getting Started with OpenClawBrain

This is the fastest honest path to a working OpenClawBrain `0.2.30` install.

## What you are installing

OpenClawBrain is a native OpenClaw plugin that gives agents local, inspectable memory. It remembers durable corrections, preferences, workflows, and context, then learns when that memory should affect a future turn.

The production route brain is `route-policy-v3`: a compact learned route function backed by redacted route frames, SQLite evidence, shadow decisions, replay/eval cases, calibration, gated promotion, and rollback lineage.

The 0.2.30 runtime includes Memory Authority resolution, automatic Memory Graph Maintenance, and the real OpenClawBrain-owned Codex Telegram bridge. A retrieved memory is not automatically injected just because it is relevant. It must still be current enough, scoped correctly, safe to use, not superseded, not tombstoned, and compatible with the current user instruction. The graph maintenance layer then keeps long-lived memory healthy with dry-run proposals, deterministic duplicate/edge cleanup, tombstone-aware recapture checks, and proof. The Codex bridge reads local Codex thread messages, tails watched replies, and can send trusted local Telegram replies into exact Codex threads when enabled.

## Before you start

You need:

- OpenClaw `2026.5.2` or later
- a running OpenClaw gateway
- local Ollama or another local OpenAI-compatible endpoint for the full default learning path

## 1) Install or upgrade the plugin

Use the same command for a fresh install or an upgrade. `--force` replaces an older local copy.

```bash
openclaw plugins install clawhub:openclawbrain@0.2.30 --force
openclaw plugins enable openclawbrain
```

If ClawHub metadata is still propagating, install the GitHub release archive:

```bash
curl -L -o /tmp/openclawbrain-0.2.30.tgz \
  https://github.com/jonathangu/openclawbrain/releases/download/v0.2.30/openclawbrain-0.2.30.tgz
openclaw plugins install /tmp/openclawbrain-0.2.30.tgz --force
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

## 6) Try Codex continuity from Telegram/OpenClaw

OpenClawBrain owns these commands without modifying the OpenClaw core checkout:

```text
/brain codex status
/brain codex threads
/brain codex last --latest
/brain codex messages --latest --limit 5
/brain codex bind <thread-id>
/brain codex tail --bound
/brain codex reply Please continue and tell me when tests pass.
/brain codex handoff --bound
```

Recent-message copy is direct transport from Codex rollout JSONL, not an LLM summary. The public package keeps Telegram-to-Codex writes disabled by default; Jonathan's local profiles can enable them with trusted sender checks, exact bound threads, write allowlists, and high-risk refusal.

## 7) Try memory

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

## 8) Try graph maintenance

Graph maintenance is separate from Memory Authority. It curates the graph over time; it does not decide turn-level use.

By default, the service also runs graph maintenance passively in the background. The automatic loop creates dry-run/proposal records and may apply only deterministic low-risk repairs, such as exact duplicate consolidation, bad-edge retirement, and observation-only feedback rows. Anything that could change authority, scope, privacy, tombstones, or meaning remains review-gated.

```text
/brain graph health
/brain graph dry-run
/brain graph proposals
/brain graph explain <proposalId>
/brain graph apply <proposalId>
```

HTTP equivalents:

```text
/plugins/openclawbrain/graph/health
/plugins/openclawbrain/graph/dry-run
/plugins/openclawbrain/graph/proposals
/plugins/openclawbrain/graph/explain?proposalId=...
```

The explicit commands are still useful when you want to inspect what the background loop found, force a dry-run immediately, or approve/reject review-gated proposals.

## Compatibility note

Older file-backed modes still exist:

- `proof-only`
- `conservative`
- `active`

They are supported for compatibility, but the product center is the local SQLite memory graph in `balanced` or `aggressive` mode plus route-policy-v3 learned routing.
