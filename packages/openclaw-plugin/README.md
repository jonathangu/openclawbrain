# OpenClawBrain

**Evidence, not vibes, for agent memory.**

OpenClawBrain is a native [OpenClaw](https://docs.openclaw.ai) plugin for agents that should not need the same correction twice. It remembers durable corrections, preferences, workflows, and context, then brings back only the small slice that matters for the current turn.

Imagine an agent that understands “close it out” means evidence, not vibes. That is the product shape: less magic memory, more accountable local system.

> **LLM decides semantic meaning. Code enforces trust boundaries. SQLite stores the graph and evidence.**
>
> **Ollama proposes; code disposes.**

## What it does

- **Remembers durable lessons.** Corrections, preferences, workflows, and context become scoped local memory nodes instead of disappearing at the end of a session.
- **Keeps turn context small.** It does not dump your whole history into every turn. It retrieves candidates locally and attaches only a bounded XML memory slice when memory is likely to help.
- **Checks authority before use.** Retrieved memories can be attached, weakened, verified, confirmed, suppressed, or kept audit-only depending on staleness, scope, privacy, supersession, and current user guidance.
- **Stays local-first.** SQLite stores the graph and evidence. FTS5 powers local search. Raw transcript upload is hard-disabled.
- **Shows its work.** You can check status, run health checks, inspect proof events, search memory, view the graph, and review route decisions.
- **Learns on the standard local path.** OpenClawBrain points at local Ollama by default. Local models propose structured JSON; code validates, redacts, scopes, thresholds, and writes.

## Why it exists

Most agents are smart but forgetful. They can do good work inside one turn, then make the same mistake again tomorrow.

The usual fix is to stuff more text into every turn. That works badly. Context gets bloated, latency goes up, and the agent still lacks accountable memory.

OpenClawBrain takes a different approach:

1. route first: should memory participate?
2. search SQLite FTS locally
3. resolve whether retrieved memories still have authority
4. rank and attach only a small bounded memory block
5. record outcomes so memory can improve
6. show proof instead of asking for blind trust

## Current release

- **Current package release:** `0.2.26`
- **Recommended mode:** `balanced`
- **Requires:** OpenClaw `2026.5.2` or later
- **Live E2E proof:** turn → capture audit → strict distillation/storage → SQLite/FTS retrieval → authority resolution → bounded memory context
- **Current loop:** first-class OpenClaw memory registration, v3 production route learning, Memory Authority decisions, Codex continuity status/watch/handoff surfaces, conservative fallback, aggressive audited capture, strict scoped storage, sparse context use

## Install

```bash
openclaw plugins install clawhub:openclawbrain@0.2.26
openclaw plugins enable openclawbrain
openclaw gateway restart
```

If ClawHub is rate-limited or package metadata is still propagating, install the release archive instead:

```bash
curl -L -o /tmp/openclawbrain-0.2.26.tgz \
  https://github.com/jonathangu/openclawbrain/releases/download/v0.2.26/openclawbrain-0.2.26.tgz
openclaw plugins install /tmp/openclawbrain-0.2.26.tgz --force
openclaw plugins enable openclawbrain
openclaw gateway restart
```

If you run multiple named agents/profiles, scope OpenClawBrain to all of them so each gets its own local graph:

```bash
openclaw config set plugins.entries.openclawbrain.config.scopes.agents '["main","pelican","bountiful"]' --strict-json
openclaw gateway restart
```

## Default local setup

The default OpenClawBrain setup is aimed at the full local path: balanced mode, conversation/tool hooks, local SQLite, and local Ollama on `127.0.0.1`.

```bash
openclaw config set plugins.entries.openclawbrain.config.enabled true --strict-json
openclaw config set plugins.entries.openclawbrain.config.mode '"balanced"' --strict-json
openclaw config set plugins.entries.openclawbrain.config.hooks.allowPromptContext true --strict-json
openclaw config set plugins.entries.openclawbrain.config.hooks.allowConversationAccess true --strict-json
openclaw config set plugins.entries.openclawbrain.config.hooks.allowToolObservation true --strict-json
openclaw config validate
openclaw gateway restart
```

## Default local learning models

The default route/planner/feedback/learning model is `qwen2.5:32b-instruct`, reached through a local OpenAI-compatible endpoint such as Ollama.

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

If local model calls are unavailable, the main agent keeps working. Known memories can still be searched and used as bounded context; background learning simply gets quieter or retries later.

## Check that it is live

Use `inspect` and `doctor` for runtime truth. The HTTP routes use gateway auth on normal OpenClaw installs, so access them through your authenticated dashboard/client or pass your gateway auth header when using curl.

```bash
openclaw plugins inspect openclawbrain --runtime
openclaw doctor

# Authenticated plugin routes:
# /plugins/openclawbrain/status
# /plugins/openclawbrain/doctor
# /plugins/openclawbrain/proof?limit=10
# /plugins/openclawbrain/search?query=pnpm&limit=10
# /plugins/openclawbrain/codex/status
```

## Codex continuity bridge

`0.2.26` also adds an OpenClawBrain-owned Codex continuity bridge. It does not patch OpenClaw core. The public-safe default reads local Codex SQLite in read-only mode with an explicit stale label; host-provided app-server readers can be enabled later without bundling shell/process control inside the package. It exposes quiet operator surfaces:

```text
/brain codex status
/brain codex threads [filter]
/brain codex watch [thread-id|--latest]
/brain codex handoff [thread-id]
```

Telegram-to-Codex writes stay disabled by default. `/brain codex goal` and `/brain codex steer` refuse unless a later write path is explicitly feature-flagged, sender-gated, repo-allowlisted, provenance-tagged, risk-classified, and confirmed.

For local development or Jonathan's personal Mac, update the external OpenClawBrain extension without dirtying the OpenClaw checkout:

```bash
pnpm install:local-openclaw
openclaw gateway restart
```

To update every local OpenClaw home that already has OpenClawBrain installed, use:

```bash
pnpm install:local-openclaw:all
```

## What you can inspect

| Endpoint | What it shows |
|---|---|
| `/plugins/openclawbrain/status` | whether the plugin is enabled, loaded, and how the runtime is behaving |
| `/plugins/openclawbrain/doctor` | SQLite + FTS health under the current Node runtime |
| `/plugins/openclawbrain/proof?limit=20` | recent redacted proof, route, and memory-context events |
| `/plugins/openclawbrain/graph?limit=50` | redacted memory nodes and memory edges |
| `/plugins/openclawbrain/learn?limit=50` | route examples and current learning state |
| `/plugins/openclawbrain/search?query=...&limit=20` | local memory search |
| `/plugins/openclawbrain/audit?limit=20` | recent capture/store/reject decisions and rejection distribution |
| `/plugins/openclawbrain/explain-last` | compact postmortem for the latest route and memory authority decision |
| `/plugins/openclawbrain/codex/status` | read-only Codex continuity status with stale SQLite source labeling, plus host app-server readers when configured |
| `/plugins/openclawbrain/codex/threads` | recent local Codex threads and visible goals |
| `/plugins/openclawbrain/codex/handoff` | evidence-separated Codex handoff brief |
| `/plugins/openclawbrain/codex/watches` | bridge-local watch registry and redacted audit events |

## Privacy and safety

- Local learning defaults to the local Ollama path
- Raw transcript upload is hard-disabled
- Redaction happens before storage and before model use
- The model does not write directly to memory
- Stale, superseded, private, tombstoned, or locally overridden memories do not silently become guidance
- Plugin failure does not block the main agent
- Local-first by default

## More

- [How it works](https://openclawbrain.ai/how-it-works/)
- [Install guide](https://openclawbrain.ai/install/)
- [Proof commands](https://openclawbrain.ai/proof/)
- [GitHub repo](https://github.com/jonathangu/openclawbrain)

## License

MIT
