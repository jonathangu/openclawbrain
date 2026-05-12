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
- **Maintains the graph.** Dry-run proposals find exact duplicates, bad edges, stale high-authority memories, scoped exception candidates, and tombstone recapture risks before long-lived memory drifts.
- **Bridges Codex to Telegram.** It can show recent Codex UI thread messages, follow one thread briefly, attach passive notes, send explicit trusted actions through Codex app-server, steer active work, and detach cleanly when Telegram should go quiet.
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

- **Current package release:** `0.2.33`
- **Recommended mode:** `balanced`
- **Requires:** OpenClaw `2026.5.2` or later
- **Live E2E proof:** turn → capture audit → strict distillation/storage → SQLite/FTS retrieval → authority resolution → bounded memory context
- **Current loop:** first-class OpenClaw memory registration, v3 production route learning, Memory Authority decisions, Memory Graph Maintenance proposals, Codex Telegram transcript/watch/reply/steer surfaces, conservative fallback, aggressive audited capture, strict scoped storage, sparse context use

## Install

```bash
openclaw plugins install clawhub:openclawbrain@0.2.33 --force
openclaw plugins enable openclawbrain
openclaw gateway restart
```

If ClawHub is rate-limited or package metadata is still propagating, install the release archive instead:

```bash
curl -L -o /tmp/openclawbrain-0.2.33.tgz \
  https://github.com/jonathangu/openclawbrain/releases/download/v0.2.33/openclawbrain-0.2.33.tgz
openclaw plugins install /tmp/openclawbrain-0.2.33.tgz --force
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

`0.2.33` makes the OpenClawBrain-owned Codex continuity bridge usable as a lightweight Telegram operator layer. It does not patch OpenClaw core. It reads local Codex SQLite for thread metadata, follows `threads.rollout_path` into rollout JSONL, copies final user/assistant message text directly without LLM summarization, attaches passive notes without starting Codex, sends explicit trusted actions through Codex app-server `thread/resume` plus `turn/start`, can steer active Codex turns with `turn/steer`, and can detach a chat from all matching bindings/watches.

```text
/brain codex status
/brain codex doctor
/brain codex threads [filter]
/brain codex messages [thread-id|--latest|--bound] [--limit 5] [--role assistant|user|all]
/brain codex last [thread-id|--latest|--bound]
/brain codex bind <thread-id>
/brain codex binding
/brain codex unbind
/brain codex detach
/brain codex tail [thread-id|--latest|--bound]
/brain codex watch [thread-id|--latest|--bound] --messages
/brain codex watches
/brain codex unwatch <watch-id|thread-id|latest|all>
/brain codex note <message>
/brain codex notes
/brain codex act [--with-notes] <message>
/brain codex reply <message>
/brain codex send <thread-id|--bound> <message>
/brain codex steer [thread-id|--bound] <message>
/brain codex handoff [thread-id|--latest|--bound]
```

Use `note` for context that should not make Codex act. Use `act`, `reply`, or `send` only when the operator intends to start a real Codex turn that may edit files, run tools, or request approvals under Codex's normal sandbox behavior. Plain watches stay quiet by default; `tail` is the explicit assistant-message forwarding mode. `detach` is the escape hatch.

Telegram-to-Codex writes and steering stay disabled by default in the public package. Local trusted profiles can enable the happy path with `enableTelegramWrites=true`, `enableTelegramSteer=true`, trusted sender/chat, and write allowlists. The bridge refuses `--latest` writes, refuses high-risk publish/deploy/delete/secrets/full-access wording by default, and never bypasses Codex sandbox or approval behavior.

The public package talks to Codex over a localhost WebSocket app-server endpoint, not by spawning shell commands. For a trusted local setup, run:

```bash
codex app-server --listen ws://127.0.0.1:53177
```

Then set `codexBridge.appServerUrl="ws://127.0.0.1:53177"` in the OpenClawBrain profile config.

## Memory graph maintenance

Memory Authority decides turn-level use. Graph Maintenance curates long-term graph evolution. It now runs passively as part of the OpenClawBrain service: the timer creates dry-run/proposal records automatically, applies only deterministic low-risk cleanup when configured, and leaves stale authority, privacy, tombstone, scope, and semantic changes for explicit review.

```text
/brain graph health
/brain graph dry-run
/brain graph proposals
/brain graph apply <proposalId>
/brain graph reject <proposalId>
/brain graph stale
/brain graph clusters
/brain graph tombstones
/brain graph explain <proposalId>
```

The safety boundary is intentional: graph maintenance can provide features and proposals, but `MemoryAuthorityResolver` still recomputes whether a memory can influence the current turn. Connectivity is not authority, behavioral edges are not truth evidence, and tombstoned content cannot be revived by merge, proof, proposal, or LLM distillation.

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
| `/plugins/openclawbrain/graph/health` | graph health metrics |
| `/plugins/openclawbrain/graph/dry-run` | redacted maintenance proposals without mutation |
| `/plugins/openclawbrain/graph/proposals` | graph maintenance proposal list |
| `/plugins/openclawbrain/graph/apply?proposalId=...` | apply low-risk deterministic proposals |
| `/plugins/openclawbrain/graph/explain?proposalId=...` | explain proposal evidence and safety boundary |
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
