# OpenClawBrain

**Latency-safe memory that learns from your corrections.** OpenClawBrain is a native [OpenClaw](https://docs.openclaw.ai) plugin that stores redacted memory nodes in local SQLite, retrieves only the few memories that matter, and keeps full proof/audit surfaces.

## What v0.2 does

- **Corrections can stick automatically.** With LLM distillation enabled, user corrections and preferences become scoped memory nodes instead of handwritten notes.
- **Routing stays latency-safe.** Most turns use cached policy + local SQLite retrieval. Only ambiguous or high-signal turns pay for one bounded planner call.
- **Learning happens in the background.** `agent_end`, tool observations, and background jobs update outcomes, route examples, and policy snapshots.
- **Prompts stay bounded.** Candidate retrieval happens locally; only 0-5 selected memories are injected.
- **Everything is inspectable.** Status, proof, graph, learning, and search routes expose the current state.

## Current truth

OpenClawBrain v0.2 is the memory-graph runtime described in [`FINAL_PLAN.md`](FINAL_PLAN.md), with SQLite self-checks and fallback and 54 passing plugin tests. Current package release: **v0.2.7**.

The package still keeps **legacy file-backed compatibility modes** (`proof-only`, `conservative`, `active`) for users who want the older activation-file path. The **v0.2 path** is `mode: "balanced"` or `"aggressive"`.

## Install

Requires OpenClaw `2026.4.29` or later.

```bash
openclaw plugins install clawhub:openclawbrain
openclaw plugins enable openclawbrain
```

## Configure the v0.2 path

Minimum runtime config:

```bash
openclaw config set plugins.entries.openclawbrain.config.enabled true --strict-json
openclaw config set plugins.entries.openclawbrain.config.mode '"balanced"' --strict-json
openclaw config set plugins.entries.openclawbrain.hooks.allowPromptInjection true --strict-json
openclaw config set plugins.entries.openclawbrain.hooks.allowConversationAccess true --strict-json
openclaw config validate
openclaw gateway restart
```

To enable automatic semantic distillation and same-turn planning, point the plugin at a structured JSON model endpoint. **Local Ollama is the standard path**. Set `baseUrl` to your local Ollama OpenAI-compatible v1 endpoint:

```bash
ollama list

openclaw config set plugins.entries.openclawbrain.config.llm '{
  "enabled": true,
  "baseUrl": "<your local Ollama OpenAI-compatible v1 endpoint>",
  "routeModel": "qwen3.5:9b",
  "plannerModel": "qwen3.5:9b",
  "feedbackModel": "qwen3.5:9b",
  "learningModel": "qwen3.5:9b"
}' --strict-json
openclaw config validate
openclaw gateway restart
```

If `llm.enabled` is left `false`, the plugin still exposes proof/search/graph/status and can use legacy activation-file modes, but **automatic correction capture and LLM route learning are not active**.

Privacy note: background packets, route planning inputs, and queued distillation jobs keep only **redacted user-message summaries plus hashes**, not raw user-message text.

## Verify

```bash
openclaw plugins inspect openclawbrain --json
curl http://127.0.0.1:18789/plugins/openclawbrain/status
curl http://127.0.0.1:18789/plugins/openclawbrain/doctor
curl http://127.0.0.1:18789/plugins/openclawbrain/proof?limit=10
curl 'http://127.0.0.1:18789/plugins/openclawbrain/graph?limit=10'
curl 'http://127.0.0.1:18789/plugins/openclawbrain/learn?limit=10'
curl 'http://127.0.0.1:18789/plugins/openclawbrain/search?query=pnpm&limit=10'
```

## Routes

| Endpoint | Description |
|---|---|
| `/plugins/openclawbrain/status` | Current plugin config, memory counts, routing stats, latency counters |
| `/plugins/openclawbrain/doctor` | SQLite driver and FTS5 smoke check under the running Node runtime |
| `/plugins/openclawbrain/proof?limit=20` | Recent redacted proof and route events |
| `/plugins/openclawbrain/graph?limit=50` | Redacted memory nodes and edges |
| `/plugins/openclawbrain/learn?limit=50` | Route examples and active policy snapshot |
| `/plugins/openclawbrain/search?query=...&limit=20` | Local memory search over SQLite/FTS |

## Architecture

Core runtime pieces live in `packages/openclaw-plugin/src/`:

- `memory-store.ts` — SQLite schema, FTS5 search, graph storage, route decisions, injections, proofs, jobs
- `feedback-distiller.ts` — structured JSON memory distillation
- `route-fn.ts` + `latency-controller.ts` — cache-aware routing and sync-budget control
- `memory-planner.ts` — single-call fast planner for ambiguous/high-signal turns
- `context-selector.ts` — bounded memory selection and formatting
- `learning.ts` + `route-learning.ts` — background outcomes, memory scoring, and policy snapshots
- `search.ts` — status/graph/learn/search payloads plus additive memory supplements

The design target is simple:

> **LLM decides semantic meaning. Code enforces trust boundaries. SQLite stores the graph and evidence.**

See [`FINAL_PLAN.md`](FINAL_PLAN.md), [`VISION.md`](VISION.md), and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full model.

## Safety

- Off by default
- Raw transcript storage hard-disabled (`rawTranscriptUpload: false`)
- Redaction before store and before LLM by default
- Fail-closed when prompt augmentation or conversation access is disabled
- Plugin failure does not block the main agent

## Development

```bash
pnpm install
pnpm --dir packages/openclaw-plugin check
pnpm --dir packages/openclaw-plugin build
pnpm --dir packages/openclaw-plugin test
```

Current gate: `pnpm --dir packages/openclaw-plugin test` → **54/54 pass**.

## Publish

```bash
clawhub publish packages/openclaw-plugin \
  --slug openclawbrain \
  --name "OpenClawBrain" \
  --version 0.2.7 \
  --changelog "Add SQLite self-checks and fallback and scanner-safe reliability metadata."
```

## License

MIT
