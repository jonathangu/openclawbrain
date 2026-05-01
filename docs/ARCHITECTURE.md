# Architecture

OpenClawBrain v0.2 is a native OpenClaw plugin that keeps a **local SQLite memory graph**, learns from outcomes, and injects only bounded context back into the prompt.

## Runtime shape

The important modules are:

```text
packages/openclaw-plugin/src/
├── index.ts              # plugin entry, hooks, routes, memory supplements
├── config.ts             # defaults + config normalization
├── memory-store.ts       # SQLite schema, FTS5, graph storage, proofs, jobs
├── job-queue.ts          # local async work queue
├── feedback-distiller.ts # structured JSON feedback capture
├── memory-operations.ts  # validates/applies distillation output
├── route-fn.ts           # local/cached route planning
├── latency-controller.ts # decides whether sync planner is allowed
├── memory-planner.ts     # one-call planner for ambiguous/high-signal turns
├── context-selector.ts   # selects and formats bounded context
├── learning.ts           # outcome learning, freshness decay, pruning
├── route-learning.ts     # route examples + active policy snapshot
├── search.ts             # graph/learn/search payloads + memory supplements
├── proof-store.ts        # proof mirror and status persistence
└── context-files.ts      # legacy activation-file compatibility path
```

## Two runtime lanes

### 1) v0.2 memory-graph lane

Used when:

- `config.mode` is `balanced` or `aggressive`
- OpenClaw's prompt-augmentation hook is allowed

Flow:

1. `before_prompt_build` creates a redacted turn packet.
2. `RouteFn` makes an initial local plan from cached policy + retrieval hints.
3. `LatencyController` decides whether to stay local or allow one bounded `MemoryPlanner` call.
4. Candidate memories come from local SQLite/FTS.
5. `ContextSelector` picks a small memory set and formats bounded prompt text.
6. Route decisions, injections, and proofs are recorded.
7. `agent_end`, `after_tool_call`, and the background service update outcomes and policy snapshots.

### 2) legacy compatibility lane

Used when:

- `config.mode` is `proof-only`, `conservative`, or `active`

This preserves the older activation-file behavior by reading:

- `context.md`
- `corrections.md`
- `tool-guidance.md`

That lane remains for backward compatibility, but it is no longer the product center.

## Latency model

OpenClawBrain is designed so memory does **not** add a blocking model call to every turn.

The code follows the four-tier plan from `FINAL_PLAN.md`:

- **Tier 0** — no extra sync LLM call, local cache/policy/retrieval only
- **Tier 1** — cached learned route + local retrieval
- **Tier 2** — one bounded `MemoryPlanner` call for ambiguous/high-signal turns
- **Tier 3** — async distillation, route learning, maintenance, pruning

## Storage model

`memory-store.ts` persists:

- memory nodes
- memory edges
- route decisions
- route examples
- policy snapshots
- injection records
- distillation runs
- background jobs
- proof events

Search uses SQLite FTS5 plus graph expansion and scope filters.

## LLM boundary

Current runtime support is intentionally narrow:

- local Ollama or another localhost OpenAI-compatible endpoint

The LLM never writes directly to storage. It only returns structured JSON proposals. `memory-operations.ts` validates, redacts, scopes, dedupes, and applies them.

## Hooks and surfaces

Registered hooks:

- `before_prompt_build`
- `agent_end` (when conversation access is allowed)
- `after_tool_call` (when tool observation is allowed)
- `model_call_started`
- `model_call_ended`
- `gateway_start`
- `gateway_stop`

Registered HTTP routes:

- `/plugins/openclawbrain/status`
- `/plugins/openclawbrain/proof`
- `/plugins/openclawbrain/graph`
- `/plugins/openclawbrain/learn`
- `/plugins/openclawbrain/search`

Registered additive memory surfaces:

- `registerMemoryPromptSupplement(...)`
- `registerMemoryCorpusSupplement(...)`

## Privacy and trust boundaries

- raw transcript storage stays off
- redaction happens before persistence
- redaction before LLM is on by default
- proof rows and status surfaces remain inspectable
- plugin failure should not block the main OpenClaw run

## Product invariant

The design center is unchanged:

> **LLM decides semantic meaning. Code enforces trust boundaries. SQLite stores the graph and evidence.**
