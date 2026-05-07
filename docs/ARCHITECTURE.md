# Architecture

OpenClawBrain `0.2.21` is a native OpenClaw plugin that keeps a **local SQLite memory graph**, learns from outcomes, and injects only bounded context back into the prompt.

The core invariant is unchanged:

> **LLM decides semantic meaning. Code enforces trust boundaries. SQLite stores the graph and evidence.**

## Runtime shape

The important runtime modules are:

```text
packages/openclaw-plugin/src/
├── index.ts                         # plugin entry, hooks, routes, memory supplements
├── config.ts                        # defaults + config normalization
├── memory-store.ts                  # SQLite schema, FTS5, graph storage, proofs, jobs
├── route-fn.ts                      # v3-first route decision path with fallback handling
├── route-policy-v2.ts               # legacy learned policy fallback/rollback path
├── route-policy-v3*.ts              # v3 snapshots, normalization, calibration, eval, routing modes
├── route-policy-v3-shadow.ts        # shadow decisions and live would-have-routed evidence
├── route-policy-v3-reporting.ts     # candidate reports, activation, rollback summaries
├── route-policy-v3-datasets.ts      # replay/eval/calibration datasets
├── route-policy-v3-promotion.ts     # gated promotion and cooldown controls
├── feedback-distiller.ts            # structured feedback capture
├── memory-operations.ts             # validates/applies distillation output
├── memory-planner.ts                # bounded planner for ambiguous/high-signal turns
├── context-selector.ts              # selects and formats bounded context
├── learning.ts                      # outcome learning, freshness decay, pruning
├── search.ts                        # graph/learn/search payloads + memory supplements
├── proof-store.ts                   # proof mirror and status persistence
└── context-files.ts                 # legacy activation-file compatibility path
```

## Production route path

`route-policy-v3` is the production route brain. It is a compact learned `route_fn`, not a memory dump and not a prompt asking the model to remember better.

```text
active valid route-policy-v3 snapshot
  -> calibrated family-aware match
  -> bounded context injection + proof

v3 abstains / no safe match / invalid snapshot
  -> route-policy-v2 fallback

v2 misses or rollback required
  -> legacy heuristics as last resort
```

Abstention is intentional. If the system lacks support, confidence, or a safe match, it keeps the prompt clean.

## Learning loop

Normal work produces route-learning evidence:

```text
corrections, outcomes, route misses, handoffs
  -> redacted route frames and memory nodes
  -> SQLite evidence graph
  -> shadow decisions on live traffic
  -> replayable eval cases and labels
  -> action-family calibration
  -> candidate route-policy-v3 snapshots
  -> gated promotion or rollback
  -> bounded context injection or abstention
```

The v3 warehouse includes:

- `route_frames_v3`
- `route_shadow_decisions_v3`
- `route_calibration_examples_v3`
- `route_eval_cases_v3` and labels
- `route_action_family_stats_v3`
- `route_policy_candidate_reports_v3`

Promotion is gated by schema validation, replay, calibration, cold-start support, sync-budget checks, cooldowns, and rollback lineage.

## Runtime lanes

### 1) v0.2 memory-graph lane

Used when:

- `config.mode` is `balanced` or `aggressive`
- OpenClaw's prompt-augmentation hook is allowed

Flow:

1. `before_prompt_build` creates a redacted turn packet.
2. `RouteFn` asks active `route-policy-v3` first.
3. If v3 abstains, v2 and legacy heuristics are fallback paths.
4. Candidate memories come from local SQLite/FTS plus graph expansion.
5. `ContextSelector` picks a small memory set and formats bounded prompt text.
6. Route decisions, injections, and proofs are recorded.
7. `agent_end`, `after_tool_call`, and the background service update outcomes, route frames, shadow decisions, replay cases, candidate reports, and policy snapshots.

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
- **Tier 3** — async distillation, route learning, maintenance, pruning, replay, calibration, and promotion

## Storage model

`memory-store.ts` persists memory nodes/edges, route decisions, injection records, distillation runs, background jobs, proof events, and the v3 route-learning warehouse. Search uses SQLite FTS5 plus graph expansion and scope filters.

Raw transcript storage remains off; route-learning evidence is compact, redacted, scoped, and inspectable.

## LLM boundary

Current runtime support is intentionally narrow:

- local Ollama or another localhost OpenAI-compatible endpoint by default
- remote LLM paths are explicit/optional, not required for safe runtime operation

The LLM never writes directly to storage and never owns production routing. It only returns structured JSON proposals. Code validates, redacts, scopes, dedupes, thresholds, stores, replays, calibrates, promotes, and rolls back.

## Hooks and surfaces

Registered hooks include:

- `before_prompt_build`
- `agent_end`
- `after_tool_call`
- `model_call_started`
- `model_call_ended`
- `gateway_start`
- `gateway_stop`

Registered HTTP routes include:

- `/plugins/openclawbrain/status`
- `/plugins/openclawbrain/doctor`
- `/plugins/openclawbrain/proof`
- `/plugins/openclawbrain/graph`
- `/plugins/openclawbrain/learn`
- `/plugins/openclawbrain/search`
- `/plugins/openclawbrain/route-teacher`
- `/plugins/openclawbrain/route-counterfactuals`
- `/plugins/openclawbrain/route-policy`
- `/plugins/openclawbrain/audit`
- `/plugins/openclawbrain/explain-last`

Registered additive memory surfaces:

- `registerMemoryPromptSupplement(...)`
- `registerMemoryCorpusSupplement(...)`

## Privacy and trust boundaries

- raw transcript storage stays off
- redaction happens before persistence and before model use
- the model does not write directly to memory
- SQLite stores the graph and evidence locally
- proof rows and route-policy surfaces remain inspectable
- plugin failure should not block the main OpenClaw run
