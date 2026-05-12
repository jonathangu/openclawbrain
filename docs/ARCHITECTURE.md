# Architecture

OpenClawBrain `0.2.32` is a native OpenClaw plugin that keeps a **local SQLite memory graph**, learns from outcomes, injects only bounded context back into the prompt, exposes a quiet Codex continuity bridge, and maintains graph health without patching OpenClaw core.

The central runtime addition is the **Memory Authority** layer. Retrieval can over-include candidates, but a memory is not allowed to influence the turn until authority resolution checks freshness, scope, privacy, supersession, current instructions, validation strategy, and risk. The Memory Graph Maintenance layer is separate: it curates graph structure and evidence over time, but it never directly decides turn-level authority. The Codex continuity bridge applies the same stance to local Codex state: useful status and handoff facts are surfaced, but raw telemetry is not captured as durable memory.

The core invariant is unchanged:

> **LLM decides semantic meaning. Code enforces trust boundaries. SQLite stores the graph and evidence.**

## Runtime shape

The important runtime modules are:

```text
packages/openclaw-plugin/src/
├── index.ts                         # plugin entry, hooks, routes, memory supplements
├── config.ts                        # defaults + config normalization
├── memory-store.ts                  # SQLite schema, FTS5, graph storage, proofs, jobs
├── memory-authority.ts              # relevance vs authority, validity, verification/confirmation decisions
├── graph-maintenance.ts             # graph health, proposals, lineage, edge observations, safe mutations
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
├── codex-continuity.ts              # read-only Codex status/watch/handoff bridge
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
  -> memory authority resolver
  -> bounded context injection, verification cue, confirmation cue, or abstention + proof

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
  -> memory authority resolution
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
5. `MemoryAuthorityResolver` decides `inject`, `weak_context`, `verify_before_use`, `confirm_before_use`, `abstain`, `audit_only`, or `never_use`.
6. `ContextSelector` picks a small authorized memory set and formats bounded prompt text.
7. Route decisions, injections, authority events, and proofs are recorded.
8. `agent_end`, `after_tool_call`, and the background service update outcomes, route frames, shadow decisions, replay cases, candidate reports, and policy snapshots.

### 2) legacy compatibility lane

Used when:

- `config.mode` is `proof-only`, `conservative`, or `active`

This preserves the older activation-file behavior by reading:

- `context.md`
- `corrections.md`
- `tool-guidance.md`

That lane remains for backward compatibility, but it is no longer the product center.

### 3) Codex continuity lane

Used when:

- `config.codexBridge.enabled` is true
- Jonathan wants Telegram/OpenClaw to be the low-bandwidth operator surface around Codex UI
- local Codex state is readable through SQLite fallback or a host-provided app-server reader

Flow:

1. The public package reads Codex SQLite state in read-only mode and labels the result stale.
2. If OpenClaw later provides a host app-server reader, the bridge can use that reader without bundling shell/process control in OpenClawBrain.
3. `/brain codex status`, `/brain codex threads`, `/brain codex watch`, and `/brain codex handoff` expose concise operator views.
4. Watch processing dedupes terminal events and only notifies completion, failure, blocker, approval-needed, or auth-failure events.
5. Handoff briefs separate observed facts from Codex-reported claims.
6. Raw Codex telemetry is not stored as durable OpenClawBrain memory.
7. Telegram-to-Codex writes remain disabled unless explicitly feature-flagged with trusted sender, repo allowlist, provenance, risk, and confirmation controls.

### 4) graph maintenance lane

Used when:

- the service runs passive background graph maintenance on a timer
- an operator asks `/brain graph health`, `/brain graph dry-run`, or the corresponding authenticated HTTP routes
- the system needs to inspect duplicate nodes, bad edges, stale high-authority memories, tombstone recapture risk, scoped exception candidates, or feedback observations
- a low-risk deterministic proposal is applied automatically or explicitly

Flow:

1. `GraphMaintenanceEngine` snapshots memory nodes, validity rows, memory edges, authority events, and route teacher signals.
2. It computes graph health metrics and compiles redacted proposals.
3. The passive service loop records dry-run/proposal history on a timer.
4. Low-risk deterministic proposals, such as exact duplicate consolidation, bad edge retirement, or observation-only feedback rows, can be applied transactionally.
5. Semantic merges, stale authority changes, scoped exceptions, privacy changes, supersession, and tombstone recapture remain review-gated.
6. Applied mutations write proof, node lineage, and edge observation rows.

Boundary:

> Graph Maintenance can provide features. `MemoryAuthorityResolver` recomputes the turn-level verdict.

Important invariants:

- current user instruction outranks old memory
- connectivity is not authority
- behavioral edges are not epistemic evidence
- implicit route success cannot raise evidence confidence
- tombstoned or hard-deleted content cannot be revived by merge, proof, proposal, or LLM distillation

## Latency model

OpenClawBrain is designed so memory does **not** add a blocking model call to every turn.

The code follows the four-tier plan from `FINAL_PLAN.md`:

- **Tier 0** — no extra sync LLM call, local cache/policy/retrieval only
- **Tier 1** — cached learned route + local retrieval
- **Tier 2** — one bounded `MemoryPlanner` call for ambiguous/high-signal turns
- **Tier 3** — async distillation, route learning, maintenance, pruning, replay, calibration, and promotion

## Storage model

`memory-store.ts` persists memory nodes/edges, route decisions, injection records, distillation runs, background jobs, proof events, memory validity rows, memory authority events, graph maintenance runs/proposals, memory node lineage, memory edge observations, and the v3 route-learning warehouse. Search uses SQLite FTS5 plus graph expansion and scope filters.

`memory_validity` keeps orthogonal state for retention, behavioral availability, temporal validity, privacy class, decay policy, validation strategy, and authority scores. `memory_authority_events` records when a memory was used, weakened, verified, confirmed, suppressed, tombstoned, superseded, or withheld.

`graph_maintenance_runs` and `graph_maintenance_proposals` keep dry-run and apply history. Proposal rows include preconditions, risk factors, redacted evidence, applied diffs, rollback hints, and status. `memory_node_lineage` preserves canonical merge/split/supersession lineage. `memory_edge_observations` records behavioral, lineage, retention, temporal, scope, and epistemic observations without treating every edge weight as authority.

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
- `/plugins/openclawbrain/graph/health`
- `/plugins/openclawbrain/graph/dry-run`
- `/plugins/openclawbrain/graph/proposals`
- `/plugins/openclawbrain/graph/apply`
- `/plugins/openclawbrain/graph/reject`
- `/plugins/openclawbrain/graph/stale`
- `/plugins/openclawbrain/graph/clusters`
- `/plugins/openclawbrain/graph/tombstones`
- `/plugins/openclawbrain/graph/explain`
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
