# OpenClawBrain v0.2 — Implementation Plan

**See [FINAL_PLAN.md](./FINAL_PLAN.md) for the complete, authoritative implementation plan.**

This file is a summary. `FINAL_PLAN.md` has the full design including:
- Four execution tiers for latency control (Tier 0–3)
- LatencyController module
- MemoryPlanner single-call fast path
- Route cache, candidate cache, policy snapshot cache
- Job queue for async distillation
- Distillation runs audit table
- 8 implementation phases
- Complete test plan

---

## One-line summary

**Use LLMs for semantic distillation, but put them behind a latency-aware route layer, cache, queue, and timeout system.**

## Key principle

The LLM decides semantic meaning. The code enforces trust boundaries. SQLite stores the graph and evidence. Background learning improves the route policy.

## Source tree

```
src/
  index.ts               # wiring
  config.ts              # config schema/defaults
  redact.ts              # redaction, hashing
  llm-client.ts          # abstract LLM interface
  llm-json.ts            # structured JSON calls + validation
  feedback-distiller.ts  # LLM feedback distillation
  turn-distiller.ts      # LLM turn frame extraction
  route-fn.ts            # learned LLM route function
  context-selector.ts    # LLM selects/distills memories
  memory-planner.ts      # single-call fast planner (Tier 2)
  latency-controller.ts  # decides Tier 0/1/2/3
  memory-operations.ts   # applies LLM-proposed operations
  memory-store.ts        # SQLite CRUD, FTS5, migrations
  job-queue.ts           # async job queue
  graph.ts               # edge logic, contradiction
  learning.ts            # background jobs
  route-learning.ts      # route policy improvement
  injection.ts           # budget/format/record
  search.ts              # OpenClaw memory supplements
  status.ts              # status payloads
  routes.ts              # HTTP handlers
  memory-types.ts        # shared types
  sqlite-driver.ts       # SQLite adapter
```

## Four execution tiers

| Tier | What happens | When |
|------|-------------|------|
| 0 | Route cache + policy snapshot + SQLite retrieval | Most turns |
| 1 | Cached learned route + local retrieval | Cache-hit implementation turns |
| 2 | One fast bounded sync LLM call (700–2500ms) | High-signal, ambiguous, cache-miss turns |
| 3 | Async via agent_end/background service | Feedback capture, learning, pruning |

## 8 phases

1. SQLite store, jobs, proof backbone
2. LLM JSON infrastructure
3. Feedback distillation (async first)
4. Learned route function + latency controller
5. Retrieval + context selection
6. MemoryPlanner single-call fast path
7. Route learning + self-regulation
8. Memory supplements + release polish
