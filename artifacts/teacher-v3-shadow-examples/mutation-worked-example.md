# Teacher v3 shadow worked example: mutation

This is a shadow-only worked example from the current Teacher v3 machinery. It does **not** imply live graph mutation or promotion.

## Proposal
- `proposalId`: `prop_mutation_shadow_01`
- `proposalClass`: `mutation`
- `reviewMode`: `shadow_only`
- `rollbackKey`: `rollback:teacher-v3:mutation:shadow`
- `proposal`:
  - connect `b -> c`
  - inject an `episode_anchor` note with content `shadow note`
  - fired nodes: `a`, `b`

## Evidence
- `src/brain-core/shadow-application.ts`
  - shadow applications operate on `candidateGraph` and keep reversible operations separate from live mutation.
- `src/brain-core/teacher-v3-shadow-replay.ts`
  - mutation summaries are labeled `shadow_only`, record rollback, and set `promotionBypass: false`.
- `test/brain-core/teacher-v3-shadow-replay.test.ts`
  - exercises the same shadow replay shape on a real candidate graph.

## Replay result
- before: 3 nodes / 1 edge
- after: 4 nodes / 4 edges
- applications: 2
- replay outcome: `applied`
- summary: `Mutation replay stayed shadow-only on the candidate graph (2/2 application(s) applied) and rollback restored the base graph without any promotion bypass.`

## Why it remained shadow-only
- `mutation` is one of the shadow-only Teacher v3 classes.
- the replay summary is explicitly `shadow_only`.
- `promotionBypass` is `false`.
- the helper only mutates the candidate graph, never the live graph.

## Rollback semantics
- rollback strategy: `reset_shadow_candidate_state`
- rollback restored: `true`
- rollback returns the candidate graph to the base graph.
- after rollback: 3 nodes / 1 edge

## Still target-state
- persist the proposal record durably so Gate 1 can own the lifecycle.
- emit the proof bundle from the stored proposal + runtime/proof data.
- keep live serving off; no direct mutation promotion.
- any later promotion must still go through replay, proof, and rollback binding.
