# Teacher v3 graph-maintenance proposal lifecycle — Lane C

## Baseline inspected
- Contracts: `src/brain-core/teacher-v3-contracts.ts`
- Store/migrations: `src/brain-store/store.ts`, `src/brain-store/migrations.ts`
- Replay/shadow: `src/brain-core/teacher-v3-replay.ts`, `src/brain-core/teacher-v3-shadow-replay.ts`, `src/brain-core/shadow-application.ts`
- Proof/artifacts: `scripts/teacher-v3-proof-bundle.mjs`, `src/brain-core/teacher-v3-proposal-artifact.ts`
- Existing tests/docs: `test/brain-store/teacher-v3-proposals.test.ts`, `test/brain-core/teacher-v3-replay.test.ts`, `test/brain-core/teacher-v3-shadow-replay.test.ts`, `docs/architecture/teacher-v3*.md`

The branch already had a generic `brain_teacher_proposals` table and general Teacher proposal contracts. The missing narrow lane was a durable, replayable graph-maintenance proposal path with explicit shadow-only safety for structural mutation.

## Change
- Added `src/brain-core/teacher-v3-graph-maintenance.ts` with a narrow add-edge graph-maintenance lifecycle:
  - builds a persisted `TeacherProposal` with `proposalClass=mutation`, `proposalKind=add_edge`, `safeClassMode=shadow_only`
  - requires evidence refs, subject ids, expected effect, replay suites, rollback key, lineage, and stable idempotency identity
  - replays only against a cloned candidate graph via the existing shadow mutation substrate
  - emits a `TeacherProposalReplaySummaryV1`, detailed shadow replay summary, and lifecycle summary with rollback semantics
- Extended `TeacherProposalV1` / summaries with `safeClassMode`, defaulted from proposal class review mode.
- Hardened `BrainStore` proposal writes:
  - rejects mismatched `safeClassMode`
  - rejects replayGate review-mode mismatch
  - rejects `promotable` / `promoted` status for shadow-only classes (`mutation`, `forgetting`, `correction`)
  - derives lifecycle state on status changes, so `shadow_scored` becomes `replayed`
- Added durable example proof: `artifacts/teacher-v3-graph-maintenance/prop_graph_add_edge_01-lifecycle.json`.
- Added focused tests: `test/brain-core/teacher-v3-graph-maintenance.test.ts`.

## Measured / replayed result
Example proposal: `prop_graph_add_edge_01`

- Kind: `add_edge`
- Class: `mutation`
- Safe mode: `shadow_only`
- Subjects: `concept_teacher_v3`, `concept_graph_prior`
- Evidence: `evi_graph_maintenance_add_edge_01`
- Expected effect: retrieval better, truth risk low, token budget same
- Replay suites: `teacher-v3-graph-maintenance-shadow`, `teacher-v3-rollback-smoke`
- Rollback key: `rollback:teacher-v3:graph-maintenance:add-edge:01`
- Replay outcome: applied in shadow candidate graph
- Candidate effect: edge count +1 in shadow graph
- Rollback: restored true
- Promotion bypass: false
- Live self-editing: false

Inspect the durable JSON proof at:

`artifacts/teacher-v3-graph-maintenance/prop_graph_add_edge_01-lifecycle.json`

## Tests run
```bash
npx vitest run --dir test \
  test/brain-core/teacher-v3-graph-maintenance.test.ts \
  test/brain-store/teacher-v3-proposals.test.ts \
  test/brain-core/teacher-v3-shadow-replay.test.ts \
  test/brain-core/teacher-v3-replay.test.ts
```

Result: 4 files passed, 10 tests passed.

```bash
npx vitest run --dir test \
  test/brain-core/teacher-v3-contracts.test.ts \
  test/teacher-v3-proposal-artifact.test.ts \
  test/teacher-v3-proof-bundle.test.ts \
  test/teacher-v3-replay-outcomes.test.ts \
  test/teacher-v3-promotable-examples.test.ts \
  test/teacher-v3-shadow-worked-examples.test.ts \
  test/brain-core/teacher-v3-graph-maintenance.test.ts \
  test/brain-store/teacher-v3-proposals.test.ts
```

Result: 8 files passed, 30 tests passed.

Additional gate:
```bash
npx tsc --noEmit
```
Result: failed on pre-existing repo-wide type errors outside this lane; a filtered rerun showed no `teacher-v3-graph-maintenance`, `teacher-v3-contracts`, or `brain-store/store.ts` type errors.

## Honest boundary
This implements one narrow durable graph-maintenance proposal lifecycle. It persists/loads via the existing proposal store, replays against a shadow candidate graph, summarizes rollback, and blocks mutation promotion. It does **not** enable live graph self-editing, canary activation, npm publish, GitHub push, local install, gateway restart, or any weakening of user-explicit correction/deletion guardrails.
