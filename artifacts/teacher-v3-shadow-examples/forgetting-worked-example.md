# Teacher v3 shadow worked example: forgetting

This is a shadow-only worked example from the current Teacher v3 machinery. It does **not** imply live deletion or promotion.

## Proposal
- `proposalId`: `prop_forgetting_shadow_01`
- `proposalClass`: `forgetting`
- `reviewMode`: `shadow_only`
- `rollbackKey`: `rollback:teacher-v3:forgetting:shadow`
- current state: `retained`
- requested transition: `archive`
- target source:
  - `sourceId`: `bn_source_01`
  - `sourceKind`: `summary`
  - `authority`: `raw_source`

## Evidence
- `src/brain-core/teacher-v3-contracts.ts`
  - retention transitions are fail-closed, and teacher-driven hard delete is guarded for `user_explicit` correction memory.
- `src/brain-core/teacher-v3-shadow-replay.ts`
  - forgetting replay uses `evaluateRetentionTransitionV1`, records the decision, and keeps rollback explicit.
- `test/brain-core/teacher-v3-shadow-replay.test.ts`
  - covers the archive path and the blocked hard-delete guardrail.

## Replay result
- before: `retained`
- after: `archived`
- replay outcome: `applied`
- decision: allowed
- summary: `Forgetting replay moved bn_source_01 from retained to archived in shadow-only mode and can roll back to retained with no promotion bypass.`

## Why it remained shadow-only
- `forgetting` is one of the shadow-only Teacher v3 classes.
- the replay summary is explicitly `shadow_only`.
- `promotionBypass` is `false`.
- the helper simulates retention-state transitions; it does not perform live memory deletion.

## Rollback semantics
- rollback strategy: `restore_retention_state`
- rollback restored: `true`
- rollback returns the source from `archived` back to `retained`.

## Still target-state
- persist the proposal record durably so Gate 1 can own the lifecycle.
- emit the proof bundle from the stored proposal + runtime/proof data.
- keep hard delete blocked for `user_explicit` correction memory.
- a related boundary case stays blocked: `prop_forgetting_shadow_02` with `hard_delete` on `user_explicit` memory is denied by `deny_hard_delete_user_explicit`.
