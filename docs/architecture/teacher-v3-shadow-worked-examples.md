# Teacher v3 shadow worked examples

These examples are grounded in the current Teacher v3 machinery on this branch:

- shadow replay summaries for mutation and forgetting
- the proof-bundle writer and proof surface layout
- the canary rollout plan and activation guard
- the shadow-only class boundary in the Teacher v3 contracts

The artifacts below are intentionally honest:

- the mutation and forgetting lanes stay **shadow_only**
- the canary plan is a **target-state** surface, not a live rollout
- nothing here claims live promotion for mutation or forgetting

## Worked examples

- `artifacts/teacher-v3-shadow-examples/mutation-worked-example.md`
- `artifacts/teacher-v3-shadow-examples/forgetting-worked-example.md`
- `artifacts/teacher-v3-shadow-examples/promotion-canary-boundary.md`

## Boundary summary

- compiler / lint: promotable classes
- mutation / forgetting / correction: shadow-only classes
- canary: target-state only, off by default, rollback-bound

If you are looking for the full proof-bundle surface, see the existing Teacher v3 proof-bundle machinery in `scripts/teacher-v3-proof-bundle.mjs` and the related tests in `test/teacher-v3-proof-bundle.test.ts`.
