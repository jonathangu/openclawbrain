# Teacher v3 promotion / canary boundary

This is the short boundary note for the Teacher v3 promotion story. It makes the line explicit: mutation and forgetting stay shadow-only, while canary exists only as a target-state plan.

## What is promotable today
- `compiler`
- `lint`

## What stays shadow-only
- `mutation`
- `forgetting`
- `correction`

## Real canary plan
The current canary plan for a Teacher v3 lane is intentionally off by default:

- `surfaceState`: `target`
- `rolloutMode`: `off`
- `enabled`: `false`
- `rollbackBound`: `true`
- example binding:
  - `proposalClass`: `mutation`
  - `rollbackKey`: `rollback:teacher-v3:canary:mutation`
  - `candidatePackVersion`: `8`
  - `candidatePackId`: `candidate_pack_08`

The plan summary from the real contract surface reads:

> `mutation canary plan stays target, rolloutMode=off, enabled=false, and rollback-bound to rollback:teacher-v3:canary:mutation, candidatePackVersion=8, candidatePackId=candidate_pack_08.`

## Real activation guard
`describeTeacherCanaryActivationGuardV1(...)` makes the boundary explicit:

- an off plan is not an activation
- a requested canary is blocked until all of these are present:
  - replay summary
  - proof bundle
  - matching proof rollback binding

For the blocked case, the guard reports:

- `requested`: `true`
- `allowed`: `false`
- `blocked`: `true`
- blockers:
  - `missing replay summary`
  - `missing proof bundle`
  - `missing proof rollback binding`

## Still target-state
- keep the canary plan inspectable but off by default.
- bind the candidate pack by durable version or id.
- require replay + proof + rollback binding before any later activation.
- do not treat the canary plan as a live-serving change path.
