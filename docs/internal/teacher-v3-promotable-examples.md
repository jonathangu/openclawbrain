# Teacher v3 promotable worked examples

This page points at the checked-in worked examples for the two promotable Teacher v3 lanes:

- compiler
- lint

The examples are generated from the real Gate 1–4 machinery in this branch:

- Gate 1: `BrainStore` proposal persistence / reload
- Gate 2: the five-file Teacher v3 proof bundle writer
- Gate 3: replay summaries over candidate pack state
- Gate 4: canary rollout stays explicit, rollback-bound, and off by default

## Artifact root

All example artifacts live under:

`artifacts/teacher-v3-promotable-examples/`

The root manifest is:

- `artifacts/teacher-v3-promotable-examples/manifest.json`

## What each lane shows

| Lane | Proposal status | Review mode | Proof verdict | Rollback binding | Target-state boundary |
|---|---|---|---|---|---|
| compiler | promoted | promotable | reviewable | `rollback:teacher-v3:compiler:worked-example` | canary remains off by default; proof bundle stays reviewable and bounded |
| lint | promotable | promotable | reviewable | `rollback:teacher-v3:lint:worked-example` | lint audits public truth surfaces only; Teacher v3 proof/reporting remains target-state |

## Compiler worked example

- `artifacts/teacher-v3-promotable-examples/compiler/example.md`
- `artifacts/teacher-v3-promotable-examples/compiler/example.json`
- `artifacts/teacher-v3-promotable-examples/compiler/proof-bundle/summary.md`
- `artifacts/teacher-v3-promotable-examples/compiler/proof-bundle/status.json`
- `artifacts/teacher-v3-promotable-examples/compiler/proof-bundle/surface-map.json`
- `artifacts/teacher-v3-promotable-examples/compiler/proof-bundle/proposal-report.json`
- `artifacts/teacher-v3-promotable-examples/compiler/proof-bundle/verdict.json`

What it demonstrates:

- a compiler proposal persisted and reloaded through `BrainStore`
- replay over candidate pack state stayed promotable
- proof bundle output stayed bounded and publication-safe
- the canary plan remained target-state and off by default

## Lint worked example

- `artifacts/teacher-v3-promotable-examples/lint/example.md`
- `artifacts/teacher-v3-promotable-examples/lint/example.json`
- `artifacts/teacher-v3-promotable-examples/lint/proof-bundle/summary.md`
- `artifacts/teacher-v3-promotable-examples/lint/proof-bundle/status.json`
- `artifacts/teacher-v3-promotable-examples/lint/proof-bundle/surface-map.json`
- `artifacts/teacher-v3-promotable-examples/lint/proof-bundle/proposal-report.json`
- `artifacts/teacher-v3-promotable-examples/lint/proof-bundle/verdict.json`

What it demonstrates:

- a lint proposal that audits public release truth surfaces
- replay over candidate pack state stayed promotable
- the bundle is reviewable, but the proposal remains reviewable/promotable rather than live-mutating
- Teacher v3 proof/reporting surfaces are still target-state overlays

## Honesty boundary

These examples do **not** claim that Teacher v3 proof/reporting is a shipped live runtime truth source.
They show the reviewed surfaces, the replay summary, the proof bundle, and the rollback binding, while keeping the live/target split explicit.
