# Teacher v3 proof bundle

- bundle: `teacher-v3-lint-worked-example`
- proposal: `teacher-v3-lint-worked-example` (lint, promotable)
- review mode: **promotable**
- replay status: **promotable**
- replay score: 0.796 → 1.000 (Δ 0.204)
- candidate pack: candidate_pack_lint_08
- verdict: **reviewable**
- severity: **info**
- runtime truth: `openclawbrain status --detailed`
- proof truth: `openclawbrain proof --openclaw-home ~/.openclaw`
- docs truth: `docs/architecture/teacher-v3-proof.md`

## Canary rollout
- surface state: target
- rollout mode: off
- enabled: no
- disabled by default: yes
- rollback bound: yes
- rollback key: `rollback:teacher-v3:lint:worked-example`
- candidate pack: candidate_pack_lint_08
- binding: Lint lane: rollback-bound to rollback:teacher-v3:lint:worked-example; candidate pack candidate_pack_lint_08.
- guardrails: Keep the rollout plan target-state only until it is explicitly shipped.; Default rolloutMode stays off.; Do not use the canary plan to change live serving without separate replay, proof, and rollback binding.; Canary activation stays blocked until replay summary, proof bundle, and rollback binding are all present.; Bind any candidate pack by durable version or id, never by ad hoc display labels.; Bind the plan to an explicit rollback key before any later tranche can opt it in.

## Surface counts
- shipped surfaces: 3
- target bundle artifacts: 5
- total referenced surfaces: 8

## Replay outcomes
- captured outcomes: 2
- replay suites: release-docs-drift-smoke, teacher-v3-lint-proof-surface-smoke
- results: pass=1, warn=1, fail=0
- review modes: promotable=2, shadow_only=0
- sources: proposal_record=2, proof_bundle=0, derived=0

## Canary rollout
- rollout mode: off
- enabled: no
- candidate pack: candidate_pack_lint_08
- activation: off by default
- guard summary: canary rollout remains off by default for teacher-v3-lint-worked-example


## Gate 1 seam
- present: yes
- record source: brain_store
- note: Lint proposal round-tripped through BrainStore before proof-bundle emission.

## Publication-safe artifacts
- `ocb-t129-gate5a/artifacts/teacher-v3-promotable-examples/lint/proof-bundle/summary.md` — bounded human summary
- `ocb-t129-gate5a/artifacts/teacher-v3-promotable-examples/lint/proof-bundle/status.json` — thin machine status
- `ocb-t129-gate5a/artifacts/teacher-v3-promotable-examples/lint/proof-bundle/surface-map.json` — shipped-vs-target inventory
- `ocb-t129-gate5a/artifacts/teacher-v3-promotable-examples/lint/proof-bundle/proposal-report.json` — machine-readable proposal report
- `ocb-t129-gate5a/artifacts/teacher-v3-promotable-examples/lint/proof-bundle/verdict.json` — review verdict

## Recommendations
- Preserve the persisted proposal record seam and load it directly once Gate 1 lands.
- Keep the bundle publication-safe and bounded; never spill raw logs into the target-state artifacts.
- Thread candidate-state replay and rollback binding in the next tranche before considering canary activation.
