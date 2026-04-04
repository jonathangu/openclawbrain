# Lint worked example

- proposal: `prop_teacher_v3_lint_worked_example` (lint, promotable)
- review mode: **promotable**
- rollback key: `rollback:teacher-v3:lint:worked-example`
- replay: **promotable** / **promotable**
- replay summary: lint replay accepted on candidate_pack_lint_08; before=0.796 after=1.000 delta=0.204
- proof verdict: **reviewable** (info)
- proof bundle: `teacher-v3-lint-worked-example`
- proof bundle files: `lint/proof-bundle/summary.md`, `lint/proof-bundle/status.json`, `lint/proof-bundle/surface-map.json`, `lint/proof-bundle/proposal-report.json`, `lint/proof-bundle/verdict.json`
- gate 1 seam: yes

## What was proposed
- Keep the public release story aligned across README, docs index, changelog, and end-state docs.
- subjects: release:0.4.29, docs:public-surface-truth, docs:proof-surface-boundary

## Evidence
- `evi_lint_release_01` → `scripts/verify-release-docs-drift.mjs#verifyReleaseDocsDrift`: This deterministic lint compares the current release version in CHANGELOG.md against public release-surfaces.
- `evi_lint_readme_01` → `README.md#current-version`: Current version: **0.4.29**
- `evi_lint_docs_01` → `docs/README.md#current-release-notes`: Current release notes (0.4.29)
- `evi_lint_endstate_01` → `docs/END_STATE.md#split-package-story`: split packages `@openclawbrain/openclaw@0.4.29` and `@openclawbrain/cli@0.4.29` are published

## Counterevidence / boundary
- `cevi_lint_target_01` → `docs/architecture/teacher-v3-proof.md#target-state-surfaces`: Teacher v3 reporting may summarize and cross-reference truth, but it must not become a new source of truth for the live runtime.

## Replay summary
- before score: 0.796
- after score: 1.000
- score delta: 0.204
- class summary: Lint replay is promotable on candidate pack candidate_pack_lint_08; the bounded report-only review preserved counterevidence and replay discipline.
- replay suites: release-docs-drift-smoke, teacher-v3-lint-proof-surface-smoke

## Proof bundle + verdict surface
- verdict: **reviewable** (info)
- why: runtime, proof, and docs truth were summarized; Gate 1 persistence is already wired so the record can be loaded from storage (Captured 2 replay outcomes across 2 suites (release-docs-drift-smoke, teacher-v3-lint-proof-surface-smoke); results pass=1, warn=1, fail=0; review modes promotable=2, shadow_only=0; sources proposal_record=2, proof_bundle=0, derived=0.)
- review mode: promotable
- publication-safe artifacts: teacher-v3-proof-summary, teacher-v3-proof-status, teacher-v3-proof-surface-map, teacher-v3-proof-proposal-report, teacher-v3-proof-verdict
- rollback-bound canary: yes
- rollout mode: off
- enabled: no
- activation guard: canary rollout remains off by default for teacher-v3-lint-worked-example

## What remains target-state
- The release-docs lint is promotable, but it still only audits public surfaces; it does not mutate them.
- Teacher v3 proof/reporting surfaces remain target-state overlays, not the live runtime truth source.
- The rollback key is explicit even though the proposal stayed reviewable/promotable.
