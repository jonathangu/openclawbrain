# Compiler worked example

- proposal: `prop_teacher_v3_compiler_worked_example` (compiler, promoted)
- review mode: **promotable**
- rollback key: `rollback:teacher-v3:compiler:worked-example`
- replay: **promotable** / **promotable**
- replay summary: compiler replay accepted on candidate_pack_compiler_08; before=0.792 after=1.000 delta=0.208
- proof verdict: **reviewable** (info)
- proof bundle: `teacher-v3-compiler-worked-example`
- proof bundle files: `compiler/proof-bundle/summary.md`, `compiler/proof-bundle/status.json`, `compiler/proof-bundle/surface-map.json`, `compiler/proof-bundle/proposal-report.json`, `compiler/proof-bundle/verdict.json`
- gate 1 seam: yes

## What was proposed
- Persist compiler proposals with stable lineage, replay summaries, and bounded proof bundles.
- subjects: proposal:compiler-persistence, store:round-trip, proof-bundle:five-file-layout

## Evidence
- `evi_compiler_store_01` → `src/brain-store/store.ts#updateTeacherProposalStatus`: updateTeacherProposalStatus({ proposalId, status, proofBundle, replaySummary, canaryRollout })
- `evi_compiler_replay_01` → `src/brain-core/teacher-v3-replay.ts#buildTeacherProposalReplaySummaryV1`: Compiler replay is promotable on candidate pack ...
- `evi_compiler_proof_01` → `scripts/teacher-v3-proof-bundle.mjs#buildTeacherV3ProofBundle`: summary.md, status.json, surface-map.json, proposal-report.json, verdict.json

## Counterevidence / boundary
- `cevi_compiler_target_01` → `docs/architecture/teacher-v3-proof.md#target-state`: The target-state proof bundle is an overlay on top of the first three surfaces, not a replacement for them.

## Replay summary
- before score: 0.792
- after score: 1.000
- score delta: 0.208
- class summary: Compiler replay is promotable on candidate pack candidate_pack_compiler_08; evidence-backed lineage stays intact and the candidate graph is distinct from base state.
- replay suites: compiler-persistence-smoke, compiler-proof-bundle-smoke

## Proof bundle + verdict surface
- verdict: **reviewable** (info)
- why: runtime, proof, and docs truth were summarized; Gate 1 persistence is already wired so the record can be loaded from storage (Captured 2 replay outcomes across 2 suites (compiler-persistence-smoke, compiler-proof-bundle-smoke); results pass=1, warn=1, fail=0; review modes promotable=2, shadow_only=0; sources proposal_record=2, proof_bundle=0, derived=0.)
- review mode: promotable
- publication-safe artifacts: teacher-v3-proof-summary, teacher-v3-proof-status, teacher-v3-proof-surface-map, teacher-v3-proof-proposal-report, teacher-v3-proof-verdict
- rollback-bound canary: yes
- rollout mode: off
- enabled: no
- activation guard: canary rollout remains off by default for teacher-v3-compiler-worked-example

## What remains target-state
- The canary plan is explicit, off by default, and rollback-bound.
- The proof bundle is a review surface, not a live runtime truth source.
- Live serving still only uses promoted packs.
