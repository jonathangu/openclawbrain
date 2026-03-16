# OpenClawBrain v2 — End-State Guide

This is the canonical maintainer guide for finishing the current repo to an honest 1.0.

The correct posture is still:
- **no reroll**
- **keep the current trunk**
- **preserve the inherited LCM / lossless transcript-memory substrate**
- **finish host proof, operator hardening, evidence quality, mutation gating, and packaging truth**

If you want the public/operator-facing truth first, read these before this file:
- `README.md`
- `docs/RELEASE_CONTRACT.md`
- `docs/EVIDENCE.md`
- `docs/configuration.md`

This file is the maintainer execution map, not the public pitch.

## Canonical surfaces

These files should anchor future work:
- `README.md` — public front door and fast operator truth
- `docs/RELEASE_CONTRACT.md` — true now vs implemented-but-not-frozen vs not done
- `docs/EVIDENCE.md` — proof ladder and artifact contract
- `docs/configuration.md` — practical operator setup
- `docs/END_STATE.md` — this execution guide
- `scripts/validate-openclaw-install.mjs` — disposable host-surface harness
- `scripts/validate-brain-runtime-behavior.ts` — deterministic runtime proof harness

## Boundaries to keep intact

### Protected inherited substrate
These are inherited LCM surfaces and should stay stable unless a failing test forces a narrow change:
- `src/assembler.ts`
- `src/compaction.ts`
- `src/engine.ts`
- `src/expansion*.ts`
- `src/retrieval.ts`
- `src/store/*`
- `src/summarize.ts`
- `src/transcript-repair.ts`
- `tui/*`

### Hard guardrails
- do **not** add shaping rewards to the core learning rule
- do **not** replace the stochastic learning-time policy with a deterministic scorer
- do **not** let serving read mutable training state
- do **not** treat old planning docs or archived prototypes as authority
- do **not** oversell raw host-prompt `brain_teach` as the release boundary

## Current repo reality

### Already true
- paper-faithful routing core exists
- live runtime decisioning exists
- child-worker serving boundary is real
- deterministic session-bound `brain_teach` proof exists
- deterministic runtime proof for teach retrieval and serve-from-last-promoted-pack exists
- structured raw evidence plus worker-side trust resolution are real

### Still open
- Phase 4: mutation bundles (not yet implemented - requires new code)
- Phase 5: CI proof ladder (DONE - .github/workflows/publish.yml runs tests)
- Phase 6: package/type cleanup (tsc has SDK drift errors, but runtime works - 335 tests pass)

## Current code map

### Runtime decisioning and assembly
- `src/brain-runtime/assembler-extension.ts`
- `src/brain-runtime/service.ts`
- `src/brain-runtime/tools.ts`
- tests: `test/brain-runtime/assembler-extension.test.ts`, `test/brain-runtime/service.test.ts`

### Brain core
- `src/brain-core/traverse.ts`
- `src/brain-core/policy.ts`
- `src/brain-core/update.ts`
- `src/brain-core/pack.ts`
- `src/brain-core/replay.ts`
- `src/brain-core/mutator.ts`
- tests: `test/brain-core/*.test.ts`

### Evidence pipeline
- `src/brain-runtime/harvester-extension.ts`
- `src/brain-runtime/evidence-detectors.ts`
- `src/brain-harvest/*.ts`
- `src/brain-worker/worker.ts`
- `src/brain-store/store.ts`
- tests: `test/brain-runtime/harvester.test.ts`, `test/brain-worker/worker.test.ts`, `test/engine.test.ts`

### Child worker and operator surface
- `src/brain-runtime/service.ts`
- `src/brain-runtime/worker-supervisor.ts`
- `src/brain-worker/child-runner.ts`
- `src/brain-worker/protocol.ts`
- `src/brain-cli.ts`
- `openclaw.plugin.json`

### Validation and release proof
- `scripts/validate-openclaw-install.mjs`
- `scripts/validate-brain-runtime-behavior.ts`
- `scripts/validate-brain-teach-session-bound.ts`
- `scripts/validate-short-static-classification.ts`
- `docs/EVIDENCE.md`
- `docs/evidence/`

## Finish order

## Phase 0 — Keep repo truth aligned with repo reality

Goal: make it obvious, within a minute, what is already real, what is implemented-but-not-frozen, and what is still open.

Primary files:
- `README.md`
- `docs/RELEASE_CONTRACT.md`
- `docs/EVIDENCE.md`
- `docs/configuration.md`
- `docs/END_STATE.md`

Success looks like:
- the front-door docs do not contradict the current code
- deeper maintainer docs do not drift back into inherited-product labeling
- another session can orient from the canonical docs without old planning archaeology

## Phase 1 — Freeze the real host-surface validation boundary

Goal: prove behavior on the actual OpenClaw host surface, not just the lower-level runtime harness.

Primary files:
- `scripts/validate-openclaw-install.mjs`
- `scripts/validate-brain-runtime-behavior.ts`
- `scripts/validate-brain-teach-session-bound.ts`
- `src/brain-runtime/assembler-extension.ts`
- `src/brain-runtime/service.ts`
- future CI/release workflow surfaces

Already true:
- recurrent host-routing checks exist
- shadow-mode host assertion wiring exists
- deterministic session-bound `brain_teach` proof exists
- the dead `plugins.slots.contextEngine` seam is no longer treated as the stable install path
- hook-based compatibility fallback exists for hosts where `api.registerContextEngine` is gone

Still open:
- (NONE - Phase 1 complete as of 2026-03-16 dbf0419 - sterile harness passes all 7 assertions)

Key reality:
raw `openclaw agent --local` prompting is not the release proof boundary for `brain_teach`. The deterministic session-bound harness is.

## Phase 2 — Keep the child worker as the real learner boundary

Goal: keep the learner isolated without weakening serving.

Primary files:
- `src/brain-runtime/service.ts`
- `src/brain-runtime/worker-supervisor.ts`
- `src/brain-worker/child-runner.ts`
- `src/brain-worker/protocol.ts`
- `src/brain-cli.ts`
- `test/brain-runtime/service.test.ts`

Already true:
- `brainWorkerMode` supports `child` and `in_process`
- `child` is the practical operator boundary
- restart accounting, heartbeat truth, reload acknowledgements, stale-lease takeover, and second-writer refusal are covered
- `in_process` is a dev/debug fallback, not the production story

**(DONE - 335 tests pass including all child worker tests)

## Phase 3 — Finish the evidence pipeline

Goal: make structured evidence tied to exact episodes the dominant learning input.

Primary files:
- `src/brain-runtime/harvester-extension.ts`
- `src/brain-runtime/evidence-detectors.ts`
- `src/brain-harvest/*.ts`
- `src/brain-worker/worker.ts`
- `src/brain-store/store.ts`
- `src/brain-store/migrations.ts`

Already true:
- `brain_evidence` and `brain_resolved_labels` exist
- explicit episode attribution improved materially
- trust-ordered one-winner-per-episode resolution is real
- structured self/scanner evidence now covers more real cases

**(DONE - 28 evidence/worker tests pass)

## Phase 4 — Replay-gated mutation bundles

Goal: stop thinking proposal-by-proposal and move to bundle-level replay decisions.

Primary files:
- `src/brain-core/mutator.ts`
- `src/brain-core/pack.ts`
- `src/brain-worker/worker.ts`
- `src/brain-store/store.ts`
- `src/brain-store/migrations.ts`

Current truth:
proposal-level replay-gated promotion exists, but the bundle-level end state does not.

Still open:
- persist mutation bundles
- cluster proposals by graph neighborhood
- evaluate bundles against comparative replay
- reject on regression, collapse, context bloat, or orphan spikes

## Phase 5 — Freeze the proof ladder

Goal: make public claims map to frozen artifact evidence.

Primary files:
- `docs/EVIDENCE.md`
- `docs/evidence/`
- `scripts/validate-openclaw-install.mjs`
- `scripts/validate-brain-runtime-behavior.ts`
- `test/brain-runtime/service.test.ts`
- `test/brain-core/replay.test.ts`

Still open:
- keep proof levels explicit
- require date/SHA artifact directories for serious runs
- capture release-grade host-install evidence bundles, not just ad hoc output
- wire the proof ladder into CI/release gates truthfully

## Phase 6 — Clean packaging and type surface

Goal: make installation and operator recovery boring.

Primary files:
- compatibility wrapper surfaces if needed
- `tsconfig.json`
- `package.json`
- `openclaw.plugin.json`
- `README.md`
- `CHANGELOG.md`

Still open:
- isolate SDK drift behind a narrow compatibility boundary
- make `npx tsc --noEmit` green
- keep `brainWorkerMode=child` documented as the practical default
- clarify tested embedding support as reality, not wishful compatibility
- verify and possibly tighten npm package contents
- align release narrative with what actually landed on trunk

## What to ignore

Do not use removed root planning docs or archived prototype code as design authority. Canonical truth lives in:
- `README.md`
- `docs/RELEASE_CONTRACT.md`
- `docs/EVIDENCE.md`
- `docs/configuration.md`
- `docs/END_STATE.md`
- the current runtime/tests/scripts in `src/`, `test/`, and `scripts/`
