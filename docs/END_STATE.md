# OpenClawBrain v2 — Definitive End-State Guide

This is the canonical implementation guide for finishing the current repo to an honest 1.0.

The correct posture is:

- **no reroll**
- **keep the current trunk**
- **preserve the inherited lossless-claw substrate**
- **finish proof, operator hardening, evidence quality, mutation gating, and packaging truth**

## Canonical surfaces

These files should anchor future work:

- `README.md` — public front door and fast operator truth
- `docs/RELEASE_CONTRACT.md` — what is true now vs not frozen vs not done
- `docs/END_STATE.md` — this implementation guide
- `docs/EVIDENCE.md` — proof ladder and artifact contract
- `scripts/validate-openclaw-install.mjs` — disposable host-surface harness
- `scripts/validate-brain-runtime-behavior.ts` — deterministic runtime proof harness

## Keep these boundaries intact

### Protected inherited substrate — do not rewrite casually
These are inherited LCM / lossless-claw surfaces and should stay stable unless a failing test forces a narrow change:

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
- Do **not** add intermediate shaping rewards to the core learning rule.
- Do **not** replace the stochastic learning-time policy with a deterministic scorer.
- Do **not** let serving read mutable training state.
- Do **not** treat old planning docs or archived prototypes as authority.

## Current code map

### Runtime decisioning and assembly
- `src/brain-runtime/assembler-extension.ts`
- `src/brain-runtime/service.ts`
- `src/brain-runtime/tools.ts`
- Tests: `test/brain-runtime/assembler-extension.test.ts`, `test/brain-runtime/service.test.ts`

### Brain core
- `src/brain-core/traverse.ts`
- `src/brain-core/policy.ts`
- `src/brain-core/update.ts`
- `src/brain-core/pack.ts`
- `src/brain-core/replay.ts`
- `src/brain-core/mutator.ts`
- Tests: `test/brain-core/*.test.ts`

### Evidence pipeline
- `src/brain-runtime/harvester-extension.ts`
- `src/brain-runtime/evidence-detectors.ts`
- `src/brain-harvest/human.ts`
- `src/brain-harvest/self.ts`
- `src/brain-harvest/scanner.ts`
- `src/brain-store/store.ts`
- `src/brain-store/migrations.ts`
- `src/brain-worker/worker.ts`
- Tests: `test/brain-runtime/harvester.test.ts`, `test/brain-worker/worker.test.ts`, `test/engine.test.ts`

### Child worker and operator surface
- `src/brain-runtime/service.ts`
- `src/brain-worker/child-runner.ts`
- `src/brain-cli.ts`
- `openclaw.plugin.json`
- Tests: `test/brain-runtime/service.test.ts`

### Validation and release proof
- `scripts/validate-openclaw-install.mjs`
- `scripts/validate-brain-runtime-behavior.ts`
- `docs/EVIDENCE.md`
- `docs/evidence/`

## Finish order

## Phase 0 — Align repo truth with repo reality

Goal: make it obvious, within a minute, what is already real, what is implemented-but-not-frozen, and what is still open.

### Work in
- `README.md`
- `docs/RELEASE_CONTRACT.md`
- `docs/END_STATE.md`
- `docs/EVIDENCE.md`

### What success looks like
- the README front page does not contradict the current code
- no duplicate root planning docs compete with the canonical docs set
- another session can orient from the docs above without spelunking old plans

## Phase 1 — Finish the real host-surface validation harness

Goal: prove behavior on the actual OpenClaw host surface, not just the lower-level runtime harness.

### Main files
- `scripts/validate-openclaw-install.mjs`
- `scripts/validate-brain-runtime-behavior.ts`
- `src/brain-runtime/assembler-extension.ts`
- `src/brain-runtime/service.ts`
- `src/brain-runtime/tools.ts`
- future: `.github/workflows/validate-openclaw-install.yml`

### Already true
- recurrent host routing checks run
- shadow-mode host assertion wiring exists
- current local-Ollama harness runs end to end on the non-skipped matrix

### Still open
- deterministic host-surface `brain_teach` proof
- deterministic host-surface worker-down / last-promoted-pack fail-open proof
- explicit `skip_no_embedding` and `skip_uninitialized` assertions on the host surface
- frozen evidence bundle per run under `docs/evidence/YYYY-MM-DD/<git-sha>/`
- short-static-lookup semantic drift (`open PLAYBOOK.md` still surfacing as `use_brain` on the host path)

### Key reality to remember
`openclaw agent --local` currently exposes session targeting, timeout, delivery, and verbose controls, but no explicit deterministic “force this tool call” control. If host-surface `brain_teach` remains impossible to drive honestly through that CLI, the harness must classify that truthfully or use a lower-level host harness.

## Phase 2 — Harden the child worker

Goal: make the child worker the real learner boundary without affecting serving.

### Main files
- `src/brain-runtime/service.ts`
- future: `src/brain-runtime/worker-supervisor.ts`
- `src/brain-worker/child-runner.ts`
- future: `src/brain-worker/protocol.ts`
- `src/brain-cli.ts`

### What is already real
- `brainWorkerMode` supports `child` and `in_process`
- child worker heartbeat / PID truth already surfaces through status
- lease protection exists in the child runner

### What remains
- pull child lifecycle logic out of `service.ts` into a cleaner supervisor
- define explicit worker protocol messages (`ready`, `heartbeat`, `reload-graph`, `reload-graph-ack`, `tick-result`, `shutdown`, `fatal-error`)
- add restart accounting and better doctor/status reporting
- make `in_process` clearly dev-only
- add crash / stale-lease / second-writer / reload-ack tests

## Phase 3 — Finish the evidence pipeline

Goal: make structured evidence tied to exact episodes the dominant learning input.

### Main files
- `src/brain-runtime/harvester-extension.ts`
- `src/brain-runtime/evidence-detectors.ts`
- `src/brain-harvest/human.ts`
- `src/brain-harvest/self.ts`
- `src/brain-harvest/scanner.ts`
- `src/brain-worker/worker.ts`
- `src/brain-store/store.ts`
- `src/brain-store/migrations.ts`

### What is already real
- `brain_evidence` and `brain_resolved_labels` exist
- explicit episode attribution improved materially
- trust-ordered one-winner-per-episode resolution is real
- `brain_teach` records evidence metadata against the corrected episode path

### What remains
- expand evidence schema (`messageId`, `partId`, `toolName`, `command`, `exitCode`, `filesTouched`, `artifactPath`, `taughtNodeId`, `correctedEpisodeId`)
- push harvesters toward raw evidence only, with final label resolution in the worker
- reduce “most recent message” fallback to a genuine fallback
- build richer scanner extractors (runbook/tool-chain/reuse/bridge/issue→PR→commit)

## Phase 4 — Replay-gated mutation bundles

Goal: stop thinking proposal-by-proposal and move to bundle-level replay decisions.

### Main files
- `src/brain-core/mutator.ts`
- `src/brain-core/pack.ts`
- `src/brain-worker/worker.ts`
- `src/brain-store/store.ts`
- `src/brain-store/migrations.ts`

### Current truth
Mutation proposals and replay-gated promotion exist, but the bundle-level end state does not.

### What remains
- persist mutation bundles
- cluster proposals by graph neighborhood
- evaluate bundles against comparative replay (`base` vs `candidate`)
- reject on regression / collapse / context bloat / orphan spikes
- keep split/merge behind flags until the bundle harness is strong enough

## Phase 5 — Freeze the proof ladder

Goal: make public claims map to frozen artifact evidence.

### Main files
- `docs/EVIDENCE.md`
- `docs/evidence/`
- `scripts/validate-openclaw-install.mjs`
- `scripts/validate-brain-runtime-behavior.ts`
- `test/brain-runtime/service.test.ts`
- `test/brain-core/replay.test.ts`

### What remains
- define proof ladder levels clearly
- require date/SHA artifact directories
- capture host-install evidence bundles, not just ad hoc command output
- add release-candidate summary markdown for every serious release proof run

## Phase 6 — Clean packaging and type surface

Goal: make installation and operator recovery boring.

### Main files
- future: `src/openclaw-sdk-compat.ts` (or equivalent compatibility wrapper)
- `tsconfig.json`
- `package.json`
- `openclaw.plugin.json`
- `README.md`

### What remains
- isolate SDK drift behind a narrow compatibility boundary
- make `npx tsc --noEmit` green
- document `brainWorkerMode=child` as the practical default
- clarify embedding support as tested reality, not wishful compatibility
- verify package contents with `npm pack --dry-run`

## What to ignore now

Do not use removed root planning docs or archived prototype code as design authority. The canonical truth lives in:

- `README.md`
- `docs/RELEASE_CONTRACT.md`
- `docs/END_STATE.md`
- `docs/EVIDENCE.md`
- the current runtime/tests/scripts in `src/`, `test/`, and `scripts/`
