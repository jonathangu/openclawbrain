# OpenClawBrain v2 — Release Contract

This is the sharp truth surface for the repo.

Use these public labels consistently:
- **selective-intervention agenda**
- **live-path implemented**
- **operator-checked**

Current truthful state:
- **selective-intervention agenda:** yes
- **live-path implemented:** yes
- **operator-checked:** yes

That contract means OpenClawBrain already has a real runtime path, promoted-pack serving, fail-open behavior, and an exercised operator install / status / proof lane. It also has bounded checked scorecards for activation-first intervention and restraint. It does not yet have frozen public proof for later-preference current-choice fidelity, a second specificity / restraint lane, or tool-capability choice as a headline claim.

## 1. Safe public claims now

### Operator-checked runtime

- the one-home install / attach / `status --detailed` / `proof` lane is real
- the runtime serves only promoted packs and can fail open from the last promoted pack
- explicit `use_brain`, `shadow`, and named skip modes are real
- the child-worker learning boundary is real
- detailed status and proof surfaces expose load, pack, and runtime-guard truth for the selected home

Primary files:
- `src/brain-runtime/assembler-extension.ts`
- `src/brain-runtime/service.ts`
- `src/brain-runtime/worker-supervisor.ts`
- `src/brain-worker/child-runner.ts`
- `packages/cli/dist/src/cli.js`
- `packages/cli/dist/src/proof-command.js`

### Bounded selective-intervention proof

- the checked activation-first bundle separates unique wins from ties: `18` better, `7` tied, `0` worse on `felt_resume_25`
- restraint stayed clean: `0/65` unnecessary activations and `0/69` must-not-fire failures
- broad-live guardrail replay saw `0/403` regressions on the checked bundle
- the `0/403` number is a guardrail result, not `403` product wins

Primary artifacts:
- `artifacts/activation-first-gating-retune/T-20260419-269/scorecard.json`
- `artifacts/activation-first-gating-retune/T-20260419-269/broad-live-comparative-eval/summary.md`

### Foundation already in repo

- a paper-faithful routing core exists
- replay-gated promotion exists
- raw evidence -> resolved labels flow exists
- correction-first assembly and summary-aware routing exist

Primary files:
- `src/brain-core/traverse.ts`
- `src/brain-core/policy.ts`
- `src/brain-core/update.ts`
- `src/brain-core/replay.ts`
- `src/brain-core/pack.ts`
- `src/brain-runtime/summary-routing-policy.ts`
- `src/brain-runtime/user-memory-proposals.ts`
- `src/brain-worker/worker.ts`

These matter, but they are not by themselves the public product win.

## 2. Implemented foundation, not yet frozen as a public headline

- later-preference current-choice fidelity on the real runtime path
- a second restraint / specificity lane beyond activation-first gating
- tool-capability choice as a proved operator-facing lane
- universal attribution and dated citation surfaces
- same-gateway multi-profile and shared-write proof
- broader host/profile coverage beyond the exercised surface
- boring install / recovery behavior for another operator without repo archaeology

## 3. Release discipline

- claims should name the exact lane, command or artifact, environment, and open boundaries
- decision-quality claims must separate unique wins, ties, regressions, and keep-clean negatives
- operator proof proves install / runtime / reporting truth for one exercised host surface; it does not by itself prove broad answer quality
- if a release only improves proof or operator surfaces, say that plainly instead of expanding the product story

## Safe public summary

> OpenClawBrain already has a real live runtime path and an exercised operator proof lane. The current checked scorecards show bounded selective-intervention wins plus clean restraint on named bundles. The next proof rungs are current-choice fidelity, a second restraint / specificity lane, and only then tool-capability choice.
