# OpenClawBrain v2 — End-State Guide

This is the canonical maintainer guide for finishing the current repo to an honest 1.0.

The correct posture is:
- **no reroll**
- **keep the current trunk**
- **preserve the inherited LCM / lossless transcript-memory substrate**
- **optimize for selective intervention at important decision points, not generic "better memory"**
- **keep claims bounded and keep unique wins separate from ties**
- **keep the operator story boring, truthful, and one-home first**

If you want the public/operator-facing truth first, read these before this file:
- `README.md`
- `docs/RELEASE_CONTRACT.md`
- `docs/EVIDENCE.md`
- `docs/configuration.md`

This file is the maintainer execution map, not the public pitch.

## Canonical surfaces

These files should anchor future work:
- `README.md` — public front door and current agenda
- `docs/README.md` — docs entry point
- `CLAIMS.md` — public claims boundary
- `docs/RELEASE_CONTRACT.md` — safe public summary and frozen boundaries
- `docs/EVIDENCE.md` — proof ladder and artifact contract
- `docs/configuration.md` — practical operator setup
- `docs/proof/README.md` — proof-surface map
- `docs/END_STATE.md` — this execution guide
- `packages/openclaw/README.md` — runtime package truth
- `packages/cli/README.md` — operator CLI truth
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
- do **not** let docs drift back into broad-memory or "generally smarter" language without new proof
- do **not** count ties as product wins
- do **not** keep repo-local task artifacts or draft internal specs unless a current proof manifest, test, or operator surface still depends on them

## Current repo reality

### Already true

- a real live runtime path exists with promoted-pack serving, explicit skip / shadow modes, and fail-open behavior
- the child-worker learner boundary, replay-gated promotion, and raw evidence pipeline all exist
- split packages `@openclawbrain/openclaw@0.4.46` and `@openclawbrain/cli@0.4.46` are published
- the operator install / status / proof lane is real on the exercised host surface
- deterministic session-bound `brain_teach` proof exists
- the checked activation-first bundle separates `18` unique wins, `7` ties, and `0` regressions on reviewed felt traces
- the same bundle keeps restraint clean: `0/65` unnecessary activations and `0/69` must-not-fire failures
- the paired checked broad-live guardrail bundle shows `0/403` regressions
- the protected current-choice lane remains `full-ocb 5/5` with `regret=0` and `harm=0`
- the broader specificity/restraint cohort passes `full-ocb 12/12` with `regret=0` and `harm=0`
- the first route-level tool-capability choice proof passes for `weather.current_conditions`, with a must-fire current weather/rain case and a must-not-fire weather-definition case

### Not yet frozen

- broad memory or broad answer-quality improvement
- live weather-tool execution end to end from the capability-choice lane
- a generalized tool-capability evaluator beyond the first weather proof
- universal attribution and dated citation surfaces
- same-gateway multi-profile and broader host coverage
- boring install / recovery for another operator without repo archaeology

## Current code map

### Runtime decisioning and assembly

- `src/brain-runtime/assembler-extension.ts`
- `src/brain-runtime/service.ts`
- `src/brain-runtime/tools.ts`
- `src/brain-runtime/summary-routing-policy.ts`
- tests: `test/brain-runtime/assembler-extension.test.ts`, `test/brain-runtime/service.test.ts`

### Routing and promotion foundation

- `src/brain-core/traverse.ts`
- `src/brain-core/policy.ts`
- `src/brain-core/update.ts`
- `src/brain-core/pack.ts`
- `src/brain-core/replay.ts`
- `src/brain-core/mutator.ts`
- tests: `test/brain-core/*.test.ts`

### Evidence and supervision

- `src/brain-runtime/harvester-extension.ts`
- `src/brain-runtime/evidence-detectors.ts`
- `src/brain-harvest/*.ts`
- `src/brain-worker/worker.ts`
- `src/brain-store/store.ts`
- tests: `test/brain-runtime/harvester.test.ts`, `test/brain-worker/worker.test.ts`

### Operator surface

- `src/brain-runtime/worker-supervisor.ts`
- `src/brain-worker/child-runner.ts`
- `src/brain-worker/protocol.ts`
- `src/brain-cli.ts`
- `openclaw.plugin.json`

### Validation and proof

- `scripts/validate-openclaw-install.mjs`
- `scripts/validate-brain-runtime-behavior.ts`
- `scripts/validate-brain-teach-session-bound.ts`
- `docs/EVIDENCE.md`
- `docs/evidence/`
- `artifacts/activation-first-gating-retune/T-20260419-269/`

## Finish order

### Phase 0 — Keep repo truth aligned with repo reality

Goal: make it obvious, within a minute, what is already real, what is bounded, and what is still open.

Primary files:
- `README.md`
- `CLAIMS.md`
- `docs/RELEASE_CONTRACT.md`
- `docs/EVIDENCE.md`
- `docs/configuration.md`
- `docs/END_STATE.md`

Success looks like:
- the front-door docs do not contradict the current code or checked artifacts
- deeper maintainer docs do not drift back into inherited broad-memory labeling
- another session can orient from the canonical docs without old planning archaeology

### Phase 1 — Recover one indisputable current-choice fidelity win

Goal: make later choices stick on the real runtime path.

What this means:
- fix the later-preference redaction / prompt seam
- preserve audit/operator redaction while letting the model see the current correction text
- rerun the bounded later-preference lane cleanly

Success looks like:
- active misses recover
- keep-clean cases stay clean
- the proof language stays bounded to current-choice fidelity

### Phase 2 — Prove restraint or concrete specificity in the same policy family

Goal: show the system improves the decision boundary, not only preference recall.

Target shape:
- choose exactly one lane
- pair a positive recovery with a must-not-fire keep
- keep ties, unique wins, and regressions separated

Success looks like:
- one clean specificity or must-fire recovery
- one preserved restraint / must-not-fire keep
- no slide back into broad-memory storytelling

### Phase 3 — Keep the operator story boring

Goal: make dogfooding feel solid without repo archaeology.

Primary files:
- `README.md`
- `docs/README.md`
- `docs/configuration.md`
- `packages/openclaw/README.md`
- `packages/cli/README.md`
- install / proof scripts and status surfaces

Success looks like:
- the same one-home install / restart / `status --detailed` / `proof` lane stays canonical
- fail-open behavior stays obvious
- warmup, proof, and status expectations are explicit

### Phase 4 — Reconnect to tool-capability choice only after Phases 1 and 2

Goal: show the same policy family can govern action choice, not just context injection or abstention.

Boundary:
- one capability family
- one or two explicit tasks
- no public headline until current-choice fidelity and restraint are both real

### Phase 5 — Expand the public story only after the proof lanes are real

Goal: say only what the repo can prove, and package it cleanly.

Success looks like:
- one current-choice worked example
- one restraint / specificity worked example
- one honest explanation of what tool-capability choice still needs
- no inflation from ties or guardrails into product wins

## Tranche filter

Before starting any OpenClawBrain tranche, answer these:

1. What important decision point does this improve?
2. Is the effect on the real runtime path, not only an eval surface?
3. Is the likely outcome a unique product win, or only a tie / guardrail keep?
4. What is the paired restraint or must-not-fire guardrail?
5. Would we still want this work if we banned ourselves from saying "better memory"?

If the answers are weak, do not start the tranche.

## What to ignore

- broad "OpenClawBrain makes OpenClaw generally smarter" rhetoric
- architecture churn that does not cash out in a runtime decision
- benchmark ties presented as wins
- public story expansion ahead of proof
- tool-capability ambitions before current-choice fidelity and restraint are both real

Do not use removed root planning docs or archived prototype code as design authority. Canonical truth lives in:
- `README.md`
- `docs/RELEASE_CONTRACT.md`
- `docs/EVIDENCE.md`
- `docs/configuration.md`
- `docs/END_STATE.md`
- the current runtime/tests/scripts in `src/`, `test/`, and `scripts/`
