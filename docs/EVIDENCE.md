# OpenClawBrain v2 — Evidence Ladder

This document defines what proof must exist before public claims are treated as frozen.

The current achievable agenda is narrow on purpose:

- current-choice fidelity
- restraint / specificity
- honest operator proof surfaces
- route-level capability choice, starting with the bounded weather lane

The point is not to accumulate logs for their own sake. The point is to make the repo's claims auditable without flattening ties into wins or using vague "memory got better" language.

## What counts as evidence

Evidence should answer five questions clearly:

1. **What exact claim was being tested?**
2. **What command or harness produced the result?**
3. **What environment, model, and config did it run with?**
4. **What remains open after this run?**
5. **If the claim is comparative, what was a unique win, a tie, a regression, or a keep-clean negative?**

If a bundle cannot answer those questions quickly, or if it treats ties as product wins, it is not a good release artifact yet.

## Artifact layout

Store release and benchmark artifacts under:

```text
docs/evidence/YYYY-MM-DD/<git-sha>/
```

Each serious bundle should contain at minimum:
- `summary.md`
- `validation-report.json`
- `status.json`
- `doctor.json`
- `config-snapshot.json`
- `logs.txt`

If a routed or decision-quality path is part of the claim, also include:
- `trace.json`
- a scorecard or summary table that separates `better`, `tied`, and `worse`
- the paired negative or restraint counts when applicable

For Level 2 host/operator runs, also include the pre-run diagnostic ladder outputs:
- `status-all.txt`
- `gateway-probe.txt`
- `gateway-status.txt`
- `channels-status.txt`

If a run is partial, `summary.md` must say exactly what was and was not proven.

## Reading evidence correctly

Not every bundle under `docs/evidence/` is frozen release proof.

Four categories matter:

### 1. Current operator truth bundles
Use these when the repo is claiming install / runtime / reporting truth publicly.

### 2. Current bounded decision-lane bundles
Use these when the repo is claiming a named selective-intervention win or restraint keep publicly.

### 3. Partial proof bundles
Useful for progress, but the summary must explicitly say the run was partial and what boundary remains open.

### 4. Historical failure bundles
Useful when they truthfully capture seam drift or operator failures, but they must not be mistaken for the current success boundary.

## Proof ladder

### Level 1 — Mechanism and safety proofs

Purpose: prove the runtime, serving, learning, and fail-open primitives in isolation.

Primary surfaces:
- `test/brain-core/policy.test.ts`
- `test/brain-core/traverse.test.ts`
- `test/brain-core/update.test.ts`
- `test/brain-core/replay.test.ts`
- `test/brain-runtime/service.test.ts`
- `test/brain-runtime/harvester.test.ts`
- `src/brain-runtime/worker-supervisor.ts`
- `src/brain-worker/protocol.ts`
- `scripts/validate-brain-runtime-behavior.ts`

Required claims:
- promoted-pack serving is real
- serve-from-last-promoted-pack survives worker failure
- explicit skip and shadow modes are real
- replay gates block regressions
- child-worker supervision records restart truth
- raw harvesting preserves richer evidence before worker-side resolution

### Level 2 — Operator truth proofs

Purpose: prove install / attach / `status --detailed` / `proof` on the real OpenClaw host surface.

Primary surfaces:
- `scripts/validate-openclaw-install.mjs`
- `scripts/validate-brain-teach-session-bound.ts`
- `openclawbrain proof --openclaw-home <path>`
- checked-in bundles under `docs/evidence/`

Required claims:
- the one-home operator lane works end to end
- runtime load truth is visible
- shadow / skip surfaces remain truthful
- proof bundles capture status, doctor, config, and logs for the selected home
- the boundary stays explicit: one exercised host surface, not broad host generality

### Level 3 — Bounded selective-intervention proofs

Purpose: prove one named decision lane with paired restraint.

Required claims:
- one named positive lane
- one named keep-clean negative or must-not-fire lane
- unique wins, ties, and regressions reported separately
- exact command, artifact, and manifest made explicit
- honest boundary against broader live answer-quality claims

The first two target lanes are:
- later-preference current-choice fidelity
- restraint or concrete-specificity recovery

Current checked example:
- `artifacts/activation-first-gating-retune/T-20260419-269/scorecard.json` shows `18` better, `7` tied, `0` worse on `felt_resume_25`, plus `0/65` unnecessary activations and `0/69` must-not-fire failures
- `artifacts/activation-first-gating-retune/T-20260419-269/broad-live-comparative-eval/summary.md` shows `0/403` broad-live regressions; that is guardrail evidence, not `403` wins

### Level 4 — Tool-capability choice proofs

Purpose: prove the same policy family improves action choice, not just context injection or abstention.

Required claims:
- one capability family
- one or two explicit tasks
- unique wins, ties, and regressions reported separately
- a paired restraint boundary showing when not to intervene

This is not current frozen proof.

## Release checklist

Do not claim a release candidate is fully proven unless the bundle includes:
- exact commit SHA
- exact validation command(s) or harnesses
- model + embedding configuration used
- pass/fail results for host harness assertions
- status and doctor snapshots
- at least one trace proving the routed path being claimed
- unique wins, ties, and regressions split when the claim is comparative
- paired keep-clean negative counts when the claim is about decision quality
- a short summary of what remains open

For an operator-grade release, the proof ladder should also be enforced by CI or another repeatable release gate rather than living only as prose.

## Current proof truth

As of the current trunk:

- **Level 1:** ✅ real
- **Level 2:** ✅ current on the exercised host surface
- **Level 3:** 🟡 partially current
  - the activation-first bounded scorecard exists and is current
  - the later-preference current-choice lane is not yet frozen
  - the second specificity / restraint lane is not yet frozen
- **Level 4:** ❌ not current

Current proof surfaces to point at:
- `docs/evidence/2026-03-16/4ccd71a22418b9170128b8d948f5a95801a10380/`
- `openclawbrain proof --openclaw-home ~/.openclaw --skip-install --skip-restart`
- `artifacts/activation-first-gating-retune/T-20260419-269/scorecard.json`
- `artifacts/activation-first-gating-retune/T-20260419-269/broad-live-comparative-eval/summary.md`

Remaining boundaries (honestly scoped):
- raw prompt-driven `openclaw agent --local` is **not** the release proof boundary for `brain_teach`
- operator proof does **not** yet imply later-preference current-choice fidelity, general specificity, or tool-capability choice
- frozen operator proof does **not** yet imply universal dated citations, exact attribution on every learning path, same-gateway multi-profile support, or broad live answer-quality gains

## What CI now enforces

The current release smoke gate requires:
- tests
- a mainline publish ref with no pending changesets and a matching split release-note / changelog pair
- a fresh checked-in proof bundle for the frozen host/operator lane
- the expected proof files (`summary.md`, `validation-report.json`, `status.json`, `doctor.json`, `config-snapshot.json`, `logs.txt`, `trace.json`, and the pre-run ladder snapshots)
- the current replay / eval assertion set for the frozen host/operator lane
- package verification (`npm pack --dry-run` or stronger equivalent)

## What CI still does not enforce

The intended release gate should still grow into:
- tests
- broader evidence-ladder checks appropriate to every release claim
- host/runtime validation reruns that match the repo's public contract, rather than only checking in a fresh frozen bundle

Docs should stay honest that the current gate is a proof-freshness smoke boundary, not a full host rerun on every CI execution.
