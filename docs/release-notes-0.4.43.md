# OpenClawBrain 0.4.43

Canonical install lane:

```bash
openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
openclawbrain status --openclaw-home ~/.openclaw --detailed
openclawbrain proof --openclaw-home ~/.openclaw
```

Internal published packages:

- `@openclawbrain/openclaw@0.4.43`
- `@openclawbrain/cli@0.4.43`

## Why this release exists

`0.4.43` exists to publish the new OpenClawBrain-only quality tranche without waiting on regular OpenClaw main.

The repo now has two real new slices beyond `0.4.42`:

1. the cold-start continuation + explainable eval tranche
2. the compaction-oriented routing / compact-health / retry-identity tranche

This release packages both into the public OCB surface.

## What changed

- the shipped OCB runtime now carries the cold-start continuation work: same-family warm-start retrain truth, clearer base-vs-delta status/proof reporting, and the converged explainable eval/reporting lane
- the runtime and reporting surfaces now expose budgeted routing fit metrics even when clipping is avoided, so retrieval fit quality stays legible before overflow
- the scorecard surface now includes the first compact-health metrics: expand-before-assert, branch-heavy expand-to-source, non-fresh summary prevalence, snapshot-vs-condense share, and token reduction per compaction pass
- retry-visible identity now survives through trace / route-row / observation surfaces so downstream OpenClaw-side dedupe has a stable handoff surface

## Operator truth

This is still an OpenClawBrain release, not a regular OpenClaw-main release.

The public lane stays the same:

- run `openclawbrain install --openclaw-home ...`
- restart the gateway
- verify with `status --detailed`
- capture durable evidence with `proof`

The important change is that the OCB package now includes the new continuation/eval and compaction-safe retrieval work for any OpenClawBrain user.

## What success looks like

After upgrading, a healthy host should still show the same converged operator truth, while the shipped OCB layer gains:

1. clearer cold-start continuation lineage and explainable eval surfaces
2. visible budgeted routing / compact-health reporting
3. stable retry-visible identity handoff for downstream dedupe-aware behavior

## Focused verification

- `npx vitest run test/brain-core/cold-start-router-contracts.test.ts test/brain-core/cold-start-router-periodic-retrain.test.ts test/brain-core/cold-start-router-trainer.test.ts test/brain-core/cold-start-router-runtime.test.ts test/brain-runtime/continuous-learning-status.test.ts test/provenance-audit-chain.test.ts test/eval/openclawbrain-explainable-scorecard.test.ts test/eval/comparative-eval-runner.test.ts test/frozen-recorded-session-eval-gate.test.ts test/proof-cron.test.ts test/replay-proof-lane.test.ts`
- `npx vitest run test/brain-core/route-rows.test.ts test/brain-core/trace.test.ts test/brain-runtime/observation.test.ts test/brain-runtime/service.test.ts test/brain-runtime/summary-routing-policy.test.ts test/brain-runtime/assembler-extension.test.ts test/eval/openclawbrain-explainable-scorecard.test.ts test/route-quality-summary.test.ts`
- `npm run release:verify`

## Upgrade

```bash
openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
openclawbrain status --openclaw-home ~/.openclaw --detailed
openclawbrain proof --openclaw-home ~/.openclaw
```

This should behave like the same no-drama OCB lane as before, but with the newest continuation/eval and compaction-safe retrieval/reporting work included in the shipped package surface.
