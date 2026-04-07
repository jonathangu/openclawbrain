# OpenClawBrain 0.4.37

Canonical install lane:

```bash
openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
openclawbrain status --openclaw-home ~/.openclaw --detailed
openclawbrain proof --openclaw-home ~/.openclaw
```

Internal published packages:

- `@openclawbrain/openclaw@0.4.37`
- `@openclawbrain/cli@0.4.37`

## Why this release exists

`0.4.37` is the continuous ongoing learning release.

The shipped story is now bigger than repo substrate:

- the first bounded continuous-learning loop is part of the product
- operator-facing status/control surfaces for that loop are shipped
- the recorded-session replay/eval blocker bundle is green on the release path
- repo-wide `tsc` is green again on the final integrated release branch

Graphify remains intentionally off the live serve path. This release builds on the existing Graphify bridge; it does not turn Graphify into current-truth authority or a hot-path dependency.

## What changed

- ships route rows plus direct online supervision inside the same-family live `route_fn`
- ships Graphify delta/reorg scheduler registry plus periodic same-family retrain as the first bounded ongoing-learning loop
- ships operator-facing continuous-learning status/control surfaces for Graphify cadence, retrain/promotion visibility, queue visibility, and pause controls
- hardens the clean-install replay/eval verification path so the recorded-session acceptance packet is green on the shipped release path
- keeps the public install lane unchanged: install, restart, status, proof

## Operator truth

This is a stronger shipped product loop, not a second operator workflow.

The canonical lane is still:

- install or upgrade with `openclawbrain install --openclaw-home ...`
- restart the gateway
- verify with `status --detailed`
- capture durable evidence with `proof`

What changed is that the background improvement loop and its status/control surfaces are now part of the shipped product instead of only local repo truth.

## Focused verification

- `npm run release:verify`
- `node scripts/verify-release-docs-drift.mjs`
- `npx tsc -p tsconfig.json --noEmit --pretty false`
- `npx vitest run test/brain-runtime/continuous-learning-status.test.ts test/graphify-scheduler.test.ts test/brain-core/cold-start-router-periodic-retrain.test.ts test/graphify-final-replay-proof.test.ts test/continuous-learning-acceptance.test.ts test/brain-core/graphify-training-bridge.test.ts test/canonical-frozen-trace-set.test.ts test/replay-proof-lane.test.ts test/frozen-recorded-session-eval-gate-economics.test.ts test/frozen-recorded-session-eval-gate.test.ts test/eval/comparative-eval-runner.test.ts --reporter=dot`

## Upgrade

```bash
openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
openclawbrain status --openclaw-home ~/.openclaw --detailed
openclawbrain proof --openclaw-home ~/.openclaw
```

If you already run the canonical install lane, rerun the same lane. The public commands do not change; the shipped background loop and operator surfaces do.
