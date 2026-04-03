# OpenClawBrain 0.4.27

Canonical install lane:

```bash
openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
openclawbrain status --openclaw-home ~/.openclaw --detailed
openclawbrain proof --openclaw-home ~/.openclaw
```

Internal published packages:

- `@openclawbrain/openclaw@0.4.27`
- `@openclawbrain/cli@0.4.27`

## Why this release exists

`0.4.27` is the closed-loop repair release.

Before this release, the live background-learning loop could silently stall even when real user feedback existed. Historical serve decisions could fall out of the bounded tail before feedback arrived, some real interaction shapes were still excluded from teacher labeling, and the operator surfaces were not honest enough about where the loop was progressing versus where it was merely alive.

This release closes that seam while keeping the public contract simple: **one OpenClawBrain version, one install lane**.

## What changed

### Closed-loop learning recovery

- historical serve-decision recovery now reaches beyond the bounded tail when exact feedback-linked provenance is present
- learner/watch recovery no longer depends only on whatever happened to still be in the most recent tail slice
- real historical supervision can therefore reconnect to the original serve decisions instead of silently disappearing from the learning path

### Teacher / materialization fixes

- `message_delivered` interactions are now admitted into the teacher labeler so real exported feedback is not dropped on that interaction-shape seam
- serve-decision inputs passed into learning/materialization are compacted so the loop does not need to haul oversized raw decision payloads by default
- this keeps the repaired path practical enough to materialize and promote candidate packs again on the exercised host

### More honest operator truth

- status/learning surfaces separate harvested artifacts, eligible feedback, matched decisions, supervised trajectories, router updates, and promoted-pack state more cleanly
- runtime truth remains explicit about split-surface uncertainty instead of pretending the global CLI path and installed extension path are automatically identical
- the result is a more honest operator surface: healthy daemon/watch activity is no longer confused with healthy end-to-end learning progress

## Live verification truth

On the exercised host after deploy, one-shot learning produced:

- `supervisionCount = 58`
- `routerUpdateCount = 65`
- `materialized = pack-213597b7`
- `promoted = true`

Canonical detailed status/proof truth after deploy included:

- `serve       state=serving_active_pack`
- `serving     pack=pack-213597b7`
- `learnFlow   harvested=762 eligible=83 loaded=yes pack=pack-213597b7 matched=3111 supervised=32 updated=65`
- `traced      present=yes ... supervision=58 updates=65 ... pack=pack-213597b7`

Canonical proof bundle:

- `artifacts/operator-proof-20260403-160724Z`
- verdict: `success_and_proven`

GitHub issue cluster closed by this repair:

- `#9`
- `#10`
- `#11`
- `#12`
- `#13`
- `#14`

Final verification also closed the two older lingering issues that were already resolved in trunk but still open on GitHub:

- `#1`
- `#4`

## Verification

- `node --test packages/cli/dist/test/status-learning-path.test.js packages/openclaw/dist/test/status-learning-path.test.js`
- `npx vitest run test/index-brain-teach-session-binding.test.ts test/brain-runtime/assembler-extension.test.ts test/proof-smoke-gate.test.ts`
- `node scripts/verify-proof-smoke.mjs`
- `npm view @openclawbrain/openclaw version`
- `npm view @openclawbrain/cli version`
- canonical local install + detailed status/proof verification on `~/.openclaw`

## Upgrade

```bash
openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
openclawbrain status --openclaw-home ~/.openclaw --detailed
openclawbrain proof --openclaw-home ~/.openclaw
```

If you are already on the canonical install lane, rerun the same lane. Do not use the retired compatibility package path.
