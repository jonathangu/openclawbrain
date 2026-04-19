# OpenClawBrain 0.4.44

Canonical install lane:

```bash
openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
openclawbrain status --openclaw-home ~/.openclaw --detailed
openclawbrain proof --openclaw-home ~/.openclaw
```

Internal published packages:

- `@openclawbrain/openclaw@0.4.44`
- `@openclawbrain/cli@0.4.44`

## Why this release exists

`0.4.44` exists to ship the post-win OpenClawBrain tranche that turned the binary-gate publishable win into stronger package truth.

This release packages four connected slices into the public OCB surface:

1. hard binary-gate promotion checks plus automatic threshold selection
2. route-decision summaries on runtime / proof / economics surfaces
3. compact runtime-mode proof tables with explicit learned-route deltas
4. a safe report-only teacher proposal artifact lane for promotable `compiler` / `lint` proposals

## What changed

- hardens the binary-gate promotion flow so reviewed must-fire/trap truth and broad-live vetoes are part of the release-facing promotion boundary instead of hand-checked side evidence
- adds automatic activation-threshold selection and fixes the merged abstention truth surface so known duplicate trap overlap no longer distorts scoring
- makes route-decision summaries and compact runtime-mode proof tables visible in the proof lane using the familiar `no_brain`, `vector_only`, `graph_prior_only`, and `learned_route` vocabulary
- adds a report-only teacher proposal artifact surface with evidence refs, replay hooks, proof linkage, rollback linkage, and markdown rendering, without claiming live teacher-driven mutation
- aligns README, docs, and canonical public site surfaces to `0.4.44`

## Operator truth

This is still an OpenClawBrain release, not a regular OpenClaw main release.

The public lane stays the same:

- run `openclawbrain install --openclaw-home ...`
- restart the gateway
- verify with `status --detailed`
- capture durable evidence with `proof`

The important change is that the shipped package now includes the new hardening/proof tranche users can already see on the public proof surfaces.

## Honest boundary

This release improves promotion hardening, proof visibility, and safe teacher reporting.
It does **not** claim:

- broad online proof
- universal learned-route superiority
- live teacher-driven graph mutation

## What success looks like

After upgrading, a healthy host should still show the same converged operator truth, while the shipped OCB layer gains:

1. stronger binary-gate promotion safety
2. more legible route-decision and runtime-mode proof surfaces
3. a real report-only teacher proposal review lane

## Focused verification

- `npm test -- test/activation-first-gating-retune-runner.test.ts test/scripts/grade-binary-gate-v2-splits.test.ts test/scripts/build-binary-gate-v2-tranches.test.ts test/brain-core/cold-start-router-replay-gate.test.ts`
- `npm run test:learned-route-mission`
- `npm test -- test/brain-core/route-decision-event.test.ts test/brain-runtime/service.test.ts test/economics-scorecard.test.ts test/proof-cron.test.ts`
- `npm test -- test/teacher-v3-proposal-artifact.test.ts`
- `npm run release:verify`

## Upgrade

```bash
openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
openclawbrain status --openclaw-home ~/.openclaw --detailed
openclawbrain proof --openclaw-home ~/.openclaw
```

This should behave like the same no-drama OCB lane as before, but with the new post-win hardening, proof packaging, and report-only teacher artifact work included in the shipped package surface.
