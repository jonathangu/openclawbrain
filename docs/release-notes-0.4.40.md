# OpenClawBrain 0.4.40

Canonical install lane:

```bash
openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
openclawbrain status --openclaw-home ~/.openclaw --detailed
openclawbrain proof --openclaw-home ~/.openclaw
```

Internal published packages:

- `@openclawbrain/openclaw@0.4.40`
- `@openclawbrain/cli@0.4.40`

## Why this release exists

`0.4.40` exists because the unified operator/proof tranche already landed on trunk, but the public split-package release surface still told the older `0.4.39` story.

The install-convergence repair from `0.4.39` stays intact. What changes here is the public release truth: the bounded-anytime summary, economics scorecard, route-quality summary, and provenance audit chain are now the operator/proof surfaces the repo leads with for this tranche.

## What changed

- `status --detailed` now exposes a bounded-anytime summary with deadline posture, clip/fail-open rates, and recent branch-behavior context
- `status --detailed` also exposes a route-quality summary with replay verdict, `STOP_LOCAL` health, tool-action-priors health, control posture, and rollback/proof linkage when available
- proof-cron health/nightly outputs now write bounded economics scorecards (`economics-scorecard.json` / `.md`) with explicit measured / derived / proxy labels
- `openclawbrain proof --openclaw-home ...` now captures a provenance audit chain (`provenance-audit-chain.md` / `.json`) that ties serve-decision rows to attribution truth, learning-update truth, and promotion/proof truth
- README/docs/release surfaces now point at `0.4.40` instead of leaving the new operator/proof tranche repo-only

## Operator truth

This is a release-surface alignment cut, not a new install workflow.

The canonical lane stays the same:

- run `openclawbrain install --openclaw-home ...`
- restart the gateway
- verify `status --detailed`
- capture durable evidence with `proof`

The honest public change is that the verification lane now has stronger bounded summaries and audit surfaces around it. The live serve path is still promoted OpenClawBrain packs; these additions are operator/proof surfaces, not a new hot-path dependency.

## Proof boundary

The underlying truth/proof tranche was already validated locally before this release cut:

- combined regression battery for bounded-anytime, economics, route-quality, teacher-v3 proof, provenance, and runtime service surfaces
- fresh operator-proof bundle with verdict `success_and_proven`
- fresh teacher-v3 proof bundle that keeps shipped-vs-target proof packaging explicit

See `task-artifacts/T-20260407-180/closeout.md`, `artifacts/operator-proof-20260408-012317Z`, and `artifacts/teacher-v3-proof/teacher-v3-proof-20260408-012317Z` for the tranche evidence that this release packages into the public version story.

## Focused verification

- `node scripts/release-plan.mjs`
- `npm run release:verify:docs-drift`

## Upgrade

```bash
openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
openclawbrain status --openclaw-home ~/.openclaw --detailed
openclawbrain proof --openclaw-home ~/.openclaw
```

If a host is already healthy on `0.4.39`, the upgrade path is still this same lane. `0.4.40` mainly makes the operator proof and release surfaces tell the fuller truth about what the shipped system now exposes.
