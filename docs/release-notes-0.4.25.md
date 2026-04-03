# OpenClawBrain 0.4.25

Canonical install lane:

```bash
openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
openclawbrain status --openclaw-home ~/.openclaw --detailed
openclawbrain proof --openclaw-home ~/.openclaw
```

Internal published packages:

- `@openclawbrain/openclaw@0.4.25`
- `@openclawbrain/cli@0.4.25`

## Why this release exists

`0.4.25` is the operator-proof and status-surface hardening release.

The main product job here was to make the default proof path boring and trustworthy again, then expose the right thin operator truth on canonical surfaces. Before this release, proof could degrade around install/restart choreography and the canonical status/proof surfaces did not show live feedback/coverage truth directly.

This release closes that seam while keeping the public story simple: **one OpenClawBrain version, one install lane**.

## What changed

### Default proof path hardening

- `openclawbrain proof --openclaw-home ~/.openclaw` now avoids the old redundant restart choreography when install already handled restart or explicitly reported that no restart was required
- the default proof lane now completes cleanly on the live host and emits full proof artifacts (`summary.md`, `steps.json`, `verdict.json`) without the earlier skip-flag ritual
- proof still fails closed when runtime truth is genuinely missing instead of papering over real breakage

### Canonical status feedback surfaces

- `openclawbrain status --detailed` now exposes a thin `feedback` line
- it also exposes an adjacent `attrCover` line for attribution / teacher-queue coverage truth
- these lines are conservative by design: they show the truth directly, including zero/partial coverage, instead of inventing helpfulness claims

### Thin proof/operator readout

- proof health surfaces now include a thin operator readout (`helping`, `summary`, `where`, `why`, `stale/missing`)
- proof-cron health/nightly summaries now carry thin truth lines for:
  - feedback counts / traced-route coverage
  - attribution coverage
  - replay freshness

## Live verification truth

On the exercised host, this release shape was verified with:

- default proof bundle `operator-proof-20260402-installed-hardened` → `success_and_proven`
- deployed status-feedback bundle `operator-proof-20260402-installed-status-feedback` → `success_and_proven`
- canonical installed `status --detailed` now includes:
  - `feedback    helpful=0 irrelevant=0 harmful=0 supervisedTraceCount=0 routeTraceCount=0`
  - `attrCover   completedWithoutEvaluation=0 ready=0 delayed=0 budgetDeferred=0`

Important nuance: this means the surfaces are now truthful and visible; it does **not** mean the host is already showing rich nonzero learning feedback. The operator surface is fixed first so later learning improvements can be judged honestly.

## Verification

- `npm run release:plan -- --json`
- `npm run release:verify`
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
