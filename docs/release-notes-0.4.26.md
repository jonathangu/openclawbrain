# OpenClawBrain 0.4.26

Canonical install lane:

```bash
openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
openclawbrain status --openclaw-home ~/.openclaw --detailed
openclawbrain proof --openclaw-home ~/.openclaw
```

Internal published packages:

- `@openclawbrain/openclaw@0.4.26`
- `@openclawbrain/cli@0.4.26`

## Why this release exists

`0.4.26` ships the post-0.4.25 hardening tranche.

The previous release fixed the proof lane and introduced the thin operator surfaces. This release finishes the immediate follow-through:
- operator-proof capture no longer dirties the repo by default
- teacher status no longer misreports lag/staleness from double-sampled live watch state
- `feedback` and `attrCover` now surface real host truth on split-package hosts instead of falsely reading all-zero when the legacy Brain tables are empty

This is still the same product rule: **one OpenClawBrain version, one install lane**.

## What changed

### Proof artifact hygiene

- `scripts/capture-openclawbrain-operator-proof.mjs` now defaults to the shared workspace-sibling artifacts root instead of repo-root `./artifacts/operator-proof-*`
- generated proof/runtime scratch paths are ignored more cleanly so routine local proof runs stop dirtying the checkout

### Teacher status truth

- current-profile status/report now reuses one shared operator snapshot instead of sampling live watch state twice during a single status command
- this removes the misleading seam where detailed status could say teacher was lagging/stale while the live JSON/watch snapshot still showed a healthy watch heartbeat
- the host teacher path is therefore represented more honestly: fresh watch + real `no_teacher_artifacts` no-op when no teachable material exists

### Feedback / attribution coverage truth

- the traced-learning bridge now falls back truthfully when legacy Brain tables are empty on split-package hosts
- `feedback` can derive historical active-pack supervision from router artifacts
- `attrCover` can derive sparse-feedback queue truth from watch teacher-snapshot notes
- the surface stays conservative: it exposes real historical supervision and queue pressure without pretending the latest export had fresh human labels

## Live verification truth

On the exercised host after deploy, canonical `status --detailed` now shows:

- `feedback    helpful=32 irrelevant=0 harmful=0 supervisedTraceCount=32 routeTraceCount=214`
- `attrCover   completedWithoutEvaluation=0 ready=83 delayed=0 budgetDeferred=51`
- `teacherProof ... freshness=fresh queue=0/8 running=no noOp=no_teacher_artifacts`

Canonical proof also reran successfully:

- `artifacts/operator-proof-20260402-t109-post-release`
- verdict: `success_and_proven`

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

If you are already on the canonical install lane, rerun the same lane.
