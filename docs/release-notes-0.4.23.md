# OpenClawBrain 0.4.23

Canonical install lane:

```bash
openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
openclawbrain status --openclaw-home ~/.openclaw --detailed
openclawbrain proof --openclaw-home ~/.openclaw
```

Internal published packages:

- `@openclawbrain/openclaw@0.4.22`
- `@openclawbrain/cli@0.4.23`

## Why this release exists

This release publishes the single-source-of-truth follow-up after `0.4.22`.

The underlying runtime was already working, but the operator-facing story still had two ugly seams:

- watcher freshness could flip to `stale_snapshot` on a knife-edge heartbeat threshold
- context-memory-management and health truth were fragmented across status, proof, and docs

`0.4.23` makes those operator surfaces tell the truth more coherently.

## What changed

### Watcher freshness truth

- watcher freshness now reports a structured `lagging` state instead of jumping straight from healthy to `stale_snapshot`
- near-threshold heartbeat jitter is treated as aging/lagging rather than a fake split-brain stale failure
- proof/status surfaces now read the same structured watch truth

### Operator health truth

- proof health snapshots and nightly aggregates now consume a shared `operatorHealth` contract
- operator-facing output now makes `partial`, `unknown`, `stale`, and `unhealthy` semantics explicit instead of implying stronger confidence than the live probe supports
- latest operator-health truth is carried through the nightly aggregate surface

### Context-management truth

- `openclawbrain status` now exposes a canonical `contextManagement` model
- that model describes:
  - summary spine + protected fresh tail hot context
  - summary freshness vs non-fresh states
  - expand-to-source behavior
  - prefetch lifecycle
  - budget controls
- stale docs/operator seams were removed, including the phantom `openclawbrain context` command claim

## Important caveats

- this is a CLI/operator-surface release; the published runtime payload stays `@openclawbrain/openclaw@0.4.22`
- the public release story is still **OpenClawBrain 0.4.23** with one canonical install lane; internal split-package versions remain maintainer detail

## Verification

- `npm run release:plan -- --json`
- `npm run release:verify`
- `npm view @openclawbrain/cli version`
- `npm view @openclawbrain/openclaw version`

## Upgrade

```bash
openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
openclawbrain status --openclaw-home ~/.openclaw --detailed
```

If you are already on the canonical install lane, rerun the same lane. Do not use the retired compatibility package path.
