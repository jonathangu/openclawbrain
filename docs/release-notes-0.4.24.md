# OpenClawBrain 0.4.24

Canonical install lane:

```bash
openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
openclawbrain status --openclaw-home ~/.openclaw --detailed
openclawbrain proof --openclaw-home ~/.openclaw
```

Internal published packages:

- `@openclawbrain/openclaw@0.4.24`
- `@openclawbrain/cli@0.4.24`

## Why this release exists

This release repairs the accidental mixed `0.4.23` publish state.

Jon’s product rule is explicit: OpenClawBrain should present one single visible version number and one single install path to users. `0.4.23` shipped the CLI follow-up without republishing the runtime payload, which left the public split-package surface mixed even though the intended product story was singular.

`0.4.24` fixes that cleanly by realigning both published packages on the same visible version.

## What changed

### One-version repair

- `@openclawbrain/openclaw` is republished at `0.4.24`
- `@openclawbrain/cli` is published at `0.4.24`
- the public release story returns to a single visible version: **OpenClawBrain 0.4.24**
- the canonical install lane remains one front door: `openclawbrain install --openclaw-home ...`

### Included operator fixes

This release carries forward the single-source-of-truth tranche fixes:

- watcher freshness now reports a structured `lagging` state instead of knife-edge `stale_snapshot` flips
- proof/status/nightly surfaces share an explicit `operatorHealth` contract
- `openclawbrain status` exposes a canonical `contextManagement` model for hot context, freshness, prefetch lifecycle, expand-to-source behavior, and budget controls
- stale docs/operator seams are cleaned up, including the phantom `openclawbrain context` command claim

## Important caveats

- `0.4.23` exists historically as an already-published mixed package state and cannot be unpublished retroactively; `0.4.24` is the clean repair release that restores the intended product contract
- users should follow the canonical install lane and treat split-package version details as maintainer-only implementation detail

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
```

If you are already on the canonical install lane, rerun the same lane. Do not use the retired compatibility package path.
