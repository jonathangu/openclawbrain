# OpenClawBrain 0.4.30

Canonical install lane:

```bash
openclawbrain install --openclaw-home ./openclaw-cormorantai
openclaw gateway restart
openclawbrain status --openclaw-home ./openclaw-cormorantai --detailed
openclawbrain proof --openclaw-home ./openclaw-cormorantai
```

Internal published packages:

- `@openclawbrain/openclaw@0.4.30`
- `@openclawbrain/cli@0.4.30`

## Why this release exists

`0.4.30` is the runtime-surface convergence release. The operator footgun behind issue `#15` was real: OpenClawBrain can run across two live code surfaces — the daemon-side CLI runtime and the selected OpenClaw home's installed hook/runtime-guard — so upgrades and hotfixes could look done when only one side actually moved.

This release closes that seam on the canonical operator lane.

## What changed

- status/proof now surface daemon-vs-hook split-runtime skew more explicitly
- converge/install verification now blocks half-converged daemon vs installed-hook states instead of reading like success
- proof capture treats half-converged runtime surfaces as blocking truth, not a soft warning
- docs/help/examples now treat explicit custom homes like `./openclaw-cormorantai` as first-class through `--openclaw-home`
- the one-version / one-install-lane public story stays intact

## Operator truth

Use the same exact `--openclaw-home` path through install, status, proof, rollback, and troubleshooting. That path can be:

- the default `~/.openclaw`
- a profile-specific home like `~/.openclaw-example`
- an explicit nonstandard path like `./openclaw-cormorantai`

The important thing is not the shape of the path; it is pinning the same chosen home consistently.

## Upgrade

```bash
openclawbrain install --openclaw-home ./openclaw-cormorantai
openclaw gateway restart
openclawbrain status --openclaw-home ./openclaw-cormorantai --detailed
openclawbrain proof --openclaw-home ./openclaw-cormorantai
```
