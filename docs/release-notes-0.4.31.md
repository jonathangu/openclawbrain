# OpenClawBrain 0.4.31

Canonical install lane:

```bash
openclawbrain install --openclaw-home ./openclaw-cormorantai
openclaw gateway restart
openclawbrain status --openclaw-home ./openclaw-cormorantai --detailed
openclawbrain proof --openclaw-home ./openclaw-cormorantai
```

Internal published packages:

- `@openclawbrain/openclaw@0.4.31`
- `@openclawbrain/cli@0.4.31`

## Why this release exists

`0.4.31` closes the last daemon-vs-hook proof seam from issue `#16`.

The operator lane was already honest enough to say when the daemon-side CLI runtime and the selected OpenClaw home's installed hook were split surfaces. But on a healthy shared host, the installed hook version was getting dropped before the current-profile status/proof surface was rendered. That left the host stuck at `split_path_version_unverified` / `success_but_proof_incomplete` even when the on-disk hook already had a concrete proven version.

This release keeps that version identity all the way through detailed status and proof.

## What changed

- preserves installed-hook `packageVersion` in the current-profile status/report object instead of dropping it during report shaping
- makes detailed status print the hook as `@openclawbrain/openclaw@<version>` when package/manifest truth is already present on disk
- lets proof bundles positively prove daemon-vs-hook same-version convergence on a healthy shared host
- keeps restart/profile-token inference warnings explicit without confusing them for Brain-runtime failure

## Operator truth

Use the same exact `--openclaw-home` path through install, status, proof, rollback, and troubleshooting. That path can be:

- the default `~/.openclaw`
- a profile-specific home like `~/.openclaw-example`
- an explicit nonstandard path like `./openclaw-cormorantai`

The important thing is not the shape of the path; it is pinning the same chosen home consistently.

On a healthy host, the expected status/proof boundary is now concrete rather than ambiguous:

- `surface ... skew=split_path_same_version converge=converged`
- daemon surface includes `@openclawbrain/cli@<version>`
- hook surface includes `@openclawbrain/openclaw@<version>`
- `openclawbrain proof --openclaw-home <path>` can end `success_and_proven`

## Upgrade

```bash
openclawbrain install --openclaw-home ./openclaw-cormorantai
openclaw gateway restart
openclawbrain status --openclaw-home ./openclaw-cormorantai --detailed
openclawbrain proof --openclaw-home ./openclaw-cormorantai
```
