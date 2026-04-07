# OpenClawBrain 0.4.36

Canonical install lane:

```bash
openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
openclawbrain status --openclaw-home ~/.openclaw --detailed
openclawbrain proof --openclaw-home ~/.openclaw
```

Internal published packages:

- `@openclawbrain/openclaw@0.4.36`
- `@openclawbrain/cli@0.4.36`

## Why this release exists

`0.4.36` is the Graphify + hardening follow-up release.

Graphify remains intentionally off the live serve path. The bridge belongs in the artifact-first compiler / diagnostic lane, not as current-truth authority and not as a hot-path dependency. This release packages the follow-up publish after the Graphify integration and release hardening work landed.

## What changed

- keeps the Graphify bridge lanes off-path and preserves the cold-start / maintenance-diagnostics boundary
- preserves exact dependency pinning and the dependency-policy guard so the release posture stays narrow
- keeps the hardened release verification path intact while repinning the OpenClaw peer to `2026.4.5`
- aligns the repo docs, public site, and Jon badge to the new published version

## Operator truth

This is a follow-up release, not a new operator workflow.

The canonical lane is still:

- install or upgrade with `openclawbrain install --openclaw-home ...`
- restart the gateway
- verify with `status --detailed`
- capture durable evidence with `proof`

The difference is the shipped truth: Graphify stays off the serve path, and the release/posture hardening now travels with the published version.

## Focused verification

- `npm run release:verify`
- `node scripts/verify-release-docs-drift.mjs`
- focused Graphify proof / replay checks from the follow-up lane

## Upgrade

```bash
openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
openclawbrain status --openclaw-home ~/.openclaw --detailed
openclawbrain proof --openclaw-home ~/.openclaw
```

If you already run the canonical install lane, rerun the same lane. Graphify stays off-path; the public install story does not change.
