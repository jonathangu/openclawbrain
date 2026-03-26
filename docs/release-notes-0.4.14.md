# OpenClawBrain 0.4.14 / 0.4.3 split-package release notes

Published packages:

- `@openclawbrain/openclaw@0.4.3`
- `@openclawbrain/cli@0.4.14`

This release carries the canonical new-user split-package lane forward after the payload-sync work landed in the publishable package payloads.

## Why this release exists

`0.4.13 / 0.4.2` closed the source-specific `STOP_LOCAL` gap on the split lane, but the public operator story still had two release-side seams:

1. the CLI now converges install/update/repair through `openclawbrain install --openclaw-home <path>`, yet the release workflow still only published the plugin package
2. the plugin/runtime payload now carries additional config and runtime-truth surface, but the versioning flow did not automatically keep `packages/openclaw/openclaw.plugin.json` aligned with `packages/openclaw/package.json`

This release closes those seams on the canonical public lane:

```bash
openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
openclawbrain status --openclaw-home ~/.openclaw --detailed
openclawbrain proof --openclaw-home ~/.openclaw
```

## What changed

### `@openclawbrain/cli@0.4.14`

- `openclawbrain install` is now the explicit public front door for one OpenClaw home, with converge logic that installs or refreshes `@openclawbrain/openclaw`, repairs hook wiring, and only restarts when runtime-affecting state changed
- operator docs and CLI help now describe `proof` as the current follow-up surface, while keeping `--proof` framed as the intended future add-on to `install`
- proof and status surfaces stay aligned with the selected `--openclaw-home`, so install, restart, verify, and durable evidence all read as one operator lane

### `@openclawbrain/openclaw@0.4.3`

- the published plugin manifest now exposes the bounded-runtime config surface the runtime already honors, including compile deadline, retrieval budget fraction, max per-node fanout, and frontier size
- runtime payload truth now preserves bounded-context and provenance details through trace, teacher, worker, and status surfaces instead of leaving the split package behind the repo state
- the package release path now keeps `openclaw.plugin.json` version-locked to `packages/openclaw/package.json`, so publish verification stays truthful after versioning

## Proof coverage

Focused package-facing proof added or exercised in-repo:

- `packages/cli/dist/test/install-converge.test.js`
  - proves the CLI converge planner only requires restart when install state actually changed
- `packages/openclaw/dist/test/runtime-budget-forwarding.test.js`
  - proves bounded-runtime controls reach the published package payload
- `packages/openclaw/dist/test/teacher-decision-match.test.js`
  - proves published runtime truth still materializes decision-match state through the split payload

## Verification summary

Execute before publish:

```bash
npm test
npm run release:verify:openclaw
npm run release:verify:cli
```

Publish in this order so the front-door CLI never points at an older plugin payload than the one it is meant to converge:

```bash
npm publish ./packages/openclaw
npm publish ./packages/cli
```
