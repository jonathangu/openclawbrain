# OpenClawBrain 0.4.39

Canonical install lane:

```bash
openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
openclawbrain status --openclaw-home ~/.openclaw --detailed
openclawbrain proof --openclaw-home ~/.openclaw
```

Internal published packages:

- `@openclawbrain/openclaw@0.4.39`
- `@openclawbrain/cli@0.4.39`

## Why this release exists

`0.4.39` exists because the install story still had a real operator seam after `0.4.38`.

The public lane was back to working after the `0.4.37` CLI tarball hotfix, but a real upgrade could still leave the host half-converged when the daemon/runtime side moved ahead and the installed hook/plugin package for the selected OpenClaw home stayed old.

That is not an acceptable install experience. The public install lane should converge the selected home or fail loudly.

## What changed

- `install` now refreshes the authoritative native plugin state when the installed hook package version lags the daemon/runtime version for the same selected home
- half-converged daemon-vs-installed-hook states remain explicit blocking truth on the status/proof surfaces
- README, operator docs, and public install/upgrade/troubleshooting pages now say plainly that the same four-command lane repairs stale-hook skew too
- the continuous-learning loop, operator controls, and replay/eval hardening from `0.4.37` / `0.4.38` remain intact

## Operator truth

This is a behavior fix, not a new workflow.

The canonical lane stays the same:

- run `openclawbrain install --openclaw-home ...`
- restart the gateway
- verify `status --detailed`
- capture durable evidence with `proof`

What changed is that `install` now treats a stale installed hook/plugin version as part of the same repair path instead of preserving the stale native plugin record and leaving the host half-converged.

## Focused verification

- `npx vitest run test/install-converge-seam.test.ts`
- `node --test packages/cli/dist/test/install-converge.test.js`
- `npm run release:verify`
- real-host install lane on `~/.openclaw`

## Upgrade

```bash
openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
openclawbrain status --openclaw-home ~/.openclaw --detailed
openclawbrain proof --openclaw-home ~/.openclaw
```

If a host already had daemon-vs-hook skew from a prior upgrade, rerun the same lane on the same selected home. This release makes that public repair path reconcile the stale installed hook too.
