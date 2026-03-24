# Release notes — 0.4.12

`0.4.12` upgrades the public operator lane for OpenClawBrain.

## Package

- `@openclawbrain/cli@0.4.12`

## What changed

- added a first-class `openclawbrain proof --openclaw-home <path>` command
- proof bundles now capture `summary.md`, `steps.json`, `verdict.json`, raw per-step logs, startup breadcrumbs, and runtime-load-proof snapshots
- install and healthy-status guidance now point to proof capture when an operator needs durable evidence
- daemon launchd payloads no longer depend on ephemeral `~/.npm/_npx` cache paths
- daemon status now reports the configured runtime path, program arguments, and command explicitly
- docs now align around one canonical install / verify / proof story

## Canonical commands

```bash
openclaw plugins install @openclawbrain/openclaw
npx @openclawbrain/cli install --openclaw-home ~/.openclaw
openclaw gateway restart
npx @openclawbrain/cli status --openclaw-home ~/.openclaw --detailed
npx @openclawbrain/cli proof --openclaw-home ~/.openclaw --skip-install --skip-restart
```
