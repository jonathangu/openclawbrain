# OpenClawBrain docs

Start here if you want the current truthful operator lane.

## Start here

If you want to try OpenClawBrain for the first time, use this command path:

```bash
npx @openclawbrain/cli@0.4.48 install --openclaw-home ~/.openclaw
openclaw gateway restart
npx @openclawbrain/cli@0.4.48 status --openclaw-home ~/.openclaw --detailed
npx @openclawbrain/cli@0.4.48 proof --openclaw-home ~/.openclaw
```

That same path is also the upgrade and repair path. It proves the operator install / runtime / reporting lane for one OpenClaw home; it does not by itself prove broad decision-quality gains.

## What OpenClawBrain Is Right Now

OpenClawBrain is a selective intervention layer for OpenClaw.

The current achievable agenda is:

- current-choice fidelity
- explicit-preference precedence for current durable choices
- restraint / specificity
- honest proof surfaces
- operator-story quality
- capability-choice generalization after the first weather lane

## New users

- [Quick start](getting-started/quick-start.md)
- [Lifecycle](lifecycle.md)
- [Troubleshooting](operating/troubleshooting.md)
- [Current release notes (0.4.48)](release-notes-0.4.48.md)

## Proof and claims

- [Proof map](proof/README.md)
- [Claims boundary](../CLAIMS.md)
- [Release contract](RELEASE_CONTRACT.md)
- [Evidence ladder](EVIDENCE.md)

## If you want the deeper explanation

- [How it works](https://openclawbrain.ai/how-it-works/)
- [Architecture overview](architecture/overview.md)

## Maintainer notes

- [End-state guide](END_STATE.md)
