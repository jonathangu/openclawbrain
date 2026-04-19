# OpenClawBrain docs

Start here.

## Start here

If you want to try OpenClawBrain for the first time, use this command path:

```bash
npx @openclawbrain/cli@0.4.44 openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
npx @openclawbrain/cli@0.4.44 openclawbrain status --openclaw-home ~/.openclaw --detailed
npx @openclawbrain/cli@0.4.44 openclawbrain proof --openclaw-home ~/.openclaw
```

That same path is also the upgrade and repair path.

## New users

- [Quick start](getting-started/quick-start.md)
- [Lifecycle](lifecycle.md)
- [Troubleshooting](operating/troubleshooting.md)
- [Current release notes](release-notes-0.4.44.md)

## What OpenClawBrain is

OpenClawBrain is a memory layer for OpenClaw.

It helps the agent:
- remember corrections
- carry forward preferences
- reuse successful past work
- keep the live prompt small and focused

## If you want the deeper explanation

- [How it works](https://openclawbrain.ai/how-it-works/)
- [Architecture overview](architecture/overview.md)
- [Proof docs](proof/README.md)

## Maintainer notes

These are useful, but not the first stop for a newcomer:

- [Claims boundary](../CLAIMS.md)
- [Release contract](RELEASE_CONTRACT.md)
- [End state notes](END_STATE.md)
- [Evidence notes](EVIDENCE.md)
