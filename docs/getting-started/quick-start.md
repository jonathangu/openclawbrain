# Quick start

This is the shortest supported path from a working OpenClaw install to a verified OpenClawBrain install.

## Before you start

- OpenClaw is already installed and working
- Node.js 20+
- npm

## Install and verify

The public install story is three commands to install or update, then one command to verify.

```bash
openclaw plugins install @openclawbrain/openclaw
npx @openclawbrain/cli install --openclaw-home ~/.openclaw
openclaw gateway restart
npx @openclawbrain/cli status --openclaw-home ~/.openclaw --detailed
```

The first three commands install or update OpenClawBrain. The last command verifies the selected OpenClaw home.

## What success looks like

- `status --detailed` reports the selected OpenClaw home as attached
- after the first promoted pack exists, detailed status also reports `serveState=serving_active_pack`

If the install does not look healthy yet, go straight to [Troubleshooting](../operating/troubleshooting.md).

## What to read next

- [Lifecycle](../lifecycle.md) for rollback, detach, and uninstall
- [Configuration guide](../configuration.md) for embeddings and advanced operator commands
- [Architecture overview](../architecture/overview.md) if you want the system design before reading code
