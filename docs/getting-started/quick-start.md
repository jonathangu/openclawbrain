# Quick start

This is the shortest supported path from a working OpenClaw install to a verified OpenClawBrain install.

## Before you start

- OpenClaw is already installed and working
- Node.js 20+
- npm

## Install and verify

The public operator story has two literal OpenClawBrain actions: **Install OpenClawBrain** on a host that does not have it yet, and **Update OpenClawBrain** on a host that already has it installed.

### Install OpenClawBrain

```bash
openclaw plugins install @openclawbrain/openclaw
npx -y @openclawbrain/cli install --openclaw-home ~/.openclaw
openclaw gateway restart
npx -y @openclawbrain/cli status --openclaw-home ~/.openclaw --detailed
npx -y @openclawbrain/cli proof --openclaw-home ~/.openclaw --skip-install --skip-restart
```

### Update OpenClawBrain

```bash
openclaw plugins update openclawbrain
npx -y @openclawbrain/cli install --openclaw-home ~/.openclaw
openclaw gateway restart
npx -y @openclawbrain/cli status --openclaw-home ~/.openclaw --detailed
npx -y @openclawbrain/cli proof --openclaw-home ~/.openclaw --skip-install --skip-restart
```

The real install command is `openclaw plugins install @openclawbrain/openclaw`. The real update command is `openclaw plugins update openclawbrain`. For an already-installed host, the plugin update is only step 1. You still need to rerun the CLI `install` command so the activation root and native package plugin wiring stay correct for that OpenClaw home. `status --detailed` is the quick verify surface. `proof` writes `summary.md`, `steps.json`, `verdict.json`, raw step logs, and proof pointers under one bundle directory.

## What success looks like

- `status --detailed` reports the selected OpenClaw home as attached
- after the first promoted pack exists, detailed status also reports `serveState=serving_active_pack`

If the install does not look healthy yet, go straight to [Troubleshooting](../operating/troubleshooting.md).

## What to read next

- [Lifecycle](../lifecycle.md) for rollback, detach, and uninstall
- [Configuration guide](../configuration.md) for embeddings and advanced operator commands
- [Architecture overview](../architecture/overview.md) if you want the system design before reading code
