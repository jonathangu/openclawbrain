# Quick start

This is the shortest supported path from a working OpenClaw install to a verified OpenClawBrain install.

## Before you start

- OpenClaw is already installed and working
- Node.js 20+
- npm

## Install and verify

Keep the same `--openclaw-home` value through the whole flow.

```bash
npx @openclawbrain/cli@0.4.44 install --openclaw-home ~/.openclaw
openclaw gateway restart
npx @openclawbrain/cli@0.4.44 status --openclaw-home ~/.openclaw --detailed
npx @openclawbrain/cli@0.4.44 proof --openclaw-home ~/.openclaw
```

What these commands do:

1. attach OpenClawBrain to one OpenClaw home
2. restart the gateway so the runtime reloads
3. check detailed status
4. save a proof bundle if you want a durable record

## What success looks like

Look for these signals in `status --detailed`:

- `STATUS ok`
- `loadProof=status_probe_ready`
- `surface ... converge=converged`

If the install does not look healthy yet, go straight to [Troubleshooting](../operating/troubleshooting.md).

## Next

- [Lifecycle](../lifecycle.md)
- [Troubleshooting](../operating/troubleshooting.md)
- [Configuration guide](../configuration.md)
