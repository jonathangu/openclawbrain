# OpenClawBrain 0.4.4 release notes

`0.4.4` is a focused operator CLI patch for durable reinstall truth.

## What changed

- operator CLI advances to `@openclawbrain/cli@0.4.4`
- reinstall now normalizes `plugins.allow` and `plugins.entries.openclawbrain` correctly instead of reintroducing the legacy `openclaw` hint during the split-package flow
- status output now reports the canonical `openclawbrain` install identity instead of the legacy `openclaw` install hint

## Public lane

```bash
openclaw plugins install @openclawbrain/openclaw@0.4.0
npx @openclawbrain/cli@0.4.4 install --openclaw-home ~/.openclaw
openclaw gateway restart
npx @openclawbrain/cli@0.4.4 status --openclaw-home ~/.openclaw --detailed
```

## Scope

This patch ships the CLI-side reinstall/config truth fix.

The separate host-side `openclaw` vs `openclawbrain` warning suppression lives in the OpenClaw host patch lane. On patched hosts that warning is gone; older host builds may still emit the warning until the host fix is released there.
