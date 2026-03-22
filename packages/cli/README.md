# @openclawbrain/cli

`@openclawbrain/cli@0.4.8` is the published operator CLI package for OpenClawBrain.

Primary public flow:

```bash
openclaw plugins install @openclawbrain/openclaw@0.4.0
npx @openclawbrain/cli@0.4.8 install --openclaw-home ~/.openclaw
openclaw gateway restart
npx @openclawbrain/cli@0.4.8 status --openclaw-home ~/.openclaw --detailed
```

Patch note for `0.4.4`: the CLI now normalizes `plugins.allow` / `plugins.entries.openclawbrain` correctly on reinstall and reports the canonical `openclawbrain` install identity in status output.

Host release note: patched OpenClaw host builds no longer emit the old `openclaw` vs `openclawbrain` mismatch warning. Older host builds may still show that warning until the host-side alias fix is released there.

This package carries the `openclawbrain` CLI, daemon controls, import/export helpers, and install/status/operator management code. `@openclawbrain/openclaw` is the plugin/runtime payload.

## Commands

```bash
npx @openclawbrain/cli@0.4.8 install --openclaw-home ~/.openclaw
npx @openclawbrain/cli@0.4.8 status --openclaw-home ~/.openclaw --detailed
npx @openclawbrain/cli@0.4.8 rollback --activation-root /var/openclawbrain/activation --dry-run
npx @openclawbrain/cli@0.4.8 daemon status --activation-root /var/openclawbrain/activation
```

If the CLI is already on your `PATH`, `openclawbrain ...` is the same command surface. The docs lead with `npx` because that is the clean-host public-registry lane that already passed on `redogfood`.

The old `openclawbrain-ops` alias stays wired to the same entrypoint for compatibility.
