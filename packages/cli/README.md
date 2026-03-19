# @openclawbrain/cli

`@openclawbrain/cli@0.4.1` is the published operator CLI package for OpenClawBrain.

Primary public flow:

```bash
openclaw plugins install @openclawbrain/openclaw@0.4.0
npx @openclawbrain/cli@0.4.1 openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
npx @openclawbrain/cli@0.4.1 openclawbrain status --openclaw-home ~/.openclaw --detailed
```

Patch note for `0.4.1`: rerunning `openclawbrain install --shared` against a native package install that is already pinned to the requested activation root is now a successful no-op instead of a false repin failure.

Current caveat: some hosts still warn about a plugin id mismatch because the plugin manifest uses `openclawbrain` while the package/entry hint uses `openclaw`. The install still works; treat that warning as currently cosmetic.

This package carries the `openclawbrain` CLI, daemon controls, import/export helpers, and install/status/operator management code. `@openclawbrain/openclaw` is the plugin/runtime payload.

## Commands

```bash
npx @openclawbrain/cli@0.4.1 openclawbrain install --openclaw-home ~/.openclaw
npx @openclawbrain/cli@0.4.1 openclawbrain status --openclaw-home ~/.openclaw --detailed
npx @openclawbrain/cli@0.4.1 openclawbrain rollback --activation-root /var/openclawbrain/activation --dry-run
npx @openclawbrain/cli@0.4.1 openclawbrain daemon status --activation-root /var/openclawbrain/activation
```

If the CLI is already on your `PATH`, `openclawbrain ...` is the same command surface. The docs lead with `npx` because that is the clean-host public-registry lane that already passed on `redogfood`.

The old `openclawbrain-ops` alias stays wired to the same entrypoint for compatibility.
