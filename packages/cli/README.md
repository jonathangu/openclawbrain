# @openclawbrain/cli

`@openclawbrain/cli@0.4.2` is the published operator CLI package for OpenClawBrain.

Primary public flow:

```bash
openclaw plugins install @openclawbrain/openclaw@0.4.0
npx @openclawbrain/cli@0.4.2 install --openclaw-home ~/.openclaw
openclaw gateway restart
npx @openclawbrain/cli@0.4.2 status --openclaw-home ~/.openclaw --detailed
```

Patch note for `0.4.2`: the CLI now persists declared attachment policy during install/attach so later `status` reads stop underreporting shared installs as `policy=null` / `undeclared`, and the package tarball now carries the full operator module surface plus traced-learning bridge needed for the canonical brain-store status path.

Current caveat: some hosts still warn about a plugin id mismatch because the plugin manifest uses `openclawbrain` while the package/entry hint uses `openclaw`. The install still works; treat that warning as currently cosmetic.

This package carries the `openclawbrain` CLI, daemon controls, import/export helpers, and install/status/operator management code. `@openclawbrain/openclaw` is the plugin/runtime payload.

## Commands

```bash
npx @openclawbrain/cli@0.4.2 install --openclaw-home ~/.openclaw
npx @openclawbrain/cli@0.4.2 status --openclaw-home ~/.openclaw --detailed
npx @openclawbrain/cli@0.4.2 rollback --activation-root /var/openclawbrain/activation --dry-run
npx @openclawbrain/cli@0.4.2 daemon status --activation-root /var/openclawbrain/activation
```

If the CLI is already on your `PATH`, `openclawbrain ...` is the same command surface. The docs lead with `npx` because that is the clean-host public-registry lane that already passed on `redogfood`.

The old `openclawbrain-ops` alias stays wired to the same entrypoint for compatibility.
