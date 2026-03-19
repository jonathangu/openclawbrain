# @openclawbrain/cli

Staged operator split for OpenClawBrain.

- This package carries the `openclawbrain` CLI, daemon controls, import/export helpers, and install/status/operator management code.
- `@openclawbrain/openclaw` is the plugin/runtime payload.
- Public install docs are not switched yet; this package exists in-repo so the split can be verified before release.

## Commands

```bash
openclawbrain install --openclaw-home ~/.openclaw
openclawbrain status --openclaw-home ~/.openclaw --detailed
openclawbrain rollback --activation-root /var/openclawbrain/activation --dry-run
openclawbrain daemon status --activation-root /var/openclawbrain/activation
```

The old `openclawbrain-ops` alias stays wired to the same entrypoint for staged compatibility.
