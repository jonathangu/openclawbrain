# Lifecycle

This guide covers the supported install, verify, proof, rollback, detach, and uninstall flow for OpenClawBrain.

## Install or refresh one OpenClaw home

Keep the same `--openclaw-home` value through the whole lifecycle. The public lane stays pinned to one OpenClaw home.

```bash
openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
openclawbrain status --openclaw-home ~/.openclaw --detailed
```

`install` is the public front door for the selected home. It writes or repairs the hook for that home and pins the activation root the runtime serves from. `status --detailed` verifies that selected home.

When you need durable operator evidence today, run:

```bash
openclawbrain proof --openclaw-home ~/.openclaw
```

The intended canonical lane is the same install command with optional `--proof`. Until that flag lands cleanly across the operator surfaces, proof stays a separate follow-up command.

If you are explicitly managing the native plugin package yourself, use OpenClaw's plugin manager for `@openclawbrain/openclaw`, then rerun `openclawbrain install --openclaw-home ~/.openclaw`.

## Verify and prove

Look for these checkpoints in `status --detailed`:

- `STATUS ok`
- `loadProof=status_probe_ready`
- `attachTruth ... runtime=proven`

When you need a durable bundle, run the `proof` command above after install/restart or rerun it later with `--skip-install --skip-restart` to capture the current operator state without replaying lifecycle steps. When your installed proof surface still expects the explicit replay guards, use those flags there. The public story stays on the same selected `--openclaw-home`.

## Roll back

Preview the rollback first:

```bash
openclawbrain rollback --openclaw-home ~/.openclaw --dry-run
```

Apply the rollback only after the preview looks correct:

```bash
openclawbrain rollback --openclaw-home ~/.openclaw
```

Rollback moves the serve path back to the previous promoted pack when one is available.

## Detach and keep data

`detach` removes the OpenClaw profile hook and keeps OpenClawBrain data in place.

```bash
openclawbrain detach --openclaw-home ~/.openclaw
openclaw gateway restart
```

## Uninstall and keep data

```bash
openclawbrain uninstall --openclaw-home ~/.openclaw --keep-data
openclaw gateway restart
```

## Uninstall and purge data

```bash
openclawbrain uninstall --openclaw-home ~/.openclaw --purge-data
openclaw gateway restart
```

## Notes

- Restart the gateway after install, detach, or uninstall so the running profile picks up the new hook state.
- `rollback`, `status`, and `learn` do not need a gateway restart.
- The plugin package itself is managed by OpenClaw's plugin manager. Removing the hook does not remove the installed package files.
- If the running gateway still behaves like nothing changed, restart it first before assuming the hook edit failed.

Next docs:

- [Quick start](getting-started/quick-start.md)
- [Troubleshooting](operating/troubleshooting.md)
- [Configuration guide](configuration.md)
