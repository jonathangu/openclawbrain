# Lifecycle

This guide covers the supported install, verify, proof, rollback, detach, and uninstall flow for OpenClawBrain.

## Install or update

Keep the same `--openclaw-home` value through the whole lifecycle. The public lane stays pinned to one OpenClaw home.

### Fresh install

```bash
openclaw plugins install @openclawbrain/openclaw
npx -y @openclawbrain/cli install --openclaw-home ~/.openclaw
openclaw gateway restart
npx -y @openclawbrain/cli status --openclaw-home ~/.openclaw --detailed
npx -y @openclawbrain/cli proof --openclaw-home ~/.openclaw --skip-install --skip-restart
```

### Update an existing host

```bash
openclaw plugins update openclawbrain
npx -y @openclawbrain/cli install --openclaw-home ~/.openclaw
openclaw gateway restart
npx -y @openclawbrain/cli status --openclaw-home ~/.openclaw --detailed
npx -y @openclawbrain/cli proof --openclaw-home ~/.openclaw --skip-install --skip-restart
```

For an already-installed host, the plugin update is only step 1. You still need to rerun the CLI `install` command so the activation root and native package plugin wiring stay correct for that OpenClaw home. `status --detailed` verifies the selected OpenClaw home. `proof` captures the durable operator bundle when you need explicit install/restart/status evidence.

## Verify and prove

Look for these checkpoints in `status --detailed`:

- `STATUS ok`
- `loadProof=status_probe_ready`
- `attachTruth ... runtime=proven`

When you need a durable bundle, run the `proof` command above after install/restart or rerun it later with `--skip-install --skip-restart` to capture the current operator state without replaying lifecycle steps.

## Roll back

Preview the rollback first:

```bash
npx @openclawbrain/cli rollback --openclaw-home ~/.openclaw --dry-run
```

Apply the rollback only after the preview looks correct:

```bash
npx @openclawbrain/cli rollback --openclaw-home ~/.openclaw
```

Rollback moves the serve path back to the previous promoted pack when one is available.

## Detach and keep data

`detach` removes the OpenClaw profile hook and keeps OpenClawBrain data in place.

```bash
npx @openclawbrain/cli detach --openclaw-home ~/.openclaw
openclaw gateway restart
```

## Uninstall and keep data

```bash
npx @openclawbrain/cli uninstall --openclaw-home ~/.openclaw --keep-data
openclaw gateway restart
```

## Uninstall and purge data

```bash
npx @openclawbrain/cli uninstall --openclaw-home ~/.openclaw --purge-data
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
