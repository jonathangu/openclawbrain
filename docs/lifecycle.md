# Lifecycle

This guide covers the supported install, verify, rollback, detach, and uninstall flow for OpenClawBrain.

## Install or update

The public install story is three commands to install or update, then one command to verify.

```bash
openclaw plugins install @openclawbrain/openclaw
npx @openclawbrain/cli install --openclaw-home ~/.openclaw
openclaw gateway restart
npx @openclawbrain/cli status --openclaw-home ~/.openclaw --detailed
```

The first three commands install or update OpenClawBrain. The last command verifies the selected OpenClaw home.

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
