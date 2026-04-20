# Lifecycle

This guide covers install, verify, rollback, detach, and uninstall.

## Install or refresh one OpenClaw home

Use the same home path through the whole flow.

```bash
npx @openclawbrain/cli@0.4.45 install --openclaw-home ~/.openclaw
openclaw gateway restart
npx @openclawbrain/cli@0.4.45 status --openclaw-home ~/.openclaw --detailed
npx @openclawbrain/cli@0.4.45 proof --openclaw-home ~/.openclaw
```

That same command path is used for:
- first install
- upgrade
- repair

## Verify

Use `status --detailed` for the fast check.
Use `proof` when you want a saved bundle.

Healthy installs should show:
- `STATUS ok`
- `loadProof=status_probe_ready`
- `surface ... converge=converged`

## Roll back

Preview first:

```bash
npx @openclawbrain/cli@0.4.45 rollback --openclaw-home ~/.openclaw --dry-run
```

Apply the rollback:

```bash
npx @openclawbrain/cli@0.4.45 rollback --openclaw-home ~/.openclaw
```

## Detach and keep data

```bash
npx @openclawbrain/cli@0.4.45 detach --openclaw-home ~/.openclaw
openclaw gateway restart
```

## Uninstall and keep data

```bash
npx @openclawbrain/cli@0.4.45 uninstall --openclaw-home ~/.openclaw --keep-data
openclaw gateway restart
```

## Uninstall and purge data

```bash
npx @openclawbrain/cli@0.4.45 uninstall --openclaw-home ~/.openclaw --purge-data
openclaw gateway restart
```

## Notes

- keep the same `--openclaw-home` value through install, verify, rollback, detach, and uninstall
- restart the gateway after install, detach, or uninstall
- if something looks wrong, rerun install first, then recheck status

## Next

- [Quick start](getting-started/quick-start.md)
- [Troubleshooting](operating/troubleshooting.md)
- [Configuration guide](configuration.md)
