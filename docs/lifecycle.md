# Lifecycle

This guide covers the supported install, verify, proof, rollback, detach, and uninstall flow for OpenClawBrain.

## Install or refresh one OpenClaw home

Keep the same `--openclaw-home` value through the whole lifecycle. The public lane stays pinned to one OpenClaw home.

```bash
openclawbrain install --openclaw-home ./openclaw-cormorantai
openclawbrain status --openclaw-home ./openclaw-cormorantai --detailed
```

`install` is the public front door for the selected home. It repairs the installed hook/runtime-guard surface for that home and re-checks it against the separate daemon runtime surface for the same activation root:

- the **installed hook/runtime-guard** surface that OpenClaw loads from the selected `--openclaw-home`
- the **daemon runtime** surface that background watch/learner work runs from for that activation root

`status --detailed` verifies both surfaces for the selected home. Fresh homes default to the cold-start prior. If the selected home already has user history, rerunning install rebuilds the stronger generic prior underneath the saved preferences, corrections, and habits instead of wiping them.

If you ever do manual hook or daemon surgery, the safe recovery lane is still the same command:

```bash
openclawbrain install --openclaw-home ./openclaw-cormorantai
```

The selected home can be the default `~/.openclaw`, a profile-specific home, or an explicit nonstandard path like `./openclaw-cormorantai`. The important part is that install, status, rollback, detach, uninstall, and proof all stay pinned to the same exact `--openclaw-home` value.

Safe converge lane for upgrades or hotfixes:

1. Update the global packages that own the daemon/runtime surface.
2. If this activation root runs the managed background daemon, restart that daemon-side surface.
3. Run `openclawbrain install --openclaw-home <path>` to reconverge the selected hook/runtime-guard surface.
4. Run `openclawbrain status --openclaw-home <path> --detailed` and confirm the `surface` line reports `converge=converged`.
5. Run `openclawbrain proof --openclaw-home <path>` when you need a durable bundle that captures the same surface truth.

When you need durable operator evidence today, run:

```bash
openclawbrain proof --openclaw-home ./openclaw-cormorantai
```

The intended canonical lane is the same install command with optional `--proof`. Until that flag lands cleanly across the operator surfaces, proof stays a separate follow-up command.

If you do manual plugin surgery anyway, rerun `openclawbrain install --openclaw-home ./openclaw-cormorantai` (or the exact home you are operating on) before trusting the host again. The public story remains one install lane.

## Verify and prove

Look for these checkpoints in `status --detailed`:

- `STATUS ok`
- `loadProof=status_probe_ready`
- `attachTruth ... runtime=proven`
- `surface ... converge=converged`

If `surface ... converge=half_converged` appears, treat that as a failed converge. One side of the split runtime moved without the other. Refresh the daemon-side CLI/runtime surface if needed, then rerun `openclawbrain install --openclaw-home <path>` for the same selected home before trusting the host again.

When you need a durable bundle, run the `proof` command above after install/restart or rerun it later with `--skip-install --skip-restart` to capture the current operator state without replaying lifecycle steps. When your installed proof surface still expects the explicit replay guards, use those flags there. The public story stays on the same selected `--openclaw-home`.

## Roll back

Preview the rollback first:

```bash
openclawbrain rollback --openclaw-home ./openclaw-cormorantai --dry-run
```

Apply the rollback only after the preview looks correct:

```bash
openclawbrain rollback --openclaw-home ./openclaw-cormorantai
```

Rollback moves the serve path back to the previous promoted pack when one is available.

## Detach and keep data

`detach` removes the OpenClaw profile hook and keeps OpenClawBrain data in place.

```bash
openclawbrain detach --openclaw-home ./openclaw-cormorantai
openclaw gateway restart
```

## Uninstall and keep data

```bash
openclawbrain uninstall --openclaw-home ./openclaw-cormorantai --keep-data
openclaw gateway restart
```

## Uninstall and purge data

```bash
openclawbrain uninstall --openclaw-home ./openclaw-cormorantai --purge-data
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
