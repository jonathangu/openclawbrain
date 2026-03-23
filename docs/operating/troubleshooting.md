# Troubleshooting

Start every operator investigation with the canonical verify command:

```bash
npx @openclawbrain/cli status --openclaw-home ~/.openclaw --detailed
```

## `BRAIN NOT YET LOADED` appears after plugin install

Cause:

The plugin package is installed, but the activation root has not been pinned for the selected OpenClaw home yet.

Fix:

```bash
openclaw plugins install @openclawbrain/openclaw
npx @openclawbrain/cli install --openclaw-home ~/.openclaw
openclaw gateway restart
npx @openclawbrain/cli status --openclaw-home ~/.openclaw --detailed
```

## `status --detailed` does not show the selected home as attached

Cause:

The install step may have targeted a different OpenClaw home, or the gateway may still be running the old hook state.

Fix:

- rerun the install command against the intended `--openclaw-home`
- restart the gateway
- verify the same home path again

## `serveState=fail_open_static_context`

Cause:

The runtime is serving without a usable promoted pack. Common reasons are first-run state, missing embeddings, or a compile failure that forced the memory layer to step aside.

Fix:

- confirm the install is attached
- run a few real turns so the export and learning path have material to process
- inspect `npx @openclawbrain/cli learn --openclaw-home ~/.openclaw --json`
- review `~/.openclawbrain/extension-errors.log` if the extension reported compile errors

## Plugin install warns about `openclaw` versus `openclawbrain`

Cause:

Older OpenClaw host builds may still emit the historical plugin-id mismatch warning during plugin install.

Fix:

Treat the warning as cosmetic if the canonical verify command reports a healthy attached install. If install or status actually fails, troubleshoot that failure directly rather than assuming the warning is harmless.

## Detach or uninstall ran, but the running agent still behaves the same

Cause:

The hook state changed on disk, but the gateway has not reloaded it yet.

Fix:

```bash
openclaw gateway restart
```

## You need one shared activation root across multiple profiles

Cause:

Same-gateway multi-profile attachment and shared-root concurrent write safety are not yet part of the public claims boundary.

Fix:

Use separate OpenClaw homes and dedicated activation roots for the supported path today.

## Next docs

- [Quick start](../getting-started/quick-start.md)
- [Lifecycle](../lifecycle.md)
- [Configuration guide](../configuration.md)
- [Fail-open design](../architecture/fail-open.md)
