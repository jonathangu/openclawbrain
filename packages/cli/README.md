# @openclawbrain/cli

Operator CLI for OpenClawBrain. Use it with `@openclawbrain/openclaw`.

The public install story is three commands to install or update, one command to verify, and one official proof command when you need a durable operator bundle.

```bash
openclaw plugins install @openclawbrain/openclaw
npx @openclawbrain/cli install --openclaw-home ~/.openclaw
openclaw gateway restart
npx @openclawbrain/cli status --openclaw-home ~/.openclaw --detailed
npx @openclawbrain/cli proof --openclaw-home ~/.openclaw --skip-install --skip-restart
```

The first three commands install or update OpenClawBrain. `status --detailed` is the quick verify surface. `proof` writes `summary.md`, `steps.json`, `verdict.json`, raw step logs, and proof pointers under one bundle directory.

## Common commands

```bash
npx @openclawbrain/cli rollback --openclaw-home ~/.openclaw --dry-run
npx @openclawbrain/cli detach --openclaw-home ~/.openclaw
npx @openclawbrain/cli uninstall --openclaw-home ~/.openclaw --keep-data
npx @openclawbrain/cli learn --openclaw-home ~/.openclaw --json
npx @openclawbrain/cli daemon status --activation-root ~/.openclawbrain/activation
```

## Docs

- [Repo README](../../README.md)
- [Quick start](../../docs/getting-started/quick-start.md)
- [Lifecycle](../../docs/lifecycle.md)
- [Troubleshooting](../../docs/operating/troubleshooting.md)
