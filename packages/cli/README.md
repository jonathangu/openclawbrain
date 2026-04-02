# @openclawbrain/cli

Operator CLI internals for OpenClawBrain.

The public front door is one command pinned to one OpenClaw home:

```bash
openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
openclawbrain status --openclaw-home ~/.openclaw --detailed
```

`install` is the public front door for the selected home. It writes or repairs the hook for that home and pins the activation root the runtime serves from. `status --detailed` is the quick verify surface.

When you need durable operator evidence today, run:

```bash
openclawbrain proof --openclaw-home ~/.openclaw
```

The intended canonical lane is the same install command with optional `--proof`. Until that lands cleanly across the operator surfaces, proof stays a separate follow-up command. `proof` writes `summary.md`, `steps.json`, `verdict.json`, raw step logs, and proof pointers under one bundle directory.

This package is part of the internal split architecture. Public docs should lead with OpenClawBrain and the `openclawbrain install` lane, not with package-pair trivia.

## Common commands

```bash
openclawbrain rollback --openclaw-home ~/.openclaw --dry-run
openclawbrain detach --openclaw-home ~/.openclaw
openclawbrain uninstall --openclaw-home ~/.openclaw --keep-data
openclawbrain learn --openclaw-home ~/.openclaw --json
openclawbrain daemon status --activation-root ~/.openclawbrain/activation
```

## Docs

- [Repo README](../../README.md)
- [Quick start](../../docs/getting-started/quick-start.md)
- [Lifecycle](../../docs/lifecycle.md)
- [Troubleshooting](../../docs/operating/troubleshooting.md)
