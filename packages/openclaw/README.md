# @openclawbrain/openclaw

Plugin and runtime payload for OpenClawBrain.

Install it together with `@openclawbrain/cli`. The public install story is three commands to install or update, then one command to verify.

```bash
openclaw plugins install @openclawbrain/openclaw
npx @openclawbrain/cli install --openclaw-home ~/.openclaw
openclaw gateway restart
npx @openclawbrain/cli status --openclaw-home ~/.openclaw --detailed
```

The first three commands install or update OpenClawBrain. The last command verifies the selected OpenClaw home.

## What this package contains

- the OpenClaw plugin manifest
- the installed extension runtime guard
- `compileRuntimeContext()` and related runtime load helpers

If this package is installed before the CLI pins the activation root, the extension fails open and logs `BRAIN NOT YET LOADED` instead of blocking the agent.

## Docs

- [Repo README](../../README.md)
- [Quick start](../../docs/getting-started/quick-start.md)
- [Lifecycle](../../docs/lifecycle.md)
- [Troubleshooting](../../docs/operating/troubleshooting.md)
