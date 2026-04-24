# @openclawbrain/openclaw

Plugin and runtime payload internals for the current OpenClawBrain selective-intervention lane.

Most operators should start with the `openclawbrain` front door, not with manual package management:

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

This package is still the runtime payload under the hood. If you are explicitly managing the native package layer yourself, use OpenClaw's plugin manager for `@openclawbrain/openclaw`, then rerun `openclawbrain install --openclaw-home ~/.openclaw`.

Those surfaces prove install / runtime / reporting truth for one selected home. They are not, by themselves, a broad answer-quality claim.

Public docs should lead with the OpenClawBrain product lane and the `openclawbrain install` command, not with direct runtime-package management.

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
