# Quick start

This is the shortest supported path from a working OpenClaw install to a verified OpenClawBrain install.

## Before you start

- OpenClaw is already installed and working
- Node.js 20+
- npm

## Install and verify

Keep the same `--openclaw-home` value through install, restart, status, and proof. The public operator story is one command front door for one OpenClaw home.

That home does **not** have to be `~/.openclaw`. Explicit nonstandard homes like `./openclaw-cormorantai` are first-class as long as you keep the same path pinned through the whole flow.

```bash
openclawbrain install --openclaw-home ./openclaw-cormorantai
openclaw gateway restart
openclawbrain status --openclaw-home ./openclaw-cormorantai --detailed
```

`install` is the public front door. It converges the selected home's installed hook/runtime-guard surface and the daemon runtime surface onto one coherent state for that activation root. `status --detailed` is the quick verify surface.

Fresh homes default to the cold-start prior. If you are upgrading an existing home, rerun the same install lane; it rebuilds the stronger generic prior underneath your saved preferences instead of resetting them.

Activation and teacher wiring are separate checks. `BRAIN LOADED` and an attached home prove the brain hook is live for that OpenClaw home. They do **not** by themselves prove that an optional teacher model is wired. Teacher wiring uses the dedicated config fields `brainTeacherEnabled`, `brainTeacherProvider`, and `brainTeacherModel`, and the same `status --detailed` surface should report `teacherConfigured`, `teacherProvider`, `teacherModel`, and `teacherConfigError`.

When you need durable operator evidence today, run:

```bash
openclawbrain proof --openclaw-home ./openclaw-cormorantai
```

The intended canonical lane is `openclawbrain install --openclaw-home <your-home> --proof`. Until that flag lands cleanly across the operator surfaces, proof stays a separate follow-up command. `proof` writes `summary.md`, `steps.json`, `verdict.json`, raw step logs, and proof pointers under one bundle directory.

If you manually change plugin files anyway, treat that as maintainer-only surgery and rerun `openclawbrain install --openclaw-home ./openclaw-cormorantai` (or whatever exact home you are operating on) before trusting the host again.

In the detailed status output, look for `surface ... converge=converged`. If it says `converge=half_converged`, the daemon runtime and installed hook/runtime-guard drifted apart and the safe lane is to rerun install for that same `--openclaw-home`.

## What success looks like

- `status --detailed` reports the selected OpenClaw home as attached
- `loadProof=status_probe_ready`
- after the first promoted pack exists, detailed status also reports `serveState=serving_active_pack`
- if you configured a teacher, detailed status reports `teacherConfigured=true`, the expected `teacherProvider` and `teacherModel`, and `teacherConfigError=null`

If the install does not look healthy yet, go straight to [Troubleshooting](../operating/troubleshooting.md).

## What to read next

- [Lifecycle](../lifecycle.md) for rollback, detach, and uninstall
- [Configuration guide](../configuration.md) for embeddings and advanced operator commands
- [Architecture overview](../architecture/overview.md) if you want the system design before reading code
