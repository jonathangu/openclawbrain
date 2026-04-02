# Quick start

This is the shortest supported path from a working OpenClaw install to a verified OpenClawBrain install.

## Before you start

- OpenClaw is already installed and working
- Node.js 20+
- npm

## Install and verify

Keep the same `--openclaw-home` value through install, restart, status, and proof. The public operator story is one command front door for one OpenClaw home.

```bash
openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
openclawbrain status --openclaw-home ~/.openclaw --detailed
```

`install` is the public front door. It writes or repairs the hook for the selected home and pins the activation root the runtime serves from. `status --detailed` is the quick verify surface.

Activation and teacher wiring are separate checks. `BRAIN LOADED` and an attached home prove the brain hook is live for that OpenClaw home. They do **not** by themselves prove that an optional teacher model is wired. Teacher wiring uses the dedicated config fields `brainTeacherEnabled`, `brainTeacherProvider`, and `brainTeacherModel`, and the same `status --detailed` surface should report `teacherConfigured`, `teacherProvider`, `teacherModel`, and `teacherConfigError`.

When you need durable operator evidence today, run:

```bash
openclawbrain proof --openclaw-home ~/.openclaw
```

The intended canonical lane is `openclawbrain install --openclaw-home ~/.openclaw --proof`. Until that flag lands cleanly across the operator surfaces, proof stays a separate follow-up command. `proof` writes `summary.md`, `steps.json`, `verdict.json`, raw step logs, and proof pointers under one bundle directory.

If you manually change plugin files anyway, treat that as maintainer-only surgery and rerun `openclawbrain install --openclaw-home ~/.openclaw` before trusting the host again.

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
