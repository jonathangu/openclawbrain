# Troubleshooting

Start every operator investigation with the canonical verify command:

```bash
openclawbrain status --openclaw-home ~/.openclaw --detailed
```

When you need a durable install/restart/status bundle instead of an ad hoc terminal check, run:

```bash
openclawbrain proof --openclaw-home ~/.openclaw
```

## `BRAIN NOT YET LOADED` appears after install

Cause:

The hook for the selected OpenClaw home is not pinned yet, or the gateway has not reloaded it yet.

Fix:

```bash
openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
openclawbrain status --openclaw-home ~/.openclaw --detailed
```

If you bypassed the install lane and changed plugin files directly, rerun `openclawbrain install --openclaw-home ~/.openclaw` for the same home afterward.

## `status --detailed` shows `surface ... converge=half_converged`

Cause:

The two live runtime surfaces drifted apart:

- the selected home's installed hook/runtime-guard moved one way
- the daemon runtime used for background watch/learner work moved another way

This is the main operator seam during upgrades or hotfixes.

Fix:

```bash
openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
openclawbrain status --openclaw-home ~/.openclaw --detailed
```

Do not treat one-sided daemon or hook edits as converged just because the other surface still loads.

## `status --detailed` does not show the selected home as attached

Cause:

The install step may have targeted a different OpenClaw home, or the gateway may still be running the old hook state.

Fix:

- rerun the install command against the intended `--openclaw-home`
- restart the gateway
- verify the same home path again

## `status --detailed` shows the home as attached, but `teacherConfigured=false`

Cause:

Brain activation and teacher wiring are separate checks. The runtime hook may be loaded correctly for the selected OpenClaw home while the optional teacher is still unset, points at the wrong provider/model, or fails model resolution.

Fix:

- confirm the teacher config uses the dedicated fields `brainTeacherEnabled`, `brainTeacherProvider`, and `brainTeacherModel`
- remember that adding a model to Ollama only makes it available; it does not automatically select it as the OpenClawBrain teacher
- restart the gateway after changing teacher config
- rerun `openclawbrain status --openclaw-home ~/.openclaw --detailed`
- verify `teacherConfigured=true`, the expected `teacherProvider` and `teacherModel`, and `teacherConfigError=null`

## `serveState=fail_open_static_context`

Cause:

The runtime is serving without a usable promoted pack. Common reasons are first-run state, missing embeddings, or a compile failure that forced the memory layer to step aside.

Fix:

- confirm the install is attached
- run a few real turns so the export and learning path have material to process
- inspect `openclawbrain learn --openclaw-home ~/.openclaw --json`
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
