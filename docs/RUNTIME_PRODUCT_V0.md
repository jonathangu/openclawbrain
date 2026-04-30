# OpenClawBrain v0 runtime product shape

OpenClawBrain v0 is a native OpenClaw plugin plus one user-facing CLI package. It is intentionally conservative: staying silent is a successful product action.

## Package layout

- `packages/runtime-policy` — pure deterministic selected product policy.
- `packages/proof-store` — profile-local JSONL proof events and status rendering.
- `packages/openclaw-plugin` — native OpenClaw plugin with `openclaw.plugin.json`, prompt hooks, lifecycle hooks, and a status HTTP route.
- `packages/openclaw-integration` — profile-bound integration adapter export that reuses the native plugin/config surface.
- `packages/installer` — thin wrapper around documented `openclaw plugins ...` / `openclaw config set ...` paths.
- `packages/cli` — `openclawbrain` command shell for install/enable/disable/status/proof/doctor/uninstall.

## Hook strategy

v0 is not a context engine replacement. It uses prompt hooks for bounded same-turn context:

- `stay_silent` / `proof_only` — return nothing and write proof.
- `correction_only` — return bounded `prependContext` containing only the correction payload.
- `full_context` — return bounded `appendContext` containing the selected context summary.
- Tool-heavy verification — inject a read-only verification hint before prompt build; do not use `before_tool_call` unless a later version needs to rewrite/block/approve an actual tool call.
- Next-turn context — use `api.enqueueNextTurnInjection(...)` when the event explicitly asks to queue once for the next turn.

The plugin also observes `model_call_started`, `model_call_ended`, `agent_end`, `gateway_start`, and `gateway_stop` for status/proof-adjacent telemetry without raw prompt/response content.

## OpenClaw config shape

OpenClawBrain config lives under the plugin entry, never as an unknown root key:

```json5
{
  plugins: {
    entries: {
      openclawbrain: {
        enabled: true,
        hooks: {
          allowPromptInjection: true,
          allowConversationAccess: true,
        },
        config: {
          enabled: true,
          mode: "conservative",
          openclawProfile: "main",
          activationRoot: "~/.openclawbrain/activation/main",
          proofEvents: true,
          rawTranscriptUpload: false,
          scopes: { agents: ["main"], sessionKeys: [] },
        },
      },
    },
  },
}
```

This matches OpenClaw's plugin model: plugin schemas are declared by `openclaw.plugin.json`, and operators configure plugin-owned payloads at `plugins.entries.<id>.config`.

## Trust defaults

- Raw transcript upload: never.
- Proof events: local, redacted JSONL.
- Proof scope: `openclawProfile` + `agentId` + `sessionKeyHash`.
- Prompt injection disabled: fail closed and write `stay_silent` proof.
- Default mode: `conservative`.
- Disable path: `openclawbrain disable --profile <profile>`.

## First-run target

```bash
npm install -g openclawbrain
openclawbrain install
openclawbrain enable --profile main --agent main
openclawbrain status --profile main
openclawbrain proof --profile main
```

Native OpenClaw flow:

```bash
openclaw plugins install openclawbrain
openclaw plugins enable openclawbrain
openclaw config set plugins.entries.openclawbrain.config.mode conservative
openclaw gateway restart
openclaw plugins inspect openclawbrain --json
```

Local development can link the plugin with:

```bash
openclaw plugins install -l ./packages/openclaw-plugin
```

The public install path is not complete until this works from npm without repo knowledge. `v0.1.0` in this repo is a local productization scaffold, not a published release.
