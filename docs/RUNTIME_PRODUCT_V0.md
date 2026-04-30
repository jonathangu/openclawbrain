# OpenClawBrain v0 runtime product shape

OpenClawBrain v0 is a native OpenClaw plugin plus one user-facing CLI package. It is intentionally conservative: staying silent is a successful product action.

## Package layout

- `packages/runtime-policy` — pure deterministic selected product policy.
- `packages/proof-store` — profile-local JSONL proof events and status rendering.
- `packages/openclaw-plugin` — native OpenClaw plugin with `openclaw.plugin.json` and `api.on(...)` hooks.
- `packages/openclaw-integration` — profile-bound integration adapter export that reuses the native plugin/config surface.
- `packages/installer` — thin wrapper around documented `openclaw plugins ...` / `openclaw config set ...` paths.
- `packages/cli` — `openclawbrain` command shell for install/enable/disable/status/proof/doctor/uninstall.

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
          activationRoot: "~/.openclawbrain/activation/main",
          proofEvents: true,
          rawTranscriptUpload: false,
          scopes: { agents: ["main"] },
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
- Profile isolation: one activation root per profile.
- Default mode: `conservative`.
- Disable path: `openclawbrain disable --profile <profile>`.

## First-run target

```bash
npm install -g openclawbrain
openclawbrain install
openclawbrain enable --profile main
openclawbrain status --profile main
openclawbrain proof --profile main
```

The public install path is not complete until this works from npm without repo knowledge. `v0.1.0` in this repo is a local productization scaffold, not a published release.
