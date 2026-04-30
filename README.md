# OpenClawBrain

A local, profile-bound selective intervention layer for OpenClaw.

OpenClawBrain helps an OpenClaw profile:

- remember user corrections,
- continue bounded work,
- supply relevant context,
- stay silent on direct answers,
- show local proof of what it did.

It is not a general intelligence layer. It is not a cloud memory service. It does not upload raw transcripts.

## Install target

```bash
npm install -g openclawbrain
openclawbrain install
openclawbrain enable --profile main
openclawbrain status --profile main
```

Then, after real usage:

```bash
openclawbrain proof --profile main
```

## Runtime shape

OpenClawBrain ships as one user-visible package with internal workspaces:

- `packages/runtime-policy` — pure deterministic selected product policy.
- `packages/proof-store` — local redacted proof/status store.
- `packages/openclaw-plugin` — native OpenClaw plugin entry and manifest.
- `packages/openclaw-integration` — compatibility adapter export for the profile-bound OpenClaw integration surface.
- `packages/installer` — thin wrapper around OpenClaw plugin/config commands.
- `packages/cli` — `openclawbrain` command shell.

OpenClaw config belongs under the plugin entry:

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

Do not configure OpenClawBrain with an unknown root `openclawbrain` key.

## Proof

```bash
openclawbrain proof --profile main
```

Proof events are local, redacted JSONL. The proof surface reports decisions like `stay_silent`, `correction_only`, `full_context`, and `proof_only` without storing raw private transcripts.

## Disable / uninstall

```bash
openclawbrain disable --profile main
openclawbrain uninstall --profile main
```

`v0.1.0` in this repository is a local productization scaffold, not a published npm release yet.
