# OpenClawBrain v0.1 runtime product shape

OpenClawBrain v0.1 is a pure native OpenClaw plugin. The product is intentionally conservative: staying silent is a successful product action.

## Product package

The publishable package is:

```text
packages/openclaw-plugin
```

Package name:

```text
openclawbrain
```

The v0.1 product path does not require a separate CLI, installer, runtime adapter, or OpenClaw context-engine replacement. Older scaffold packages can remain as reference, but install/enable/status/proof must work through the native OpenClaw plugin lifecycle.

## Hook strategy

v0.1 uses prompt hooks for bounded same-turn context:

- `stay_silent` / `proof_only` — return nothing and write proof.
- `correction_only` — return bounded `prependContext` containing only correction guidance.
- `full_context` — return bounded `prependContext` containing selected local activation context.
- Tool-heavy verification — inject a read-only verification hint before prompt build; do not use `before_tool_call` in v0.1.

Registered hooks:

- `before_prompt_build`
- `model_call_started`
- `model_call_ended`
- `gateway_start`
- `gateway_stop`

`agent_turn_prepare` is only registered when the runtime explicitly advertises support. `agent_end` is optional and gated behind `plugins.entries.openclawbrain.hooks.allowConversationAccess=true`.

Prompt context updates requires:

```bash
openclaw config set plugins.entries.openclawbrain.config.hooks.allowPromptContext true --strict-json
```

If prompt context augmentation is disabled, OpenClawBrain fails closed and writes `stay_silent` proof.

## First-class plugin surfaces

The plugin is not hook-only. It registers:

- service: `openclawbrain`
- HTTP route: `/plugins/openclawbrain/status`
- HTTP route: `/plugins/openclawbrain/proof?limit=20`

The proof route returns bounded redacted recent proof events only.

## OpenClaw config shape

OpenClawBrain config lives under the plugin entry, never as an unknown root key:

```json5
{
  plugins: {
    entries: {
      openclawbrain: {
        enabled: true,
        hooks: {
          allowPromptContext: true,
          allowConversationAccess: false,
        },
        config: {
          enabled: true,
          mode: "conservative",
          activationRoot: "~/.openclawbrain/activation/${agentId}",
          proofEvents: true,
          proofRetentionEvents: 1000,
          maxContextChars: 3000,
          includeActivationContext: true,
          rawTranscriptUpload: false,
          scopes: { agents: ["main"] },
        },
      },
    },
  },
}
```

## Trust defaults

- Enabled default: `false`.
- Mode default: `conservative`.
- Raw transcript upload: const `false`; `true` fails closed.
- Raw transcript storage: no.
- Raw user-text storage: no.
- Proof events: local redacted JSONL.
- Activation files: fixed local filenames only.
- Activation root: agent-scoped.
- Symlinks and oversized activation files: rejected before reading.

## First-run target

```bash
openclaw plugins install clawhub:openclawbrain
openclaw plugins enable openclawbrain
openclaw config set plugins.entries.openclawbrain.config.enabled true --strict-json
openclaw config set plugins.entries.openclawbrain.config.mode '"conservative"' --strict-json
openclaw config set plugins.entries.openclawbrain.config.hooks.allowPromptContext true --strict-json
openclaw config validate
openclaw gateway restart
openclaw plugins inspect openclawbrain --json
```

Local development can link the plugin with:

```bash
pnpm --dir packages/openclaw-plugin build
openclaw plugins install -l ./packages/openclaw-plugin
```

The public install path is complete only after tarball/fresh install, live OpenClaw inspect, a real turn, status/proof verification, and disable path pass. `v0.1.1` is the native plugin release candidate. Public release follows `docs/RELEASE_RUNBOOK.md`: replace the legacy ClawHub Skill with this Code Plugin, verify provenance, then fresh-install `clawhub:openclawbrain`.
