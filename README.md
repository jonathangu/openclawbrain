# OpenClawBrain

OpenClawBrain is a local, agent-scoped selective intervention plugin for OpenClaw.

It helps an OpenClaw agent:

- remember corrections,
- continue bounded work,
- use relevant local context,
- verify before claiming on tool-heavy turns,
- stay silent on direct answers,
- expose local proof of what it did.

It is not a cloud memory service, a general intelligence layer, or a replacement for OpenClaw's memory/context engine. It does not upload raw transcripts or store raw user text.

## Product package

The v0.1 product is one native OpenClaw plugin package:

```text
packages/openclaw-plugin
```

Publishable package name:

```text
openclawbrain
```

The older `packages/runtime-policy`, `packages/proof-store`, `packages/openclaw-integration`, `packages/installer`, and `packages/cli` scaffolds are not the v0.1 product path. Their useful implementation ideas have been collapsed into `packages/openclaw-plugin`.

## Install / enable

```bash
openclaw plugins install clawhub:openclawbrain
openclaw plugins enable openclawbrain
openclaw config set plugins.entries.openclawbrain.config.enabled true --strict-json
openclaw config set plugins.entries.openclawbrain.config.mode '"conservative"' --strict-json
openclaw config set plugins.entries.openclawbrain.hooks.allowPromptInjection true --strict-json
openclaw config validate
openclaw gateway restart
openclaw plugins inspect openclawbrain --json
```

Local development:

```bash
pnpm --dir packages/openclaw-plugin build
openclaw plugins install -l ./packages/openclaw-plugin
```

Do not configure OpenClawBrain with a root `openclawbrain` key. Runtime config belongs only under:

```text
plugins.entries.openclawbrain.config
```

## Local activation files

OpenClawBrain reads only fixed local files under the agent activation root:

```text
~/.openclawbrain/activation/${agentId}/context.md
~/.openclawbrain/activation/${agentId}/corrections.md
~/.openclawbrain/activation/${agentId}/tool-guidance.md
```

Files are read lazily only when policy fires. Symlinks and oversized files are rejected before reading.

## Status and proof

The plugin registers first-class OpenClaw surfaces beyond hooks:

```text
/plugins/openclawbrain/status
/plugins/openclawbrain/proof?limit=20
```

Proof events are local redacted JSONL under the activation root. Proof records use precise privacy claims:

- `rawTranscriptStored: false`
- `rawUserTextStored: false`
- `redactionApplied: true`
- `hashesOnlyForUserText: true`

## Development gates

```bash
pnpm --dir packages/openclaw-plugin check
pnpm --dir packages/openclaw-plugin build
npm pack --dry-run --workspace packages/openclaw-plugin
pnpm test:product
pnpm ocb:traces:production-status
pnpm ocb:e2e:smoke
pnpm ocb:e2e:production
```

`v0.1.1` is the native plugin release candidate for ClawHub. The legacy ClawHub `openclawbrain` Skill is being replaced by this Code Plugin so the canonical install path is `openclaw plugins install clawhub:openclawbrain`. npm remains optional and is not part of v0.1.
