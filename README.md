# OpenClawBrain

**Memory that learns from your corrections.** A plugin for [OpenClaw](https://docs.openclaw.ai) that remembers what you correct, carries useful context across sessions, and keeps prompts smaller.

If you use the same OpenClaw agent day-to-day, you probably correct the same things twice. OpenClawBrain fixes that.

## What it does

- **Corrections stick.** You correct a workflow once; OpenClawBrain brings that lesson into future turns.
- **Context carries forward.** Relevant context from previous sessions is available, not dumped into every prompt.
- **Prompts stay small.** Only useful context is injected, not your entire history.
- **Stays silent when not needed.** On simple direct answers, it does nothing. No overhead.
- **You can verify it.** Status and proof routes let you see what it decided and why.

## Install

Requires [OpenClaw](https://docs.openclaw.ai) 2026.4.29 or later.

```bash
openclaw plugins install clawhub:openclawbrain
openclaw plugins enable openclawbrain
```

Then configure it:

```bash
openclaw config set plugins.entries.openclawbrain.config.enabled true --strict-json
openclaw config set plugins.entries.openclawbrain.config.mode '"conservative"' --strict-json
openclaw config set plugins.entries.openclawbrain.hooks.allowPromptInjection true --strict-json
openclaw config validate
openclaw gateway restart
```

Verify it's running:

```bash
openclaw plugins inspect openclawbrain --json
curl http://127.0.0.1:18789/plugins/openclawbrain/status
```

## How it works

OpenClawBrain is a native OpenClaw plugin. It hooks into the agent's pre-prompt build and runs locally on your machine. It does not upload data anywhere.

1. A turn comes in with a user message.
2. OpenClawBrain classifies the turn (direct answer, correction follow-up, continuation, tool-heavy, etc.).
3. Based on its mode, it decides whether to stay silent or inject bounded context.
4. If it injects, it reads local files and clips/redacts before returning.
5. It writes a local proof event so you can audit what it did.

Context comes from files on your machine:

```
~/.openclawbrain/activation/<agent-id>/context.md
~/.openclawbrain/activation/<agent-id>/corrections.md
~/.openclawbrain/activation/<agent-id>/tool-guidance.md
```

Create these files and fill them with what you want the agent to remember. Symlinks and oversized files are rejected before reading.

## Configuration

All config lives under `plugins.entries.openclawbrain.config` — never a root `openclawbrain` key.

| Setting | Default | Description |
|---------|---------|-------------|
| `enabled` | `false` | Must be `true` to do anything. |
| `mode` | `"conservative"` | `off`, `proof-only`, `conservative`, or `active`. Conservative stays silent unless bounded correction/context is useful. |
| `activationRoot` | `"~/.openclawbrain/activation/${agentId}"` | Where context/corrections/tool-guidance files live. |
| `proofEvents` | `true` | Write local proof events for auditing. |
| `proofRetentionEvents` | `1000` | Max proof events before old ones rotate out. |
| `maxContextChars` | `3000` | Max characters injected from activation files. |
| `includeActivationContext` | `true` | Whether to read activation context files. |

## Safety

- **Off by default.** `enabled` is `false` until you turn it on.
- **Local only.** No network calls, no data upload, no cloud service.
- **Redacted storage.** Proof events store hashes of user text, never raw text. `rawTranscriptStored: false`, `rawUserTextStored: false`, `redactionApplied: true`.
- **Fail-closed.** If `rawTranscriptUpload` is set to `true`, the plugin shuts down entirely. If `allowPromptInjection` is `false`, it returns no mutation.
- **Not a single point of failure.** If the plugin errors, the agent runs normally without it.

## Routes

| Endpoint | Description |
|----------|-------------|
| `/plugins/openclawbrain/status` | Current plugin status, config, last decision. |
| `/plugins/openclawbrain/proof?limit=20` | Recent proof events. |

## Development

```bash
# Install deps
pnpm install

# Build plugin
pnpm --dir packages/openclaw-plugin build

# Run tests (12 tests, all must pass)
pnpm --dir packages/openclaw-plugin test

# Check types
pnpm --dir packages/openclaw-plugin check
```

### Architecture

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

### Publishing

```bash
clawhub package publish packages/openclaw-plugin \
  --source-repo jonathangu/openclawbrain \
  --source-commit <sha> \
  --source-ref refs/tags/openclawbrain-v<version> \
  --source-path packages/openclaw-plugin
```

## License

MIT — see [LICENSE](LICENSE).
