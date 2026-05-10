# OpenClaw Codex Continuity Bridge Operator Guide

Status: implemented in `openclawbrain@0.2.24` as an OpenClawBrain-owned external plugin surface. The older local OpenClaw bundled-plugin branch is prototype/reference only.

## Product Contract

Codex UI remains the high-bandwidth coding workbench. OpenClaw and Telegram are the low-bandwidth operator surface. OpenClawBrain is the authority layer that decides what state matters, what should stay quiet, and what should become durable memory.

The bridge is intentionally not a second coding UI. It exposes concise status, explicit watched completion notifications, evidence-separated handoff briefs, and a gated write path that is disabled by default.

The bridge must not require Jonathan's personal OpenClaw checkout to carry local core edits. Personal OpenClaw should stay stock/upstream-trackable so it can be upgraded freely.

## Operator Workflow

1. From Telegram, ask `/brain codex status` to see whether Codex is active, what the latest thread is, and whether the bridge is reading live app-server state or stale SQLite fallback.
2. Ask `/brain codex threads` to list recent Codex threads with thread ids.
3. Ask `/brain codex watch <thread-id>` when you want one completion, failure, blocker, approval-needed, or auth-failure notification for a thread.
4. Ask `/brain codex handoff <thread-id>` when returning to the Mac and wanting a brief that separates observed facts from Codex-reported claims.
5. Keep Telegram-to-Codex writes disabled until repo allowlists, trusted senders, provenance, and app-server write capabilities are explicitly configured.

## API Routes

Routes are registered by the `openclawbrain` plugin and require gateway authentication.

- `GET /plugins/openclawbrain/codex/status`
- `GET /plugins/openclawbrain/codex/threads`
- `GET /plugins/openclawbrain/codex/handoff`
- `GET /plugins/openclawbrain/codex/watches`

The write path is intentionally not exposed as a mutating route in `0.2.24`. `/brain codex goal` and `/brain codex steer` refuse by default.

## Telegram Commands

- `/brain codex status`
- `/brain codex threads [filter]`
- `/brain codex watch [thread-id|--latest]`
- `/brain codex handoff [thread-id]`
- `/brain codex goal <goal text>`

`/brain codex goal` is present but rejected by default because `codexBridge.enableTelegramWrites` defaults to `false`. Existing `/codex steer` behavior remains owned by the native Codex conversation binding; OpenClawBrain does not replace that control path.

## Safety Model

Read-only status is allowed first. Watched notifications write only bridge-local state. Handoff briefs are generated from observed bridge state and explicitly label Codex claims as reported unless independently verified.

Telegram-to-Codex writes require:

- `codexBridge.enableTelegramWrites = true`
- trusted sender match when `trustedTelegramSenders` is configured
- repo path under `repoAllowlist`
- provenance metadata with `requestedBy` and `requestId`
- acceptable risk class
- confirmation for risky wording
- unambiguous thread selection
- explicitly confirmed app-server write method, for example `turn/start`

SQLite is read-only fallback only. It is never used as a write path.

## Config

Example disabled-by-default config:

```toml
[plugins.entries.openclawbrain.config.codexBridge]
enabled = true
notifyChannel = "telegram"
notifyTarget = "<telegram-chat-id>"
enableTelegramWrites = false
repoAllowlist = []
trustedTelegramSenders = []
```

Example future write-mode config:

```toml
[plugins.entries.openclawbrain.config.codexBridge]
enabled = true
notifyChannel = "telegram"
notifyTarget = "<telegram-chat-id>"
enableTelegramWrites = true
repoAllowlist = ["/Users/guclaw/openclaw", "/Users/guclaw/.openclaw/workspace/openclawbrain"]
trustedTelegramSenders = ["<jonathan-telegram-user-id>"]
```

Keep write mode off unless the threat model is acceptable for the machine.

## Memory Boundaries

OpenClawBrain should store durable operating truths:

- Codex UI is the high-bandwidth workbench.
- OpenClaw and Telegram are the mobile operator surface.
- Telegram summaries should be concise.
- Notify only on explicit watched completion, failure, blocker, approval-needed, or auth-failure events.
- Handoff briefs must separate observed facts from Codex-reported claims.

OpenClawBrain should not store durable raw telemetry:

- raw Codex messages
- command output
- full diffs
- transient thread status
- temporary watch requests
- secrets or auth failures
- failed guesses about user intent

Current explicit instruction still overrides these durable defaults.

## Remaining Risks

- Codex app-server protocol is experimental; the bridge falls back to SQLite and labels that state stale.
- Thread selection for writes remains conservative and may refuse ambiguous requests.
- Telegram account compromise is still a remote-control risk; high-risk actions should require local Mac approval before write mode is enabled.
- The bridge does not independently verify Codex claims unless it observes supporting evidence.
- Live notification delivery depends on the OpenClaw gateway/plugin service being loaded and the Telegram outbound adapter being healthy.
- If a future feature truly needs a new OpenClaw host capability, ship it through a small upstream PR; do not block the personal bridge workflow on a long-lived OpenClaw fork.

## Local Install / Update

Use the OpenClawBrain-owned installer from this repo:

```bash
pnpm install:local-openclaw
openclaw gateway restart
```

That copies the built plugin into `~/.openclaw/extensions/openclawbrain`, installs runtime dependencies inside the extension directory, and updates `~/.openclaw/plugins/installs.json`. It does not edit `/Users/guclaw/openclaw`.

To update every local OpenClaw home that already has OpenClawBrain installed:

```bash
pnpm install:local-openclaw:all
```
