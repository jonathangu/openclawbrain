# OpenClaw Codex Continuity Bridge Operator Guide

Status: implemented in the local OpenClaw `codex` bundled plugin.

## Product Contract

Codex UI remains the high-bandwidth coding workbench. OpenClaw and Telegram are the low-bandwidth operator surface. OpenClawBrain is the authority layer that decides what state matters, what should stay quiet, and what should become durable memory.

The bridge is intentionally not a second coding UI. It exposes concise status, explicit watched completion notifications, evidence-separated handoff briefs, and a gated write path that is disabled by default.

## Operator Workflow

1. From Telegram, ask `/codex status` to see whether Codex is active, what the latest thread is, and whether the bridge is reading live app-server state or stale SQLite fallback.
2. Ask `/codex threads` to list recent Codex threads with thread ids.
3. Ask `/codex watch <thread-id>` when you want exactly one completion, failure, blocker, approval-needed, or auth-failure notification for a thread.
4. Ask `/codex handoff <thread-id>` when returning to the Mac and wanting a brief that separates observed facts from Codex-reported claims.
5. Keep Telegram-to-Codex writes disabled until repo allowlists, trusted senders, provenance, and app-server write capabilities are explicitly configured.

## API Routes

All routes are registered by the Codex plugin and require gateway authentication.

- `GET /codex/status`
- `GET /codex/threads`
- `POST /codex/watch`
- `POST /codex/watch/check`
- `POST /codex/handoff`
- `POST /codex/goal`
- `POST /codex/steer`

The write routes exist for the final phase but refuse by default unless the feature flag and all safety gates pass.

## Telegram Commands

- `/codex status`
- `/codex threads`
- `/codex watch [thread-id]`
- `/codex handoff [thread-id]`
- `/codex goal <goal text>`

`/codex goal` is present but rejected by default because `codexBridge.enableTelegramWrites` defaults to `false`. Existing `/codex steer` behavior remains owned by the native Codex conversation binding; the bridge write route for steering is the authenticated HTTP route.

## Safety Model

Read-only status is allowed first. Watched notifications write only bridge-local state. Handoff briefs are generated from observed bridge state and explicitly label Codex claims as reported unless independently verified.

Telegram-to-Codex writes require:

- `codexBridge.enableTelegramWrites = true`
- trusted sender match when `trustedTelegramSenders` is configured
- repo path under `allowedRepos`
- provenance metadata with `requestedBy` and `requestId`
- acceptable risk class
- confirmation for risky wording
- unambiguous thread selection
- explicitly confirmed app-server write method, for example `turn/start`

SQLite is read-only fallback only. It is never used as a write path.

## Config

Example disabled-by-default config:

```toml
[plugins.entries.codex.config.codexBridge]
enabled = true
notifyChannel = "telegram"
notifyTarget = "<telegram-chat-id>"
enableTelegramWrites = false
allowedRepos = []
trustedTelegramSenders = []
confirmedWriteMethods = []
```

Example future write-mode config:

```toml
[plugins.entries.codex.config.codexBridge]
enabled = true
notifyChannel = "telegram"
notifyTarget = "<telegram-chat-id>"
enableTelegramWrites = true
allowedRepos = ["/Users/guclaw/openclaw", "/Users/guclaw/.openclaw/workspace/openclawbrain"]
trustedTelegramSenders = ["<jonathan-telegram-user-id>"]
confirmedWriteMethods = ["turn/start"]
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
