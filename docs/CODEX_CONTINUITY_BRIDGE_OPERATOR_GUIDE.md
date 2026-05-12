# OpenClawBrain Codex Telegram Bridge Operator Guide

Status: implemented in `openclawbrain@0.2.33` as an OpenClawBrain-owned plugin surface. It does not modify OpenClaw core.

## Product Contract

Codex UI is the high-bandwidth coding workbench when Jonathan is at the computer. OpenClaw and Telegram are the mobile operator surface when he is away. OpenClawBrain is the continuity layer between them: it can show recent Codex messages, follow one thread briefly, attach passive notes, send explicit actions into an exact Codex thread, steer active work, detach cleanly, and leave local audit/proof without turning Telegram into a noisy second IDE.

The bridge has two lanes:

- **Transcript lane:** read recent messages from Codex SQLite `threads.rollout_path` plus rollout JSONL, copy final user/assistant text directly, and forward watched assistant messages to Telegram.
- **Note lane:** attach passive operator context to a bound Codex thread without starting or steering Codex.
- **Control lane:** write an explicit action into a specific Codex thread through Codex app-server `thread/resume` plus `turn/start`, or steer an active Codex turn through `turn/steer`.

The bridge never writes Codex SQLite or rollout JSONL. It never stores raw Codex transcript as durable OpenClawBrain memory.

## Daily Workflow

1. Ask `/brain codex threads` to find the thread id you care about.
2. Ask `/brain codex last <thread-id>` to copy the latest assistant reply into Telegram.
3. Ask `/brain codex bind <thread-id>` to attach this Telegram chat to that exact Codex thread.
4. Ask `/brain codex tail --bound` when you want new completed assistant replies forwarded for a short window.
5. Ask `/brain codex note <message>` when you want to attach context without asking Codex to act.
6. Ask `/brain codex act <message>` to send a real instruction to the bound idle thread.
7. Ask `/brain codex steer <message>` only when Codex is actively working and you need to redirect the current turn.
8. Ask `/brain codex detach` when you want Telegram to stop forwarding and forget the chat binding.
9. Ask `/brain codex handoff --bound` when returning to the computer and wanting observed facts separated from Codex-reported claims.

## Telegram Commands

```text
/brain codex status
/brain codex doctor
/brain codex threads [filter]
/brain codex messages [thread-id|--latest|--bound] [--limit 5] [--role assistant|user|all]
/brain codex last [thread-id|--latest|--bound]
/brain codex bind <thread-id>
/brain codex binding
/brain codex unbind
/brain codex detach
/brain codex tail [thread-id|--latest|--bound]
/brain codex watch [thread-id|--latest|--bound] --messages
/brain codex watch [thread-id|--latest|--bound]
/brain codex watches
/brain codex unwatch <watch-id|thread-id|latest|all>
/brain codex note <message>
/brain codex notes
/brain codex act [--with-notes] <message>
/brain codex reply <message>
/brain codex send <thread-id|--bound> <message>
/brain codex steer [thread-id|--bound] <message>
/brain codex handoff [thread-id|--latest|--bound]
```

`note` is passive and never calls Codex app-server. `act`, `reply`, and `send` are actions: they may edit files, run tools, or request approvals under Codex's normal sandbox. `reply` uses the bound thread. `send` requires an explicit full thread id or `--bound`. The bridge intentionally rejects `--latest` writes because wrong-thread writes are the failure mode that would destroy trust.

`steer` also uses an exact or bound thread, but it requires a currently in-progress Codex turn. If the thread is idle, use `reply` instead. `goal` remains a future phase because new-thread creation requires repo/model/sandbox selection.

## API Routes

Routes are registered by the `openclawbrain` plugin and require gateway authentication.

- `GET /plugins/openclawbrain/codex/status`
- `GET /plugins/openclawbrain/codex/threads`
- `GET /plugins/openclawbrain/codex/messages?threadId=<id>&limit=5&role=assistant`
- `GET /plugins/openclawbrain/codex/handoff?threadId=<id>`
- `GET /plugins/openclawbrain/codex/watches`

Mutating writes are exposed through the authenticated `/brain codex act`, `/brain codex reply`, and `/brain codex send` command path, not public unauthenticated HTTP.

## Safety Model

Public package defaults are safe:

- direct message copy is on;
- message watches are on;
- forwarding mode defaults to `redacted`;
- Telegram-to-Codex writes default to off;
- high-risk Telegram writes default to off.

Jonathan's local profiles can enable writes without patching OpenClaw core. The happy path is meant to be fast:

- sender/chat is trusted by OpenClaw or listed in `trustedTelegramSenders`;
- the Telegram chat is bound to an exact thread, or `send` uses an explicit full thread id;
- the repo path is under `writeAllowlist` or `repoAllowlist`;
- the message is not high risk;
- Codex app-server accepts `thread/resume` and `turn/start`.

High-risk wording such as publish, deploy, delete, production, token, secret, password, full-access, or yolo is refused from Telegram unless `highRiskTelegramWrites` is explicitly enabled. Even then, Codex sandbox and approval behavior still applies.

## Config

Public-safe default:

```toml
[plugins.entries.openclawbrain.config.codexBridge]
enabled = true
messageWatchesEnabled = true
directMessageCopyEnabled = true
telegramForwardingMode = "redacted"
enableTelegramWrites = false
enableTelegramSteer = false
highRiskTelegramWrites = false
repoAllowlist = []
writeAllowlist = []
trustedTelegramSenders = []
```

Jonathan local trusted setup:

```toml
[plugins.entries.openclawbrain.config.codexBridge]
enabled = true
messageWatchesEnabled = true
directMessageCopyEnabled = true
telegramForwardingMode = "raw_trusted"
enableTelegramWrites = true
enableTelegramSteer = true
highRiskTelegramWrites = false
trustedTelegramSenders = ["<trusted-telegram-user-or-chat-id>"]
writeAllowlist = ["/Users/guclaw"]
appServerUrl = "ws://127.0.0.1:53177"
appServerTimeoutMs = 30000
```

Run Codex app-server beside OpenClaw with:

```bash
codex app-server --listen ws://127.0.0.1:53177
```

## Memory Boundaries

OpenClawBrain should store durable operating truths:

- Codex UI is the high-bandwidth workbench.
- OpenClaw and Telegram are the mobile operator surface.
- Telegram should receive direct copied messages only when requested or watched.
- Codex bridge notifications should stay concise.
- Handoff briefs must separate observed facts from Codex-reported claims.

OpenClawBrain should not store durable raw telemetry:

- raw Codex messages;
- command output;
- full diffs;
- rollout JSONL contents;
- temporary watch requests;
- secrets or auth failures.

Current explicit instruction still overrides these durable defaults.

## Remaining Risks

- Codex app-server is experimental and may change.
- Transcript reads depend on Codex continuing to write `threads.rollout_path` and rollout JSONL.
- Forwarding raw trusted Codex text to Telegram exports that text to Telegram, which is an external durable system.
- If Codex UI/app-server is not running, writes fail or become `possibly_sent` after timeout.
- Active-turn steer and new-thread goal creation need stronger event integration before they should be exposed.

## Local Install / Update

Use the OpenClawBrain-owned installer from this repo:

```bash
pnpm --dir /Users/guclaw/.openclaw/workspace/openclawbrain install:local-openclaw:all
openclaw gateway restart
```

This installs the plugin into local OpenClaw homes without dirtying `/Users/guclaw/openclaw`.
