# Architecture

OpenClawBrain is a native OpenClaw plugin that runs entirely on the user's machine. It hooks into the agent's pre-prompt build cycle and optionally injects bounded, redacted context.

## Plugin structure

```
packages/openclaw-plugin/
├── src/
│   ├── index.ts          # Plugin entry: hooks, routes, service registration
│   ├── config.ts         # Config resolution (plugins.entries.openclawbrain.config)
│   ├── policy.ts         # Turn classification and decision policy
│   ├── context-files.ts  # Read/redact/clip activation files
│   ├── proof-store.ts    # Local proof event append/read/status
│   ├── redact.ts         # Text redaction, hashing, safe string
│   └── status.ts         # Status payload builder
├── dist/                 # Built JS (committed for ClawHub installs)
├── test/
│   └── index.test.mjs    # 12 tests covering all paths
├── openclaw.plugin.json  # Plugin manifest (family, config schema, hooks)
└── package.json          # Package metadata with ClawHub compat/build fields
```

## Turn classification

When a turn arrives, the plugin classifies it into one of six slices:

| Slice | Example | Default policy |
|-------|---------|----------------|
| `direct-answer` | "What is 2+2?" | stay silent |
| `continuation` | "continue" / "keep going" | inject context |
| `correction-follow-up` | "use family inbox, not work" | inject corrections |
| `retrieval-heavy` | "what did we decide about X?" | inject context |
| `tool-heavy` | "run tests and check results" | inject context + verification hint |
| `stale-memory-conflict` | "that's wrong, use X instead" | inject corrections |

Classification is heuristic-based on the turn type (if provided by OpenClaw) and redacted prompt patterns. It does not make model calls.

## Decision policy

The policy engine takes the classification, the mode, and the config, and returns one of:

- **`stay_silent`** — No context injected. Proof event written.
- **`proof_only`** — Log what was considered but don't inject.
- **`correction_only`** — Inject `corrections.md` content only.
- **`full_context`** — Inject `context.md` + `corrections.md` + `tool-guidance.md` as relevant.

Mode affects which slices are acted on:

| Mode | Behavior |
|------|----------|
| `off` | Always stay silent. |
| `proof-only` | Classify and log, never inject. |
| `conservative` | Inject only on corrections, continuation, tool-heavy, and stale memory. Stay silent on direct answers and unknown. |
| `active` | Inject on most slices (conservative + more). |

## Hooks

The plugin registers these OpenClaw hooks:

- **`before_prompt_build`** — Main hook. Classifies the turn, decides policy, reads context, returns `{ prependContext: "..." }` or `{}`.
- **`agent_turn_prepare`** — Optional secondary hook (same logic, gated by `supportsHook`).
- **`model_call_started`** — Writes a telemetry proof event.
- **`model_call_ended`** — Writes a telemetry proof event.
- **`gateway_start`** — Writes status to the activation root.
- **`gateway_stop`** — Writes status to the activation root.
- **`agent_end`** — Optional, gated by `hooks.allowConversationAccess` and `supportsHook`.

## Injection mechanism

When the policy decides to inject context:

1. Read activation files (`context.md`, `corrections.md`, `tool-guidance.md`) from the agent's activation root.
2. Reject symlinks, oversized files, and non-regular files before reading.
3. Redact sensitive values (emails, tokens, phone numbers, URLs) from the content.
4. Clip to `maxContextChars`.
5. Build a bounded injection string with headers indicating the slice and decision.
6. Return `{ prependContext: injectionText }` to the OpenClaw hook caller.

If anything fails, return `{}` (no mutation). The agent runs normally.

## Proof events

Every decision writes a local proof event as JSONL under the activation root:

```json
{
  "schemaVersion": "ocb.proof.event.v1",
  "pluginVersion": "0.1.1",
  "agentId": "main",
  "decisionKind": "correction_only",
  "reasonCode": "correction_follow_up_detected",
  "slice": "correction-follow-up",
  "mode": "conservative",
  "rawTranscriptStored": false,
  "rawUserTextStored": false,
  "redactionApplied": true,
  "hashesOnlyForUserText": true
}
```

Privacy claims in every proof event:
- `rawTranscriptStored: false` — raw transcripts are never written.
- `rawUserTextStored: false` — raw user text is never written.
- `redactionApplied: true` — sensitive values are removed before any storage.
- `hashesOnlyForUserText: true` — user text appears only as SHA-256 hashes.

Proof events rotate after `proofRetentionEvents` (default 1000).

## HTTP routes

Two routes are registered for the gateway:

- `GET /plugins/openclawbrain/status` — Returns plugin status, config, last decision kind, and timestamp.
- `GET /plugins/openclawbrain/proof?limit=N` — Returns the most recent N proof events (max 100).

Both require gateway authentication.

## Redaction

The redactor covers:
- Email addresses → `[redacted-email]`
- Token/secret-like patterns → `[redacted-secret]`
- Phone numbers → `[redacted-phone]`
- URLs → `[redacted-url]`
- API key patterns → `[redacted-secret]`

Redaction happens before any storage or injection. Original content is never persisted.

## Config resolution

Config is resolved from `plugins.entries.openclawbrain.config` only. A root `openclawbrain` config key is intentionally ignored. This prevents namespace collisions with other plugins and keeps config discoverable under the standard plugin entry point.

`rawTranscriptUpload` is a `const: false` field. If set to `true`, the plugin fails closed and returns empty on all turns.
