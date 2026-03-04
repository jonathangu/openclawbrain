---
name: openclawbrain-context-injector
description: prepend OpenClawBrain context on message:preprocessed
homepage: docs/openclawbrain-openclaw-hooks.md
metadata:
  openclaw:
    emoji: 🧠
    events:
      - message:preprocessed
    requires:
      bins:
        - python3
      config:
        - workspace.dir
---

# OpenClawBrain Context Injector Hook

- **Event**: `message:preprocessed`
- **Entry**: `handler.ts`
- **When enabled**: prepend retrieved brain context to `event.context.bodyForAgent`.
- **Intent**: provide low-friction OpenClaw integration for OpenClawBrain without changing user prompts.

## Behavior

On each `message:preprocessed` event, the hook does:

1. Skip if `message` is empty.
2. Skip slash-commands: if trimmed message starts with `/`.
3. Resolve agentId from workspace config:
   - match `event.context.workspaceDir` against `event.context.cfg.agents.list[].workspace`.
   - fallback: `main`.
4. Compute state path:
   - `~/.openclawbrain/<agentId>/state.json`.
5. Compute `chatId`:
   - `${channelId}:${conversationId}` when both exist.
6. Budget:
   - default 12,000 chars.
   - upgrade to 20,000 chars when message indicates recall/correction context.
7. Invoke:
   - `python3 -m openclawbrain.openclaw_adapter.query_brain <state-path> <message> --format prompt --exclude-bootstrap --redact --max-prompt-context-chars <budget>`
   - include `--chat-id` when available.
   - command timeout: 2s, fail-open.
8. On success, prepend returned text to `event.context.bodyForAgent`.
9. If the user message starts with `Correction:`, `Fix:`, `Teaching:`, or `Note:`, invoke `capture_feedback` (best-effort, fail-open).

## Fail-open

If command execution fails for any reason (timeout, parsing error, missing state, etc.),
append nothing and return the original event.

## Recommended defaults

Keep bootstrap exclusions and redaction enabled:
- `--exclude-bootstrap`
- `--redact`

The emitted prompt block is a `[BRAIN_CONTEXT ...]` section and is marked as data-only context.
