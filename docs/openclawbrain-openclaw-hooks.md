# OpenClawBrain OpenClaw Hooks

This is the recommended OpenClaw integration path for brain-backed context retrieval.
It keeps OpenClaw behavior **fail-open**:

- If the hook cannot run, OpenClaw keeps its original prompt untouched.
- No user-visible errors are introduced from the hook.

## What changes when enabled

When `openclawbrain-context-injector` is enabled on `message:preprocessed`:

- The hook runs `query_brain.py` for each non-slash user message.
- The returned prompt block is prepended to `event.context.bodyForAgent`.
- The injected block is marked as prompt data (`[BRAIN_CONTEXT ...]`) and is intended to be used as context, not instruction text.
- You keep normal OpenClaw flow and fall back to standard behavior if retrieval fails.
- If a user message starts with `Correction:`, `Fix:`, `Teaching:`, or `Note:`, the hook best-effort calls `capture_feedback` (fail-open).

## Always-on learning (recommended default experience)

Brain-first context injection is only half the loop. The other half is **same-turn learning**:

- When a user clearly corrects the agent, call `capture_feedback --kind CORRECTION` in the same turn.
- When a user teaches a durable rule/fact, call `capture_feedback --kind TEACHING`.
- Use `--message-id` or `--dedup-key` so retries cannot double-inject.

OpenClawBrain provides the adapter CLIs; your OpenClaw agent prompt/policy should make this automatic so operators do not need to say “inject teaching” or “log this.”

## Install and enable (recommended)

```bash
openclaw hooks install /path/to/openclawbrain/integrations/openclaw/hooks/openclawbrain-context-injector
openclaw hooks enable openclawbrain-context-injector
openclaw gateway restart
```

Note: `--link` is dev-only. If you use it, set `hooks.internal.load.extraDirs` to the parent hooks directory (the directory that contains `openclawbrain-context-injector/`), then restart the gateway.

Why the gateway restart? OpenClaw loads hook manifests at gateway start, so a restart is required for discovery.

## Verify

```bash
openclaw hooks check
openclaw hooks list
openclaw hooks info openclawbrain-context-injector
```

“Ready” means the hook is discovered, eligible, and enabled (no missing requirements such as `python3` or `workspace.dir`).

To disable:

```bash
openclaw hooks disable openclawbrain-context-injector
openclaw gateway restart
```

Troubleshooting:

- If `openclaw hooks info openclawbrain-context-injector` is not found, run `openclaw hooks list` and restart the gateway. Ensure the hook exists under `~/.openclaw/hooks/`.

## Runtime context budget

- **Default budget**: 20,000 chars.
- **Recall/correction messages** (for example: `remember`, `last time`, `earlier`, `we decided`, `correction`, `audit`) use **80,000** chars.
- Budgets are tunable if you maintain a custom hook or wrapper.

## Security notes

- `--exclude-bootstrap` is enabled so AGENTS/USER/memory bootstrap files are not re-injected from the brain context.
- `--redact` is enabled so common token-like patterns are masked before injection.
- Prompt data is marked as data-only context (`[BRAIN_CONTEXT ...]`), not instruction text.
- Fail-open: if the hook errors or times out, OpenClaw continues unchanged.
- Use OpenClaw `chat_id` consistently to preserve per-turn firing logs for later `capture_feedback`.

Troubleshooting: [docs/openclaw-integration-troubleshooting.md](openclaw-integration-troubleshooting.md)
