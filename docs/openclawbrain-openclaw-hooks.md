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

## Install and enable

```bash
openclaw hooks install /path/to/openclawbrain/integrations/openclaw/hooks/openclawbrain-context-injector
openclaw hooks enable openclawbrain-context-injector
openclaw gateway restart
```

Note: `--link` is dev-only. If you use it, set `hooks.internal.load.extraDirs` to the parent hooks directory (the directory that contains `openclawbrain-context-injector/`), then restart the gateway.

To disable:

```bash
openclaw hooks disable openclawbrain-context-injector
```

Troubleshooting:

- If `openclaw hooks info openclawbrain-context-injector` is not found, run `openclaw hooks list` and restart the gateway. Ensure the hook exists under `~/.openclaw/hooks/`.

## Runtime context budget

- **Default budget**: 12,000 chars.
- **Recall/correction messages** (for example: `remember`, `last time`, `earlier`, `we decided`, `correction`, `audit`) use **20,000** chars.

## Security notes

- `--exclude-bootstrap` is enabled so AGENTS/USER/memory bootstrap files are not re-injected from the brain context.
- `--redact` is enabled so common token-like patterns are masked before injection.
- Prompt data is data-only context (not instruction text).
- Keep secrets out of user-visible prompts and training data; redaction is conservative, not perfect.
- Use OpenClaw `chat_id` consistently to preserve per-turn firing logs for later `capture_feedback`.
