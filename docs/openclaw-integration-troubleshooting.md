# OpenClaw Integration Troubleshooting

This page covers the most common integration failures and quick fixes.

## 1) Hook not discovered (wrong path or no restart)

**Symptom:** `openclaw hooks info openclawbrain-context-injector` says not found.

**Fix:**

```bash
openclaw hooks install /path/to/openclawbrain/integrations/openclaw/hooks/openclawbrain-context-injector
openclaw gateway restart
openclaw hooks list
```

Make sure the hook exists under `~/.openclaw/hooks/` after install.

## 2) Hook discovered but not eligible

**Symptom:** `openclaw hooks check` shows the hook as not eligible.

**Likely causes:** missing `python3`, or missing OpenClaw config `workspace.dir`.

**Fix:**

- Install `python3` so the hook can execute.
- Ensure your OpenClaw config sets `workspace.dir` (the hook requires it).
- Re-run `openclaw hooks check` after fixing requirements.

## 3) Daemon not running / `daemon.sock` missing

**Symptom:** No context is injected and `~/.openclawbrain/main/daemon.sock` does not exist.

**Fix:**

```bash
openclawbrain serve --state ~/.openclawbrain/main/state.json
```

If you prefer the explicit subcommand, use:

```bash
openclawbrain serve start --state ~/.openclawbrain/main/state.json
```

## 4) Slow query -> hook times out (2s) and fails open

**Symptom:** OpenClaw answers as usual, but no brain context appears. This can happen when the hook exceeds its 2s timeout.

**Fix:**

- Ensure the daemon is running (so queries are hot).
- Reduce brain size or trim context budget if needed.
- Check that the state file is local (not on a slow network mount).

## 5) Slash commands are skipped

**Symptom:** Messages starting with `/` never receive brain context.

**Explanation:** This is by design. The hook skips slash commands.

## 6) How to tell if injection is happening

**Simple test:**

1. Send a normal (non-slash) message that includes “remember”.
2. Look for a `[BRAIN_CONTEXT ...]` block at the top of `bodyForAgent`.

If your OpenClaw deployment logs prompts, search gateway logs for `[BRAIN_CONTEXT`. If you do not log prompts, temporarily enable request logging and repeat the test.
