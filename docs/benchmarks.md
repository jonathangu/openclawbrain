# Benchmarks

## LLM-call metrics (OpenClaw session JSONL)

This harness estimates how many LLM API calls were made in a session log and how they map to user-visible exchanges.

**Definition**: a **turn** equals one LLM API call (one assistant completion).

### Run

```bash
python3 benchmarks/openclaw_llm_calls/analyze_session_jsonl.py \
  ~/.openclaw/agents/main/sessions/session-2026-03-01.jsonl \
  --json-out /tmp/openclaw_llm_calls.json
```

### Output

- A human-readable table printed to stdout.
- A JSON summary containing:
  - total LLM call count (turns)
  - prompt/completion/total tokens (when usage metadata exists)
  - LLM calls per user-visible exchange

If your session log includes token usage metadata, the script aggregates it. Otherwise, token counts will be zero but turn counts still work.
