# OpenClawBrain Ops Recipes

Practical operator runbooks for cutovers and large replays.

**OpenClaw path note:** OpenClaw agent session logs typically live at `~/.openclaw/agents/<agent>/sessions` (e.g., `~/.openclaw/agents/main/sessions`). You can pass that directory directly via `--sessions <dir>`.

## Cutover fast

Use this when you need improved retrieval quickly and can defer full replay/harvest.

1. Run fast-learning only:

```bash
openclawbrain replay \
  --state ~/.openclawbrain/main/state.json \
  --sessions /path/to/sessions \
  --fast-learning \
  --stop-after-fast-learning \
  --resume \
  --checkpoint ~/.openclawbrain/main/replay_checkpoint.json
```

2. Start the daemon so serving traffic uses the latest state:

```bash
python3 -m openclawbrain.socket_server --state ~/.openclawbrain/main/state.json
```

3. Run full replay/harvest later (off-peak):

```bash
openclawbrain replay \
  --state ~/.openclawbrain/main/state.json \
  --sessions /path/to/sessions \
  --full-learning \
  --resume \
  --checkpoint ~/.openclawbrain/main/replay_checkpoint.json
```

`examples/ops/cutover_then_background_full_learning.sh` automates this sequence.

## Parallel replay

For large histories, run replay in parallel workers and checkpoint frequently:

```bash
openclawbrain replay \
  --state ~/.openclawbrain/main/state.json \
  --sessions /path/to/sessions \
  --full-learning \
  --replay-workers 4 \
  --workers 4 \
  --checkpoint ~/.openclawbrain/main/replay_checkpoint.json \
  --checkpoint-every-seconds 60 \
  --checkpoint-every 50 \
  --persist-state-every-seconds 300 \
  --resume
```

Notes:
- `--replay-workers` controls edge-replay parallelism.
- `merge_batches` in replay output indicates how many merge windows were applied.
- Use both `--checkpoint-every-seconds` and `--checkpoint-every` for long runs so restarts resume from recent progress.

Use `examples/ops/replay_last_days.sh` when you want a bounded replay window.

## Prompt caching

For stable prompt caching, build the appendix from fired nodes and append it late:

```bash
python3 examples/openclaw_adapter/query_brain.py \
  ~/.openclawbrain/main/state.json \
  "user query summary" \
  --format prompt
```

Guidelines:
- Append the `[BRAIN_CONTEXT v1]...[/BRAIN_CONTEXT]` block near the end of the final prompt.
- Keep stable node ordering (do not reshuffle lines between retries).
- Avoid echoing/paraphrasing this block earlier in the prompt; duplicated context reduces cache stability.

## Media memory

OpenClaw session logs often store uploads as `[media attached: ...]` stubs. Those stubs are low-signal by themselves, so replay should ingest allowlisted `toolResult` text (OCR/transcripts/captions) when present.

```bash
openclawbrain replay \
  --state ~/.openclawbrain/main/state.json \
  --sessions /path/to/sessions \
  --include-tool-results \
  --tool-result-allowlist image,openai-whisper,openai-whisper-api,openai-whisper-local,summarize \
  --tool-result-max-chars 20000
```

Flags:
- `--include-tool-results` enables attachment of tool outputs to media-stub user turns.
- `--tool-result-allowlist` limits ingestion to trusted tool names.
- `--tool-result-max-chars` bounds appended text size per interaction.
