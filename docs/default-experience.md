# Default Brain-Building Experience (macOS)

This is the **default, complete brain-building pipeline** for OpenClawBrain. It is designed to be reproducible, one-command, and **local-embedding-first** for OpenClaw operators.

## Goals

- Local embeddings everywhere (BGE-large via `fastembed`), **no OpenAI embeddings**.
- Re-embed existing states before learning so all vectors are consistent.
- Full replay pipeline (`--mode full`) with tool results included.
- Structural maintenance tasks run every cycle.
- Async route-teacher labeling + route model training.

## Cost & spend notes

This pipeline intentionally does **LLM work** in replay and labeling:

- **Replay `--mode full`** runs fast-learning transcript mining + edge replay + harvest. The fast-learning phase calls an LLM to extract learning events, so it will incur LLM usage.
- **Async teacher labeling** (`async-route-pg --teacher openai --teacher-model gpt-5-mini`) uses the OpenAI API to label route decisions and generate training traces.

The script **never uses OpenAI embeddings**. All embedding work is forced to local BGE-large (`BAAI/bge-large-en-v1.5`). If you do not want any OpenAI usage at all, set `--teacher none` and use a non-LLM replay mode; otherwise supply `OPENAI_API_KEY`.

## One-command orchestration

The operator script below is the default macOS path for a complete brain build across all agents in `~/.openclaw/openclaw.json`.

- Script: `examples/ops/default_experience.sh`
- It is idempotent and logs per agent under `~/.openclawbrain/<agent>/scratch/`.

### What it does (per agent)

1. **Re-embed** the state with local BGE-large embeddings.
2. **Replay** full pipeline with tool results included.
3. **Maintain** structural tasks: `health,decay,scale,split,merge,prune,connect`.
4. **Async route teacher labeling** (OpenAI teacher, `gpt-5-mini`) for the last 168 hours.
5. **Train route model** from the generated traces.

### Run it

```bash
examples/ops/default_experience.sh
```

### Environment

If present, the script sources:

```
~/.openclaw/credentials/env/openclawbrain.env
```

This is the place to set `OPENAI_API_KEY` (required for replay `--mode full` and async teacher labeling). The script does not print secrets.

## Notes

- The script expects existing agent state files at `~/.openclawbrain/<agent>/state.json`.
- The script does **not** start or stop `launchd` services; it only builds artifacts.
- If you need a clean rebuild + cutover workflow, use `examples/ops/rebuild_then_cutover.sh` after this completes.
