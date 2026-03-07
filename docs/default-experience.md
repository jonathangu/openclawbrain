# Default Brain-Building Experience (macOS)

This is the **default, complete brain-building pipeline** for OpenClawBrain. It is designed to be reproducible, one-command, and **local-embedding-first** for OpenClaw operators.

## Goals

- Local embeddings everywhere (BGE-large via `fastembed`), **no OpenAI embeddings**.
- Re-embed existing states before learning so all vectors are consistent.
- Full replay pipeline (`--mode full`) with tool results included.
- Structural maintenance tasks run every cycle.
- Optional async route-teacher traces + route model training (off by default).

## Cost & spend notes

This pipeline intentionally does **LLM work** in replay. Teacher labeling is optional.

- **Replay `--mode full`** runs fast-learning transcript mining + edge replay + harvest. The fast-learning phase calls an LLM to extract learning events, so it will incur LLM usage.
- **Optional async teacher labeling** (`async-route-pg`) uses a teacher model to label route decisions and generate training traces when enabled.

The script **never uses OpenAI embeddings**. All embedding work is forced to local BGE-large (`BAAI/bge-large-en-v1.5`). If you want to avoid OpenAI entirely, use Ollama for replay/teacher labeling or set `--teacher none` and use a non-LLM replay mode.

## Local LLM (Ollama)

You can run the pipeline without OpenAI by using Ollama for replay fast-learning. Optional teacher labeling can also use Ollama. This is a full opt-out of OpenAI usage (embeddings are already local).

Install and start Ollama:

```bash
brew install ollama
ollama serve
```

Pull the recommended model:

```bash
ollama pull qwen3.5:9b-q4_K_M
```

Run replay with Ollama:

```bash
openclawbrain replay --state ./brain/state.json --sessions ./sessions/ --mode full --llm ollama
```

Optional: run async-route-pg with Ollama:

```bash
openclawbrain async-route-pg --state ./brain/state.json --teacher ollama --teacher-model qwen3.5:9b-q4_K_M
```

## One-command orchestration

The canonical unattended entrypoint is now the CLI subcommand:

```bash
openclawbrain build-all
```

This runs the default brain-building pipeline across all agents listed in `~/.openclaw/openclaw.json` and logs per agent under `~/.openclawbrain/<agent>/scratch/`.

The legacy script remains available for environments that prefer shell orchestration:

- `examples/ops/default_experience.sh`

### What it does (per agent)

1. **Re-embed** the state with local BGE-large embeddings.
2. **Replay** full pipeline with tool results included.
3. **Maintain** structural tasks: `health,decay,scale,split,merge,prune,connect`.
4. **Optional:** async route teacher labeling for recent queries (only when enabled).
5. **Optional:** train a route model from generated traces.

### Run it

```bash
openclawbrain build-all
```

Use a smaller fast-learning model with explicit worker parallelism:

```bash
openclawbrain build-all --llm ollama --llm-model qwen3.5:9b-q4_K_M --workers 8 --embed-model BAAI/bge-large-en-v1.5
```

For faster iteration, use prioritized and bounded replay when rebuilding large histories:

```bash
openclawbrain build-all \\
  --llm ollama \\
  --llm-model qwen3.5:9b-q4_K_M \\
  --workers 8 \\
  --replay-priority tool \\
  --replay-sample-rate 0.25 \\
  --replay-since-hours 24 \\
  --replay-max-interactions 50000 \\
  --advance-offsets-on-skip
```

This pattern only keeps recent tool-relevant interactions and can significantly cut replay runtime. Keep `--advance-offsets-on-skip` enabled for one-pass resumable runs; otherwise repeated passes may be required to eventually process skipped interactions.

## Optional: Async teacher traces

By default, the script runs only the core steps (re-embed → replay → maintain). To enable high-cadence teacher traces and route-model training, set env vars before running the script.

Minimum enablement:

```bash
ENABLE_ASYNC_TEACHER=1 \
TEACHER_PROVIDER=ollama \
TEACHER_MODEL=qwen3.5:9b-q4_K_M \
examples/ops/default_experience.sh
```

Key env vars (all optional; defaults are shown in the script):
- `ENABLE_ASYNC_TEACHER` (`0` by default)
- `TEACHER_PROVIDER` (`none` by default; set to `ollama` or `openai`)
- `TEACHER_MODEL`
- `SINCE_HOURS`
- `MAX_DECISION_POINTS`
- `SAMPLE_RATE`
- `MAX_QUERIES`

For more detailed trace workflows (side traces, combining runs, training/apply), see `docs/teacher-traces.md`.

### Environment

If present, the script sources:

```
~/.openclaw/credentials/env/openclawbrain.env
```

This is the place to set `OPENCLAWBRAIN_DEFAULT_LLM=ollama` and `OPENCLAWBRAIN_OLLAMA_MODEL=qwen3.5:9b-q4_K_M` so replay auto-selection stays local. The script does not print secrets.

When `openclawbrain replay --llm auto` is used (the default), LLM selection is:

The shipped 9B tag is only the default fallback; operators can still upgrade to any other Ollama model with `--llm-model`, `--teacher-model`, `OPENCLAWBRAIN_OLLAMA_MODEL`, or `OLLAMA_MODEL`.

- If `OPENCLAWBRAIN_DEFAULT_LLM` is set to `none`, `openai`, `ollama`, or `openrouter`, that choice is honored.
- Otherwise, prefer Ollama when `OPENCLAWBRAIN_OLLAMA_MODEL` or `OLLAMA_MODEL` is set.
- Else prefer OpenAI when `OPENAI_API_KEY` is set.
- Else, no LLM is used.

You can also control parallelism with:

- `PARALLEL_AGENTS` (default `2`): number of agent pipelines to run concurrently. Parallelism is across agents only; steps within a single agent remain sequential.

## Logging and audit artifacts

Each agent run writes auditable artifacts under `~/.openclawbrain/<agent>/scratch/` with a shared timestamp prefix.

- `default-experience.<ts>.log` captures the full run (stdout/stderr) with readable section headers.
- `default-experience.<ts>.status_before.json` and `default-experience.<ts>.status_after.json` capture `openclawbrain status --json` snapshots.
- `default-experience.<ts>.maintain.json` captures maintain output.
- If async teacher is enabled, `default-experience.<ts>.async-route-pg.json` and `default-experience.<ts>.train-route-model.json` capture machine-readable step outputs.
- `route_traces.jsonl` and `route_model.npz` are only produced when async teacher is enabled.
- `state.pre-default-experience.<ts>.json` is the pre-run state backup.
- `default-experience.<ts>.manifest.json` summarizes paths and embeds the before/after status objects.
- `build-all.<ts>.events.jsonl` is the durable run-level stream for all `build-all` agents/steps. It is append-only and includes `run_start`, `agent_start`, `agent_end`, and step start/end records.
- `build-all.<ts>.stall_audit.jsonl` captures step-level stall timeouts, retries, and skips.
- The root `~/.openclawbrain/scratch/build-all.<ts>.manifest.json` is written immediately at start with status `running`, then updated as each agent completes so partial progress is visible after crashes, and set to `complete` on completion with summary counts.

## Notes

- The script expects existing agent state files at `~/.openclawbrain/<agent>/state.json`.
- The script does **not** start or stop `launchd` services; it only builds artifacts.
- If you need a clean rebuild + cutover workflow, use `examples/ops/rebuild_then_cutover.sh` after this completes.
- Optional stall guard: `openclawbrain build-all --step-stall-timeout-seconds 1800` terminates stalled steps, retries once, then skips with a JSONL audit record.
- `build-all` now snapshots `openclawbrain status --json` before replay and enforces embedding compatibility:
  - it requires both `embedder_dim` and `index_dim` to match (if both are present),
  - it prints a clear warning and continues when `--reembed` is enabled,
  - otherwise it fails fast with a suggestion to rerun with `--reembed` or `openclawbrain reembed --state ...`.

## Tool provenance edges

Replay now builds first-class tool action/evidence nodes by default. This helps the brain learn tool usage quickly while keeping evidence out of prompt context unless explicitly requested.

- Tool actions are always connected to the fired nodes that caused them.
- Tool evidence is stored only for allowlisted tools and is redacted/truncated to avoid secrets or graph bloat.
- Tool evidence nodes (`tool_evidence::...`) are excluded from traversal context by default.
- Learning nodes with `session_pointer` (or session + line range metadata) automatically link to matching tool evidence nodes.

To view tool evidence during query/traverse:

```bash
openclawbrain query --provenance "your query"
```

To disable tool edge creation during replay:

```bash
openclawbrain replay --no-tool-edges --mode edges-only --sessions <path>
```
