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
ollama pull qwen2.5:32b-instruct
```

Run replay with Ollama:

```bash
openclawbrain replay --state ./brain/state.json --sessions ./sessions/ --mode full --llm ollama
```

Optional: run async-route-pg with Ollama:

```bash
openclawbrain async-route-pg --state ./brain/state.json --teacher ollama --teacher-model qwen2.5:32b-instruct
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
openclawbrain build-all --llm ollama --llm-model qwen2.5:7b-instruct --workers 8 --embed-model BAAI/bge-large-en-v1.5
```

## Optional: Async teacher traces

By default, the script runs only the core steps (re-embed → replay → maintain). To enable high-cadence teacher traces and route-model training, set env vars before running the script.

Minimum enablement:

```bash
ENABLE_ASYNC_TEACHER=1 \
TEACHER_PROVIDER=ollama \
TEACHER_MODEL=qwen2.5:32b-instruct \
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

This is the place to set `OPENCLAWBRAIN_DEFAULT_LLM=ollama` and `OPENCLAWBRAIN_OLLAMA_MODEL=qwen2.5:32b-instruct` so replay auto-selection stays local. The script does not print secrets.

When `openclawbrain replay --llm auto` is used (the default), LLM selection is:

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

## Notes

- The script expects existing agent state files at `~/.openclawbrain/<agent>/state.json`.
- The script does **not** start or stop `launchd` services; it only builds artifacts.
- If you need a clean rebuild + cutover workflow, use `examples/ops/rebuild_then_cutover.sh` after this completes.

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
