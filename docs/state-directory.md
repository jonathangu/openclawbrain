# State Directory Layout

OpenClawBrain stores per-agent state under `~/.openclawbrain/<agent>/`. The directory is portable and safe to back up while the daemon is stopped.

## On-disk artifacts

`state.json`: Primary brain state (nodes, edges, embeddings, metadata).

`state.bak`: Last-good snapshot written before major maintenance or writes.

`graph.json`: Legacy graph-only export for compatibility and debugging.

`index.json`: Legacy index metadata for compatibility with older tooling.

`texts.json`: Legacy node text export (parallel to graph/index).

`fired_log.jsonl`: Optional append-only log of fired nodes per chat/session (from adapters or daemon).

`learning_events.jsonl`: Append-only learning sidecar used for fast-learning and replay pipelines.

`injected_feedback.jsonl`: Dedup ledger of injected feedback/corrections to avoid re-applying the same event.

`journal.jsonl`: Append-only query/learn/health telemetry used by replay and async teachers.

`replay_checkpoint.json`: Replay resume checkpoint (last processed offsets, timestamps, and worker state).

`daemon.sock`: Unix socket used by `openclawbrain serve` for hot-path queries.

`daemon.pid`: PID file for the daemon process (used for clean stop/restart).

`scratch/`: Temporary outputs (route traces, build-all events, staged exports). Safe to prune when idle.

## What grows over time

- `state.json` grows as new nodes/edges are added.
- `journal.jsonl`, `fired_log.jsonl`, `learning_events.jsonl`, and `injected_feedback.jsonl` are append-only and grow continuously.
- `replay_checkpoint.json` changes during replay and stays small.
- `scratch/` can grow quickly if batch jobs are running.

## Maintenance impact

`openclawbrain maintain` (or `replay` with pruning) can:

- Compact `state.json` by pruning dormant edges/nodes.
- Reduce size of hot-path traversals by demoting or deleting low-signal edges.
- Leave append-only JSONL files intact unless explicitly rotated.

Recommended: rotate or archive `journal.jsonl` and `learning_events.jsonl` periodically if they become large, then resume with fresh files.

## Backup, restore, migrate

Backup:

- Stop the daemon.
- Copy the entire directory (`state.json`, sidecars, and logs).

Restore:

- Stop the daemon.
- Replace `state.json` with `state.bak` or a backup snapshot.
- Keep JSONL sidecars if you want learning history; discard them if you need a clean slate.

Migrate:

- Move `~/.openclawbrain/<agent>/` to a new machine or path.
- Update any service or hook configs that reference the state path.
- Start the daemon against the new `state.json` path.
