# Findings — history ingestion + runtime-policy training

Date: 2026-03-06
Repo: `openclawbrain`

## What was actually wrong

1. History ingestion only treated session directories as top-level `*.jsonl` files, so it missed:
   - `sessions.json` session indices
   - nested Codex rollout trees such as `~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl`
   - direct Codex sqlite state databases whose `threads.rollout_path` values point at rollout logs
2. Codex rollout JSONL lines were not replayable because `replay.py` only understood OpenClaw-style message envelopes, not Codex `response_item` messages and function-call records.
3. Runtime-policy training had dishonest export paths:
   - `replay --traces-out` wrote placeholder traces without decision points or query vectors
   - `replay --labels-out` wrote no labels
   - `harvest --traces-out` wrote placeholder traces
4. `train-route-model` failed late with a generic message instead of telling operators which required fields were missing.

## Fix implemented

- Added shared session-source resolution in `openclawbrain/session_sources.py`.
- Extended replay/session ingestion to resolve `sessions.json`, nested Codex rollouts, and direct Codex sqlite state files.
- Extended replay parsing to understand Codex `response_item` message/function-call/function-call-output records.
- Replaced fake runtime-policy export paths with explicit CLI rejections for:
  - `replay --traces-out`
  - `replay --labels-out`
  - `harvest --traces-out`
- Kept `harvest --labels-out` as the valid harvested-label path.
- Made `train-route-model` fail fast with a detailed readiness summary that reports missing `query_vector`, missing decision points/candidates, and insufficient indexed candidates.

## Remaining constraints

- `sessions.json` entries that only point to ACP/Codex metadata are resolved through the local Codex stores (`CODEX_HOME`, `~/.codex/sessions`, `state_*.sqlite`). If those local Codex artifacts are gone, replay still cannot reconstruct deleted histories.
- This change does not make `replay` or `harvest` synthesize route traces; the supported trainable trace producer remains `async-route-pg`.

## Validation plan executed

- Added regression coverage for nested Codex rollouts.
- Added regression coverage for `sessions.json` expansion and sqlite-backed rollout discovery.
- Added regression coverage for Codex rollout parsing of messages + function-call records.
- Added regression coverage for new CLI export rejections.
- Added regression coverage for `train-route-model` readiness errors.

## Deployment / prod

- No production deployment performed.
- No production state or live agent session stores were modified by this repo change.
