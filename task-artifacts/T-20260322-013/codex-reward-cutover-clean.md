# T-20260322-013 Reward Cutover Report

## Summary
- Added durable `brain_observations` storage and store APIs for full-turn reward evaluation.
- Replaced trace-slice teacher judging with teacher-v2 full-turn observation judging.
- Cut the worker over to observation evaluation -> teacher evidence/label -> existing update/decay/promotion flow.
- Preserved the explicit correction path in `observeUserTurn()` / `teachUserCorrection()` and kept it separate from terminal teacher rewards.
- Removed the old regex/self/scanner harvesting modules and replaced engine post-turn harvesting with observation recording.

## Focused Verification
- `node --experimental-transform-types --input-type=module <<'EOF' ... teacher observation script ... EOF`
  Result: `teacher-observation ok`
  Coverage: teacher-v2 input materialization and JSON result parsing.
- `node --experimental-transform-types --input-type=module <<'EOF' ... worker observation script ... EOF`
  Result: `worker-observation-flow ok`
  Coverage: good retrieval + bad generation, good retrieval + tool failure, ambiguous follow-up holdoff/attachment.
- `node --experimental-transform-types --input-type=module <<'EOF' ... service observation/restart/explicit-correction script ... EOF`
  Result: `service-observation-restart-explicit ok`
  Coverage: recorded observations, next-user follow-up, process restart with pending observations, explicit correction still immediate.
- `node --experimental-transform-types --input-type=module <<'EOF' ... engine afterTurn script ... EOF`
  Result: `engine-turn-observation ok`
  Coverage: post-turn observation recording and structured tool-result capture.
- `node --experimental-transform-types --input-type=module <<'EOF' ... fail-open status script ... EOF`
  Result: `fail-open-status ok`
  Coverage: teacher misconfiguration remains fail-open and status reporting stays truthful.

## Environment Note
- `npm ci --offline`
  Result: failed with `EPERM` on `/Users/example/.npm/_cacache` because the cache contains root-owned files, so a local `vitest` install could not be materialized in this sandbox.

## Residual Risks
- The cutover was executed with direct Node TS-transform verification instead of the repo's `vitest` runner because the local dependency install is blocked by the npm cache ownership issue above.
- Observation maturity currently uses `trainerIntervalMs` as the no-follow-up grace window; if operators want a different delay, that will need a dedicated config knob.
- Evidence tables remain in place for teacher evidence/tracing and historical compatibility, even though the old message-harvest reward path was removed.

## Artifact Note
- Writing the requested files directly under `/Users/example/.openclaw/workspace/task-artifacts/...` and `/Users/example/.openclaw/workspace/task-status/...` was blocked by sandbox permissions, so fallback copies were written inside this worktree.

## Commit Note
- `git commit -m "Cut over brain rewards to observations and teacher-v2"`
  Result: failed with `fatal: Unable to create '/Users/example/.openclaw/workspace/openclawbrain/.git/worktrees/T-20260322-013-reward-cutover-clean/index.lock': Operation not permitted`
  Reason: the worktree's shared git metadata directory is outside the writable sandbox.
