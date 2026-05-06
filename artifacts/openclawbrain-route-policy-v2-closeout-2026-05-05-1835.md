# OpenClawBrain route policy v2 closeout — 2026-05-05 18:35 PDT

## Objective
Implement the route-teacher/counterfactual learning loop end to end, with Part 8 completed deeply: distill examples into a compact deterministic learned `route_fn`, store it safely, update it through gated learning, publish the release, and verify live install.

## Shipped version
- Repo commit: `95b9c59a43d9e809333b8767a92ac40bcb2c5d79`
- Tag: `v0.2.18`
- ClawHub package: `openclawbrain@0.2.18`
- ClawHub release id: `rd766s3xxj7e4acakxayc7yjqx8678b6`

## What changed
- Added `docs/ROUTE_TEACHER_MASTER_PLAN_PART2.md` covering policy representation, storage, update flow, gating, and proof surfaces.
- Added `packages/openclaw-plugin/src/route-policy-v2.ts` for policy distillation, validation, scoring, and gated activation.
- Upgraded storage to `SCHEMA_VERSION = 6`.
- Added `route_frames` table for compact redacted turn-level routing features.
- Added route decision audit fields for:
  - `route_frame_id`
  - `policy_rule_id`
  - `candidate_count`
  - `reason_code`
  - `injection_payload_hash`
- Updated runtime routing so `RouteFn` first evaluates active `route-policy-v2` rules, records the matched rule id, and falls back conservatively when no safe rule matches.
- Updated proof/search payloads to expose actual route, active policy snapshot, matched rule, graph snapshot, teacher verdict, and counterfactual summary.
- Added deep tests covering policy validation, activation, matched-rule recording, and budget/broad-rule rejection.

## Learned route_fn design
### Stored representation
The learned route function is stored as a `route-policy-v2` snapshot in SQLite (`route_policy_snapshots_v2`).

A snapshot contains:
- lifecycle state: `candidate` / `shadow` / `active` / `rejected`
- ordered compact rules
- global budgets
- evaluation summary
- supporting example ids

Each rule contains only compact structured signals, not raw transcript text:
- match predicates derived from turn/task/intent signals
- chosen route (`no_memory`, `retrieve_memory`, `retrieve_and_distill`, `high_confidence_correction_only`, etc.)
- allowed memory types
- query templates
- graph depth
- sync planner allowance
- confidence
- evidence ids

### Training/update flow
1. Runtime records a route decision and a compact redacted route frame.
2. Route teacher critiques actual route vs graph-grounded alternatives.
3. Counterfactuals are generated from graph snapshots and outcomes.
4. `route_training_examples_v2` accumulates support/harm evidence.
5. Distiller compresses examples into a candidate `route-policy-v2` snapshot.
6. Validator rejects unsafe/broad/over-budget policies.
7. Safe snapshots become `shadow` or `active` according to config gates.
8. Active snapshot is used deterministically by `RouteFn` on future turns.
9. Runtime records which rule matched so later outcomes can reinforce or harm that rule indirectly through new examples.

### Safety invariants
- No raw user text/transcript storage in route frames or policy rules.
- Policy validation rejects broad retrieval rules.
- Policy validation rejects sync-planner budget overflow.
- Teacher output is validated and fail-closed.
- Runtime defaults stay latency-safe and do not require sync LLM by default.
- Shadow snapshots are ignored by runtime until activated.

## Verification
### Local package gates
- `pnpm --dir packages/openclaw-plugin test`
- Result: `79/79` passing

### Packaging
- `git diff --check`
- `npm pack --workspace packages/openclaw-plugin --pack-destination /tmp`
- Output tarball: `/tmp/openclawbrain-0.2.18.tgz`

### Fresh temp-HOME install
- Installed tarball into a clean temporary OpenClaw home.
- Verified plugin loaded at version `0.2.18`.

### Live install/runtime
- `openclaw plugins inspect openclawbrain --runtime --json`
- Verified:
  - version `0.2.18`
  - status `loaded`
  - enabled `true`
  - activated `true`
  - services: `openclawbrain`
  - `httpRoutes: 11`
  - `hookCount: 5`
- `openclaw gateway status`
- Verified gateway running and reachable.

### Installed-code smoke test
Using the live installed code under `~/.openclaw/extensions/openclawbrain/dist/*`:
- inserted a route training example
- distilled an active policy snapshot
- planned a matching turn
- observed a recorded `matchedPolicyRuleId`

Observed smoke output:
```json
{
  "snapshotStatus": "active",
  "route": "no_memory",
  "matchedPolicyRuleId": "34a49249fe02591a"
}
```

### Publish verification
- `clawhub package inspect openclawbrain --version 0.2.18 --json`
- Verified latest version/tag/source metadata for `0.2.18`.
- Requested package rescan: `sd749mzx4m1a9jwwv56dcrd8g9867624`.

## Notes
- GitHub release `v0.2.18` was created and includes `openclawbrain-0.2.18.tgz` for fallback distribution.
- ClawHub scan status was still `pending` immediately after publish/rescan.
