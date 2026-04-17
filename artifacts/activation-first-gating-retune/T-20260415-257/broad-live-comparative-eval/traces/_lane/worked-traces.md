# Worked Traces

- traces included: 8/403
- selection rule: highest bundle score spread first, then trace id; turns ordered by per-turn score spread
- note: qualityScore and winnerMode are internal deterministic replay diagnostics, not the public/operator scorecard.
- source manifest: `extracted-semantic-rich-live-535` (frozen_recorded_session_eval_manifest.v1, 26eec14b9bb8)
- omitted traces: 395 (see _lane/summary-tables.json for the complete table)

## live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-011

- bundle dir: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-011`
- learned_route vs approved prior: `worse`
- learned_route vs no_brain floor: `better`
- diagnostic winner: `graph_prior_only`
- diagnostic top score modes: `vector_only`, `graph_prior_only`
- score spread: 100

| mode | diagnostic quality | compile ok | required-context recall | promotions | warnings |
| --- | ---: | ---: | ---: | ---: | ---: |
| graph_prior_only | 100 | 1/1 | 1/1 | 0 | 1 |
| vector_only | 100 | 1/1 | 1/1 | 0 | 1 |
| learned_route | 40 | 1/1 | 0/1 | 0 | 2 |
| no_brain | 0 | 0/1 | 0/1 | 0 | 1 |

### turn-1

- user: Pre-compaction memory flush. Store durable memories now (use memory/2026-03-05.md; create memory/ if needed). IMPORTANT: If the file alre...
- expected phrases: `no_reply`
- feedback kinds: none
- learned_route vs approved prior: `worse`
- diagnostic top modes: `vector_only`, `graph_prior_only` (spread 100)

| mode | phase | diagnostic quality | compile | required-context recall | activation | source | latency ms | promoted | selection | context preview |
| --- | --- | ---: | --- | ---: | --- | --- | ---: | --- | --- | --- |
| graph_prior_only | eval | 100 | yes | 1/1 | n/a | none | 5 | no | b8bf3b4c382c | Teaching feedback on live-session session 670a1ed3-b4c8-4dfe-8a1f-2d1526663457-seed: user: Pre-compaction memory flush. Store durable mem... |
| vector_only | eval | 100 | yes | 1/1 | n/a | none | 5 | no | f398a8b9c50c | Teaching feedback on live-session session 670a1ed3-b4c8-4dfe-8a1f-2d1526663457-seed: user: Pre-compaction memory flush. Store durable mem... |
| learned_route | eval | 40 | yes | 0/1 | yes | learned_route_artifact:candidate_override:router-artifact-t-20260415-257-activation-first-gating-only-v1@0.0.1 | 10 | no | f2d12b7746d1 | Pointer-aware init keeps fast boot first with anchors memory/2026-03-05-1953.md, MEMORY.md, AGENTS.md, SOUL.md, +4 more, working set TASK... |
| no_brain | eval | 0 | no | 0/1 | n/a | none | 0 | no | none | none |

## live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-014

- bundle dir: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-014`
- learned_route vs approved prior: `worse`
- learned_route vs no_brain floor: `better`
- diagnostic winner: `graph_prior_only`
- diagnostic top score modes: `vector_only`, `graph_prior_only`
- score spread: 100

| mode | diagnostic quality | compile ok | required-context recall | promotions | warnings |
| --- | ---: | ---: | ---: | ---: | ---: |
| graph_prior_only | 100 | 1/1 | 1/1 | 0 | 1 |
| vector_only | 100 | 1/1 | 1/1 | 0 | 1 |
| learned_route | 40 | 1/1 | 0/1 | 0 | 2 |
| no_brain | 0 | 0/1 | 0/1 | 0 | 1 |

### turn-1

- user: Pre-compaction memory flush. Store durable memories now (use memory/2026-03-05.md; create memory/ if needed). IMPORTANT: If the file alre...
- expected phrases: `no_reply`
- feedback kinds: none
- learned_route vs approved prior: `worse`
- diagnostic top modes: `vector_only`, `graph_prior_only` (spread 100)

| mode | phase | diagnostic quality | compile | required-context recall | activation | source | latency ms | promoted | selection | context preview |
| --- | --- | ---: | --- | ---: | --- | --- | ---: | --- | --- | --- |
| graph_prior_only | eval | 100 | yes | 1/1 | n/a | none | 5 | no | a51c47d35034 | Teaching feedback on live-session session 670a1ed3-b4c8-4dfe-8a1f-2d1526663457-seed: user: Pre-compaction memory flush. Store durable mem... |
| vector_only | eval | 100 | yes | 1/1 | n/a | none | 5 | no | 45887e5c3471 | Teaching feedback on live-session session 670a1ed3-b4c8-4dfe-8a1f-2d1526663457-seed: user: Pre-compaction memory flush. Store durable mem... |
| learned_route | eval | 40 | yes | 0/1 | yes | learned_route_artifact:candidate_override:router-artifact-t-20260415-257-activation-first-gating-only-v1@0.0.1 | 10 | no | 6af1f2fdb95a | Pointer-aware init keeps fast boot first with anchors memory/2026-03-05-1953.md, MEMORY.md, AGENTS.md, SOUL.md, +4 more, working set TASK... |
| no_brain | eval | 0 | no | 0/1 | n/a | none | 0 | no | none | none |

## live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-017

- bundle dir: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-017`
- learned_route vs approved prior: `worse`
- learned_route vs no_brain floor: `better`
- diagnostic winner: `graph_prior_only`
- diagnostic top score modes: `vector_only`, `graph_prior_only`
- score spread: 100

| mode | diagnostic quality | compile ok | required-context recall | promotions | warnings |
| --- | ---: | ---: | ---: | ---: | ---: |
| graph_prior_only | 100 | 1/1 | 1/1 | 0 | 1 |
| vector_only | 100 | 1/1 | 1/1 | 0 | 1 |
| learned_route | 40 | 1/1 | 0/1 | 0 | 2 |
| no_brain | 0 | 0/1 | 0/1 | 0 | 1 |

### turn-1

- user: Pre-compaction memory flush. Store durable memories now (use memory/2026-03-05.md; create memory/ if needed). IMPORTANT: If the file alre...
- expected phrases: `no_reply`
- feedback kinds: none
- learned_route vs approved prior: `worse`
- diagnostic top modes: `vector_only`, `graph_prior_only` (spread 100)

| mode | phase | diagnostic quality | compile | required-context recall | activation | source | latency ms | promoted | selection | context preview |
| --- | --- | ---: | --- | ---: | --- | --- | ---: | --- | --- | --- |
| graph_prior_only | eval | 100 | yes | 1/1 | n/a | none | 5 | no | 078858d3ed44 | Teaching feedback on live-session session 670a1ed3-b4c8-4dfe-8a1f-2d1526663457-seed: user: Pre-compaction memory flush. Store durable mem... |
| vector_only | eval | 100 | yes | 1/1 | n/a | none | 10 | no | ba02cbdc6bcf | Teaching feedback on live-session session 670a1ed3-b4c8-4dfe-8a1f-2d1526663457-seed: user: Pre-compaction memory flush. Store durable mem... |
| learned_route | eval | 40 | yes | 0/1 | yes | learned_route_artifact:candidate_override:router-artifact-t-20260415-257-activation-first-gating-only-v1@0.0.1 | 10 | no | 17e703ffb0c2 | Pointer-aware init keeps fast boot first with anchors memory/2026-03-05-1953.md, MEMORY.md, AGENTS.md, SOUL.md, +4 more, working set TASK... |
| no_brain | eval | 0 | no | 0/1 | n/a | none | 0 | no | none | none |

## live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-014

- bundle dir: `live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-014`
- learned_route vs approved prior: `worse`
- learned_route vs no_brain floor: `better`
- diagnostic winner: `graph_prior_only`
- diagnostic top score modes: `vector_only`, `graph_prior_only`
- score spread: 100

| mode | diagnostic quality | compile ok | required-context recall | promotions | warnings |
| --- | ---: | ---: | ---: | ---: | ---: |
| graph_prior_only | 100 | 1/1 | 1/1 | 0 | 1 |
| vector_only | 100 | 1/1 | 1/1 | 0 | 1 |
| learned_route | 40 | 1/1 | 0/1 | 0 | 2 |
| no_brain | 0 | 0/1 | 0/1 | 0 | 1 |

### turn-1

- user: Pre-compaction memory flush. Store durable memories now (use memory/2026-03-03.md; create memory/ if needed). IMPORTANT: If the file alre...
- expected phrases: `no_reply`
- feedback kinds: none
- learned_route vs approved prior: `worse`
- diagnostic top modes: `vector_only`, `graph_prior_only` (spread 100)

| mode | phase | diagnostic quality | compile | required-context recall | activation | source | latency ms | promoted | selection | context preview |
| --- | --- | ---: | --- | ---: | --- | --- | ---: | --- | --- | --- |
| graph_prior_only | eval | 100 | yes | 1/1 | n/a | none | 5 | no | 8cdded98483c | Teaching feedback on live-session session 9fe29ce5-b989-46df-bb1c-d79eb7599c65-seed: user: Pre-compaction memory flush. Store durable mem... |
| vector_only | eval | 100 | yes | 1/1 | n/a | none | 5 | no | a4b9c81babb0 | Teaching feedback on live-session session 9fe29ce5-b989-46df-bb1c-d79eb7599c65-seed: user: Pre-compaction memory flush. Store durable mem... |
| learned_route | eval | 40 | yes | 0/1 | yes | learned_route_artifact:candidate_override:router-artifact-t-20260415-257-activation-first-gating-only-v1@0.0.1 | 10 | no | 43206f310194 | Pointer-aware init keeps fast boot first with anchors MEMORY.md, AGENTS.md, SOUL.md, USER.md, +3 more, working set TASKS.md, task-artifac... |
| no_brain | eval | 0 | no | 0/1 | n/a | none | 0 | no | none | none |

## live-main-0856fc42-5677-417a-94a6-eeed26a9d994-window-003

- bundle dir: `live-main-0856fc42-5677-417a-94a6-eeed26a9d994-window-003`
- learned_route vs approved prior: `worse`
- learned_route vs no_brain floor: `better`
- diagnostic winner: `graph_prior_only`
- diagnostic top score modes: `vector_only`, `graph_prior_only`
- score spread: 100

| mode | diagnostic quality | compile ok | required-context recall | promotions | warnings |
| --- | ---: | ---: | ---: | ---: | ---: |
| graph_prior_only | 100 | 1/1 | 3/3 | 0 | 1 |
| vector_only | 100 | 1/1 | 3/3 | 0 | 1 |
| learned_route | 40 | 1/1 | 0/3 | 0 | 2 |
| no_brain | 0 | 0/1 | 0/3 | 0 | 1 |

### turn-1

- user: Exec denied (gateway id=d5f4d62c-fb2a-4717-8e25-73fb1c15efa2, approval-timeout (obfuscation-detected)): python3 - <<'PY' from pathlib imp...
- expected phrases: `T-20260403-129`, `/capture-openclawbrain-operator-proof.mjs`, `/proof/docs`
- feedback kinds: none
- learned_route vs approved prior: `worse`
- diagnostic top modes: `vector_only`, `graph_prior_only` (spread 100)

| mode | phase | diagnostic quality | compile | required-context recall | activation | source | latency ms | promoted | selection | context preview |
| --- | --- | ---: | --- | ---: | --- | --- | ---: | --- | --- | --- |
| graph_prior_only | eval | 100 | yes | 3/3 | n/a | none | 5 | no | 28a99ec60d72 | Teaching feedback on live-session session 0856fc42-5677-417a-94a6-eeed26a9d994-seed: assistant: Done. - Added `scripts/teacher-v3-proof-b... |
| vector_only | eval | 100 | yes | 3/3 | n/a | none | 5 | no | 30a7ad5cbf3c | Teaching feedback on live-session session 0856fc42-5677-417a-94a6-eeed26a9d994-seed: assistant: Done. - Added `scripts/teacher-v3-proof-b... |
| learned_route | eval | 40 | yes | 0/3 | yes | learned_route_artifact:candidate_override:router-artifact-t-20260415-257-activation-first-gating-only-v1@0.0.1 | 10 | no | d74d0a3a78c6 | Pointer-aware init keeps fast boot first with anchors memory/2026-04-04-boot-check.md, MEMORY.md, AGENTS.md, SOUL.md, +4 more, working se... |
| no_brain | eval | 0 | no | 0/3 | n/a | none | 0 | no | none | none |

## live-pelican-072a9d79-0a6d-4d33-aa9c-b4474dc2a3b3-window-002

- bundle dir: `live-pelican-072a9d79-0a6d-4d33-aa9c-b4474dc2a3b3-window-002`
- learned_route vs approved prior: `worse`
- learned_route vs no_brain floor: `better`
- diagnostic winner: `graph_prior_only`
- diagnostic top score modes: `vector_only`, `graph_prior_only`
- score spread: 100

| mode | diagnostic quality | compile ok | required-context recall | promotions | warnings |
| --- | ---: | ---: | ---: | ---: | ---: |
| graph_prior_only | 100 | 1/1 | 3/3 | 0 | 1 |
| vector_only | 100 | 1/1 | 3/3 | 0 | 1 |
| learned_route | 40 | 1/1 | 0/3 | 0 | 2 |
| no_brain | 0 | 0/1 | 0/3 | 0 | 1 |

### turn-1

- user: Exec denied (gateway id=5256aa70-d463-4b6d-9937-e7af81662d97, approval-timeout (obfuscation-detected)): python3 - <<'PY' from pathlib imp...
- expected phrases: `/pel087_core36_alert_suite.py`, `/test_core_capture_state_machine.py`, `/test_pel087_core36_alert_suite.py`
- feedback kinds: none
- learned_route vs approved prior: `worse`
- diagnostic top modes: `vector_only`, `graph_prior_only` (spread 100)

| mode | phase | diagnostic quality | compile | required-context recall | activation | source | latency ms | promoted | selection | context preview |
| --- | --- | ---: | --- | ---: | --- | --- | ---: | --- | --- | --- |
| graph_prior_only | eval | 100 | yes | 3/3 | n/a | none | 10 | no | 8fe7ae8ff582 | Teaching feedback on live-session session 072a9d79-0a6d-4d33-aa9c-b4474dc2a3b3-seed: assistant: I’ve got the state machine module in plac... |
| vector_only | eval | 100 | yes | 3/3 | n/a | none | 10 | no | a4e02fcca53d | Teaching feedback on live-session session 072a9d79-0a6d-4d33-aa9c-b4474dc2a3b3-seed: assistant: I’ve got the state machine module in plac... |
| learned_route | eval | 40 | yes | 0/3 | yes | learned_route_artifact:candidate_override:router-artifact-t-20260415-257-activation-first-gating-only-v1@0.0.1 | 10 | no | 95e2935b6b15 | Pointer-aware init keeps fast boot first with anchors memory/2026-04-09-pelican-alerts.md, MEMORY.md, AGENTS.md, SOUL.md, +4 more, workin... |
| no_brain | eval | 0 | no | 0/3 | n/a | none | 0 | no | none | none |

## live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-015

- bundle dir: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-015`
- learned_route vs approved prior: `tied`
- learned_route vs no_brain floor: `better`
- diagnostic winner: `vector_only`
- diagnostic top score modes: `vector_only`
- score spread: 100

| mode | diagnostic quality | compile ok | required-context recall | promotions | warnings |
| --- | ---: | ---: | ---: | ---: | ---: |
| vector_only | 100 | 1/1 | 1/1 | 0 | 1 |
| graph_prior_only | 40 | 1/1 | 0/1 | 0 | 1 |
| learned_route | 40 | 1/1 | 0/1 | 0 | 2 |
| no_brain | 0 | 0/1 | 0/1 | 0 | 1 |

### turn-1

- user: [sessionId: d4fe5adc-c09d-4b95-86b2-b106a2a97c64] A cron job "v5-gbm-training-monitor" just completed successfully. Result: Training is s...
- expected phrases: `/15`
- feedback kinds: none
- learned_route vs approved prior: `tied`
- diagnostic top modes: `vector_only` (spread 100)

| mode | phase | diagnostic quality | compile | required-context recall | activation | source | latency ms | promoted | selection | context preview |
| --- | --- | ---: | --- | ---: | --- | --- | ---: | --- | --- | --- |
| vector_only | eval | 100 | yes | 1/1 | n/a | none | 5 | no | f3607665091f | Teaching feedback on live-session session 09658232-8ff6-4788-a42d-e5e49e5404bb-seed: user: [sessionId: ffc1b986-38d1-494c-b5a2-3cac96543f... |
| graph_prior_only | eval | 40 | yes | 0/1 | n/a | none | 5 | no | 0ad5fcde6725 | Teaching feedback on live-session session 09658232-8ff6-4788-a42d-e5e49e5404bb-seed: user: [sessionId: ffc1b986-38d1-494c-b5a2-3cac96543f... |
| learned_route | eval | 40 | yes | 0/1 | yes | learned_route_artifact:candidate_override:router-artifact-t-20260415-257-activation-first-gating-only-v1@0.0.1 | 10 | no | d20e58c30ab0 | Pointer-aware init keeps fast boot first with anchors memory/2026-02-27-crabpath-polish.md, MEMORY.md, AGENTS.md, SOUL.md, +4 more, worki... |
| no_brain | eval | 0 | no | 0/1 | n/a | none | 0 | no | none | none |

## live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-016

- bundle dir: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-016`
- learned_route vs approved prior: `worse`
- learned_route vs no_brain floor: `better`
- diagnostic winner: `graph_prior_only`
- diagnostic top score modes: `vector_only`, `graph_prior_only`
- score spread: 100

| mode | diagnostic quality | compile ok | required-context recall | promotions | warnings |
| --- | ---: | ---: | ---: | ---: | ---: |
| graph_prior_only | 100 | 1/1 | 1/1 | 0 | 1 |
| vector_only | 100 | 1/1 | 1/1 | 0 | 1 |
| learned_route | 40 | 1/1 | 0/1 | 0 | 2 |
| no_brain | 0 | 0/1 | 0/1 | 0 | 1 |

### turn-1

- user: [sessionId: 09ac4735-5e22-4e2d-a190-4b942c71ef51] A cron job "v5-gbm-training-monitor" just completed successfully. Result: Good — HEARTB...
- expected phrases: `/15`
- feedback kinds: none
- learned_route vs approved prior: `worse`
- diagnostic top modes: `vector_only`, `graph_prior_only` (spread 100)

| mode | phase | diagnostic quality | compile | required-context recall | activation | source | latency ms | promoted | selection | context preview |
| --- | --- | ---: | --- | ---: | --- | --- | ---: | --- | --- | --- |
| graph_prior_only | eval | 100 | yes | 1/1 | n/a | none | 10 | no | 4fb3d0663611 | Teaching feedback on live-session session 09658232-8ff6-4788-a42d-e5e49e5404bb-seed: user: [sessionId: eeb5903a-1991-4471-87ba-565408d3db... |
| vector_only | eval | 100 | yes | 1/1 | n/a | none | 5 | no | 53e53e671b90 | Teaching feedback on live-session session 09658232-8ff6-4788-a42d-e5e49e5404bb-seed: user: [sessionId: eeb5903a-1991-4471-87ba-565408d3db... |
| learned_route | eval | 40 | yes | 0/1 | yes | learned_route_artifact:candidate_override:router-artifact-t-20260415-257-activation-first-gating-only-v1@0.0.1 | 10 | no | 93e75db6632f | Pointer-aware init keeps fast boot first with anchors memory/2026-02-27-crabpath-polish.md, MEMORY.md, AGENTS.md, SOUL.md, +4 more, worki... |
| no_brain | eval | 0 | no | 0/1 | n/a | none | 0 | no | none | none |
