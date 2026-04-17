# Worked Traces

- traces included: 8/25
- selection rule: highest bundle score spread first, then trace id; turns ordered by per-turn score spread
- note: qualityScore and winnerMode are internal deterministic replay diagnostics, not the public/operator scorecard.
- source manifest: `felt_resume_25-eval` (frozen_recorded_session_eval_manifest.v1, 0c68fa167a58)
- omitted traces: 17 (see _lane/summary-tables.json for the complete table)

## live-main-971973d8-2a63-4883-a18f-bfa883f844ea-window-003

- bundle dir: `live-main-971973d8-2a63-4883-a18f-bfa883f844ea-window-003`
- learned_route vs approved prior: `worse`
- learned_route vs no_brain floor: `better`
- diagnostic winner: `graph_prior_only`
- diagnostic top score modes: `vector_only`, `graph_prior_only`
- score spread: 70

| mode | diagnostic quality | compile ok | required-context recall | promotions | warnings |
| --- | ---: | ---: | ---: | ---: | ---: |
| graph_prior_only | 70 | 1/1 | 1/2 | 0 | 1 |
| vector_only | 70 | 1/1 | 1/2 | 0 | 1 |
| learned_route | 40 | 1/1 | 0/2 | 0 | 2 |
| no_brain | 0 | 0/1 | 0/2 | 0 | 1 |

### turn-1

- user: Continue where you left off, but narrow scope hard. Produce a usable Lane A result even if code changes are too risky. Minimum required o...
- expected phrases: `/Users/guclaw/.openclaw/workspace/openclawbrain/package.json"}`, `/Users/guclaw/.openclaw/workspace/openclawbrain`
- feedback kinds: none
- learned_route vs approved prior: `worse`
- diagnostic top modes: `vector_only`, `graph_prior_only` (spread 70)

| mode | phase | diagnostic quality | compile | required-context recall | activation | source | latency ms | promoted | selection | context preview |
| --- | --- | ---: | --- | ---: | --- | --- | ---: | --- | --- | --- |
| graph_prior_only | eval | 70 | yes | 1/2 | n/a | none | 10 | no | be417c92c79e | Teaching feedback on live-session session 971973d8-2a63-4883-a18f-bfa883f844ea-seed: user: You are running as a subagent (depth 1/1). Res... |
| vector_only | eval | 70 | yes | 1/2 | n/a | none | 10 | no | 142e00b1561a | Teaching feedback on live-session session 971973d8-2a63-4883-a18f-bfa883f844ea-seed: user: You are running as a subagent (depth 1/1). Res... |
| learned_route | eval | 40 | yes | 0/2 | yes | learned_route_artifact:candidate_override:router-artifact-t-20260415-257-activation-first-gating-only-v1@0.0.1 | 10 | no | fc8b1ded2438 | Pointer-aware init keeps fast boot first with anchors memory/2026-04-06-boot-check.md, MEMORY.md, AGENTS.md, SOUL.md, +4 more, working se... |
| no_brain | eval | 0 | no | 0/2 | n/a | none | 0 | no | none | none |

## live-main-4c69091d-1290-4bcd-a74c-7166c46e5670-window-002

- bundle dir: `live-main-4c69091d-1290-4bcd-a74c-7166c46e5670-window-002`
- learned_route vs approved prior: `worse`
- learned_route vs no_brain floor: `better`
- diagnostic winner: `graph_prior_only`
- diagnostic top score modes: `vector_only`, `graph_prior_only`
- score spread: 60

| mode | diagnostic quality | compile ok | required-context recall | promotions | warnings |
| --- | ---: | ---: | ---: | ---: | ---: |
| graph_prior_only | 60 | 1/1 | 1/3 | 0 | 1 |
| vector_only | 60 | 1/1 | 1/3 | 0 | 1 |
| learned_route | 40 | 1/1 | 0/3 | 0 | 2 |
| no_brain | 0 | 0/1 | 0/3 | 0 | 1 |

### turn-1

- user: Continue where you left off, but narrow scope hard. Produce a usable Lane B result even if code changes are too risky. Minimum required o...
- expected phrases: `T-20260406-161`, `/Users/guclaw/.openclaw/workspace/task-artifacts/T-20260406-161/lane-b-online-pg.md`, `/Users/guclaw/.openclaw/workspace/task-status/T-20260406-161/lane-b.json.`
- feedback kinds: none
- learned_route vs approved prior: `worse`
- diagnostic top modes: `vector_only`, `graph_prior_only` (spread 60)

| mode | phase | diagnostic quality | compile | required-context recall | activation | source | latency ms | promoted | selection | context preview |
| --- | --- | ---: | --- | ---: | --- | --- | ---: | --- | --- | --- |
| graph_prior_only | eval | 60 | yes | 1/3 | n/a | none | 10 | no | 12e6a730264f | Teaching feedback on live-session session 4c69091d-1290-4bcd-a74c-7166c46e5670-seed: user: You are running as a subagent (depth 1/1). Res... |
| vector_only | eval | 60 | yes | 1/3 | n/a | none | 10 | no | 48538b14f947 | Teaching feedback on live-session session 4c69091d-1290-4bcd-a74c-7166c46e5670-seed: user: You are running as a subagent (depth 1/1). Res... |
| learned_route | eval | 40 | yes | 0/3 | yes | learned_route_artifact:candidate_override:router-artifact-t-20260415-257-activation-first-gating-only-v1@0.0.1 | 10 | no | c0b6ef13bc38 | Pointer-aware init keeps fast boot first with anchors memory/2026-04-06-boot-check.md, MEMORY.md, AGENTS.md, SOUL.md, +4 more, working se... |
| no_brain | eval | 0 | no | 0/3 | n/a | none | 0 | no | none | none |

## live-main-685b2c1a-b082-4f5a-a284-ff9623440da6-window-002

- bundle dir: `live-main-685b2c1a-b082-4f5a-a284-ff9623440da6-window-002`
- learned_route vs approved prior: `worse`
- learned_route vs no_brain floor: `better`
- diagnostic winner: `graph_prior_only`
- diagnostic top score modes: `vector_only`, `graph_prior_only`
- score spread: 60

| mode | diagnostic quality | compile ok | required-context recall | promotions | warnings |
| --- | ---: | ---: | ---: | ---: | ---: |
| graph_prior_only | 60 | 1/1 | 1/3 | 0 | 1 |
| vector_only | 60 | 1/1 | 1/3 | 0 | 1 |
| learned_route | 40 | 1/1 | 0/3 | 0 | 2 |
| no_brain | 0 | 0/1 | 0/3 | 0 | 1 |

### turn-1

- user: Continue where you left off. The previous model attempt failed or timed out.
- expected phrases: `/Users/guclaw/.openclaw/workspace`, `/Users/guclaw/.openclaw/workspace/openclawbrain`, `/dev/null`
- feedback kinds: none
- learned_route vs approved prior: `worse`
- diagnostic top modes: `vector_only`, `graph_prior_only` (spread 60)

| mode | phase | diagnostic quality | compile | required-context recall | activation | source | latency ms | promoted | selection | context preview |
| --- | --- | ---: | --- | ---: | --- | --- | ---: | --- | --- | --- |
| graph_prior_only | eval | 60 | yes | 1/3 | n/a | none | 10 | no | 0d4c7b0f3298 | Teaching feedback on live-session session 685b2c1a-b082-4f5a-a284-ff9623440da6-seed: user: You are running as a subagent (depth 1/1). Res... |
| vector_only | eval | 60 | yes | 1/3 | n/a | none | 10 | no | bd157876ec02 | Teaching feedback on live-session session 685b2c1a-b082-4f5a-a284-ff9623440da6-seed: user: You are running as a subagent (depth 1/1). Res... |
| learned_route | eval | 40 | yes | 0/3 | yes | learned_route_artifact:candidate_override:router-artifact-t-20260415-257-activation-first-gating-only-v1@0.0.1 | 10 | no | dec439adfedb | Pointer-aware init keeps fast boot first with anchors MEMORY.md, AGENTS.md, SOUL.md, USER.md, +3 more, working set TASKS.md, AGENTS_ACTIV... |
| no_brain | eval | 0 | no | 0/3 | n/a | none | 0 | no | none | none |

## live-main-983f0a77-69b8-40b2-922b-c7dc44d4c7e9-window-007

- bundle dir: `live-main-983f0a77-69b8-40b2-922b-c7dc44d4c7e9-window-007`
- learned_route vs approved prior: `worse`
- learned_route vs no_brain floor: `better`
- diagnostic winner: `graph_prior_only`
- diagnostic top score modes: `graph_prior_only`
- score spread: 60

| mode | diagnostic quality | compile ok | required-context recall | promotions | warnings |
| --- | ---: | ---: | ---: | ---: | ---: |
| graph_prior_only | 60 | 1/1 | 1/3 | 0 | 1 |
| learned_route | 40 | 1/1 | 0/3 | 0 | 2 |
| vector_only | 40 | 1/1 | 0/3 | 0 | 1 |
| no_brain | 0 | 0/1 | 0/3 | 0 | 1 |

### turn-1

- user: Continue where you left off. The previous model attempt failed or timed out.
- expected phrases: `T-20260407-173`, `/Users/guclaw/.openclaw/workspace/HEARTBEAT.md"}`, `/Users/guclaw/.openclaw/workspace/task-artifacts/T-20260407-173/master-plan.md"}`
- feedback kinds: none
- learned_route vs approved prior: `worse`
- diagnostic top modes: `graph_prior_only` (spread 60)

| mode | phase | diagnostic quality | compile | required-context recall | activation | source | latency ms | promoted | selection | context preview |
| --- | --- | ---: | --- | ---: | --- | --- | ---: | --- | --- | --- |
| graph_prior_only | eval | 60 | yes | 1/3 | n/a | none | 10 | no | 29a7f2eeb5d6 | Interaction operator_override on live-session session 983f0a77-69b8-40b2-922b-c7dc44d4c7e9-seed. Message: cue-2-seed-message. || Interact... |
| learned_route | eval | 40 | yes | 0/3 | yes | learned_route_artifact:candidate_override:router-artifact-t-20260415-257-activation-first-gating-only-v1@0.0.1 | 10 | no | 628b41627c86 | Pointer-aware init keeps fast boot first with anchors memory/2026-04-07-boot-check.md, MEMORY.md, AGENTS.md, SOUL.md, +4 more, working se... |
| vector_only | eval | 40 | yes | 0/3 | n/a | none | 10 | no | 9b8ebeb1e50f | Interaction operator_override on live-session session 983f0a77-69b8-40b2-922b-c7dc44d4c7e9-seed. Message: cue-2-seed-message. || Interact... |
| no_brain | eval | 0 | no | 0/3 | n/a | none | 0 | no | none | none |

## live-bountiful-55dd01ce-c43c-4b90-a4c6-c2fa97115709-window-002

- bundle dir: `live-bountiful-55dd01ce-c43c-4b90-a4c6-c2fa97115709-window-002`
- learned_route vs approved prior: `tied`
- learned_route vs no_brain floor: `better`
- diagnostic winner: `graph_prior_only`
- diagnostic top score modes: `vector_only`, `graph_prior_only`, `learned_route`
- score spread: 40

| mode | diagnostic quality | compile ok | required-context recall | promotions | warnings |
| --- | ---: | ---: | ---: | ---: | ---: |
| graph_prior_only | 40 | 1/1 | 0/3 | 0 | 1 |
| learned_route | 40 | 1/1 | 0/3 | 0 | 2 |
| vector_only | 40 | 1/1 | 0/3 | 0 | 1 |
| no_brain | 0 | 0/1 | 0/3 | 0 | 1 |

### turn-1

- user: Conversation info (untrusted metadata): ```json { "message_id": "7508", "sender_id": "8518484672", "conversation_label": "BOUNTIFUL id:-5...
- expected phrases: `keep going finish job`, `feedback keep going finish`, `going finish job premature`
- feedback kinds: none
- learned_route vs approved prior: `tied`
- diagnostic top modes: `vector_only`, `graph_prior_only`, `learned_route` (spread 40)

| mode | phase | diagnostic quality | compile | required-context recall | activation | source | latency ms | promoted | selection | context preview |
| --- | --- | ---: | --- | ---: | --- | --- | ---: | --- | --- | --- |
| graph_prior_only | eval | 40 | yes | 0/3 | n/a | none | 10 | no | 7f7dc36bb35e | Pointer-aware init keeps fast boot first with anchors MEMORY.md, AGENTS.md, SOUL.md, USER.md, +3 more, working set TASKS.md, task-artifac... |
| learned_route | eval | 40 | yes | 0/3 | yes | learned_route_artifact:candidate_override:router-artifact-t-20260415-257-activation-first-gating-only-v1@0.0.1 | 10 | no | 5d4ec2162352 | Pointer-aware init keeps fast boot first with anchors MEMORY.md, AGENTS.md, SOUL.md, USER.md, +3 more, working set TASKS.md, task-artifac... |
| vector_only | eval | 40 | yes | 0/3 | n/a | none | 10 | no | cc91b8bf9114 | Pointer-aware init keeps fast boot first with anchors MEMORY.md, AGENTS.md, SOUL.md, USER.md, +3 more, working set TASKS.md, task-artifac... |
| no_brain | eval | 0 | no | 0/3 | n/a | none | 0 | no | none | none |

## live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-002

- bundle dir: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-002`
- learned_route vs approved prior: `tied`
- learned_route vs no_brain floor: `better`
- diagnostic winner: `graph_prior_only`
- diagnostic top score modes: `vector_only`, `graph_prior_only`, `learned_route`
- score spread: 40

| mode | diagnostic quality | compile ok | required-context recall | promotions | warnings |
| --- | ---: | ---: | ---: | ---: | ---: |
| graph_prior_only | 40 | 1/1 | 0/3 | 0 | 1 |
| learned_route | 40 | 1/1 | 0/3 | 0 | 2 |
| vector_only | 40 | 1/1 | 0/3 | 0 | 1 |
| no_brain | 0 | 0/1 | 0/3 | 0 | 1 |

### turn-1

- user: You are running a boot check. Follow BOOT.md instructions exactly. BOOT.md: # BOOT.md — Bountiful startup 1. Read `TASKS.md`. 2. Inspect ...
- expected phrases: `/process/session`, `/write/edit`, `/<task-id>/<worker-id>.md`
- feedback kinds: none
- learned_route vs approved prior: `tied`
- diagnostic top modes: `vector_only`, `graph_prior_only`, `learned_route` (spread 40)

| mode | phase | diagnostic quality | compile | required-context recall | activation | source | latency ms | promoted | selection | context preview |
| --- | --- | ---: | --- | ---: | --- | --- | ---: | --- | --- | --- |
| graph_prior_only | eval | 40 | yes | 0/3 | n/a | none | 5 | no | 80852cd6a422 | Teaching feedback on live-session session 8578588b-f6c3-4605-abef-80a728fb6bf3-seed: user: A new session was started via /new or /reset. ... |
| learned_route | eval | 40 | yes | 0/3 | yes | learned_route_artifact:candidate_override:router-artifact-t-20260415-257-activation-first-gating-only-v1@0.0.1 | 5 | no | 3cd7e418c4db | Interaction operator_override on live-session session 8578588b-f6c3-4605-abef-80a728fb6bf3-seed. Message: cue-2-seed-message. |
| vector_only | eval | 40 | yes | 0/3 | n/a | none | 5 | no | 70d36a02874e | Teaching feedback on live-session session 8578588b-f6c3-4605-abef-80a728fb6bf3-seed: user: A new session was started via /new or /reset. ... |
| no_brain | eval | 0 | no | 0/3 | n/a | none | 0 | no | none | none |

## live-main-1f25d4e1-770f-4106-a3d1-14910d8fde3d-window-002

- bundle dir: `live-main-1f25d4e1-770f-4106-a3d1-14910d8fde3d-window-002`
- learned_route vs approved prior: `tied`
- learned_route vs no_brain floor: `better`
- diagnostic winner: `graph_prior_only`
- diagnostic top score modes: `vector_only`, `graph_prior_only`, `learned_route`
- score spread: 40

| mode | diagnostic quality | compile ok | required-context recall | promotions | warnings |
| --- | ---: | ---: | ---: | ---: | ---: |
| graph_prior_only | 40 | 1/1 | 0/2 | 0 | 1 |
| learned_route | 40 | 1/1 | 0/2 | 0 | 2 |
| vector_only | 40 | 1/1 | 0/2 | 0 | 1 |
| no_brain | 0 | 0/1 | 0/2 | 0 | 1 |

### turn-1

- user: Continue where you left off. The previous model attempt failed or timed out.
- expected phrases: `/Users/guclaw/.openclaw/workspace/openclawbrain/src/brain-store/store.ts"}`, `/Users/guclaw/.openclaw/workspace/openclawbrain/src/brain-worker/worker.ts"}`
- feedback kinds: none
- learned_route vs approved prior: `tied`
- diagnostic top modes: `vector_only`, `graph_prior_only`, `learned_route` (spread 40)

| mode | phase | diagnostic quality | compile | required-context recall | activation | source | latency ms | promoted | selection | context preview |
| --- | --- | ---: | --- | ---: | --- | --- | ---: | --- | --- | --- |
| graph_prior_only | eval | 40 | yes | 0/2 | n/a | none | 10 | no | 79427a3e6077 | Pointer-aware init keeps fast boot first with anchors MEMORY.md, AGENTS.md, SOUL.md, USER.md, +3 more, working set TASKS.md, AGENTS_ACTIV... |
| learned_route | eval | 40 | yes | 0/2 | yes | learned_route_artifact:candidate_override:router-artifact-t-20260415-257-activation-first-gating-only-v1@0.0.1 | 10 | no | 6fe6754860a9 | Pointer-aware init keeps fast boot first with anchors MEMORY.md, AGENTS.md, SOUL.md, USER.md, +3 more, working set TASKS.md, AGENTS_ACTIV... |
| vector_only | eval | 40 | yes | 0/2 | n/a | none | 10 | no | 4b233d7b56a5 | Pointer-aware init keeps fast boot first with anchors MEMORY.md, AGENTS.md, SOUL.md, USER.md, +3 more, working set TASKS.md, AGENTS_ACTIV... |
| no_brain | eval | 0 | no | 0/2 | n/a | none | 0 | no | none | none |

## live-main-2b388c4b-24bf-4e37-b956-c1907568c6ad-window-002

- bundle dir: `live-main-2b388c4b-24bf-4e37-b956-c1907568c6ad-window-002`
- learned_route vs approved prior: `tied`
- learned_route vs no_brain floor: `better`
- diagnostic winner: `graph_prior_only`
- diagnostic top score modes: `vector_only`, `graph_prior_only`, `learned_route`
- score spread: 40

| mode | diagnostic quality | compile ok | required-context recall | promotions | warnings |
| --- | ---: | ---: | ---: | ---: | ---: |
| graph_prior_only | 40 | 1/1 | 0/2 | 0 | 1 |
| learned_route | 40 | 1/1 | 0/2 | 0 | 2 |
| vector_only | 40 | 1/1 | 0/2 | 0 | 1 |
| no_brain | 0 | 0/1 | 0/2 | 0 | 1 |

### turn-1

- user: Continue where you left off. The previous model attempt failed or timed out.
- expected phrases: `/src`, `route_fn`
- feedback kinds: none
- learned_route vs approved prior: `tied`
- diagnostic top modes: `vector_only`, `graph_prior_only`, `learned_route` (spread 40)

| mode | phase | diagnostic quality | compile | required-context recall | activation | source | latency ms | promoted | selection | context preview |
| --- | --- | ---: | --- | ---: | --- | --- | ---: | --- | --- | --- |
| graph_prior_only | eval | 40 | yes | 0/2 | n/a | none | 10 | no | 11c671b4a6a1 | Teaching feedback on live-session session 2b388c4b-24bf-4e37-b956-c1907568c6ad-seed: user: You are running as a subagent (depth 1/1). Res... |
| learned_route | eval | 40 | yes | 0/2 | yes | learned_route_artifact:candidate_override:router-artifact-t-20260415-257-activation-first-gating-only-v1@0.0.1 | 10 | no | 780aab512e87 | Pointer-aware init keeps fast boot first with anchors MEMORY.md, AGENTS.md, SOUL.md, USER.md, +3 more, working set TASKS.md, AGENTS_ACTIV... |
| vector_only | eval | 40 | yes | 0/2 | n/a | none | 10 | no | 7b2341d5a003 | Teaching feedback on live-session session 2b388c4b-24bf-4e37-b956-c1907568c6ad-seed: user: You are running as a subagent (depth 1/1). Res... |
| no_brain | eval | 0 | no | 0/2 | n/a | none | 0 | no | none | none |
