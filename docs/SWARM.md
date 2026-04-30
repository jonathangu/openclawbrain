# OpenClawBrain Scratch Rebuild Swarm Charter

Launched: 2026-04-29 20:11 PT
Owner: GUCLAW principal orchestrator
Repo: `/Users/guclaw/.openclaw/workspace/openclawbrain`

## Prime directive

OpenClawBrain is being rebuilt from scratch. This is a clean-room rebuild. Workers must not resurrect, inspect, quote, adapt, or depend on old OpenClawBrain implementation code, backups, docs, proof bundles, media, archived plans, deleted artifacts, old package installs, or old OCB-specific memory.

The only allowed OpenClawBrain source material at launch is:

1. the current `README.md`,
2. this swarm charter,
3. Jonathan's current direction: rebuild totally from scratch; prepare a huge parallel swarm; make sure every lane knows the plan and its place,
4. live OpenClaw runtime/docs/source when needed to understand integration surfaces, as OpenClaw itself is not old OpenClawBrain.

## North star

Build a small, reliable intervention layer for OpenClaw that helps at important decision points by learning:

- when to help,
- what tiny piece of context/workflow/restraint is useful,
- when to stay quiet,
- and how to show honest evidence for the choice.

## First milestone

Define and then build the smallest dogfoodable loop that improves one real OpenClaw decision path and honestly shows whether it helped.

No implementation code should land before the first decision path, intervention contract, proof surface, and restraint behavior are crisp enough to test.

## Operating rules for workers

1. Work in parallel, but do not edit shared source files unless explicitly assigned by GUCLAW principal.
2. Do not edit `TASKS.md`; it is principal-only.
3. Default output path: `/Users/guclaw/.openclaw/workspace/task-artifacts/openclawbrain/swarm-20260429/<lane-id>.md`.
4. Write proposals, contracts, maps, test plans, and risk notes; do not implement code unless your lane explicitly says to create a prototype artifact outside the repo.
5. Keep claims bounded. Mark assumptions. Prefer observable runtime evidence over abstractions.
6. If old OCB material appears, treat it as tainted and ignore it.
7. If blocked, write the blocker and the next smallest question/action in your artifact.

## Shared shape every artifact should include

- Lane ID and mission
- Key findings / proposal
- Recommended first dogfood path impact
- Concrete interfaces or acceptance tests, if applicable
- Risks / unknowns
- Next action for GUCLAW principal

## Swarm lanes

### L01 — Decision Path Scout
Find 3-5 candidate first dogfoodable OpenClaw decision paths. Rank by usefulness, observability, implementation size, and trust risk.

### L02 — Runtime Hook Mapper
Inspect current OpenClaw docs/source for safe integration points where a small intervention layer could observe a decision, suggest an action, or stay silent. Use only OpenClaw runtime/source, not old OCB.

### L03 — Intervention Contract Designer
Define the minimal request/response contract for an intervention: inputs, outputs, confidence/restraint, evidence, and reversibility.

### L04 — Restraint / STOP_LOCAL Designer
Make restraint first-class. Define when the brain should suppress itself, decline to retrieve, or advise no action.

### L05 — Proof Surface Designer
Design the operator-facing evidence surface: what to log/show so Jon can trust why a choice was made without leaking secrets or creating noise.

### L06 — Data Model Minimalist
Propose the smallest durable storage model needed for the first loop: observations, interventions, outcomes, and feedback.

### L07 — Evaluation Harness Designer
Define offline/online tests proving the first loop helped, did nothing, or harmed. Include counterfactual and regression checks.

### L08 — Install / Dogfood UX
Design install/status/proof commands or surfaces that make the rebuild boringly reliable to dogfood from day one.

### L09 — Safety / Privacy / Secrets
Threat-model the first loop for secret leakage, overreach, noisy interruption, stale memory, and unsafe autonomy.

### L10 — Product Narrative / Claims
Write the bounded public/internal claim language for the scratch rebuild: what it is, what it is not, and evidence required before claiming success.

### L11 — Architecture Skeleton
Propose a minimal clean-room repo structure and modules for the first loop, without old implementation inheritance.

### L12 — OpenClaw Config Surface
Inspect OpenClaw config/docs/schema paths relevant to extensions/hooks/status only; propose how OCB should be configured safely.

### L13 — Memory / Context Policy
Define what kinds of context OCB may learn, retain, forget, or refuse to use in the first milestone.

### L14 — Workflow Reuse Lane
Define how OCB can learn tiny workflow choices (tool/agent/path choice) without becoming a broad generic planner.

### L15 — Operator Feedback Loop
Design the smallest feedback mechanism: thumbs up/down, explicit correction, outcome capture, and how that changes future behavior.

### L16 — Failure Modes / Kill Switch
Define kill switches, rollback behavior, degradation mode, and how to prove OCB is inactive when disabled.

### L17 — First Build Plan
Turn the best likely pieces into a proposed day-one implementation sequence with acceptance gates.

### L18 — Synthesis Prep
Prepare a synthesis rubric for combining lane outputs into one build plan without averaging everything into mush.
