# OpenClawBrain 1.0 Execution Plan V5 — Agent-Finishable Evidence Program

**Date:** 2026-04-28  
**Status:** Agent execution contract + evidence plan  
**Purpose:** Make OpenClawBrain’s scoreboard pipeline finishable by an autonomous OpenClaw/Codex 5.5 agent without allowing fake evidence, architecture drift, or hand-edited results.

---

## 0. The hard truth

OpenClawBrain is not blocked on more runtime sophistication.

It is blocked on evidence that memory activation improves user-visible outcomes more than it harms, distracts, or costs.

The next phase must answer two questions:

1. **When OpenClawBrain fires, does the user get a better result?**
2. **When OpenClawBrain stays silent, was that restraint correct, or did it miss useful context?**

The project’s next deliverable is not a richer memory runtime.

It is a scoreboard that can honestly answer whether OpenClawBrain should become:

- full selective-context engine
- correction-sticky product
- correction+heuristics product
- hybrid default + slice-gated full OCB
- runtime health/verification layer
- or paused until better traces exist

---

## 1. Final operating principle

> **Build the scoreboard before building the cathedral.**

But the scoreboard must be agent-proof.

An autonomous coding agent must be able to finish the engineering pipeline end to end, while being forced to mark real evidence as incomplete unless real traces and real judging exist.

Operational rule:

> **The agent must always finish the pipeline, but it must never fake the evidence.**

---

## 2. Two completion gates

This plan separates engineering completion from evidence completion.

This is the most important agent-safety distinction.

## 2.1 Engineering E2E complete

Engineering E2E is complete when the full machinery works, even on smoke data.

Engineering E2E is complete when:

- `docs/results/*` contracts exist
- ledger schema validates rows
- trace manifest validates coverage and provenance
- ablation harness can run all four backends
- backend outputs are saved uniformly
- blind judge packets are generated
- judged ledger can be ingested
- `/results` is generated from ledger data
- product decision memo is generated from thresholds
- `RUN_STATE.json` is written
- tests pass
- `pnpm ocb:e2e:smoke` completes end to end

Engineering E2E may use synthetic or fixture traces.

But smoke data must be clearly labeled:

```text
NOT PRODUCT EVIDENCE
SYNTHETIC PIPELINE VALIDATION ONLY
```

## 2.2 Evidence E2E complete

Evidence E2E is complete only when real evidence exists.

Evidence E2E requires:

- 40 admitted real redacted traces
- required slice counts satisfied
- provenance and privacy metadata present
- all four backends run against all admitted traces
- blind/labeled judging complete
- `ledger-judged.jsonl` exists
- `/results` regenerated from the judged ledger
- thresholds applied
- `docs/results/30_DAY_DECISION.md` produced

Synthetic, fixture, repo-derived, or smoke traces cannot count as product proof unless admitted under trace rules.

---

## 3. Agent non-fabrication rule

The agent must not invent:

- real user traces
- provenance
- privacy approval
- judge scores
- cost measurements
- model IDs
- memory snapshots
- backend wins
- product evidence
- final product decisions from nonexistent data

If required evidence is unavailable, the agent must produce blocker artifacts and mark:

```json
{
  "engineering_e2e_complete": true,
  "evidence_e2e_complete": false
}
```

The correct incomplete state is not failure. It is honest state.

---

## 4. Agent stop conditions

The agent must stop OpenClawBrain runtime architecture work if any of these are incomplete:

- ledger schema
- trace validation
- ablation harness
- blind judge packet generation
- judged ledger import
- results generation
- threshold decision logic
- smoke E2E command

The agent must not build new OpenClawBrain runtime architecture until the scoreboard exists.

Allowed before the scoreboard:

- docs/results contracts
- results schema package
- trace validator
- smoke traces
- eval harness
- backend adapters
- blind judging flow
- results generator
- decision generator
- run-state tracking
- tests

Not allowed before the scoreboard:

- new learned-route architecture
- new Graphify/teacher/compiler lanes
- new proof/status surfaces unrelated to outcome utility
- threshold tuning on current eval
- marketing copy around unproven product claims

---

## 5. Canonical commands

The agent must create these commands or map equivalent commands in `docs/results/COMMANDS.md`.

Preferred commands:

```bash
pnpm ocb:results:schema-test
pnpm ocb:traces:validate
pnpm ocb:eval:run
pnpm ocb:eval:make-blind-packets
pnpm ocb:judgments:import
pnpm ocb:ledger:validate
pnpm ocb:results:generate
pnpm ocb:decision:generate
pnpm ocb:e2e:smoke
```

If the repo cannot support exactly these names, the agent must document equivalents in:

```text
docs/results/COMMANDS.md
```

Each command must be non-interactive and reproducible.

---

## 6. Smoke mode

Smoke mode must be implemented first.

Smoke mode uses 4–8 explicitly synthetic traces to prove the pipeline works.

Smoke traces must include:

```json
{
  "provenance_type": "synthetic",
  "mode": "smoke",
  "counts_as_product_evidence": false
}
```

Smoke mode must output:

```text
eval/results/<run-id>/ledger-draft.jsonl
eval/results/<run-id>/blind-judge-packets/
eval/results/<run-id>/ledger-judged.synthetic.jsonl
docs/results/index.md
docs/results/summary.json
docs/results/30_DAY_DECISION.synthetic.md
eval/results/<run-id>/RUN_STATE.json
```

Smoke `/results` and synthetic decision memo must clearly display:

```text
NOT PRODUCT EVIDENCE
SYNTHETIC PIPELINE VALIDATION ONLY
```

Smoke mode can set:

```json
{
  "engineering_e2e_complete": true,
  "evidence_e2e_complete": false
}
```

---

## 7. Production mode

Production mode requires the full admitted trace manifest.

Production mode must fail closed if:

- fewer than 40 admitted traces exist
- slice minimums are not met
- provenance metadata is missing
- `privacy_scrubbed=false` without explicit approval
- tool-heavy traces lack fixtures/read-only mode
- reproducibility metadata is missing
- raw cost fields are missing without `cost_measurement_mode`
- memory snapshot metadata is missing
- blind packets are missing
- judge scores are missing
- judge disagreement threshold is exceeded

Production mode may generate blocker artifacts, but it must not claim product proof.

---

## 8. Required run state

Every run must write:

```text
eval/results/<run-id>/RUN_STATE.json
```

Schema:

```json
{
  "run_id": "...",
  "mode": "smoke|production",
  "engineering_e2e_complete": false,
  "evidence_e2e_complete": false,
  "trace_count": 0,
  "real_trace_count": 0,
  "synthetic_trace_count": 0,
  "all_slice_minimums_met": false,
  "all_backends_run": false,
  "blind_packets_generated": false,
  "judging_complete": false,
  "ledger_valid": false,
  "results_generated": false,
  "decision_generated": false,
  "judge_disagreement_within_threshold": false,
  "blockers": []
}
```

Run state prevents fake completion and enables resume.

---

## 9. Required blocker artifacts

If the agent cannot complete production E2E, it must still produce:

```text
docs/results/BLOCKERS.md
docs/results/NEXT_DATA_NEEDED.md
docs/results/PARTIAL_COMPLETION.md
eval/results/<run-id>/RUN_STATE.json
```

## 9.1 `BLOCKERS.md`

Must list any missing:

- real traces
- slice coverage
- provenance
- privacy approval
- judge scores
- backend adapters
- cost metadata
- memory snapshot metadata
- prompt/model/config hashes
- failed tests
- unsafe tool traces
- threshold conflicts

## 9.2 `NEXT_DATA_NEEDED.md`

Must list exact inputs required to move from Engineering E2E to Evidence E2E:

- number of real traces needed by slice
- required provenance fields
- judging packets awaiting review
- cost metadata gaps
- memory snapshot metadata gaps
- tool fixture gaps

## 9.3 `PARTIAL_COMPLETION.md`

Must state:

- which PR phases are done
- which commands pass
- which commands fail
- current run id
- current `RUN_STATE` summary
- whether outputs are smoke-only or production evidence

---

# Part I — Evaluation mechanics

## 10. Core metrics

The scoring model has clean non-overlapping components.

```text
activation_utility = quality_delta - harm_delta - cost_penalty
```

All-trace product metric:

```text
net_task_utility =
  activation_utility                         if memory_fired
  - abstention_regret_penalty                if memory_did_not_fire_but_should_have
  0                                          if correct_abstention
```

Default:

```text
abstention_regret_penalty = abstention_regret * 0.5
```

Report both:

1. `activation_utility_when_fired`
2. `net_task_utility_all_traces`

Product decisions use **net task utility**, not fire-conditioned activation utility alone.

---

## 11. Quality delta

`quality_delta` captures user-visible answer improvement only.

It must not include memory harm or cost.

Raw quality subscores:

| Field | Meaning | Range |
|---|---|---:|
| correctness_delta | Did factual/procedural correctness improve? | -2..+2 |
| usefulness_delta | Did practical usefulness improve? | -2..+2 |
| specificity_delta | Did relevant prior context make the answer less generic? | -1..+1 |

```text
raw_quality_delta = correctness_delta + usefulness_delta + specificity_delta
```

Normalize:

```text
raw_quality_delta <= -4  -> -2
raw_quality_delta -3..-2 -> -1
raw_quality_delta -1..+1 ->  0
raw_quality_delta +2..+3 -> +1
raw_quality_delta >= +4  -> +2
```

Ledger stores both raw and normalized values.

---

## 12. Harm delta

`harm_delta` is memory-related harm only.

| Score | Meaning |
|---:|---|
| 0 | no memory-related harm |
| 1 | mild distraction / unnecessary context |
| 2 | wrong, stale, misleading, or confusing context |
| 3 | serious unsafe, creepy, or trust-breaking memory use |

Do not use `harm_delta` for general answer quality.

---

## 13. Cost penalty

`cost_penalty` is measured or pre-bucketed cost only.

| Score | Meaning |
|---:|---|
| 0 | negligible |
| 0.25 | small |
| 0.5 | noticeable |
| 1 | high relative to gain |

Cost penalty must be backed by raw cost fields:

- `input_tokens`
- `output_tokens`
- `memory_tokens`
- `latency_ms`
- `estimated_cost_usd`
- `cost_measurement_mode`

---

## 14. Net task utility

```text
if memory_fired:
  net_task_utility = activation_utility
elif should_have_fired:
  net_task_utility = -1 * abstention_regret * 0.5
else:
  net_task_utility = 0
```

This compares conservative and aggressive systems fairly.

---

## 15. Abstention regret

Applies only when `memory_fired=false`.

| Score | Meaning |
|---:|---|
| 0 | correct abstention / no useful memory available |
| 1 | small missed improvement |
| 2 | clear missed improvement |
| 3 | major missed correction/preference/workflow |

---

## 16. False fire

A false fire occurs when:

```text
memory_fired = true
should_have_fired = false
```

False-fire severity is captured by `harm_delta` and judge notes.

---

## 17. Authority precedence rules

Stale-memory-conflict judging must use fixed authority order:

1. Current user instruction in the active task
2. Newer explicit correction
3. User-approved stable preference
4. Current trusted external/source evidence
5. Older memory
6. Inferred preference

If lower-authority memory beats higher-authority current evidence or correction, score harm.

---

## 18. Priority slices

Primary priority slices:

1. Correction-follow-up
2. Continuation
3. Stale-memory-conflict

Secondary slices:

4. Retrieval-heavy
5. Tool-heavy
6. Direct-answer

Product thresholds use primary priority slices unless explicitly stated.

---

## 19. Baseline definitions

## 19.1 `none`

`none` means:

- no OpenClawBrain memory activation
- normal in-session context remains available
- normal user-provided prompt context remains available
- normal tool availability remains available
- no extra hidden memory/context injection

`none` must not be crippled.

## 19.2 `correction-only`

Eligible:

- explicit user correction
- explicit stable preference
- explicit instruction to remember
- explicit override of prior behavior

Not eligible:

- inferred preference
- weak semantic association
- graph neighbor
- teacher guess
- workflow hint unless explicitly corrected/preferred by user

## 19.3 `correction+heuristics`

Eligible:

- all correction-only items
- deterministic recency/authority rules
- exact project/person/task match
- stale-memory avoidance
- conservative STOP behavior
- small approved workflow-hint whitelist

## 19.4 `full-ocb`

Eligible:

- current full learned route path
- graph/context pack selection
- workflow/tool hints
- STOP_LOCAL
- attribution

---

## 20. Trace set requirements

Minimum production trace set: **40 real redacted traces**.

| Slice | Minimum traces |
|---|---:|
| Direct-answer | 6 |
| Continuation | 6 |
| Correction-follow-up | 8 |
| Retrieval-heavy | 6 |
| Tool-heavy | 6 |
| Stale-memory-conflict | 8 |
| **Total** | **40** |

Admission criteria:

- provenance metadata exists
- privacy-scrubbed or explicit approval exists
- user-visible task is identifiable
- slice label assigned before backend runs
- expected memory opportunity/non-opportunity labeled before backend runs
- not authored purely to favor OCB

Anti-gaming constraints:

- at least 60% real user/agent behavior not authored by OCB team
- at least 25% ambiguity/correction/contradiction/partial failure
- no trace admitted without provenance
- do not over-sample repo/design-doc-adjacent traces
- do not exclude ugly traces because OCB may fail them

---

## 21. Tool-heavy trace safety

Tool-heavy traces must be fixture-backed or read-only.

No eval run may:

- send email
- change calendars
- modify repos
- charge money
- mutate external state
- post messages
- delete files
- write to production systems

Allowed:

- mocked tool fixtures
- recorded/replayed tool outputs
- local read-only inspection
- synthetic tool transcripts

---

## 22. Blind judging

Blind randomized judging is non-negotiable where feasible.

For each trace:

1. run all four backends
2. remove backend labels from outputs
3. randomize output order per trace
4. present as Output A/B/C/D
5. judges score quality without backend identity
6. labeled audit scores memory harm/firing behavior
7. backend identity revealed only during summary generation

Judging modes:

- `blind_quality`
- `labeled_harm_audit`
- `cost_audit`

---

## 23. Judge disagreement threshold

Pause or re-judge if:

> judges disagree on utility sign — positive / neutral / negative — in more than **30%** of judged backend outputs.

Also report:

- mean absolute utility disagreement
- per-slice disagreement rate
- examples of high-disagreement traces

---

## 24. Required metrics by slice

Every backend must report:

- sample count
- mean/median activation utility when fired
- mean/median net task utility across all traces
- positive net utility rate
- negative net utility rate
- fire rate
- false-fire rate
- false-fire harm mean
- abstention regret mean
- stale-memory-conflict harm rate
- cost penalty mean
- cost per utility point
- judge disagreement rate
- bootstrap/confidence interval when possible

No single headline without per-slice tables.

---

# Part II — Ablation and thresholds

## 25. Ablation ladder

Run same traces through:

1. `none`
2. `correction-only`
3. `correction+heuristics`
4. `full-ocb`

All backends use the same:

- trace input
- model id unless explicitly testing model variation
- prompt harness
- memory snapshot where applicable
- tool fixtures
- evaluation harness commit

---

## 26. Product thresholds

## 26.1 Full OCB remains flagship only if all are true

1. Full OCB beats correction-only by at least **25% mean net task utility** across primary priority slices.
2. Correction-only captures **<75%** of full OCB’s mean net task utility across primary priority slices.
3. Full OCB wins in at least **2 of 3 primary priority slices**.
4. Full OCB does not increase false-fire harm by more than **5 percentage points** vs correction-only.
5. Full OCB has positive mean net task utility in stale-memory-conflict tasks.
6. Full OCB does not regress correction-follow-up net task utility.
7. Full OCB’s cost per utility point is not worse than correction-only by more than **25%** without corresponding gain.

## 26.2 Correction-only becomes default if any are true

1. Correction-only captures **>=75%** of full OCB’s mean net task utility across primary priority slices.
2. Correction-only wins or ties full OCB in correction-follow-up and stale-memory-conflict.
3. Full OCB introduces material stale-memory or false-fire harm.
4. Full OCB’s gains are concentrated only in secondary or low-volume slices.

## 26.3 Correction+heuristics becomes default if all are true

1. It materially beats correction-only in mean net task utility.
2. It captures **>=85%** of full OCB’s mean net task utility across primary priority slices.
3. It has lower false-fire/stale-memory harm than full OCB.
4. It is easier to explain and adopt than full OCB.

## 26.4 Hybrid outcome

A hybrid outcome is allowed:

> Correction+heuristics is default, with full OCB enabled only for slices where it beats baselines without harm.

Use if full OCB wins in retrieval-heavy/tool-heavy tasks but loses or ties in primary slices.

## 26.5 Pause condition

Pause general-memory-runtime claims if:

- no backend shows positive net task utility in at least 4 slices, or
- utility-sign judge disagreement exceeds 30%, or
- traces are too weak/clean/repo-adjacent to prove product value

---

## 27. Product decision tree

At day 30, choose exactly one:

A. Full OCB remains the product  
B. Correction-sticky product becomes default  
C. Correction+heuristics product becomes default  
D. Hybrid default + slice-gated full OCB  
E. Runtime health / verification layer only  
F. Pause until better traces exist

---

# Part III — Ledger and reproducibility

## 28. Ledger schema

File:

```text
eval/results/<run-id>/ledger-judged.jsonl
```

One row per trace/backend/judge.

```json
{
  "trace_id": "trace-001",
  "source": "telegram|github|session|synthetic|repo-derived|adversarial",
  "provenance_type": "real|synthetic|repo-derived|adversarial",
  "mode": "smoke|production",
  "counts_as_product_evidence": false,
  "privacy_scrubbed": true,
  "slice": "direct-answer|continuation|correction-follow-up|retrieval-heavy|tool-heavy|stale-memory-conflict",
  "priority_class": "primary|secondary",
  "task_type": "...",

  "backend": "none|correction-only|correction+heuristics|full-ocb",
  "memory_fired": true,
  "should_have_fired": true,
  "memory_opportunity_label_source": "pre_run_manifest|labeled_audit",
  "activation_reason": "...",
  "retrieved_memory_ids": ["..."],

  "correctness_delta": 1,
  "usefulness_delta": 1,
  "specificity_delta": 1,
  "raw_quality_delta": 3,
  "normalized_quality_delta": 1,
  "quality_delta": 1,

  "harm_delta": 0,
  "cost_penalty": 0.25,
  "activation_utility": 0.75,

  "abstention_regret": 0,
  "abstention_regret_penalty": 0,
  "net_task_utility": 0.75,

  "false_fire": false,
  "stale_memory_conflict": false,

  "input_tokens": 0,
  "output_tokens": 0,
  "memory_tokens": 0,
  "latency_ms": 0,
  "estimated_cost_usd": 0,
  "cost_measurement_mode": "measured|estimated|bucketed|missing",

  "memory_snapshot_id": "snapshot-...",
  "memory_snapshot_created_at": "2026-04-28T00:00:00Z",
  "ocb_config_hash": "sha256-...",
  "model_id": "...",
  "prompt_hash": "sha256-...",
  "code_commit": "...",
  "eval_harness_commit": "...",

  "judge_mode": "blind_quality|labeled_harm_audit|cost_audit|synthetic_smoke",
  "judge_notes": "...",
  "judge_id": "judge-a",
  "created_at": "2026-04-28T00:00:00Z"
}
```

Invariants:

- activation utility is derived, not trusted
- net task utility is derived, not trusted
- raw quality fields are stored
- reproducibility metadata is required
- smoke rows cannot count as product evidence
- every `/results` claim links to ledger rows

---

## 29. Reproducibility metadata

Each run must record:

- memory snapshot id
- memory snapshot created at
- OCB config hash
- model id
- prompt hash
- code commit
- eval harness commit
- tool fixture version, if applicable

---

# Part IV — Results page

## 30. `/results` requirements

Old `/proof` must not pretend to prove user outcomes.

`/results` must be generated from the ledger and show:

- trace count by slice
- backend comparison by slice
- activation utility when fired
- net task utility across all traces
- fire rate
- false-fire rate
- abstention regret
- stale-memory-conflict harm
- cost penalty
- cost per utility point
- uncertainty/low-N warnings
- redacted examples
- judge disagreement
- negative results
- final product decision
- smoke-only warning when applicable

Must not show:

- old proof scores as outcome proof
- cherry-picked demos without denominators
- blended headline without per-slice table
- hand-edited claims not regenerated from ledger

---

# Part V — Seven PR execution sequence

Do not ask the agent to “do V5” as one amorphous task.

Execute these seven PR-sized chunks.

---

## PR 1 — Results contract docs

Create:

```text
docs/results/RUBRIC.md
docs/results/JUDGE_PROTOCOL.md
docs/results/AUTHORITY_PRECEDENCE.md
docs/results/LEDGER_SCHEMA.md
docs/results/THRESHOLDS.md
docs/results/TRACE_ADMISSION.md
docs/results/BASELINE_DEFINITIONS.md
docs/results/TOOL_TRACE_SAFETY.md
docs/results/PRODUCT_DECISION_TREE.md
docs/results/COMMANDS.md
```

Acceptance bar:

- all docs exist
- they match V5
- thresholds are unambiguous
- priority slices are fixed
- product decisions cannot be made post hoc

---

## PR 2 — `packages/results-schema`

Implement:

```text
packages/results-schema/src/ledger.ts
packages/results-schema/src/rubric.ts
packages/results-schema/src/thresholds.ts
packages/results-schema/src/summary.ts
packages/results-schema/src/uncertainty.ts
packages/results-schema/test/ledger-schema.test.ts
packages/results-schema/test/summary.test.ts
packages/results-schema/test/no-double-counting.test.ts
packages/results-schema/test/net-task-utility.test.ts
packages/results-schema/test/threshold-conflict.test.ts
```

Acceptance bar:

- invalid ledger rows fail
- enum violations fail
- `activation_utility` is derived, not trusted
- `net_task_utility` is derived, not trusted
- quality, harm, and cost cannot be double-counted
- thresholds produce exactly one recommended product outcome or a declared tie/blocker

---

## PR 3 — Trace manifest validator and smoke traces

Create:

```text
eval/traces/manifest.json
eval/traces/smoke-001/input.json
eval/traces/smoke-001/provenance.json
...
docs/results/TRACE_COVERAGE.md
```

Smoke traces must be marked:

```json
{
  "provenance_type": "synthetic",
  "mode": "smoke",
  "counts_as_product_evidence": false
}
```

Acceptance bar:

- manifest validates
- coverage checker runs
- production mode fails with fewer than 40 admitted traces
- smoke mode passes with synthetic traces but marks results as non-evidence

---

## PR 4 — `packages/eval-harness`

Implement:

```text
packages/eval-harness/src/index.ts
packages/eval-harness/src/trace.ts
packages/eval-harness/src/run.ts
packages/eval-harness/src/blind-packets.ts
packages/eval-harness/src/reproducibility.ts
packages/eval-harness/src/tool-fixtures.ts
packages/eval-harness/src/backends/none.ts
packages/eval-harness/src/backends/correction-only.ts
packages/eval-harness/src/backends/correction-heuristics.ts
packages/eval-harness/src/backends/full-ocb.ts
```

Backend interface:

```ts
interface EvalBackend {
  id: 'none' | 'correction-only' | 'correction+heuristics' | 'full-ocb';
  run(trace: EvalTrace): Promise<EvalRunResult>;
}
```

Acceptance bar:

- same trace runs against all four backends
- outputs saved uniformly
- adapters isolated
- tool-heavy traces fixture-backed/read-only
- no external mutation possible in eval mode
- reproducibility metadata captured
- `ledger-draft.jsonl` produced

---

## PR 5 — Blind judging flow

Create:

```text
eval/results/<run-id>/blind-judge-packets/
packages/eval-harness/src/blind-packets.ts
packages/eval-harness/src/judging/import-judgments.ts
```

Acceptance bar:

- backend labels hidden in blind packets
- output order randomized per trace
- mapping stored privately for recombination
- judge imports validate
- missing judge scores fail production mode
- synthetic smoke judgments allowed only in smoke mode

---

## PR 6 — `packages/results-site`

Implement:

```text
packages/results-site/src/generate.ts
packages/results-site/src/tables.ts
packages/results-site/src/examples.ts
packages/results-site/src/uncertainty.ts
packages/results-site/src/decision.ts
```

Outputs:

```text
docs/results/index.md
docs/results/summary.json
```

Acceptance bar:

- `/results` generated, not hand-written
- every table has denominators
- empty and low-N slices visible
- negative results visible
- synthetic smoke results labeled non-evidence
- product decision threshold-derived

---

## PR 7 — End-to-end smoke and production gates

Create one command:

```bash
pnpm ocb:e2e:smoke
```

It runs:

1. schema tests
2. trace validation
3. smoke eval across all four backends
4. blind packet generation
5. synthetic judgment import
6. ledger validation
7. results generation
8. decision generation
9. `RUN_STATE` write

Acceptance bar:

- one command completes smoke pipeline
- results page exists
- summary JSON exists
- synthetic decision memo exists
- `RUN_STATE.engineering_e2e_complete=true`
- `RUN_STATE.evidence_e2e_complete=false` unless real trace/judging requirements are met

---

# Part VI — 30-day production cadence

## Days 1–3 — Freeze and define

Deliver PR 1.

Exit bar:

- docs/results contracts are repository law
- commands documented
- no threshold ambiguity

## Days 4–10 — Build trace set and schema machinery

Deliver PR 2 and PR 3.

Exit bar:

- schema tests pass
- smoke traces validate
- production trace gaps are explicit

## Days 11–15 — Build ablation harness

Deliver PR 4.

Exit bar:

- all four backends run on smoke traces
- draft ledger produced

## Days 16–21 — Build judging and audit flow

Deliver PR 5.

Exit bar:

- blind packets generated
- judgments imported
- missing production judgments fail closed

## Days 22–25 — Publish results generator

Deliver PR 6.

Exit bar:

- `/results` regenerates from ledger
- synthetic outputs labeled non-evidence

## Days 26–30 — E2E smoke and product decision gate

Deliver PR 7.

Exit bar:

- engineering E2E complete
- production E2E complete only if real evidence exists
- otherwise blockers and next data needed are generated

---

# Part VII — Exact agent instruction

Use this as the execution prompt for the OpenClaw/Codex 5.5 agent:

```text
You are implementing OpenClawBrain 1.0 Execution Plan V5.

Your job is to complete the evidence scoreboard pipeline end to end.

Do not build new OpenClawBrain runtime architecture.
Do not tune memory thresholds.
Do not write marketing copy.
Do not hand-edit results.
Do not fabricate traces, provenance, costs, judge scores, memory snapshots, backend wins, or product evidence.

First implement Engineering E2E:
1. docs/results contracts
2. packages/results-schema
3. trace manifest validator
4. smoke traces marked non-evidence
5. packages/eval-harness
6. four backend adapters
7. blind judge packet generation
8. judged ledger import
9. packages/results-site
10. threshold-derived decision generator
11. RUN_STATE tracking
12. e2e smoke command

Then prepare Production Mode:
- require 40 admitted traces
- enforce slice minimums
- enforce provenance/privacy metadata
- enforce tool fixture/read-only safety
- enforce reproducibility metadata
- fail closed if production evidence is incomplete

Completion definitions:
- Engineering E2E complete means the full pipeline runs on smoke traces and generates /results.
- Evidence E2E complete means the full pipeline runs on 40 admitted real redacted traces with completed judging.

If real traces or judge scores are unavailable, do not fake them.
Instead produce:
- docs/results/BLOCKERS.md
- docs/results/NEXT_DATA_NEEDED.md
- docs/results/PARTIAL_COMPLETION.md
- RUN_STATE.json with evidence_e2e_complete=false

All generated claims must come from ledger rows.
All utility values must be derived from schema logic.
All product decisions must come from preregistered thresholds.
```

---

## 40. Final recommendation

V5 makes the plan agent-finishable by splitting completion into two gates:

```text
Gate 1: Engineering E2E
The agent can fully complete this.

Gate 2: Evidence E2E
The agent can complete this only with real traces and judge inputs.
```

This gives the best of both worlds:

- the 5.5 agent can execute autonomously
- the project remains honest about whether OpenClawBrain has earned the full product

The agent must always finish the pipeline.

It must never fake the evidence.
