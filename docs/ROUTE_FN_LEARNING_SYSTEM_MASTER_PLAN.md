# OpenClawBrain Route Function Learning System Master Plan

**Status:** canonical end-to-end plan for making the learned `route_fn`, its update loop, and its training-data/storage layer materially better  
**Owner:** GUCLAW / Jonathan  
**Date:** 2026-05-06  
**Depends on:** shipped `route-policy-v2`, implemented `route-policy-v3`, teacher/counterfactual pipeline, current v3 storage tables  
**Supersedes for forward design:** use this as the top-level execution plan above `ROUTE_LEARNING_ULTIMATE_MASTER_PLAN.md`, `ROUTE_TEACHER_MASTER_PLAN_PART2.md`, and `ROUTE_LEARNING_V3_HARDENING_MASTER_PLAN.md` when deciding what to build next.  
**Current production baseline:** `route-policy-v2` remains the safest shipped runtime default; current v3 work is the guarded learning lane.

---

## 0. Executive decision

Yes: we should make the learned route function much better.

But the right upgrade is **not** to replace the system with a bigger opaque model.

The right upgrade is to make the route-learning system stronger in **three coupled layers**:

1. **Serving route function** — smaller, more canonical, more calibrated, more explicit about abstention and fallback.
2. **Update machinery** — better supervision, safer promotion rules, stronger replay/shadow/off-policy evidence, less policy churn.
3. **Training-data/storage model** — richer, cleaner, more normalized, lineage-preserving, easier to dedupe, easier to replay, easier to audit.

The invariant stays the same:

> **runtime serving should still be a compact deterministic artifact; richer ML should live in the update path, data path, and evaluation path.**

That means the best end state is:

```text
turn outcomes + critiques + counterfactuals + shadow proposals
  -> normalized learning warehouse
  -> prototype/action learning + calibration + replay eval
  -> candidate policy distillation
  -> strict activation gates
  -> compact deterministic route_fn snapshot at runtime
```

---

## 1. The core diagnosis

The current system already has the right bones:

- route decisions are durable
- teacher/counterfactual feedback exists
- v3 frames/prototypes/pairs/bandit state exist
- active runtime can load a deterministic learned snapshot
- fallback remains fail-closed

The real weaknesses are now structural, not conceptual.

### 1.1 Weakness in the learned `route_fn`

The learned route function is still too:

- duplicate-heavy
- heuristic in confidence meaning
- broad in some rule families
- static in sparse/dense weighting
- weakly explicit about abstention
- insufficiently family-aware in thresholds and risk

### 1.2 Weakness in the update loop

The updater still:

- learns from noisy labels too eagerly
- lacks strong held-out replay discipline
- can over-credit teacher/counterfactual output
- does not separate data collection from promotion strongly enough
- lacks an explicit shadow-to-promotion policy based on measured improvement
- lacks enough anti-churn logic for policy snapshots and prototype retirement

### 1.3 Weakness in training-data storage

The storage model is still too close to a first working implementation:

- some concepts are mixed together that should be separated
- some rows are too denormalized for durable analysis
- some labels do not preserve enough uncertainty/provenance
- replay/eval slices are not yet first-class
- feature lineage for why a policy activated is not rich enough
- data quality states are not explicit enough

---

## 2. What “better” should mean

We should define success precisely.

### 2.1 Better serving `route_fn`

A better route function should be:

- **more precise** on correction/workflow/project-context turns
- **more silent** on low-signal turns
- **more calibrated** about when it should abstain
- **more compact** for the same or better behavior
- **more interpretable** per matched rule
- **more stable** across minor phrasing variants

### 2.2 Better updates

A better updater should:

- learn faster from real turns
- resist noisy one-off lessons
- activate only after evidence of improvement
- keep shadow evidence even when not in control
- maintain rollback-ready history
- avoid thrashing the active policy

### 2.3 Better training-data storage

A better storage system should:

- keep raw sensitive text out
- preserve redacted semantic signal and lineage
- separate observations from labels from examples from policies
- support replay, shadow, and calibration analysis directly
- let us answer “why did the router learn this?” without guesswork
- support future migration without data loss

---

## 3. Design principles

1. **Deterministic runtime, richer offline/update path.**
2. **Redacted durable storage only.** No raw transcript dependency.
3. **Append-only evidence, derived snapshots for serving.**
4. **Promotion by measured improvement, not vibes.**
5. **Abstention is a first-class decision, not a failure mode.**
6. **Family-specific logic beats one global threshold.**
7. **Data lineage must survive every distillation step.**
8. **All broad rules must justify their existence.**
9. **Storage should model uncertainty explicitly.**
10. **Every new learner feature must be replay-measurable before activation.**

---

## 4. Target architecture

The future route-learning system should be organized into five layers.

```text
Layer A: Observation store
Layer B: Training warehouse
Layer C: Learners + distillers
Layer D: Eval / calibration / promotion control
Layer E: Runtime route_fn serving snapshot
```

### 4.1 Layer A — Observation store

This layer stores what happened.

It should include:

- route decisions
- route frames
- selected/omitted memory outcomes
- teacher critiques
- counterfactual proposals
- shadow proposals from candidate policies
- reward/outcome resolutions
- runtime features used at decision time

This layer is append-only and should be the main audit truth.

### 4.2 Layer B — Training warehouse

This layer transforms observations into learning-ready facts.

It should contain normalized entities like:

- action prototypes
- action-family aggregates
- pairwise preference labels
- calibration examples
- replay-eval cases
- bandit feedback events
- prototype retirement state
- policy candidate metrics

This is where we make the data analytically clean.

### 4.3 Layer C — Learners + distillers

This layer should produce:

- improved action priors
- calibrated confidence mappings
- prototype merge/retirement actions
- replay summaries
- candidate policy snapshots

It should be allowed to be richer than serving, but all outputs must still distill into the serving contract.

### 4.4 Layer D — Eval / calibration / promotion control

This layer decides whether a candidate route function:

- stays rejected
- remains shadow-only
- becomes active
- replaces a prior active policy
- triggers rollback recommendations

### 4.5 Layer E — Runtime route_fn serving snapshot

This is the only layer the default runtime should need on a live turn.

It should remain:

- deterministic
- compact
- versioned
- fully inspectable
- safe to rollback

---

## 5. The target serving contract for the learned `route_fn`

The next strong route function should not just be a list of rules. It should be a compact decision bundle.

## 5.1 Serving snapshot shape

Target shape:

```ts
RoutePolicySnapshotVNext = {
  id,
  version,
  status,
  createdAt,
  parentSnapshotId?,
  lineage: {
    sourceFrameIds,
    sourcePrototypeIds,
    sourceEvalSliceIds,
    comparedAgainstSnapshotId?,
  },
  servingMode: {
    defaultFallback: 'heuristic' | 'v2' | 'no_memory_safe',
    abstainEnabled: true,
  },
  rules: RoutePolicyRuleVNext[],
  actionPriors,
  calibration,
  thresholds,
  compactness,
  replaySummary,
  activationSummary,
  budgets,
}
```

## 5.2 Rule shape goals

Each rule should carry:

- exact match conditions
- route action outcome
- calibrated confidence metadata
- why it exists
- evidence support summary
- narrowness/specificity markers
- family tag for thresholding

### 5.2.1 New rule fields to add

Add or standardize fields like:

- `family`: `silence`, `workflow`, `correction`, `project_context`, `general_retrieval`, `sync_enabling`
- `matchSpecificityScore`
- `dominanceGroupKey`
- `canonicalActionKey`
- `rawConfidence`
- `calibratedConfidence`
- `abstainBelow`
- `fallbackMode`
- `supportSummary`
- `riskFlags`
- `diagnosticNotes`

## 5.3 Serving-time routing behavior

The runtime should use a three-way decision, not just “match vs no match”:

1. **apply rule directly**
2. **abstain and defer to fallback router**
3. **safe silence / no-memory path**

This is the single most important serving-control change.

## 5.4 Family-specific thresholds

Thresholds should be stricter for:

- correction-only rules
- sync-planner-enabling rules
- broad retrieval rules
- graph depth > 0 retrieval

Thresholds can be looser for:

- well-supported repo workflow retrieval
- repeated high-value correction patterns
- high-precision silence rules on low-signal turns

## 5.5 Explicit fallback provenance

Every runtime decision should record:

- matched snapshot id
- matched rule id
- raw score
- calibrated score
- routing mode
- abstained? true/false
- fallback source if abstained
- candidate count considered
- compactness/calibration version that influenced the choice

This makes route behavior debuggable instead of interpretive.

---

## 6. The target update system

The updater should become a controlled pipeline, not just “new data in, new snapshot out”.

## 6.1 Separate update phases

Split updates into six explicit phases:

### Phase U1 — ingest

Store raw redacted observations:

- route frame
- runtime features
- actual selected route/action
- eventual outcome/reward
- teacher/counterfactual outputs
- shadow proposals from inactive policies

### Phase U2 — normalize

Canonicalize:

- task families
- signal families
- query template families
- action family keys
- repo/project/tool placeholders
- reward components
- confidence weights

### Phase U3 — label

Create training labels with uncertainty, not just hard labels:

- pairwise preference labels
- positive/negative prototype signals
- abstain-is-better labels
- route-family-specific quality labels
- “uncertain/ambiguous” labels

### Phase U4 — aggregate

Update:

- prototype priors
- family win rates
- calibration buckets
- replay corpora
- compactness diagnostics
- retirement candidates

### Phase U5 — distill candidate snapshot

Produce a candidate policy only from quality-gated aggregated inputs.

### Phase U6 — evaluate and promote

Promotion requires:

- validation gates
- replay improvement gates
- shadow agreement diagnostics
- no major harm regression
- compactness within budget
- stable enough confidence calibration

## 6.2 Snapshot churn control

Add anti-thrash logic:

- minimum evidence delta before a new candidate is generated
- candidate dedupe by stable policy body hash
- cooldown period before replacing active snapshot again
- “improvement margin” requirement above current active policy
- explicit rollback recommendation if live shadow evidence worsens sharply

## 6.3 Update modes

The updater should support distinct modes:

- **collect_only** — store data, no new candidates
- **distill_shadow** — build candidates but never activate
- **gated_active** — activate only if full gates pass
- **manual_review_required** — build candidate and attach report for operator approval

This makes rollout safer.

---

## 7. The target training-data and storage model

This is the biggest structural improvement.

The current tables are a good start, but the next system should separate **observation facts**, **derived learning rows**, and **serving artifacts** more clearly.

## 7.1 Storage principles

- append-only where possible
- normalized where it helps replay/analysis
- denormalized only for serving snapshots
- lineage preserved across every transformation
- explicit quality state on derived rows
- explicit split between observation time and label time

## 7.2 Proposed data domains

### Domain A — Observation facts

These capture what actually happened.

#### A1. `route_frames_vNext`

One row per resolved decision context.

Keep:

- coarse turn shape
- redacted summary
- feature-family hashes/tokens
- routing mode features
- selected action id
- selected route family
- reward and reward components
- policy/rule ids if a learned snapshot influenced it
- abstain/fallback metadata
- source decision ids

Add:

- `routing_mode`
- `low_signal_score`
- `exactness_score`
- `semantic_breadth_score`
- `correction_intensity_score`
- `abstained`
- `fallback_source`
- `raw_policy_score`
- `calibrated_policy_score`
- `feature_version`

#### A2. `route_shadow_decisions_vNext`

One row per inactive policy or scorer proposal on a live turn.

This is critical.

Store:

- frame id
- candidate snapshot id or learner id
- proposed route/action
- proposed raw/calibrated score
- abstain decision
- top competing rules/actions
- whether it matched the actual chosen route
- eventual realized reward from the actual turn

This gives live comparison without giving the candidate policy control.

#### A3. `route_teacher_runs`

Keep, but strengthen with:

- explicit confidence provenance
- critique family
- “teacher uncertainty” marker
- whether the critique was later validated by outcome or contradicted by later evidence

#### A4. `route_counterfactuals`

Keep, but add:

- `counterfactual_family`
- `estimated_reward_delta`
- `counterfactual_confidence`
- `is_policy_overlap_candidate`
- `validated_later?`

### Domain B — Learning warehouse

These are derived rows.

#### B1. `route_action_prototypes_vNext`

Keep the prototype concept, but enrich it.

Each prototype should have:

- canonical action key
- action family key
- normalized query family key
- sparse signature
- compact dense embedding
- support/harm priors
- reward stats
- route-family stats
- status: `active|shadow|retired|cold_start`
- retirement reason if retired
- source lineage summary
- last promotion influence timestamp

#### B2. `route_action_family_stats_vNext`

New table.

This stores aggregates at the family level rather than only the prototype level.

Why it matters:

- better cold-start behavior
- more stable priors
- easier threshold tuning
- less overreaction to tiny prototypes

Store:

- canonical action family key
- route family
- support count
- harm count
- mean reward
- variance
- abstain-win rate
- shadow-win rate
- sync cost rate
- calibration sample count

#### B3. `route_pair_examples_vNext`

Keep pairwise supervision, but add quality metadata.

Add:

- `label_confidence`
- `label_quality`: `high|medium|low|ambiguous`
- `preferred_due_to`: `teacher|outcome|counterfactual|manual|abstain_better`
- `observed_vs_imputed`: `observed|teacher_inferred|counterfactual_inferred`
- `family_margin`

This stops all pairwise labels from being treated equally.

#### B4. `route_bandit_feedback_vNext`

Keep, but make it more factorized.

Store:

- total reward
- reward components
- cost/latency penalties
- abstain gain if fallback won
- exploration indicator
- exploitation indicator
- reward confidence
- off-policy eligibility marker

#### B5. `route_calibration_examples_vNext`

New table.

This should become first-class rather than implicit.

Each row should capture:

- frame id
- snapshot id / rule id / prototype id
- raw predicted score
- route family
- routing mode
- observed success label
- comparable? yes/no
- holdout split id
- evaluation slice id

This makes calibration reproducible.

#### B6. `route_eval_cases_vNext`

New frozen replay corpus table.

Each row should represent a curated evaluation case with:

- stable case id
- slice membership
- expected preferred route family
- quality notes
- whether human-reviewed
- whether safe for promotion gates

#### B7. `route_eval_case_labels_vNext`

New table for multi-source labels on eval cases.

Sources can be:

- outcome-derived
- teacher-derived
- manual review
- consensus

This lets replay use stronger gold labels over time.

### Domain C — Serving artifacts and promotion control

#### C1. `route_policy_snapshots_vNext`

Keep, but add:

- parent snapshot id
- stable body hash
- calibration id / summary hash
- replay summary id / hash
- compactness summary
- promotion mode
- activation gate decision summary
- rollback recommendation fields

#### C2. `route_policy_candidate_reports_vNext`

New table.

Each candidate should have a durable report row that explains:

- what changed from previous snapshot
- why it was generated
- replay results vs prior active
- compactness change
- calibration change
- projected risks
- activation or rejection reason

This becomes the operator-facing truth.

---

## 8. The exact improvements to the learned `route_fn`

## 8.1 Rule deduplication and canonicalization

This is still the first implementation priority.

Add:

- canonical action keys
- canonical query-family keys
- rule clustering by semantic equivalence
- evidence merging
- support/harm aggregation across sibling variants

## 8.2 Dominance pruning

A broader rule should die if a narrower rule already wins with equal or better evidence.

Prune when:

- same route behavior
- same or worse calibrated confidence
- same or worse evidence base
- broader match surface
- no special safety value

## 8.3 Compactness budgets

Promotion should enforce compactness limits such as:

- max total rules
- max broad-rule share
- max singleton-rule share
- min average evidence per active rule
- max duplicate-family count

## 8.4 Family-aware scoring

A route function should not treat all route families equally.

Add route families like:

- silence
- correction retrieval
- workflow retrieval
- project-context retrieval
- broad semantic retrieval
- sync-enabling retrieval

Each family gets:

- threshold policy
- risk policy
- calibration bucket space
- replay metrics

## 8.5 Routing mode detection

Before choosing a rule, compute a cheap mode:

- `exact_correction`
- `repo_workflow`
- `semantic_planning`
- `project_context`
- `casual_silence`
- `ambiguous_general`

Then use that mode to adjust:

- sparse vs dense weighting
- abstain thresholds
- risk penalties
- graph-depth willingness

## 8.6 Calibrated abstention

The route function must be allowed to say:

- “I see a possible rule, but confidence is not good enough”
- “fallback should decide”
- “silence is safer here”

This is what prevents v3-style learned routing from overclaiming control.

## 8.7 Better provenance in serving decisions

Every matched decision should explain itself in a way a human can inspect quickly.

Desired explanation shape:

- matched family
- top match signals
- raw score
- calibrated score
- threshold
- abstained/fallback reason
- compactness/calibration generation ids
- top alternative rule family

---

## 9. The exact improvements to the updater

## 9.1 Better reward shaping

Move from a mostly flat reward to a structured reward vector.

Suggested components:

- retrieval_help_gain
- correction_prevention_gain
- workflow_acceleration_gain
- accepted_memory_gain
- noisy_injection_penalty
- unnecessary_sync_penalty
- graph_overreach_penalty
- latency_penalty
- abstain_success_gain
- ambiguity_penalty
- teacher_confidence_weight
- validator_confidence_weight

Then derive scalar reward from a versioned weighting config.

That keeps old rows reusable if reward policy changes.

## 9.2 Better label quality control

Not all labels are equally trustworthy.

Downweight when:

- teacher confidence is low
- counterfactual confidence is low
- reward is near zero
- the turn is ambiguous
- supervision is only inferred, not observed

Upweight when:

- observed correction or observed tool success strongly supports the label
- repeated similar outcomes agree
- shadow proposals repeatedly lose or win on the same family

## 9.3 Better cold-start behavior

New prototypes should not get too much power too early.

Add:

- `cold_start` prototype status
- family-level priors backing cold-start prototypes
- minimum evidence before eligibility for active-policy distillation
- higher abstain penalty against under-supported prototypes

## 9.4 Prototype retirement

Retire prototypes when they are:

- repeatedly harmful
- redundant with a stronger sibling
- stale and unsupported
- consistently shadow-losing

Retirement should be explainable and reversible.

## 9.5 Shadow-first update discipline

For any nontrivial change in scoring or distillation logic:

1. log shadow decisions first
2. replay candidate policy on frozen corpus
3. compare against active policy and fallback baselines
4. only then allow activation

## 9.6 Off-policy caution

Do not let partial feedback masquerade as full labels.

Use conservative evidence buckets:

- overlap-only replay
- disagreement analysis
- abstain win/loss analysis
- later doubly-robust estimates if needed

---

## 10. The exact improvements to the training-data model

## 10.1 Separate facts from labels from policy

The most important storage cleanup is conceptual:

- **facts** = what happened
- **labels** = what we infer/prefer from what happened
- **policy** = what we serve because of those labels

The system gets much cleaner if we never blur these.

## 10.2 Version every feature space

Rows derived from features should store:

- feature schema version
- reward schema version
- calibration version
- routing-mode version
- distillation version

Without this, old data becomes hard to compare after learner changes.

## 10.3 Add dataset/slice membership

Every learning row that matters for calibration or replay should be assignable to:

- train
- calibration
- replay_eval
- shadow_audit
- manual_review

This avoids accidental data leakage.

## 10.4 Data quality states

Every derived example should be marked with something like:

- `trusted`
- `usable`
- `weak`
- `ambiguous`
- `excluded`

This is much better than pretending all rows are equally valid.

## 10.5 Better query family storage

Do not store too many literal query surface forms as if they are distinct knowledge.

Instead store:

- canonical query family
- representative examples
- placeholderized forms
- family frequency
- family win rate

## 10.6 Explicit lineage links

Every rule in an active snapshot should be traceable through:

- source prototypes
- source frames
- source pair examples
- source calibration rows
- source replay slices

A human should be able to inspect one rule and understand why it exists.

---

## 11. Evaluation and promotion framework

## 11.1 Promotion should require both safety and usefulness

A candidate should not activate only because it is “not terrible.”

Require at least one meaningful improvement axis:

- better correction recall
- better workflow recall
- lower noisy injection at similar recall
- equal utility at lower sync/latency cost
- better abstain behavior on ambiguous turns

## 11.2 Replay metrics to standardize

Track at least:

- route agreement with labeled eval cases
- correction recall
- workflow recall
- project-context recall
- silence precision
- harmful injection rate
- abstain rate
- sync-planner overuse rate
- reward-weighted agreement
- projected value delta vs active policy

## 11.3 Shadow metrics to standardize

Track:

- candidate-vs-active disagreement rate
- candidate win rate on later-observed helpful turns
- candidate loss rate on later-observed harmful turns
- abstain win rate
- family-specific disagreement heatmap

## 11.4 Rollback criteria

Recommend rollback or shadow-only demotion when:

- harmful injection rate spikes
- abstain collapses on ambiguous turns
- sync cost grows without utility gain
- correction family loses precision sharply
- compactness bloats with no measured value gain

---

## 12. Concrete execution roadmap

## Phase 1 — Route function cleanup

Goal: make the serving artifact smaller and saner.

Build:

- canonical action keys
- canonical query-family normalization
- merge duplicate rule candidates
- dominance pruning
- compactness diagnostics and gates
- richer rule provenance

Done when:

- active candidates are visibly smaller
- duplicate-family counts fall
- tests show pruning/merge behavior is correct

## Phase 2 — Calibration + abstention

Goal: make learned routing trustworthy enough to know when not to act.

Build:

- calibration example store
- route-family calibration buckets
- routing-mode-aware thresholds
- explicit abstain + fallback recording
- runtime raw vs calibrated score proof

Done when:

- replay shows more reliable confidence bands
- abstention is explicit and test-covered

## Phase 3 — Replay + shadow + promotion control

Goal: stop activating based on weak evidence.

Build:

- frozen eval corpus
- eval case labels
- shadow-decision logging
- candidate report rows
- improvement-gated activation policy

Done when:

- every candidate has a durable comparison report
- activation reasons are measurable and inspectable

## Phase 4 — Training warehouse normalization

Goal: make storage future-proof and analytically clean.

Build:

- vNext tables for calibration/eval/family stats/shadow reports
- dataset split membership
- feature/reward/distillation versioning
- data quality states
- stronger lineage links

Done when:

- replay/calibration/promotion no longer rely on ad hoc reconstruction

## Phase 5 — Adaptive routing modes + better online updates

Goal: improve quality after the safety substrate is in place.

Build:

- routing mode detector
- dynamic sparse/dense/prior weighting
- structured reward policy versioning
- cold-start prototype logic
- prototype retirement policy
- stronger abstain-win reward handling

Done when:

- adaptive logic measurably helps exact vs semantic turn families

## Phase 6 — Optional stronger learner behind the same contract

Only after the above.

Possible additions:

- stronger dense compatibility scorer
- better off-policy estimators
- lightweight learned reranker in shadow mode
- teacher distillation improvements

This phase is optional and should not happen early.

---

## 13. Code map / implementation impact

Likely primary implementation files:

- `packages/openclaw-plugin/src/route-fn.ts`
- `packages/openclaw-plugin/src/route-policy-v3.ts`
- `packages/openclaw-plugin/src/route-policy-v3-normalize.ts`
- `packages/openclaw-plugin/src/route-policy-v3-calibration.ts`
- `packages/openclaw-plugin/src/route-policy-v3-eval.ts`
- `packages/openclaw-plugin/src/route-policy-v3-routing-mode.ts`
- `packages/openclaw-plugin/src/route-teacher.ts`
- `packages/openclaw-plugin/src/memory-store.ts`
- `packages/openclaw-plugin/src/memory-types.ts`
- proof / inspect surfaces that expose policy and candidate reports

Likely new modules over time:

- `route-policy-v3-shadow.ts`
- `route-policy-v3-reporting.ts`
- `route-policy-v3-datasets.ts`
- `route-policy-v3-reward.ts`
- `route-policy-v3-promotion.ts`

---

## 14. Migration strategy

Do not try to big-bang replace the current storage.

Use staged migration:

1. keep current v3 tables live
2. add vNext companion tables where new concepts are needed
3. dual-write where safe
4. backfill derived rows from old v3 frames where possible
5. keep runtime serving on the stable snapshot contract
6. only retire old paths once replay/proof parity is reached

This avoids breaking current learning while upgrading the system.

---

## 15. Non-goals

For now, do **not**:

- replace runtime with a direct neural router
- introduce an ANN service for serving route candidates
- store raw user text for better training
- let teacher or counterfactual outputs activate policy directly without replay/shadow evidence
- optimize for tiny offline benchmark gains at the expense of auditability

---

## 16. Final recommendation

The best next move is:

1. **clean the serving route function first**
2. **make confidence/abstention real**
3. **build replay/shadow/promotion discipline**
4. **normalize the training warehouse**
5. **only then add stronger adaptivity**

In short:

> **first make the learned route function smaller and more trustworthy; then make the updater more evidence-based; then make the storage layer more like a real learning warehouse.**

That is the highest-leverage path to a route-learning system that is:

- smarter
- safer
- easier to debug
- easier to evolve
- and still compatible with OpenClawBrain’s core fail-closed product contract.
