# OpenClawBrain Route Learning v3 Hardening Master Plan

**Status:** post-v3 implementation hardening + evaluation + rollout plan  
**Owner:** GUCLAW / Jonathan  
**Date:** 2026-05-06  
**Depends on:** shipped `route-policy-v2`, implemented `route-policy-v3`, `route_frames_v3`, `route_action_prototypes_v3`, `route_pair_examples_v3`, `route_bandit_feedback_v3`, `route_bandit_state_v3`, `route_policy_snapshots_v3`  
**Purpose:** turn the new v3 route-learning stack from a working guarded v1 into a high-confidence production routing layer with cleaner policy distillation, better calibration, better evaluation, safer activation, and stronger online learning quality.

---

## 0. Executive summary

The current v3 route-learning stack is a strong first real system:

- it stores route outcomes as structured v3 learning data
- it maintains a learned action catalog and bandit state
- it distills into auditable `route-policy-v3` snapshots
- it can activate or reject snapshots through gates
- it keeps heuristics / v2 fallback as a safety net

That is the correct architecture direction.

But it is not yet the final form.

The next stage should **not** replace the current system with a bigger opaque scorer. Instead it should harden the existing architecture along five dimensions:

1. **policy quality** — dedupe, merge, compress, and simplify the serving snapshot
2. **calibration** — make confidence scores meaningful enough to support abstention and routing control
3. **evaluation** — add offline, shadow, and live metrics that tell us if v3 is truly beating heuristics/v2
4. **adaptive hybrid retrieval/routing** — make sparse/dense weighting and route candidate scoring query-aware
5. **online learning quality** — improve pairwise supervision, reward shaping, and off-policy safety before increasing v3 control

The end state is still the same invariant:

> **runtime uses a compact deterministic learned policy snapshot; richer ML is teacher/update machinery, not the serving contract.**

---

## 1. Current state and key weaknesses

## 1.1 What is already good

The implemented v3 system already has the right major pieces:

```text
route_decisions / route_frames
  -> route teacher + counterfactuals + lessons
  -> ingestRouteLearningArtifactsV3
  -> route_frames_v3 + action_prototypes_v3 + pair_examples_v3 + bandit_feedback_v3 + bandit_state_v3
  -> maybeDistillAndStorePolicyV3
  -> route_policy_snapshots_v3
  -> RouteFn loads active v3 snapshot first, else falls back
```

Key strengths:

- no synchronous LLM needed at serving time
- full auditability from route decision to learned policy snapshot
- partial-feedback-compatible learning loop
- fail-closed activation gates
- compact deterministic serving artifact
- explicit policy snapshot versioning and rollback path

## 1.2 What is still weak

Deep testing surfaced four meaningful weaknesses:

### A. Distilled policy snapshots are too duplicate-heavy

Observed pattern:

- many rules differ only trivially
- multiple prototype variants collapse to nearly the same route behavior
- serving snapshots can be large without becoming more decisive

Why it matters:

- harder to interpret
- harder to calibrate
- weaker match specificity
- greater risk of weird accidental matches

### B. Confidence is not yet truly calibrated

Observed pattern:

- rule confidence is computed heuristically from support/harm/pair stats
- score thresholds are directionally useful but not yet probabilistically meaningful
- active v3 can still defer to heuristics often, which is safe but reveals that scores are not yet trustworthy enough to consistently steer routing

Why it matters:

- we need a real abstain decision
- shadow-vs-active comparisons require confidence calibration
- activation and rollback policy should depend on calibrated reliability, not just raw heuristic score

### C. Evaluation is still too narrow

Current checks are meaningful but incomplete:

- unit/integration tests pass
- targeted probes show reject/activate behavior
- one real counterfactual-label bug was caught and fixed

Still missing:

- stable offline eval set
- shadow evaluation logs comparing v2/heuristics/v3 on the same turns
- explicit win/loss metrics by route type and query family
- off-policy evaluation discipline before increasing v3 control

### D. The hybrid sparse/dense design is static and underused

Current state:

- v3 stores sparse signatures and compact hashed dense embeddings
- distillation uses both symbolic and learned priors
- runtime still mostly consumes distilled rules, not a richer adaptive hybrid scorer

Weakness:

- exact-code / correction / repo-jargon turns and broad-semantic planning turns should not be treated with one static weighting pattern

### E. Online update quality is good but still naive

Current state:

- chosen action feedback is stored
- action stats are accumulated
- pairwise examples are derived from teacher/counterfactuals/outcomes

Still weak:

- reward shaping is simple
- pairwise labels can still be noisy even after the recent bug fix
- there is no explicit off-policy correction layer
- no confidence-aware downweighting for ambiguous or low-signal lessons

---

## 2. Hardening goals

The next iteration should achieve the following.

## 2.1 Serving policy goals

The active `route-policy-v3` snapshot should become:

- smaller
- more canonical
- more interpretable
- more decisive
- less duplicate-heavy
- safer to match

## 2.2 Confidence and activation goals

The system should know when to:

- trust v3
- defer to heuristics/v2
- remain shadow-only
- reject a candidate snapshot

## 2.3 Evaluation goals

We should be able to answer with evidence:

- when does v3 beat heuristics?
- when does v3 beat v2?
- which route families improve most?
- where is v3 still harmful or noisy?
- what kinds of turns should remain heuristic-led?

## 2.4 Learning goals

The online learner should improve with more data without:

- exploding policy size
n- amplifying noisy pairwise labels
- overfitting tiny route families
- activating regressions from partial feedback artifacts

---

## 3. Workstream A — Distilled policy cleanup and dedup

This is the highest-ROI immediate improvement.

## 3.1 Problem

Current snapshots can contain many near-duplicate rules because prototypes are created from multiple evidence paths:

- actual chosen route
- teacher preferred route
- counterfactual route
- silence alternative

These are useful internally but should not all survive independently into the serving snapshot.

## 3.2 Goal

Distillation should merge semantically equivalent or near-equivalent rule candidates into one canonical serving rule whenever possible.

## 3.3 Plan

### A1. Add canonical action keys

Create a canonical identity for route actions based on:

```text
route
+ sorted(memory_types)
+ graph_depth
+ sync_planner
+ normalized_query_template_family
+ normalized task / signal family
```

This should be stricter than the current prototype id logic for storage merge, but more semantic than raw evidence-specific variants.

### A2. Normalize query template families harder

Current query template family storage is still too literal.

Improve by:

- lowercasing and punctuation normalization
- stemming or template token normalization for route-query families
- replacing repo-specific literals with typed placeholders when safe
- collapsing minor phrasing variants into one route-query family

Examples:

- `test workflow`
- `test workflow before commit`
- `repo test command workflow`

should become a smaller number of canonical route-query families if they imply the same retrieval intent.

### A3. Merge duplicate rule candidates before activation

Add a distillation pass that:

- groups rules by canonical action key
- unions evidence ids
- sums / averages support-harm-prior features
- chooses top task/signal/project matches
- keeps the strongest or most specific version
- demotes broad or low-support sibling variants

### A4. Add rule dominance pruning

If Rule B is strictly broader and not clearly stronger than Rule A, prune Rule B.

Dominance logic example:

- same route behavior
- same or worse priors
- same or broader task/signal conditions
- no extra safety benefit

Then B should not survive in the active snapshot.

### A5. Add compactness gates

Snapshot activation should include compactness metrics such as:

- total rule count
- duplicate-family count
- average evidence per rule
- singleton-rule share
- broad-rule share

Candidate can be rejected or shadowed if it is too bloated relative to its evidence base.

## 3.4 Deliverables

- new canonical rule merge pass in `route-policy-v3.ts`
- query-template normalization helper
- rule dominance pruning helper
- compactness stats in `evalSummary`
- tests for dedup, merge, dominance pruning, and snapshot compactness

---

## 4. Workstream B — Confidence calibration and abstention

This is the highest-value safety/control improvement.

## 4.1 Problem

Current confidence is a heuristic blend of:

- support count
- harm count
- pair win rate
- bandit mean reward

This is sensible but not calibrated.

A score of `0.62` does not yet mean anything stable like “this rule is right 62% of the time.”

## 4.2 Goal

Build a calibrated confidence layer so RouteFn can do three-way serving:

1. **use v3 rule**
2. **defer to heuristic/v2**
3. **explicit abstain / no-memory safe path**

## 4.3 Plan

### B1. Create a held-out calibration set

Maintain a fixed route-eval slice separate from the examples used for distillation.

This slice should be stratified by:

- route kind
- task type
- correction-heavy vs workflow-heavy vs gratitude/low-signal
- sync vs non-sync

### B2. Add calibration model on top of raw rule score

Possible simple approaches:

- Platt scaling
- isotonic regression
- bucketed empirical reliability tables

Start simple: reliability buckets are likely enough.

### B3. Add explicit abstain thresholds

At runtime, after scoring the best v3 rule:

- if calibrated confidence >= high threshold -> apply rule directly
- if in middle band -> allow v3 policy boost but keep heuristics in control
- if below low threshold -> abstain and defer to heuristics/v2

### B4. Add route-family-specific thresholds

Do not use one threshold for all routes.

Use stricter activation thresholds for:

- `high_confidence_correction_only`
- sync-planner-enabling rules
- broad retrieval rules

Use looser thresholds for:

- silence rules on low-signal turns
n- well-supported workflow retrieval rules

### B5. Record calibration decisions in route_decisions

Store more proof fields:

- raw v3 score
- calibrated score
- abstained?: boolean
- fallback_source: heuristic | v2 | no_memory_safe

## 4.4 Deliverables

- calibration data extraction path
- calibration metadata in policy snapshot or companion table
- runtime abstain logic in `RouteFn`
- proof/audit visibility for raw vs calibrated score vs fallback decision
- tests for abstention and threshold families

---

## 5. Workstream C — Evaluation, shadowing, and off-policy safety

This is the most important improvement for making confident rollout decisions.

## 5.1 Problem

Tests show correctness of implementation, but not enough of comparative routing value.

We need durable answers to:

- “is v3 better than v2?”
- “is it better for the right classes of turns?”
- “is it only better in toy/obvious cases?”

## 5.2 Goal

Build a repeatable evaluation stack that covers:

- offline replay eval
- shadow live eval
- off-policy sanity checks
- route-family breakouts

## 5.3 Plan

### C1. Build a frozen route-eval corpus

Curate a compact but representative eval set from past route decisions, including:

- missed recall
- correction retrieval success
- workflow retrieval success
- noisy/over-injection
- gratitude / low-signal should-stay-silent
- ambiguous planning turns

Every eval case should have:

- turn frame
- candidate memory summary context
- actual route decision
- outcome/reward
- teacher verdict when available
- human judgment override slot if needed

### C2. Add snapshot-vs-baseline replay evaluation

For a candidate snapshot, replay the eval set and compare:

- v3 snapshot route
- active v2 route
- heuristic route
- actual historical chosen route
- teacher preferred route

Metrics:

- route accuracy vs eval label
- harmful injection rate
- missed useful retrieval rate
- correction recall
- workflow recall
- silence precision
- sync planner overuse rate

### C3. Add shadow-mode logging in production

When v3 is not controlling the turn, still log what it *would* have done.

Store:

- actual chosen route
- v3 proposed route
- v2 proposed route
- heuristic route
- eventual reward/outcome

This gives live disagreement analysis and better off-policy evidence.

### C4. Add off-policy estimators for candidate policies

Start with conservative estimators:

- disagreement buckets
- replay-on-overlap
- doubly robust style approximations later if worth it

Because route learning lives under partial feedback, this is important before letting v3 become more aggressive.

### C5. Add activation policy based on empirical improvement, not just safety

A candidate should not activate only because it is safe enough.

It should also show meaningful gains on at least one of:

- correction recall
- workflow recall
- lower noisy injection rate at same recall
- equal quality at lower sync/latency cost

## 5.4 Deliverables

- route eval corpus artifact
- replay evaluator
- shadow comparison logging
- dashboard / proof route for candidate-vs-active comparison
- activation criteria that require improvement, not mere acceptability

---

## 6. Workstream D — Adaptive hybrid sparse/dense routing

This is the biggest retrieval-quality improvement.

## 6.1 Problem

The current hybrid design is useful but still mostly static.

Different turn shapes need different weighting:

- exact repo/tool/package-manager corrections
- broad semantic planning
- gratitude / acknowledgments
- context-heavy project questions
- preference capture phrasing

## 6.2 Goal

Make the hybrid route scorer query-aware.

## 6.3 Plan

### D1. Add query-feature extraction for routing mode selection

Compute low-cost features such as:

- exact-token density
- acronym/code/jargon density
- repo/tool token presence
- length / verbosity
- semantic breadth estimate
- correction cue intensity
- direct memory reference intensity
- low-signal / acknowledgment probability

### D2. Add routing mode selection

Before scoring prototypes, choose one of:

- sparse-heavy
- dense-heavy
- balanced
- silence-biased
- correction-biased

This can be a deterministic meta-policy at first.

### D3. Add dynamic hybrid scoring weights

Instead of fixed sparse/dense/bonus weights, use weights conditioned on routing mode.

Example:

- correction/jargon-heavy turn -> sparse 0.75 / dense 0.15 / priors 0.10
- broad planning turn -> sparse 0.35 / dense 0.45 / priors 0.20
- gratitude/low-signal -> sparse 0.20 / dense 0.10 / silence prior 0.70

### D4. Add candidate rerank pass before distillation and/or shadow evaluation

Use a lightweight rerank stage over top candidate actions using richer features:

- more exact signal overlap
- route-hint compatibility
- evidence support
- negative evidence penalties
- graph depth reasonableness penalty

Do not put a heavy reranker in the serving loop unless clearly necessary. It can remain an offline/shadow teacher.

### D5. Add retrieval-family diagnostics

Track which scoring lane produced wins:

- sparse-led win
- dense-led win
- prior-led win
- abstain-led win

This will tell us whether the dense lane is actually adding value.

## 6.4 Deliverables

- query-feature extractor
- routing mode selector
- dynamic scoring weights
- hybrid lane diagnostics
- tests for exact-vs-semantic-vs-low-signal behavior

---

## 7. Workstream E — Online learning quality and reward shaping

This is the highest-value learning improvement after evaluation.

## 7.1 Problem

The current learner updates from real outcomes, but reward and pairwise supervision are still fairly crude.

## 7.2 Goal

Make the learner more sample-efficient, less noisy, and more aligned with true route quality.

## 7.3 Plan

### E1. Expand reward decomposition

Current reward components are useful but should become more explicit and stable.

Add or refine:

- correction-prevention gain
- workflow execution gain
- project-context hit gain
- noisy-injection penalty
- over-broad memory-type penalty
- graph overreach penalty
- unnecessary sync penalty
- latency penalty
- “should have stayed silent” gain

### E2. Add ambiguity-aware weighting

Some route outcomes are inherently ambiguous.

If the outcome signal is weak or mixed:

- lower bandit update weight
- lower pair-example margin weight
- avoid producing high-confidence rules from ambiguous examples

### E3. Add teacher confidence and validator confidence as separate signals

Distinguish:

- teacher semantic confidence
- schema/constraint validation confidence
- empirical route family reliability

This helps prevent one overconfident teacher output from dominating learning.

### E4. Improve counterfactual quality controls

Counterfactuals are valuable but noisy.

Add stricter filters:

- require minimum graph evidence
- require route-kind consistency with turn shape
- downweight broad generic counterfactuals
- prevent too many synthetic winner actions from flooding pair examples

### E5. Add stale prototype retirement

Some prototypes will become outdated or low-value.

Retire prototypes that are:

- repeatedly dominated
- unsupported for long periods
- consistently harmful
- never survive distillation

### E6. Add bounded exploration policy for online updates

If future v3 runtime takes more control, introduce explicit bounded exploration:

- only explore within safe route families
- never explore beyond hard safety gates
- log exploration decisions separately

## 7.4 Deliverables

- richer reward component logic
- ambiguity-aware update weighting
- counterfactual quality filters
- prototype retirement pass
- tests for noisy/ambiguous update handling

---

## 8. Workstream F — Serving/runtime hardening

## 8.1 Goal

Keep the runtime cheap, deterministic, and explainable while giving v3 more meaningful authority.

## 8.2 Plan

### F1. Add explicit serving modes

Per-agent or global serving mode:

- `v2_only`
- `v3_shadow`
- `v3_calibrated_assist`
- `v3_primary_with_abstain`

### F2. Improve RouteFn explanation payload

For every route decision, record:

- active serving mode
- v3 raw score
- v3 calibrated score
- v3 abstain reason if any
- winning scorer lane
- fallback path used

### F3. Add policy lineage and rollback tooling

Allow explicit inspection of:

- which frames/pairs/prototypes produced a rule
- which snapshots were rejected and why
- why the active snapshot won over its predecessor

### F4. Add live policy compactness guard

Even if a candidate is safe, do not let it activate if it is too large, too duplicate-heavy, or too unstable relative to prior active policy.

---

## 9. Rollout plan

## Phase 1 — Immediate hardening

Ship first:

1. rule dedup / merge / dominance pruning
2. compactness metrics and gates
3. calibration dataset + abstain threshold wiring
4. route decision proof fields for raw/calibrated/fallback attribution

## Phase 2 — Evaluation infrastructure

5. frozen route-eval corpus
6. replay evaluator
7. shadow logging and disagreement analysis
8. activation criteria requiring actual empirical improvement

## Phase 3 — Adaptive hybrid scoring

9. query-feature extraction
10. dynamic sparse/dense weighting
11. lane diagnostics
12. optional lightweight offline rerank teacher

## Phase 4 — Online learning refinement

13. richer reward shaping
14. ambiguity-aware updates
15. prototype retirement
16. bounded safe exploration

## Phase 5 — Controlled rollout

17. `v3_shadow`
18. `v3_calibrated_assist`
19. `v3_primary_with_abstain` for narrow route families first
20. broader v3 authority only after metrics prove it

---

## 10. Concrete code-level plan

## Files likely to change

### Distillation / learning core
- `packages/openclaw-plugin/src/route-policy-v3.ts`
- `packages/openclaw-plugin/src/route-teacher.ts`
- `packages/openclaw-plugin/src/route-fn.ts`
- `packages/openclaw-plugin/src/memory-types.ts`
- `packages/openclaw-plugin/src/memory-store.ts`
- `packages/openclaw-plugin/src/config.ts`

### Tests
- `packages/openclaw-plugin/test/learning.test.mjs`
- new replay / calibration / compactness tests if split out

### Docs / operator surfaces
- `README.md`
- route-learning docs
- proof / inspect routes if expanded

## New likely helpers/modules
- `route-policy-v3-calibration.ts`
- `route-policy-v3-eval.ts`
- `route-policy-v3-normalize.ts`
- `route-policy-v3-compactness.ts`
- `route-policy-v3-routing-mode.ts`

---

## 11. Success criteria

The hardening work is successful when:

- active v3 snapshots are smaller and less duplicate-heavy
- v3 confidence is calibrated enough to support abstention
- route decisions clearly explain why v3 acted or deferred
- candidate activation requires both safety and measurable quality improvement
- shadow/live evaluation shows specific route families where v3 beats v2/heuristics
- online updates improve route quality without inflating policy size or harm rate

---

## 12. Non-goals

This plan does **not** aim to:

- replace serving with a black-box neural router
- store raw user turns or transcript text
- require ANN/vector infra for route serving
- remove deterministic runtime routing safeguards
- force v3 to decide every turn

---

## 13. Final recommendation

The next best move is:

> **make v3 cleaner, calibrated, and measurable before making it more powerful.**

That means the most important next implementation slice is:

1. dedup/merge/distillation cleanup  
2. calibration + abstention  
3. replay/shadow evaluation  
4. only then adaptive hybrid weighting and stronger online exploration

This keeps the architecture honest:

- richer ML for learning
- deterministic compact policy for serving
- proof surfaces that show whether the learner actually got better

---

## 14. External xAI review notes

I sent a condensed version of this plan through the available xAI-backed path (`x_search`, provider `xai`, model `grok-4-1-fast`) because direct interactive Grok web submission was blocked by browser automation issues during this session.

Main takeaways from the xAI review:

- **The ordering is basically right.** Highest ROI remains:
  1. dedup / canonicalize / dominance-prune rules
  2. calibrate confidence + route-family abstain thresholds
  3. frozen replay eval corpus + shadow logging + off-policy checks
  4. dynamic sparse/dense weighting
  5. reward-shaping / counterfactual / prototype-retirement refinements

- **Add stronger compactness discipline.** xAI explicitly reinforced that duplicate-heavy rules can hide quality problems and make all downstream eval/calibration worse.

- **Use off-policy estimators after shadow logging.** Suggested conservative importance-weighted or doubly-robust style evaluation before increasing learned-policy authority.

- **Track more risks explicitly.** External review highlighted additional concerns worth making first-class:
  - distribution shift / query drift
  - reward gaming from overly simple shaping
  - policy bloat returning after dedup
  - interpretability erosion from over-complex snapshots
  - overfitting to teacher/counterfactual noise
  - cold-start prototypes
  - adversarial query patterns

- **Optional later upgrades, not first slice:**
  - hierarchical prototype lookup
  - stronger contextual bandits like LinUCB / NeuralTS
  - route-family ensemble uncertainty
  - drift-triggered retraining

Conclusion from external review: the plan should stay focused on **clean rules -> calibrated control -> measurable improvement** before making the learner more expressive.
