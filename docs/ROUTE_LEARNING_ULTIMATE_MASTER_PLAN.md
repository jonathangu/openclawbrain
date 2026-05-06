# OpenClawBrain Ultimate Route Learning Master Plan

**Status:** research-backed architecture plan for the next route-learning generation  
**Owner:** GUCLAW / Jonathan  
**Date:** 2026-05-05  
**Supersedes:** `docs/ROUTE_TEACHER_MASTER_PLAN.md` Part 1 + `docs/ROUTE_TEACHER_MASTER_PLAN_PART2.md` only for future route-learning evolution, not for the shipped `route-policy-v2` baseline.  
**Current shipped baseline:** `route-policy-v2` in `0.2.18` remains the safe production default.

## 0. Executive decision

Yes, there is a better ML direction than pure deterministic rule distillation alone.

But the right move is **not**:

- not a pure black-box neural router
- not dense-only embeddings
- not replacing the safety/rule contract with an opaque scorer

The right move is:

> **hybrid route learning** = hard safety gates + hybrid sparse/dense route representations + contextual bandit online learning + distillation back into an auditable compact serving policy.

In plain English:

- let ML learn generalization and similarity
- let online bandits learn from partial real-world feedback
- let deterministic compact policy snapshots remain the production contract

That gives us better recall and adaptation **without giving up fail-closed behavior, auditability, or cheap runtime**.

## 1. Why this plan exists

`route-policy-v2` solved Part 8 correctly for the first production version:

- the learned `route_fn` is stored as a compact JSON snapshot
- route examples are stored separately
- runtime is deterministic and cheap
- policy activation is gated and auditable

That was the right first ship.

But `v2` still learns mostly through **symbolic compression of lessons**. It can miss higher-order similarity like:

- `pnpm install broken`
- `dependency resolution weirdness`
- `workspace package manager mismatch`
- `npm vs pnpm correction`

These are similar situations even when token overlap is weak.

The next generation should learn that latent similarity **while preserving the rule-based runtime contract**.

## 2. Research takeaways from the online pass

### 2.1 Two-tower / dual-encoder systems are the right general pattern

Uber’s production writeup explains why two-tower models are attractive for retrieval/matching: separate query/item towers, cheap similarity scoring, and precomputation on one side for scalable serving.

Relevance here:

- turn/query side = current redacted turn + route frame
- route side = candidate route/action/prototype
- score = dot product or bilinear compatibility

Source:
- `https://www.uber.com/us/en/blog/innovative-recommendation-applications-using-two-tower-embeddings/`

### 2.2 Pure offline supervised routing is the wrong mental model for deployment

Recent routing work argues that routing is an **online partial-feedback** problem, not a full-label supervised classification problem.

Two especially relevant papers:

- **PILOT** learns a shared embedding space for queries and LLMs, then refines routing through contextual bandit feedback under budget constraints.
- **BaRP** explicitly trains routing under the same bandit-feedback restriction seen at deployment and shows gains over offline routers.

Relevance here:

- OpenClawBrain only sees the outcome of the route it actually chose
- we rarely observe full labels for every alternative route
- route learning should therefore treat production updates as contextual bandit learning, not pretend we have oracle labels

Sources:
- `https://aclanthology.org/2025.findings-emnlp.1301/`
- `https://arxiv.org/abs/2510.07429`

### 2.3 Hybrid sparse+dense retrieval keeps winning in production because each lane fixes the other

Hybrid retrieval literature keeps converging on the same practical lesson:

- sparse/lexical signals are better for exactness, rare terms, and interpretability
- dense embeddings are better for semantic generalization
- hybrid systems balance both

That matters here because route decisions depend on both:

- exact symbolic cues: `pnpm`, `build`, `test`, `thanks`, `quote`, repo hints, tool hints
- semantic cues: “this is the kind of turn where memory would help even if tokens differ”

Source:
- `https://mbrenndoerfer.com/writing/hybrid-retrieval-combining-sparse-dense-methods-effective-information-retrieval`

### 2.4 Stronger two-tower systems usually add lightweight cross-lane correction, not just a raw dot product

Recent work like CS3 argues that plain isolated two-tower systems leave performance on the table and benefit from lightweight cross-tower synchronization and downstream knowledge sharing, while preserving low-latency serving.

Relevance here:

- pure `q · r` is a good baseline
- but OpenClawBrain should plan for richer compatibility scoring and distillation feedback from downstream outcomes

Source:
- `https://arxiv.org/abs/2604.19269`

### 2.5 Distilling a stronger learner back into interpretable rules is a real pattern

Recent distillation work reinforces the core idea that you can train a stronger teacher and then compress it into an interpretable student/rule system.

Relevance here:

- learned scorer = teacher
- compact route policy snapshot = interpretable student
- activation requires fidelity and safety gates

Sources:
- `https://arxiv.org/abs/2507.07848v1`
- `https://openreview.net/forum?id=qy024FMO1L`

## 3. Core architectural decision

The future route stack should have **four layers**.

```text
Layer 0: Hard gates and invariants
Layer 1: Hybrid route candidate scorer (sparse + dense)
Layer 2: Contextual bandit online updater
Layer 3: Distilled compact serving policy snapshot
```

### 3.1 Layer 0 — Hard gates and invariants

These stay non-negotiable and code-enforced:

- no raw transcript storage
- redaction before storage/model use
- sync LLM planner budget limits
- bounded graph depth
- safe default silence / no-memory fallback
- policy validation and fail-closed activation
- runtime must still function when learning components are stale or unavailable

### 3.2 Layer 1 — Hybrid route candidate scorer

This is the ML generalization layer.

It scores candidate route actions using both:

- **sparse symbolic features**: task type, turn signals, repo hints, project hints, tool hints, exact tokens, safety flags
- **dense learned features**: compact dense embeddings of the redacted turn frame and candidate route prototype

### 3.3 Layer 2 — Contextual bandit updater

This is the online adaptation layer.

It learns from:

- whether retrieval helped
- whether memory was noisy
- whether a correction was prevented or surfaced faster
- latency/cost penalties
- whether the user effectively accepted or rejected the route outcome

This layer should assume **partial feedback only**.

### 3.4 Layer 3 — Distilled compact policy snapshot

This remains the production runtime contract.

The scorer and bandit are allowed to be richer internally, but the production-default serving artifact remains:

- deterministic
- versioned
- auditable
- bounded
- rollbackable

## 4. What should actually be learned

We should not treat route learning as a single giant multiclass label.

Instead, learn a **factorized route action**:

1. **route kind**
   - `no_memory`
   - `capture_only`
   - `retrieve_memory`
   - `retrieve_and_distill`
   - `high_confidence_correction_only`
2. **memory type set**
   - correction / preference / workflow / project_fact / tool_convention / etc.
3. **graph depth**
   - `0 | 1 | 2`
4. **sync planner mode**
   - `no | never_unless_ambiguous | allowed | prefer`
5. **query template family**
   - symbolic template choices, not free-form user text

This factorization matters because the combinatorial action space is otherwise too sparse and sample-inefficient.

## 5. Serving architecture recommendation

## 5.1 Important practicality note

OpenClawBrain does **not** have a billion-route catalog.

So we do **not** need ANN or a big vector index for route serving at first.

The candidate set is small enough to score exactly.

That means the right architecture is:

```text
exact candidate enumeration
  -> hard-gate filter
  -> hybrid scorer
  -> bandit adjustment / uncertainty penalty
  -> conservative winner selection
  -> record decision + reward later
```

### 5.2 Candidate set

Candidate actions come from three sources:

1. hand-coded safe baseline actions
2. distilled policy snapshot rules
3. learned route prototypes

### 5.3 Final runtime score

For candidate action `a` and current turn frame `x`:

```text
score(a, x) =
  gate(a, x)
  + λ_rule * rule_match(a, x)
  + λ_sparse * sparse_match(a, x)
  + λ_dense * dense_compat(a, x)
  + λ_bandit * bandit_bonus(a, x)
  - λ_cost * estimated_cost(a)
  - λ_risk * risk_penalty(a, x)
```

Where:

- `rule_match` = current compact policy signal
- `sparse_match` = exact/interpretable feature overlap
- `dense_compat` = bilinear or dot-product learned compatibility
- `bandit_bonus` = exploration/exploitation estimate
- `risk_penalty` = conservative damping for noisy/unsafe/high-uncertainty actions

### 5.4 Dense score form

Start simple:

```text
dense_compat(a, x) = q(x)^T W r(a)
```

Where:

- `q(x)` = turn/query embedding
- `r(a)` = route/action embedding
- `W` = learned bilinear matrix

Why bilinear over raw dot product?

- raw dot product is fine for a baseline
- bilinear scoring is still cheap
- bilinear scoring can represent asymmetric compatibility better
- it matches your instinct: “learn a two-sided weighted dot product based on the rules”

## 6. Representation design

### 6.1 Query-side representation `q(x)`

Built from redacted structured route frames only.

Inputs:

- task type
- turn signal tokens
- repo/project hints
- tool activity hints
- current route-policy-v2 match info
- recent route outcome summary
- ambiguity flags
- bounded graph summary features
- optional local text embedding of the redacted turn summary, not raw transcript

Recommended final query representation:

```text
q(x) = concat(
  sparse_signal_vector(x),
  dense_turn_embedding(x),
  scalar_context_features(x)
)
```

### 6.2 Route-side representation `r(a)`

Each candidate action should have a stable route prototype.

Inputs:

- route kind
- memory types
- graph depth
- sync planner mode
- query template family
- route prototype tokens
- empirical reward / harm priors

Recommended route representation:

```text
r(a) = concat(
  sparse_route_signature(a),
  dense_route_embedding(a),
  scalar_action_features(a)
)
```

### 6.3 Why hybrid, not dense-only

Dense-only would throw away too much exactness.

We still want exact symbolic wins for things like:

- `pnpm`
- `npm`
- `build`
- `test`
- `quote`
- `thanks`
- repo/package manager/tool conventions

So the route learner should be **hybrid by design**, not hybrid as an afterthought.

## 7. Storage plan

Keep `v2` tables intact. Add new `v3` learning tables beside them.

## 7.1 `route_frames_v3`

Purpose: canonical training/analysis frame for each route decision.

Stores:

- frame id
- agent id
- created at
- route decision id
- redacted turn summary
- task type
- turn signals json
- project/repo/tool hints json
- graph summary json
- policy snapshot id / matched rule id
- chosen action id
- chosen route parts
- outcome summary json
- reward summary json
- payload hash / dedupe hash

No raw user text.

## 7.2 `route_action_prototypes_v3`

Purpose: stable catalog of candidate actions / route prototypes.

Stores:

- action id
- route kind
- memory types json
- graph depth
- sync planner mode
- query template family
- sparse signature json
- dense embedding json/blob
- support / harm priors
- status (`active|shadow|retired`)
- provenance (`handwritten|distilled|learned`)

## 7.3 `route_pair_examples_v3`

Purpose: offline ranking/distillation training pairs.

Stores:

- example id
- frame id
- positive action id
- negative action id
- label source (`teacher|counterfactual|manual|outcome|bandit`)
- margin weight
- evidence ids json

## 7.4 `route_bandit_feedback_v3`

Purpose: online reward log for chosen actions only.

Stores:

- feedback id
- frame id
- chosen action id
- reward scalar
- reward components json
- cost scalar
- latency scalar
- accepted / rejected / ambiguous outcome
- learning bucket flag
- timestamp

## 7.5 `route_bandit_state_v3`

Purpose: persistent online learner state.

Stores:

- learner version
- feature schema version
- global/shared parameters
- per-action parameters
- covariance / precision matrices or equivalent summaries
- exploration coefficient
- last updated at

This lets the learner resume without recomputing everything.

## 7.6 `route_policy_snapshots_v3`

Purpose: next-generation compact serving artifact.

Stores:

- snapshot id
- version = `route-policy-v3`
- status = `candidate|shadow|active|rejected|retired`
- rules json
- action priors json
- confidence calibration json
- eval summary json
- teacher/scorer provenance
- source examples / source prototype ids
- fidelity-to-scorer summary

## 8. Learning pipeline

## 8.1 Offline bootstrap

Bootstrap the learner from existing data:

- `route_training_examples_v2`
- redacted route frames
- teacher critiques
- counterfactual outcomes
- manual eval sets

Build:

- positives: actions that helped, or should have helped
- negatives: actions that harmed, stayed noisy, or clearly lost to a counterfactual
- pairwise preferences instead of brittle one-label classification where possible

## 8.2 Pretraining objective

Train the hybrid scorer to rank better actions above worse ones.

Primary objectives:

- pairwise ranking loss
- contrastive loss over `(frame, action)` pairs
- optional logistic head for `helpful vs harmful`

## 8.3 Online update objective

Use contextual bandit updates after deployment.

Reward should be decomposed, not monolithic.

Example:

```text
reward =
  + retrieval_help_gain
  + correction_prevention_gain
  + accepted_memory_gain
  - noisy_injection_penalty
  - unnecessary_sync_penalty
  - graph_overreach_penalty
  - latency_penalty
```

Keep both:

- scalar reward for learning
- reward components for audit/debugging

## 8.4 Learning bucket

Do not update online from 100% of live traffic at first.

Adopt a small explicit **learning bucket** for:

- exploration
- reward estimation
- safe online adaptation

Outside the learning bucket, production stays greedy/conservative.

## 9. Distillation plan

This is the most important design choice.

The scorer is not the final source of truth.

The compact policy snapshot is.

## 9.1 Distillation target

Distill the richer scorer into a compact student artifact containing:

- ordered rules
- calibrated action priors
- optional ambiguity regions
- explicit suppression rules
- provenance and fidelity metrics

## 9.2 Distillation algorithm

1. sample route frames from recent traffic and eval sets
2. score all candidate actions with the hybrid scorer
3. identify high-margin, stable regions of behavior
4. extract compact rules or prototypes that cover those regions
5. fit confidence / harm priors per rule
6. reject rules with poor fidelity or unsafe breadth

## 9.3 Activation gate

A `route-policy-v3` candidate can activate only if it passes all of:

- schema validation
- privacy validation
- budget validation
- fidelity-to-teacher/scorer threshold
- no-regression eval threshold
- harm/noise threshold
- calibration sanity checks
- rollback compatibility

## 10. Runtime modes

Support four explicit modes.

### Mode A — `v2_only`
Current safe baseline.

### Mode B — `v2 + v3_shadow`
`v2` serves; `v3` scorer runs in shadow and records disagreements.

### Mode C — `v3_distilled_active`
Distilled `route-policy-v3` serves; scorer remains shadow/supporting.

### Mode D — `hybrid_live_ranker_high_ambiguity_only`
Only for carefully bounded ambiguous cases, allow the live scorer to break ties inside a safe candidate set.

Default recommendation: ship in order **A -> B -> C**, and delay **D** until there is strong evidence.

## 11. Evaluation plan

We should evaluate this like a routing system, not just a classifier.

## 11.1 Offline metrics

- route accuracy against adjudicated examples
- pairwise ranking accuracy
- NDCG / MRR over candidate actions
- calibration error
- harmful retrieval rate
- noisy injection rate
- unnecessary sync planner rate
- exact-cue robustness (`pnpm`, repo rules, tool conventions, etc.)

## 11.2 Counterfactual / off-policy metrics

- how often scorer prefers the judged-better route
- policy regret estimates on logged data
- shadow disagreement analysis vs `v2`

## 11.3 Online metrics

- user-visible correction reduction
- memory-help rate
- harmful/noisy injection rate
- latency overhead
- exploration regret inside learning bucket
- activation rollback rate

## 12. Rollout plan

### Phase 0 — Research + schema groundwork
- finalize schema and feature contract
- define reward components
- add immutable logging for `route_frames_v3` and bandit feedback

### Phase 1 — Offline scorer only
- build hybrid scorer from historical data
- no serving impact
- generate shadow explanations and eval reports

### Phase 2 — Shadow scorer in runtime
- serve `v2`
- run hybrid scorer in shadow
- record disagreement, confidence, and counterfactual wins

### Phase 3 — Learning bucket bandit
- small traffic slice only
- online update state enabled
- conservative exploration only among safe candidate actions

### Phase 4 — Distilled `route-policy-v3`
- distill learned behavior into compact snapshot
- activate only after gates pass

### Phase 5 — Optional ambiguity-only live tie-breaker
- only if metrics justify it
- only within a safe candidate set

## 13. Recommended exact implementation order

1. **Add `route_frames_v3` and `route_bandit_feedback_v3`**
   - get the data contract right first
2. **Define candidate action prototypes**
   - explicit route catalog, factorized action space
3. **Build offline hybrid scorer**
   - sparse + dense + bilinear, no serving changes
4. **Add shadow runtime scoring + explanation endpoint**
   - prove whether it is better before rollout
5. **Add bandit learning bucket**
   - safe online adaptation
6. **Add distiller to `route-policy-v3`**
   - compact student artifact
7. **Add activation gates and rollback**
   - only then let it serve

## 14. Non-goals / anti-goals

Do **not** do these:

- do not store raw user text for embedding training
- do not activate dense-only routing as the default
- do not let the live scorer bypass safety gates
- do not make the runtime depend on synchronous extra LLM calls
- do not treat all counterfactuals as equally trustworthy labels
- do not train one giant opaque route classifier and call it done

## 15. Crisp recommendation

If Jonathan asks, “what is the best possible next move?” the answer is:

> Build a **hybrid sparse+dense two-tower route scorer** trained offline on redacted route frames and action prototypes, refine it with a **contextual bandit learning bucket**, and **distill it back into `route-policy-v3` compact snapshots** for production serving.

That is the best blend of:

- better generalization
- online adaptation
- auditability
- low latency
- fail-closed behavior
- operator trust

## 16. Appendix: source list for this plan

- Uber two-tower production writeup: `https://www.uber.com/us/en/blog/innovative-recommendation-applications-using-two-tower-embeddings/`
- PILOT / contextual bandit shared embedding routing: `https://aclanthology.org/2025.findings-emnlp.1301/`
- BaRP / bandit-feedback routing with preference-tunable inference: `https://arxiv.org/abs/2510.07429`
- CS3 / stronger online two-tower recommendation with lightweight synchronization: `https://arxiv.org/abs/2604.19269`
- Hybrid sparse+dense retrieval overview: `https://mbrenndoerfer.com/writing/hybrid-retrieval-combining-sparse-dense-methods-effective-information-retrieval`
- Interpretable policy distillation from stronger experts: `https://arxiv.org/abs/2507.07848v1`
- Neural-to-rule distillation for interpretable rule bases: `https://openreview.net/forum?id=qy024FMO1L`
