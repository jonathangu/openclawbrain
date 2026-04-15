# Learned Route Metric Spec and Labeling Plan

Status: draft v1
Owner: GuClaw / Jonathan
Updated: 2026-04-15

## Purpose

Define the metrics OpenClawBrain should actually optimize for, the trace cohorts those metrics should run on, and the first-party labeling plan needed to make learned-route improvement real instead of cosmetic.

This spec exists because several replay surfaces were mixing together:
- retrieval diagnostics
- broad live-history regression checks
- product-purpose economics

Those are all useful, but they are not the same thing.

The new contract is:
- optimize learned routing for incremental utility over `graph_prior_only`
- use broad live-history cohorts as regression guardrails
- keep winner-mode / quality-score style surfaces explicitly diagnostic only

## Product principle

OpenClawBrain should be judged by long-run task economics, not by whether it retrieves more text or wins a diagnostic score table.

In practice that means the primary question is:

> When is learned routing actually worth it versus `graph_prior_only`?

## Metric stack

### Tier 1: primary optimize-over metric

#### 1. Net utility delta vs `graph_prior_only`

For each trace:

`utility = success_proxy - α * prompt_cost - β * latency_cost - γ * tool_failure_cost - δ * regression_penalty`

Primary offline objective:

`mean(utility_learned_route - utility_graph_prior_only)`

Required reporting:
- `better / tied / worse` vs `graph_prior_only`
- net utility delta
- tie-or-better rate
- regression rate

Interpretation:
- this is the main measure of whether learned routing is worth doing at all
- this should be the top-line metric in product-facing replay reporting

### Tier 2: router-quality metrics

#### 2. Memory-needed uplift

Evaluate on a labeled hard set where memory genuinely matters.

Required reporting:
- `better / tied / worse` on memory-needed traces only
- win rate on memory-needed traces
- loss rate on memory-needed traces

Question answered:
- does learned routing help where memory is actually needed?

#### 3. Activation precision

Whenever learned route departs from the prior, how often was that intervention beneficial?

`activation_precision = beneficial_activations / total_nontrivial_activations`

Question answered:
- does the router know when to act?

#### 4. Activation recall

Of the traces where learned routing would have helped, how often did the router actually activate it?

`activation_recall = beneficial_activations_caught / oracle_beneficial_cases`

Question answered:
- is the router missing real opportunities?

### Tier 3: broad live guardrails

#### 5. Broad live regression rate

Run on semantic-rich owned live-history cohorts.

Required reporting:
- `worse / total`
- tie-or-better rate
- no-brain-floor regression count

Question answered:
- does learned routing stay out of the way on broad real traffic?

This is a guardrail, not the main optimization target.

#### 6. Fail-open safety rate

Track degraded-serve behavior when routing or retrieval is incomplete.

Required reporting:
- degraded-but-acceptable rate
- catastrophic regression count
- fail-open rate

Question answered:
- does OCB fail safely when it cannot help?

### Tier 4: economics and efficiency

#### 7. Cost per incremental win

Required reporting:
- extra prompt tokens per win
- extra latency per win
- extra retrieval/tool hops per win

Question answered:
- are learned-route wins cheap enough to keep?

#### 8. Calibration

Required reporting:
- predicted uplift vs realized uplift
- calibration curve or bucketed expected vs observed uplift

Question answered:
- does the router know its own uncertainty?

### Tier 5: diagnostics only

These remain useful, but should never outrank the optimize-over metrics above:
- phrase-hit rate
- compile-ok rate
- `winnerMode`
- top-rank counts
- raw `qualityScore`
- tie-heavy broad replay counts by themselves

## Reporting contract

### Product-facing replay reporting must lead with
1. focus trace cohort name
2. `learned_route` vs `graph_prior_only` better / tied / worse
3. tie-or-better rate
4. regression rate
5. required-context recall delta when available
6. cost / latency delta when available

### Product-facing replay reporting must not lead with
- diagnostic winner-mode counts
- raw top-score counts
- mean quality score without cost context
- broad trace ties as if they were evidence of learned-route strength

### Diagnostic reporting is still allowed
But it must be labeled clearly as internal diagnostics only.

## Trace cohort taxonomy

The system should maintain distinct replay cohorts with different purposes.

### A. Hard memory lane (primary optimize lane)
Use for actual learned-route optimization.

Properties:
- human-rich
- semantically dependent on prior context
- real retrieval value available
- not dominated by system wrappers or operational scaffolding

Primary metrics:
- net utility delta
- memory-needed uplift
- activation precision / recall

### B. Semantic-rich broad live lane (primary guardrail lane)
Use for large real owned traffic where regressions must stay near zero.

Properties:
- owned first-party live-history traces
- filtered to remove continuation-only and wrapper-heavy junk
- still broad and realistic

Primary metrics:
- broad live regression rate
- tie-or-better rate
- recall delta

### C. Operational wrapper lane
Use for abstention / STOP behavior.

Properties:
- continuation prompts
- subagent launch scaffolding
- heartbeat/task wrapper prompts
- task-id/path-heavy operational recovery traces

Primary metrics:
- abstain / prior precision
- unnecessary activation rate
- regression rate

### D. Fail-open lane
Use for degraded retrieval or partial-memory cases.

Primary metrics:
- degraded-but-acceptable rate
- fail-open safety rate
- catastrophic regression count

## Focus order for recurring reporting

Default ordering:
1. semantic-rich owned live-history lane
2. stratified-rich owned live-history lane
3. broader mixed owned live-history lane
4. canonical equivalent-only frozen sets as fallback only

This order is intentionally biased toward owned real traces rather than replay-equivalent fixtures.

## Labeling plan

### Phase 1: small first-party labeled hard set
Target size:
- 100 to 200 traces

Each trace gets the following labels.

#### Core labels
- `memory_needed`: yes / no / unclear
- `wrapper_noise`: yes / no
- `continuation_only`: yes / no
- `operational_recovery`: yes / no
- `human_semantic_task`: yes / no
- `true_regression_if_learned_loses`: yes / no / unclear

#### Outcome labels
- `oracle_best_mode`: `graph_prior_only` / `learned_route` / `tie` / `unclear`
- `oracle_reason`: short free text
- `cost_sensitive`: low / medium / high

#### Retrieval labels
- `expected_context_type`: direct fact / recent work / plan state / correction / artifact pointer / multi-hop
- `retrieval_helped`: yes / no / unclear
- `retrieval_overkill`: yes / no / unclear

### Phase 2: hard negative mining
Prioritize labeling of:
- learned-route losses
- learned-route ties with materially higher context cost
- traces where the broad cohort says learned route is fine but the hard cohort says it is not

These are the most valuable training examples.

## Optimization implications

### What learned route should learn
The model should not simply retrieve more.
It should learn:
- when to retrieve
- when to stop
- when to stay near `graph_prior_only`

This makes learnable `STOP` and `STOP_LOCAL` first-class success behavior, not fallback behavior.

### Feature guidance
The route model should include explicit scalar features for:
- continuation-prompt detection
- heartbeat/system-wrapper detection
- subagent-launch scaffold detection
- task-id/path literal density
- user-message semantic density
- prior-answer insufficiency signals
- recent user-authored semantic carryover
- tool-failure / exec-wrapper density
- retrieval novelty vs prior-only context
- confidence / uncertainty of expected uplift

## Phase-1 thresholds

These are initial operating thresholds, not permanent product guarantees.

### Hard memory lane
- target: net utility delta > 0 vs `graph_prior_only`
- target: better count >= worse count
- stretch: better count materially exceeds worse count

### Semantic-rich broad live lane
- target: regression rate = 0 when possible
- target: tie-or-better rate extremely high
- target: required-context recall delta > 0

### Operational wrapper lane
- target: unnecessary activation rate falls over time
- target: learned-route losses trend toward zero

## What to change in recurring reports

Every recurring replay report should have this top block:
- focus cohort
- trace count
- better / tied / worse vs `graph_prior_only`
- tie-or-better rate
- regression rate
- recall delta
- cost per incremental win if available

The report should then include a smaller diagnostic block.

## Immediate next implementation steps

1. freeze this metric stack in reporting and dashboards
2. create the first `hard-memory-100` labeled trace set
3. add activation precision / recall reporting
4. add cost-per-incremental-win reporting
5. retrain learned route against incremental utility, not raw recall or winner-mode tables

## Decision rule

If learned route cannot produce:
- positive net utility on the hard lane, and
- near-zero regressions on the semantic-rich broad lane,

then it is not ready for broader product claims, even if diagnostic replay scores look good.
