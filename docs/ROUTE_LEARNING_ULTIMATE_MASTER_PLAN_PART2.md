# OpenClawBrain Route Learning Ultimate Master Plan Part 2

**Status:** implemented in `0.2.19`  
**Date:** 2026-05-05  
**Scope:** exact representation, storage, update loop, activation, runtime execution, and proof surfaces for `route-policy-v3`.

Part 1 established the architecture direction. Part 2 locks down the concrete machinery so the learned route function is inspectable, cheap at runtime, and safe to update.

## 1. Core invariant

The learned router is **not** a live embedding model call and **not** opaque free text.

It is a staged dataflow:

```text
route_decisions + route_frames + graph snapshots + outcomes
  -> route_teacher_runs + route_counterfactuals + route_training_examples_v2
  -> route_frames_v3 + route_action_prototypes_v3 + route_pair_examples_v3 + route_bandit_feedback_v3 + route_bandit_state_v3
  -> deterministic distiller
  -> route_policy_snapshots_v3
  -> active route-policy-v3 snapshot
  -> RouteFn deterministic scorer
```

Runtime stays synchronous-LLM-free by default. Learning happens in the background. Serving uses compact JSON snapshots only.

## 2. What is stored

### 2.1 Raw route/runtime evidence

Existing durable evidence still matters:

- `route_decisions`
- `route_frames`
- `route_graph_snapshots`
- `route_teacher_runs`
- `route_counterfactuals`
- `route_training_examples_v2`

These remain the audit trail and the safe baseline loop.

### 2.2 New v3 training/state tables

#### `route_frames_v3`
One row per resolved routed turn used for v3 learning.

Stores:
- redacted turn summary
- task type
- stable turn signals
- project/repo hints
- chosen action id
- chosen route / memory types / graph depth / sync-planner mode
- reward and decomposed reward components
- `policy_snapshot_id` and `policy_rule_id` when applicable
- payload hash for dedupe/audit

This is the compact per-turn supervised/bandit training record.

#### `route_action_prototypes_v3`
Canonical action catalog.

Each row represents a reusable action shape:

```text
(route, memory_types, graph_depth, sync_planner, query_template_family, sparse_signature, dense_embedding)
```

It also stores:
- support prior
- harm prior
- provenance (`handwritten` / `distilled` / `learned`)
- source example ids
- status (`active` / `shadow` / `retired`)

This is the bridge between raw turns and compact serving rules.

#### `route_pair_examples_v3`
Pairwise preference rows.

Each row says one action was better than another for a frame:
- positive action id
- negative action id
- label source (`teacher`, `counterfactual`, `outcome`, `bandit`, `manual`)
- margin weight
- evidence ids

This turns teacher/counterfactual judgments into ranking supervision.

#### `route_bandit_feedback_v3`
Observed partial-feedback reward for the chosen action.

Stores:
- chosen action id
- scalar reward
- reward components
- cost
- latency
- outcome label (`accepted`, `rejected`, `ambiguous`)
- learning bucket flag

This is the online-learning event log.

#### `route_bandit_state_v3`
Compact online learner state per agent.

Stores:
- learner version
- feature schema version
- exploration alpha
- shared weights
- per-action stats: count, reward sum, reward mean, variance, last reward, positive/negative counts

This is appendable/updateable state, not serving logic.

#### `route_policy_snapshots_v3`
The actual learned route function used by runtime when active.

Stores:
- `version = route-policy-v3`
- status: `candidate` / `shadow` / `active` / `rejected`
- ordered rules JSON
- action priors JSON
- global budgets JSON
- eval summary JSON
- source frame ids
- source prototype ids
- provenance metadata (`model`, `prompt_version`, `created_at`)

## 3. Learned route function representation

The runtime snapshot rule is intentionally compact:

```ts
type RoutePolicyRuleV3 = {
  id: string;
  priority?: number;
  actionId: string;
  match: {
    taskType?: TaskType | string;
    turnSignals?: string[];
    projectHint?: string;
    repoHintPresent?: boolean;
    safetySignalsAbsent?: string[];
  };
  route: RouteKind;
  memoryTypes: MemoryType[];
  queries: string[];
  graphDepth: 0 | 1 | 2;
  syncPlanner: 'no' | 'never_unless_ambiguous' | 'allowed' | 'prefer';
  confidence: number;
  evidenceIds: string[];
  priors?: {
    support?: number;
    harm?: number;
    banditMeanReward?: number;
    banditCount?: number;
    pairWinRate?: number;
  };
  reason?: string;
}
```

Important point: embeddings/prototype features help **build** the rule, but the live route function is still an auditable JSON rule set.

## 4. How training data is generated

For each resolved route decision:

1. **Actual decision path** becomes a chosen action prototype candidate.
2. **Teacher critique** proposes a better route/action when warranted.
3. **Counterfactuals** create alternate action candidates.
4. **Outcome sign** creates positive/negative reward.
5. **Negative retrieval outcomes** also create a silence-vs-retrieval preference when justified.
6. **Bandit feedback** updates per-action reward statistics.

So the v3 training corpus is not one table. It is the combination of:
- per-turn frame rows
- reusable action prototypes
- pairwise preferences
- logged bandit rewards
- v2 lessons/evidence for provenance

## 5. How the learned route fn updates

### 5.1 Ingestion

When the route teacher processes a resolved decision, the implementation now also:

- inserts one `route_frames_v3` row
- upserts chosen/teacher/counterfactual action prototypes
- inserts pairwise preference rows
- inserts bandit feedback
- updates `route_bandit_state_v3`

### 5.2 Distillation

`maybeDistillAndStorePolicyV3(...)` runs after ingestion.

It:
- reads recent `route_frames_v3`
- reads recent `route_pair_examples_v3`
- reads active prototypes
- reads bandit state
- computes action priors
- distills high-confidence compact rules
- validates budget/safety gates
- stores a snapshot as `active`, `shadow`, or `rejected`

### 5.3 Activation gates

The v3 snapshot is rejected if it is too broad or too risky.

Current gates include:
- no rules
- unsupported route / bad shape
- retrieval rule too broad
- sync planner budget overflow
- harm rate above configured threshold
- noisy action rate too high
- regression versus previous active snapshot

## 6. Runtime execution

`RouteFn` now prefers:

```text
active route-policy-v3
  else active route-policy-v2
  else legacy heuristic/freetext fallback
```

For v3 runtime scoring:
- task-type match gates first
- signal overlap boosts score
- route-hint compatibility boosts score
- learned priors adjust confidence
- chosen rule id is written back to `route_decisions.policy_rule_id`

That means every learned routing change remains explainable.

## 7. Why embeddings are present but not dominant

The implementation now stores a compact deterministic dense vector on action prototypes.

Use:
- stabilize prototype identity
- improve prototype ranking/distillation
- enable shared-generalization across similar turn shapes

Not used for:
- unbounded ANN serving
- opaque live retrieval-only routing
- bypassing rule validation or runtime gates

This preserves the original invariant:

> **LLM decides semantic meaning. Code enforces trust boundaries. SQLite stores the graph and evidence.**

## 8. Proof / operator surfaces

`/plugins/openclawbrain/route-policy` now exposes both v2 and v3 route policy state, including:
- active v2 snapshot
- active v3 snapshot
- v3 snapshot history summary
- recent v3 route frames
- recent v3 action prototypes

`/plugins/openclawbrain/explain-last` can still trace a route decision back through decision id, policy snapshot id, policy rule id, graph snapshot, teacher run, and counterfactuals.

## 9. Practical rollout interpretation

`0.2.19` ships the full v3 learning/storage/distillation/runtime path.

Operationally:
- `route-policy-v2` remains the safe fallback baseline
- `route-policy-v3` becomes active only when training evidence is good enough to pass gates
- if no active v3 snapshot exists, runtime safely falls back to v2

That gives us a real learned router without making the serving path opaque or brittle.
