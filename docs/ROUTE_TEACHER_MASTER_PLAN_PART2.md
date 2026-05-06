# OpenClawBrain Route Teacher Master Plan Part 2

**Status:** implementation plan + release checklist  
**Owner:** GUCLAW / Jonathan  
**Date:** 2026-05-05  
**Scope:** Part 8 deep implementation: how the learned `route_fn` is represented, stored, updated, validated, activated, and executed cheaply.

Part 1 established the loop: runtime route decision -> graph snapshot -> teacher critique -> counterfactuals -> route examples -> distilled policy. Part 2 makes the last mile precise: the learned route function is not free text and not a hidden model. It is a validated, versioned, auditable JSON policy snapshot stored in SQLite and executed by deterministic TypeScript.

## 1. Core invariant

The learned router is a data structure, not an LLM call.

```text
route_training_examples_v2
  -> deterministic distiller
  -> route_policy_snapshots_v2.rules_json
  -> active route-policy-v2 snapshot
  -> RouteFn deterministic rule scorer
  -> route_decisions.policy_snapshot_id + policy_rule_id
```

The runtime can improve because examples update the active policy snapshot, but a live turn still pays no synchronous teacher/distiller LLM cost.

## 2. Stored representation

### 2.1 `route_training_examples_v2`

This table stores the durable training signal. Each row is a compact lesson derived from actual outcomes, teacher critiques, counterfactuals, or manual eval.

Important fields:

- `example_kind`: prefer/avoid/missed/silence/sync/depth/type lesson.
- `task_type`: coarse task shape used for matching.
- `turn_signals_json`: redacted tokens/signals such as `test`, `build`, `plan`, `thanks`.
- `route`: target route for the lesson.
- `memory_types_json`: memory classes to retrieve or avoid.
- `query_templates_json`: reusable query templates, not raw prompts.
- `graph_depth`: bounded graph expansion request.
- `support_count`, `harm_count`, `confidence`: calibration inputs.
- `evidence_ids_json`: route decisions / teacher runs behind the lesson.

Rows are deduped by lesson shape; repeat evidence strengthens support or harm instead of bloating storage.

### 2.2 `route_policy_snapshots_v2`

This table stores the learned route function snapshot. The live runtime loads only the latest `status='active'` snapshot.

Important fields:

- `version`: always `route-policy-v2`.
- `status`: `candidate`, `shadow`, `active`, or `rejected`.
- `rules_json`: ordered deterministic rules.
- `global_budgets_json`: sync planner / injection / graph budgets.
- `eval_summary_json`: activation gate metrics and reasons.
- `example_ids_json`: training examples used to produce the snapshot.
- `model`, `prompt_version`: provenance for deterministic or LLM-assisted distillation.

### 2.3 Rule shape

A rule is the compact learned route function clause:

```ts
type RoutePolicyRuleV2 = {
  id: string;
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
};
```

Rules are intentionally small. They do not contain raw transcript text. `queries` are templates. `evidenceIds` point back to auditable records.

## 3. Distillation algorithm

The deterministic distiller does four passes.

### Pass A — Canonicalize examples

- Clamp confidence to `[0,1]`.
- Drop low-confidence examples.
- Normalize memory types and query templates.
- Bound graph depth to config max.
- Limit turn signals to stable reusable signal tokens.

### Pass B — Group by semantic route shape

Group key:

```text
example_kind + task_type + route + sorted(memory_types) + graph_depth
```

Within a group:

- `support = sum(support_count)`
- `harm = sum(harm_count)`
- `confidence = max(example confidence) + support bonus - harm penalty`
- `queries = top unique query templates`
- `signals = top unique turn signals`
- `evidenceIds = union evidence ids`

### Pass C — Convert lesson groups to rules

- `prefer_route`, `missed_recall`, `prefer_memory_type`, `prefer_graph_depth` become retrieval rules.
- `correct_silence` becomes a high-confidence `no_memory` rule.
- `avoid_route` can become a conservative `no_memory` rule only when harm dominates support.
- `prefer_sync_planner` can allow sync planner, but only within the global sync budget.
- Broad rules with neither task type nor turn signals are rejected unless they are very high-confidence silence rules.

### Pass D — Validate and gate activation

The generated policy must pass schema and budget validation before it can become `active`.

Validation rejects:

- unsupported policy version/status
- unknown route
- unknown memory type
- graph depth beyond configured counterfactual max depth
- confidence outside `[0,1]`
- retrieval rule with no memory types and no queries
- broad low-evidence rules
- sync planner rules over budget
- noisy/harm rate over threshold
- candidate regression against the previous active snapshot harms/noise summary

## 4. Activation lifecycle

The update lifecycle is explicit:

```text
new examples arrive
  -> distiller builds candidate snapshot
  -> validator attaches evalSummary.activationDecision
  -> if fail: store status='rejected'
  -> if shadowBeforeActivate: store status='shadow'
  -> else: demote old active to shadow, store status='active'
```

Activation is append-only. Old policies remain inspectable. Only one snapshot per agent is active.

## 5. Runtime execution

At runtime, `RouteFn` loads the active v2 snapshot and scores matching rules deterministically:

1. Task type match adds strong score.
2. Signal overlap adds score.
3. Route hint compatibility adds score.
4. Confidence and evidence count calibrate tie-breaks.
5. Unsafe/casual or low-confidence cases prefer silence.
6. Retrieval rules are budget-clamped.
7. The chosen rule id is recorded as `route_decisions.policy_rule_id`.

The runtime decision stores both:

- `policy_snapshot_id`: which learned route function was loaded.
- `policy_rule_id`: which rule actually influenced the decision.

This is the audit link that lets `/explain-last` and `/route-policy` show why the route function changed behavior.

## 6. Operator proof surfaces

- `/plugins/openclawbrain/route-policy` shows active snapshot, snapshot history, eval summary, rule count, example count.
- `/plugins/openclawbrain/route-teacher` shows teacher critiques.
- `/plugins/openclawbrain/route-counterfactuals` shows graph-grounded alternate routes.
- `/plugins/openclawbrain/explain-last` now links the last turn to route decision, graph snapshot, active policy rule, teacher critique, and counterfactual summary when available.

## 7. Done criteria for Part 8

- Learned policy is stored as `route-policy-v2` JSON in SQLite, not free text.
- Runtime uses active v2 snapshots without a synchronous LLM.
- Policy update path stores rejected/shadow/active snapshots with gate reasons.
- Route decisions record snapshot id and rule id.
- Tests cover policy storage, representation, activation gates, and runtime scoring.
- Release is packed, published, installed live, and verified through HTTP proof routes.
