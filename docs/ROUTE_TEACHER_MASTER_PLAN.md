# OpenClawBrain Route Teacher Master Plan

**Status:** implementation plan  
**Owner:** GUCLAW / Jonathan  
**Date:** 2026-05-05  
**Core idea:** LLM route teacher → graph-grounded counterfactuals → distilled `route_fn` policy → cheap deterministic runtime.

OpenClawBrain should not call an LLM on every turn just to decide whether memory matters. The right architecture is to use LLMs as an offline/background teacher, then distill their judgments into a compact policy that the runtime `route_fn` can execute cheaply.

> LLM decides semantic meaning. Code enforces trust boundaries. SQLite stores the graph and evidence.

---

## 1. Product target

The full route-learning vision is this:

1. Runtime `route_fn` makes a fast local routing decision.
2. OpenClawBrain records the decision, available memory graph neighborhood, skipped candidates, injected memories, latency tier, and outcome.
3. A background route teacher replays the turn with more context and more time.
4. The teacher critiques the route decision against the memory graph.
5. Counterfactuals estimate what would have happened with no memory, alternate memories, wider graph expansion, correction-only, workflow-only, or stay-silent.
6. OpenClawBrain stores route training examples.
7. A distiller compresses examples into a structured `route-policy-v2` snapshot.
8. Future runtime routing uses that policy with no synchronous LLM call unless the turn is genuinely ambiguous and worth the latency.

The router improves over time because expensive semantic judgment happens after the turn, while the live turn stays fast.

---

## 2. Why current route learning is not enough

Current route learning mostly observes the route that actually happened:

```text
actual route → actual injection → observed outcome → route example → policy text
```

That is useful, but weak. It does not fully answer:

- Was memory needed at all?
- Did we miss a better memory that was available in the graph?
- Was injection noisy even though the final answer succeeded?
- Would a slightly wider graph expansion have found the right rule?
- Should this turn become a stay-silent example?
- Was a sync planner justified, or did policy already know enough?

The upgraded loop learns from:

- actual outcomes
- LLM route critiques
- graph-grounded alternatives
- counterfactual replay
- silence examples
- missed-recall examples
- latency counterfactuals

That gives `route_fn` stronger training signal without making normal turns slower.

---

## 3. Runtime vs background split

### Runtime path

Runtime must stay simple:

```text
turn frame + structured route policy + graph stats → route decision
```

No LLM by default. Use code, cached policy, SQLite, and small budgets.

Runtime outputs:

- `routeDecisionId`
- route action
- retrieval plan
- graph search plan
- memory types requested
- selected memory ids
- skipped candidate ids
- latency tier
- whether sync planner was allowed/used
- injection payload hash

### Background path

Background can be smarter and slower:

```text
route decision + graph neighborhood + outcome → LLM teacher critique → counterfactual rows → distilled policy
```

The teacher may use a stronger local or optional remote model, but its output is never applied directly to runtime. It creates evidence and proposed policy updates. Code validates and activates policy snapshots.

---

## 4. Data model additions

### 4.1 `route_frames`

A redacted, normalized description of the turn before routing.

Fields:

```ts
type RouteFrame = {
  id: string;
  agentId: string;
  sessionKeyHash?: string;
  projectHint?: string;
  repoHint?: string;
  createdAt: string;
  turnHash: string;
  redactedTurnSummary: string;
  taskType: 'coding' | 'writing' | 'ops' | 'search' | 'planning' | 'chat' | 'unknown';
  turnSignals: string[];
  userIntentSignals: string[];
  safetySignals: string[];
  latencyBudgetMs: number;
};
```

Store the frame, not raw user text.

### 4.2 `route_decisions_v2`

The actual runtime decision.

```ts
type RouteDecisionV2 = {
  id: string;
  routeFrameId: string;
  actualRoute: 'no_memory' | 'retrieve_memory' | 'retrieve_and_distill' | 'capture_only' | 'planner';
  retrievalQueries: string[];
  memoryTypes: string[];
  graphDepth: number;
  selectedMemoryIds: string[];
  skippedCandidateIds: string[];
  candidateCount: number;
  injected: boolean;
  syncPlannerAllowed: boolean;
  syncPlannerUsed: boolean;
  latencyTier: 0 | 1 | 2 | 3;
  policySnapshotId?: string;
  reasonCode: string;
  createdAt: string;
};
```

### 4.3 `route_graph_snapshots`

The memory graph neighborhood available at routing time.

```ts
type RouteGraphSnapshot = {
  id: string;
  routeDecisionId: string;
  querySet: string[];
  candidateMemoryIds: string[];
  candidateSummaries: Array<{
    id: string;
    type: string;
    scope: string;
    redactedContent: string;
    score: number;
    freshness: number;
    graphDistance: number;
    linkedMemoryIds: string[];
  }>;
  graphStats: {
    nodeCountSeen: number;
    edgeCountSeen: number;
    maxDepth: number;
  };
};
```

This is what lets the teacher judge missed recall and graph-depth counterfactuals.

### 4.4 `route_teacher_runs`

The background LLM critique.

```ts
type RouteTeacherRun = {
  id: string;
  routeDecisionId: string;
  model: string;
  promptVersion: string;
  inputHash: string;
  outputHash: string;
  verdict: 'correct_route' | 'missed_recall' | 'over_injected' | 'should_stay_silent' | 'wrong_memory_type' | 'latency_waste' | 'unsafe' | 'unknown';
  teacherRoute: string;
  teacherMemoryIds: string[];
  teacherQueries: string[];
  teacherGraphDepth: number;
  syncPlannerWorthIt: boolean;
  confidence: number;
  rationale: string;
  validated: boolean;
  rejectionReason?: string;
  createdAt: string;
};
```

The rationale is redacted and operator-facing. It must not contain raw user text.

### 4.5 `route_counterfactuals`

Each alternative route considered by the teacher or replay engine.

```ts
type RouteCounterfactual = {
  id: string;
  routeTeacherRunId: string;
  routeDecisionId: string;
  kind:
    | 'no_memory'
    | 'actual_injection'
    | 'top_k_alternate'
    | 'broader_graph'
    | 'correction_only'
    | 'workflow_only'
    | 'preference_only'
    | 'context_only'
    | 'stay_silent'
    | 'sync_planner';
  memoryIds: string[];
  memoryTypes: string[];
  graphDepth: number;
  estimatedOutcome: 'likely_helpful' | 'likely_neutral' | 'likely_noise' | 'likely_harmful' | 'likely_missed' | 'unknown';
  confidence: number;
  rationale: string;
};
```

### 4.6 `route_training_examples_v2`

Validated lessons used for policy distillation.

```ts
type RouteTrainingExampleV2 = {
  id: string;
  routeDecisionId: string;
  routeTeacherRunId?: string;
  exampleKind:
    | 'prefer_route'
    | 'avoid_route'
    | 'missed_recall'
    | 'correct_silence'
    | 'avoid_sync_planner'
    | 'prefer_sync_planner'
    | 'prefer_memory_type'
    | 'avoid_memory_type'
    | 'prefer_graph_depth'
    | 'avoid_graph_depth';
  taskType: string;
  turnSignals: string[];
  route: string;
  memoryTypes: string[];
  queryTemplates: string[];
  graphDepth: number;
  confidence: number;
  supportCount: number;
  harmCount: number;
  source: 'actual_outcome' | 'teacher' | 'counterfactual' | 'manual_eval';
  evidenceIds: string[];
  createdAt: string;
};
```

### 4.7 `route_policy_snapshots_v2`

Structured policy loaded by runtime.

```ts
type RoutePolicyV2 = {
  version: 'route-policy-v2';
  id: string;
  createdAt: string;
  status: 'candidate' | 'active' | 'rejected' | 'shadow';
  rules: RoutePolicyRule[];
  globalBudgets: {
    maxSyncPlannerRate: number;
    maxInjectedMemories: number;
    maxInjectedChars: number;
    defaultGraphDepth: number;
  };
  evalSummary?: {
    cases: number;
    wins: number;
    ties: number;
    misses: number;
    noisyInjections: number;
    harms: number;
    p95LatencyMs: number;
  };
};

type RoutePolicyRule = {
  id: string;
  match: {
    taskType?: string;
    turnSignals?: string[];
    projectHint?: string;
    repoHintPresent?: boolean;
    safetySignalsAbsent?: string[];
  };
  route: 'no_memory' | 'retrieve_memory' | 'retrieve_and_distill' | 'capture_only' | 'planner';
  memoryTypes: string[];
  queries: string[];
  graphDepth: number;
  syncPlanner: 'no' | 'never_unless_ambiguous' | 'allowed' | 'prefer';
  confidence: number;
  evidenceIds: string[];
};
```

---

## 5. Route teacher prompt contract

The teacher must be constrained and evidence-grounded.

### Inputs

- redacted route frame
- actual route decision
- graph snapshot
- injected memories
- skipped candidates
- observed outcome
- latency data
- applicable active policy rule

### Questions

The teacher answers:

1. Was memory needed?
2. Was the actual route right?
3. Which memory ids should have been retrieved?
4. Which memory ids should have stayed out?
5. Should it have stayed silent?
6. Was a sync planner worth it?
7. Which route lesson should be learned?
8. Which counterfactuals are most important?

### Output schema

Teacher output must be strict JSON:

```json
{
  "verdict": "missed_recall",
  "teacherRoute": "retrieve_memory",
  "teacherMemoryIds": ["mem_123"],
  "teacherQueries": ["repo package manager", "test workflow"],
  "teacherGraphDepth": 1,
  "syncPlannerWorthIt": false,
  "confidence": 0.86,
  "rationale": "The turn asked to run tests and a repo-scoped package-manager correction was available but skipped.",
  "counterfactuals": [
    {
      "kind": "no_memory",
      "memoryIds": [],
      "estimatedOutcome": "likely_missed",
      "confidence": 0.78,
      "rationale": "Without memory the agent may default to npm."
    },
    {
      "kind": "correction_only",
      "memoryIds": ["mem_123"],
      "estimatedOutcome": "likely_helpful",
      "confidence": 0.9,
      "rationale": "The correction directly applies to test command selection."
    }
  ],
  "lessons": [
    {
      "kind": "prefer_route",
      "taskType": "coding",
      "turnSignals": ["test"],
      "route": "retrieve_memory",
      "memoryTypes": ["correction", "workflow"],
      "queryTemplates": ["repo package manager", "test workflow"],
      "graphDepth": 1,
      "confidence": 0.86
    }
  ]
}
```

### Validation

Code must reject teacher output when:

- memory ids were not in the graph snapshot
- confidence is missing or too low
- rationale contains raw user text
- route is unsupported
- graph depth exceeds config
- sync planner recommendation violates latency budget
- memory type is unknown
- lesson is too broad

---

## 6. Counterfactual replay design

Counterfactuals should be graph-grounded, not fantasy.

For each route decision, evaluate:

### 6.1 No-memory baseline

Question: would the agent likely have succeeded without memory?

Use when:

- actual injection happened
- outcome was success
- need to distinguish true memory win from normal success

Lesson outcomes:

- `actual_injection_helped`
- `injection_was_unnecessary`

### 6.2 Actual injection critique

Question: did the injected memory help, distract, or create risk?

Use always when injection happened.

Lesson outcomes:

- strengthen selected memories
- suppress selected memories
- adjust rule confidence

### 6.3 Missed recall

Question: were better memories available in top-K or graph neighbors?

Use when:

- no memory was injected
- user corrected the agent
- tool failed in a way memory could have helped
- teacher finds a high-confidence candidate

Lesson outcomes:

- prefer route
- prefer memory type
- prefer query template
- increase graph depth

### 6.4 Over-injection

Question: was injected context unnecessary or distracting?

Use when:

- final answer ignored memory
- user said the memory was irrelevant
- memory did not affect tool/action choice
- turn was casual/simple

Lesson outcomes:

- correct silence
- avoid memory type
- lower route confidence

### 6.5 Graph-depth counterfactual

Question: would graph expansion have found the right linked memory?

Use when:

- FTS found a related node, but not the final useful one
- linked workflow/context would have helped

Lesson outcomes:

- prefer graph depth 1/2 for specific turn shapes
- add linked-memory query template

### 6.6 Memory-type counterfactual

Question: did we ask for the wrong memory type?

Use when:

- preference was injected but workflow was needed
- context was injected but correction was needed
- workflow was injected but should stay silent

Lesson outcomes:

- prefer/avoid memory type by task type

### 6.7 Latency counterfactual

Question: was sync planning worth it?

Use when:

- sync planner was used
- sync planner was skipped but route was wrong
- route confidence was borderline

Lesson outcomes:

- avoid sync planner
- prefer sync planner for narrow ambiguous cases

---

## 7. Distillation into structured policy

A background distiller runs periodically or after enough examples accumulate.

### Inputs

- route training examples
- teacher runs
- actual outcomes
- counterfactuals
- current active policy
- dogfood eval cases

### Output

A candidate `route-policy-v2` JSON snapshot.

Example:

```json
{
  "version": "route-policy-v2",
  "rules": [
    {
      "match": {
        "taskType": "coding",
        "turnSignals": ["test", "build", "install", "dependency"]
      },
      "route": "retrieve_memory",
      "memoryTypes": ["correction", "workflow", "tool_convention"],
      "queries": ["repo package manager", "test workflow", "dependency setup"],
      "graphDepth": 1,
      "syncPlanner": "never_unless_ambiguous",
      "confidence": 0.84
    },
    {
      "match": {
        "taskType": "writing"
      },
      "route": "retrieve_memory",
      "memoryTypes": ["preference", "context"],
      "queries": ["writing style preference", "tone preference"],
      "graphDepth": 0,
      "syncPlanner": "no",
      "confidence": 0.8
    },
    {
      "match": {
        "turnSignals": ["thanks", "ok", "sounds good"]
      },
      "route": "no_memory",
      "memoryTypes": [],
      "queries": [],
      "graphDepth": 0,
      "syncPlanner": "no",
      "confidence": 0.9
    }
  ]
}
```

### Activation gates

A policy snapshot can become active only if it passes:

- schema validation
- no unknown memory types
- no unsupported routes
- max sync-planner rate bound
- max injected memories/chars bound
- eval no-harm gate
- noisy-injection rate under threshold
- missed-recall rate not worse than current policy

Candidate policies should run in shadow mode before activation.

---

## 8. Runtime `route_fn` v2 behavior

Runtime should be a deterministic scorer.

Pseudo-flow:

```ts
function routeFnV2(frame, graphStats, activePolicy) {
  const matches = findMatchingRules(frame, activePolicy.rules);
  const scored = scoreRules(matches, frame, graphStats);

  if (scored.best?.confidence >= HIGH_CONFIDENCE) {
    return buildDecisionFromRule(scored.best);
  }

  if (isUnsafeOrCasual(frame)) {
    return noMemory('safe_or_low_value');
  }

  if (scored.best?.confidence >= MEDIUM_CONFIDENCE) {
    return conservativeRetrieve(scored.best);
  }

  if (latencyBudgetAllowsPlanner(frame) && ambiguityIsWorthIt(frame)) {
    return plannerDecision('ambiguous_high_signal');
  }

  return noMemory('low_confidence');
}
```

Important runtime behavior:

- prefer silence when confidence is low
- never exceed injection budget
- never use superseded/harmful memories
- never widen graph beyond policy/config
- do not sync-plan casual/low-value turns
- record skipped candidates for teacher replay

---

## 9. Proof and operator UX

Add proof surfaces that make route learning legible.

### `/plugins/openclawbrain/route-teacher?limit=20`

Shows recent teacher runs:

- actual route
- teacher route
- verdict
- confidence
- lesson count
- validation status

### `/plugins/openclawbrain/route-counterfactuals?decisionId=...`

Shows alternatives considered:

- no memory
- actual injection
- top-K alternate
- graph-depth
- stay-silent
- latency

### `/plugins/openclawbrain/route-policy`

Shows active structured policy:

- active snapshot id
- rule count
- eval summary
- activation reason
- sync planner budget

### `/plugins/openclawbrain/explain-last`

Upgrade to include:

- active policy rule matched
- graph candidates seen
- injected vs skipped memory ids
- teacher critique if already available
- counterfactual summary if already available
- outcome and score changes

---

## 10. Evaluation plan

Route policy should not activate just because the teacher generated it.

### 10.1 Dogfood trace set

Build a redacted route eval set with cases for:

- coding/test/build/package-manager turns
- writing style requests
- project routing requests
- closeout/proof expectations
- explicit corrections
- workflow reuse
- casual turns that should stay silent
- stale/superseded memory conflicts
- missed recall cases
- over-injection cases

### 10.2 Metrics

Track:

- route accuracy
- useful injection rate
- missed recall rate
- correct silence rate
- noisy injection rate
- harmful injection rate
- stale injection rate
- sync planner rate
- p50/p95 route latency
- memory-disabled baseline delta
- active-policy vs candidate-policy delta

### 10.3 Activation gate

Candidate policy becomes active only when:

- harms do not increase
- noisy injection rate improves or stays low
- missed recall improves or stays flat
- sync planner rate stays under budget
- latency stays within budget
- at least one meaningful win exists vs current policy

---

## 11. Implementation phases

### Phase 1 — Route frame and graph snapshot capture

Goal: record enough evidence to teach from.

Tasks:

- Add `route_frames` records.
- Add `route_decisions_v2` or extend current route decision records safely.
- Add `route_graph_snapshots`.
- Record skipped candidate ids, not just injected ids.
- Store latency tier and policy snapshot id.
- Add tests for redaction and no raw user text storage.

Acceptance:

- every route decision can be replayed with redacted frame + graph candidates
- proof shows selected and skipped memory ids
- no raw user text is stored

### Phase 2 — Teacher run schema and validator

Goal: background LLM can critique route decisions, but code keeps control.

Tasks:

- Add strict teacher output schema.
- Add prompt builder with redacted graph snapshot input.
- Add validator for memory ids, routes, memory types, graph depth, confidence, rationale safety.
- Store accepted/rejected teacher runs.
- Add fixture tests for malformed teacher output.

Acceptance:

- valid teacher run stores cleanly
- invalid memory ids are rejected
- too-broad lessons are rejected
- unsafe/raw rationale is rejected

### Phase 3 — Counterfactual generator

Goal: create structured route alternatives.

Tasks:

- Implement no-memory, actual-injection, top-K alternate, broader-graph, memory-type, stay-silent, and latency counterfactuals.
- Ensure counterfactuals only reference graph candidates available at the time.
- Add teacher prompt section for counterfactual scoring.
- Store counterfactual rows.

Acceptance:

- missed recall examples identify available alternate memories
- over-injection examples produce stay-silent lessons
- graph-depth examples explain linked memory wins

### Phase 4 — Training example v2

Goal: convert teacher/counterfactual output into durable route lessons.

Tasks:

- Add `route_training_examples_v2`.
- Map actual outcomes and teacher verdicts into example kinds.
- Add support/harm counters.
- Deduplicate similar examples.
- Add confidence calibration.

Acceptance:

- positive/negative/missed/silence examples are stored separately
- duplicate examples strengthen support count instead of bloating DB
- low-confidence teacher lessons do not affect policy

### Phase 5 — Structured policy distiller

Goal: replace free-text policy snapshots with `route-policy-v2` JSON.

Tasks:

- Build distiller prompt and schema.
- Generate candidate policy from examples.
- Validate rules.
- Store candidate/shadow/active status.
- Add budget and safety gates.

Acceptance:

- candidate policy validates against schema
- rules include match, route, memory types, queries, graph depth, sync planner policy, confidence
- invalid or unsafe policy cannot activate

### Phase 6 — Runtime `route_fn` v2

Goal: execute structured policy cheaply.

Tasks:

- Add policy loader/cache.
- Match route frame to policy rules.
- Score rules deterministically.
- Use policy query templates and memory types.
- Preserve conservative fallback.
- Record active rule id in route decision.

Acceptance:

- high-confidence coding/test rule retrieves correction/workflow memory without LLM
- writing rule retrieves style preference without LLM
- casual rule stays silent
- ambiguous turn can still use bounded planner if allowed

### Phase 7 — Shadow eval and activation

Goal: do not activate worse policy.

Tasks:

- Add dogfood route eval fixture format.
- Add replay runner for current vs candidate policy.
- Add memory-disabled baseline.
- Add report generator.
- Activate candidate only if gates pass.

Acceptance:

- policy activation requires eval summary
- shadow policy can be inspected before activation
- active policy has no worse harm/noise gate

### Phase 8 — Proof UX

Goal: make route learning understandable.

Tasks:

- Add `/route-teacher` route.
- Add `/route-counterfactuals` route.
- Add `/route-policy` route.
- Upgrade `/explain-last`.
- Add compact status stats: teacher runs, candidate policies, active policy id, noisy/missed/silence counts.

Acceptance:

- operator can see why route_fn got better
- operator can inspect actual vs teacher route
- operator can inspect why a policy was activated or rejected

---

## 12. Testing plan

### Unit tests

- redacted route frame creation
- graph snapshot capture
- teacher schema validation
- teacher rejection paths
- counterfactual generation
- training example dedupe
- structured policy validation
- runtime rule matching
- conservative fallbacks
- proof route payloads

### Integration tests

- teach pnpm correction → test turn retrieves correction
- writing preference → writing turn retrieves preference
- casual thanks → no memory
- missed recall replay → teacher creates missed-recall lesson
- noisy injection replay → teacher creates stay-silent lesson
- candidate policy shadow eval → activation gate passes/fails correctly

### Privacy tests

- no raw user text in route frames
- no raw text in teacher input/output storage
- memory ids in teacher output must belong to snapshot
- remote teacher path disabled by default

---

## 13. Rollout plan

### Default behavior

- Runtime policy v1 remains active until v2 has enough examples.
- Teacher runs are off by default until config enables background route teaching.
- Shadow policy can be generated without activation.
- Activation requires eval gate.

### Config shape

```json
{
  "routeLearning": {
    "enabled": true,
    "teacher": {
      "enabled": true,
      "mode": "background",
      "model": "qwen2.5:32b-instruct",
      "maxRunsPerHour": 20
    },
    "counterfactuals": {
      "enabled": true,
      "topK": 5,
      "maxGraphDepth": 2
    },
    "policyV2": {
      "enabled": true,
      "shadowBeforeActivate": true,
      "minExamples": 25,
      "maxSyncPlannerRate": 0.05,
      "maxNoisyInjectionRate": 0.03
    }
  }
}
```

### Safety defaults

- no remote route teacher by default
- no auto-activation until eval gate exists
- no raw transcripts in teacher input
- no teacher output applied directly

---

## 14. Definition of done

This master plan is implemented when:

- [ ] Runtime route decisions capture redacted frames and graph snapshots.
- [ ] Background teacher critiques route decisions using only redacted, graph-grounded inputs.
- [ ] Counterfactuals are stored for no-memory, alternate memories, graph-depth, memory-type, stay-silent, and latency.
- [ ] Teacher output is schema-validated and fail-closed.
- [ ] Route training examples distinguish positive, negative, missed recall, and correct silence.
- [ ] Structured `route-policy-v2` snapshots are generated and validated.
- [ ] Runtime `route_fn` uses structured policy deterministically.
- [ ] Candidate policies run in shadow mode and activate only after eval gates pass.
- [ ] Proof routes explain actual route, teacher route, counterfactuals, and active policy rule.
- [ ] Most turns still avoid synchronous LLM calls.
- [ ] Dogfood eval shows fewer missed recalls, fewer noisy injections, and no harm increase.

---

## 15. Recommended immediate next steps

1. Implement route frame + graph snapshot capture.
2. Add teacher output schema and validator.
3. Add a first background route teacher job for sampled route decisions.
4. Store teacher critiques but do not activate policy from them yet.
5. Add counterfactual rows for no-memory, actual injection, top-K alternate, and stay-silent.
6. Add `/route-teacher` and extend `/explain-last` so the operator can see what the teacher is learning.
7. Only then build `route-policy-v2` distillation.

This keeps the sequence safe: first evidence, then critique, then counterfactuals, then distillation, then runtime activation.
