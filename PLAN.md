# OpenClawBrain v0.2 — Implementation Plan (LLM-First Architecture)

*The real product. LLM distillers + learned route function + deterministic guardrails.*

---

## Core architecture change

**v0.1:** Deterministic regex + flat file injection.

**v0.2 (old plan):** Deterministic regex + SQLite memory graph + heuristic scoring.

**v0.2 (this plan):** LLM distillers + learned route function + SQLite memory graph + deterministic guardrails.

The deterministic code is NOT responsible for understanding feedback or deciding context relevance. It only validates, redacts, budgets, persists, and audits. The semantic decisions come from small, structured LLM calls.

---

## Mental model

```
Turn comes in
  → LLM distills the turn into a compact TurnFrame
  → learned route fn decides whether memory is needed and what kind
  → broad retrieval gets candidate memories from SQLite/FTS
  → LLM context selector chooses/distills final context
  → prompt receives only the selected distilled context
  → after completion, LLM feedback distiller captures corrections/preferences/workflows/outcomes
  → route learner updates policy from prior successes/failures
```

---

## Source tree

```
packages/openclaw-plugin/src/
  index.ts               # plugin registration and wiring only
  config.ts              # config schema/types/defaults/resolution
  redact.ts              # redaction, hashing, safe snippets
  llm-client.ts          # abstract LLM client interface           [NEW]
  llm-json.ts            # structured JSON LLM calls with validation [NEW]
  feedback-distiller.ts  # LLM feedback distillation               [NEW, replaces capture.ts]
  turn-distiller.ts      # LLM turn frame extraction               [NEW]
  route-fn.ts            # learned LLM route function               [NEW, replaces policy.ts]
  context-selector.ts    # LLM selects + distills candidate memories [NEW]
  route-learning.ts      # background route policy improvement      [NEW]
  memory-store.ts        # schema, migrations, CRUD, FTS, proof, stats
  graph.ts               # edge creation, traversal, contradiction logic
  injection.ts           # budget enforcement, format, record injection
  search.ts              # OpenClaw memory supplement integration
  status.ts              # status payloads
  routes.ts              # HTTP route handlers and safe serialization
  memory-types.ts        # shared TS interfaces and enums
  sqlite-driver.ts       # tiny adapter around better-sqlite3
```

Key changes from v0.2 original:
- **Removed:** `capture.ts` (replaced by `feedback-distiller.ts`)
- **Removed:** `policy.ts` (replaced by `route-fn.ts`)
- **Added:** `llm-client.ts`, `llm-json.ts`, `feedback-distiller.ts`, `turn-distiller.ts`, `route-fn.ts`, `context-selector.ts`, `route-learning.ts`

---

## Module responsibilities

| File | Responsibility | Should not do |
|---|---|---|
| `index.ts` | Register hooks, services, routes, supplements | LLM calls, SQL, ranking |
| `llm-client.ts` | Abstract LLM client interface + adapters | Store, injection logic |
| `llm-json.ts` | Structured JSON calls with schema validation, retries | Business logic |
| `feedback-distiller.ts` | LLM distills feedback from turns and runs | SQL, injection |
| `turn-distiller.ts` | LLM extracts TurnFrame from user message | SQL, capture |
| `route-fn.ts` | LLM decides route, retrieval plan, injection plan | SQL |
| `context-selector.ts` | LLM selects and distills candidate memories | Capture, learning |
| `route-learning.ts` | Background learner updates route policy from outcomes | Prompt formatting |
| `memory-store.ts` | All SQL, migrations, transactions, FTS, persistence | LLM calls |
| `graph.ts` | Contradiction/supersession/related-edge logic | LLM calls |
| `injection.ts` | Budget enforcement, format, record injection | Semantic decisions |
| `search.ts` | OpenClaw memory supplement integration | Injection policy |
| `routes.ts` | HTTP route handlers | Core algorithms |

**Key design rule:** No file other than `memory-store.ts` should call `db.prepare()`. No file other than `llm-json.ts` should call the LLM. The LLM is a semantic engine. The code is the trust boundary.

---

## LLM abstraction layer

### `llm-client.ts` — abstract interface

```typescript
export interface DistillerModelClient {
  generateJson(input: {
    model: string;
    system: string;
    user: string;
    schema: JsonSchema;
    timeoutMs: number;
  }): Promise<string>;
}
```

Adapters:
- `OpenClawModelClient` — uses OpenClaw's model infrastructure
- `OpenAICompatibleClient` — direct OpenAI-compatible API
- `LocalModelClient` — local model (Ollama, etc.)
- `TestFakeModelClient` — for testing

Do NOT bake one provider directly into the memory store or route function.

### `llm-json.ts` — structured JSON calls

```typescript
export async function callJsonModel<T>({
  client,
  model,
  system,
  user,
  schema,
  timeoutMs,
  retries = 1,
}: {
  client: DistillerModelClient;
  model: string;
  system: string;
  user: string;
  schema: JsonSchema;
  timeoutMs: number;
  retries?: number;
}): Promise<T> {
  const raw = await client.generateJson({ model, system, user, schema, timeoutMs });
  const parsed = JSON.parse(raw);
  const validation = validateJsonSchema(schema, parsed);

  if (!validation.ok) {
    if (retries > 0) {
      return callJsonModel({
        client, model, system,
        user: repairPrompt(user, raw, validation.errors),
        schema, timeoutMs,
        retries: retries - 1,
      });
    }
    throw new Error(`LLM JSON validation failed: ${validation.errors.join('; ')}`);
  }

  return parsed as T;
}
```

Do not let random model text touch the store.

---

## The two LLM jobs

### Job 1: Feedback distillation

**Question it answers:** "What did this turn teach us?"

Runs in two modes:

**Immediate (during `before_prompt_build`):**
- Detects user corrections in the current message
- Captures preferences, standing instructions
- Runs BEFORE retrieval so the new memory can even be injected into the same turn

**Post-run (during `agent_end`):**
- Captures workflow patterns
- Observes tool success/failure
- Resolves whether injected memory helped
- Detects whether assistant complied with injected correction

**Feedback distiller schema:**

```typescript
export interface FeedbackDistillation {
  version: 1;
  shouldStore: boolean;
  confidence: number;
  feedbackType: 'correction' | 'preference' | 'standing_instruction' |
                'workflow' | 'context' | 'outcome' | 'none';
  memoryCandidates: Array<{
    type: 'correction' | 'preference' | 'workflow' | 'context';
    distilledText: string;
    subject: string;
    scope: 'global_user' | 'agent' | 'repo' | 'project' | 'session' | 'tool' | 'unknown';
    positive?: string;
    negative?: string;
    normalizedKey: string;
    tags: string[];
    confidence: number;
    importanceHint: number;
    retention: 'durable' | 'medium_term' | 'short_term' | 'ephemeral';
    contradictions: Array<{
      existingMemoryId?: string;
      reason: string;
      action: 'supersede_existing' | 'merge' | 'keep_both';
    }>;
  }>;
  injectionFeedback: Array<{
    injectionId: string;
    memoryId: string;
    outcome: 'helped' | 'ignored' | 'assistant_failed_to_use' |
             'user_corrected' | 'harmful' | 'unknown';
    confidence: number;
    evidence: string;
  }>;
  workflowCandidates: Array<{
    distilledWorkflow: string;
    prerequisites: string[];
    steps: string[];
    successSignal: string;
    failureSignal?: string;
    confidence: number;
  }>;
  audit: {
    modelReasonCode: 'explicit_user_correction' | 'explicit_user_preference' |
                     'implicit_outcome' | 'tool_success' | 'tool_failure' |
                     'no_durable_signal';
    storeRawTranscript: false;
    redactionNeeded: boolean;
  };
}
```

**What the LLM returns vs. what gets stored:**

| LLM returns | Gets stored |
|---|---|
| "User corrected: use pnpm instead of npm for this repo" | "Use pnpm instead of npm for this repo." |
| Distilled correction with positive/negative | Correction node with positive="pnpm", negative="npm" |
| Injection feedback: "assistant failed to use" | injection.outcome = 'corrected' |
| Workflow: "build then test then pack" | Workflow node with steps |
| Raw transcript | NEVER stored |

### Job 2: Context route decision

**Question it answers:** "For this turn, should memory be used? If yes, what kind and which?"

This is the "distilling LLM at context turn decision time." The route function does not just ask "does this look like coding?" It produces a structured semantic frame and uses that to decide recall.

---

## The learned route function

```typescript
type LearnedRouteFn = (input: RouteInput) => Promise<RouteDecision>;

export interface RouteInput {
  agentId: string;
  sessionId?: string;
  turnId?: string;
  latestUserMessage: string;
  recentMessages: DistilledMessage[];
  recentInjections: RecentInjectionSummary[];
  routePolicySnapshot: RoutePolicySnapshot;
  nearestRouteExamples: RouteExample[];
  config: {
    mode: 'conservative' | 'balanced' | 'aggressive';
    maxCandidateMemories: number;
    maxInjectedMemories: number;
    maxInjectedChars: number;
  };
}

export interface RouteDecision {
  route: 'no_memory' | 'capture_only' | 'retrieve_memory' |
         'retrieve_and_distill' | 'high_confidence_correction_only';
  confidence: number;
  turnFrame: {
    userIntent: string;
    taskType: 'coding' | 'planning' | 'debugging' | 'writing' |
              'preference_update' | 'correction' | 'general_question' | 'other';
    topicKeys: string[];
    entities: string[];
    constraints: string[];
    memoryNeed: 'none' | 'low' | 'medium' | 'high';
    memoryNeedReason: string;
  };
  retrievalPlan: {
    queries: string[];
    memoryTypes: Array<'correction' | 'preference' | 'workflow' | 'context'>;
    requiredTags: string[];
    excludedTags: string[];
    graphDepth: 0 | 1 | 2;
    maxCandidates: number;
  };
  injectionPlan: {
    maxItems: number;
    maxChars: number;
    preferredFormat: 'bullets' | 'rules' | 'workflow_steps' | 'do_dont' | 'none';
  };
  capturePlan: {
    shouldDistillFeedbackNow: boolean;
    likelyFeedbackType?: 'correction' | 'preference' | 'workflow' | 'outcome' | 'none';
  };
}
```

**What makes it "learned":**

The route function is fed:
1. A **policy snapshot** produced by prior learning runs (text-based rules)
2. **Nearest successful/failed route examples** (from SQLite)
3. **Recent injection outcomes** (from SQLite)
4. **Current turn frame** (from turn distiller)

Instead of hardcoding `if (turnType === 'coding') inject tool-guidance`, the LLM route function uses the policy + examples + outcomes to decide.

---

## Two-pass routing

### Pass A: Turn distillation + retrieval plan

**Input:** latest user message, recent message summaries, recent injection outcomes, route policy snapshot, nearest route examples

**Output:** TurnFrame + RetrievalPlan

Then the system retrieves broadly from SQLite/FTS.

### Pass B: Context selection + prompt distillation

**Input:** TurnFrame, candidate memories, recent injection history, prompt budget

**Output:** selected memory IDs, distilled prompt block, omitted memory IDs, reasoning labels, confidence

The route function should NOT be forced to select memories before seeing candidates. First it decides what to look for, then it decides what to inject.

---

## Turn-time flow (`before_prompt_build`)

```typescript
api.on('before_prompt_build', async event => {
  const turnInput = buildRouteInput(event, store);

  // Pass A: LLM distills turn + decides route
  const routeDecision = await routeFn.decide(turnInput);

  await store.recordRouteDecision({
    agentId: turnInput.agentId,
    sessionId: turnInput.sessionId,
    turnId: turnInput.turnId,
    decision: routeDecision,
    phase: 'pre_retrieval',
  });

  // Immediate feedback capture (if flagged)
  if (routeDecision.capturePlan.shouldDistillFeedbackNow) {
    const feedback = await feedbackDistiller.distillUserFeedback({
      latestUserMessage: turnInput.latestUserMessage,
      recentMessages: turnInput.recentMessages,
      recentInjections: turnInput.recentInjections,
      existingSimilarMemories: await store.findSimilar(turnInput.agentId, turnInput.latestUserMessage),
    });
    await store.applyFeedbackDistillation(feedback);
  }

  // If no memory needed, return
  if (routeDecision.route === 'no_memory' || routeDecision.route === 'capture_only') {
    return {};
  }

  // Broad retrieval from SQLite/FTS
  const candidates = await store.retrieveCandidates({
    agentId: turnInput.agentId,
    plan: routeDecision.retrievalPlan,
  });

  // Pass B: LLM selects + distills context
  const contextDecision = await contextSelector.select({
    routeDecision,
    candidates,
    recentInjections: turnInput.recentInjections,
    maxChars: routeDecision.injectionPlan.maxChars,
  });

  await store.recordContextSelection({
    agentId: turnInput.agentId,
    sessionId: turnInput.sessionId,
    turnId: turnInput.turnId,
    routeDecisionId: routeDecision.id,
    selectedMemoryIds: contextDecision.selectedMemoryIds,
    distilledContext: contextDecision.distilledContext,
    confidence: contextDecision.confidence,
  });

  if (!contextDecision.distilledContext) {
    return {};
  }

  return {
    prependContext: contextDecision.distilledContext,
  };
});
```

**Important:** The LLM decides whether context is needed and what shape it should take. The deterministic code only enforces the budget and writes the audit trail.

---

## Feedback capture flow

### Immediate (`before_prompt_build`)

For things that affect the current turn:
- "Actually, use pnpm instead of npm"
- "From now on, give me file-by-file plans"
- "No, I prefer X"

The route LLM can flag `capturePlan.shouldDistillFeedbackNow = true`. Then the feedback distiller runs.

If the feedback is a high-confidence correction, store it immediately before retrieval, so it can even be injected into the SAME turn.

### Post-run (`agent_end`)

For things that need the full run context:
- Workflow capture
- Tool success/failure
- Whether injected memory helped
- Whether assistant complied
- Whether user correction was resolved

```typescript
api.on('agent_end', async event => {
  const distillation = await feedbackDistiller.distillRunOutcome({
    agentId: resolveAgentId(event),
    sessionId: event.sessionId,
    turnId: event.turnId,
    finalMessages: event.messages,
    recentInjections: await store.getRecentInjections(event),
    toolEvents: await store.getBufferedToolEvents(event),
  });

  await store.applyFeedbackDistillation(distillation);
  await learning.observeDistillation(distillation);

  return {};
});
```

---

## Context selector schema

```typescript
export interface ContextSelection {
  shouldInject: boolean;
  confidence: number;
  selectedMemoryIds: string[];
  distilledContext: string;
  selected: Array<{
    memoryId: string;
    reason: 'directly_relevant_correction' | 'matching_user_preference' |
            'repo_workflow' | 'tool_guidance' | 'contradiction_resolution' |
            'supporting_context';
    useHow: 'must_follow' | 'prefer' | 'consider' | 'avoid';
    confidence: number;
  }>;
  omitted: Array<{
    memoryId: string;
    reason: 'irrelevant' | 'too_general' | 'superseded' | 'low_confidence' |
            'would_pollute_prompt' | 'budget';
  }>;
  audit: {
    promptBudgetUsedChars: number;
    risk: 'low' | 'medium' | 'high';
  };
}
```

**Context selector prompt (simplified):**
- Inject only memories likely to change assistant behavior
- Prefer corrections over preferences
- Prefer repo/project-specific over global generic
- Do not inject superseded memories
- Do not inject memories merely because they share keywords
- Distill selected memories into a small prompt block
- Keep context operational, not explanatory

**Output prompt block example:**
```
<openclawbrain-memory>
- Must follow: Use pnpm instead of npm for this repo.
- User preference: For implementation feedback, give file-by-file details.
</openclawbrain-memory>
```

The context selector is therefore not "search result ranking." It is a **semantic compression step**.

---

## Learned route function in practice

### Route decisions table

```sql
CREATE TABLE route_decisions (
  id TEXT PRIMARY KEY,
  agent_id TEXT NOT NULL,
  session_id TEXT,
  turn_id TEXT,
  route TEXT NOT NULL,
  confidence REAL NOT NULL,
  turn_frame_json TEXT NOT NULL,
  retrieval_plan_json TEXT NOT NULL,
  injection_plan_json TEXT NOT NULL,
  selected_memory_ids_json TEXT NOT NULL DEFAULT '[]',
  omitted_memory_ids_json TEXT NOT NULL DEFAULT '[]',
  model TEXT NOT NULL,
  prompt_version TEXT NOT NULL,
  policy_snapshot_id TEXT,
  outcome TEXT DEFAULT 'pending',
  reward REAL DEFAULT 0,
  created_at TEXT NOT NULL,
  resolved_at TEXT
);
```

### Route examples table

```sql
CREATE TABLE route_examples (
  id TEXT PRIMARY KEY,
  agent_id TEXT NOT NULL,
  turn_frame_json TEXT NOT NULL,
  route_decision_json TEXT NOT NULL,
  outcome TEXT NOT NULL,
  reward REAL NOT NULL,
  lesson TEXT NOT NULL,
  tags TEXT NOT NULL DEFAULT '[]',
  created_at TEXT NOT NULL
);
```

### Route policy snapshots table

```sql
CREATE TABLE route_policy_snapshots (
  id TEXT PRIMARY KEY,
  agent_id TEXT NOT NULL,
  policy_text TEXT NOT NULL,
  examples_json TEXT NOT NULL DEFAULT '[]',
  created_at TEXT NOT NULL,
  active INTEGER NOT NULL DEFAULT 0
);
```

### Route outcomes

Each route decision gets an outcome later:
- `accepted` — no correction after injection
- `corrected` — user corrected after injection
- `irrelevant_context` — context didn't matter
- `helpful_context` — context demonstrably helped
- `tool_success` — tools succeeded after injection
- `tool_failure` — tools failed after injection
- `no_signal` — couldn't determine

---

## Route learning loop

The background learner periodically asks an LLM:

**"Given recent successful and failed route decisions, update the route policy."**

Input:
- Top positive route examples
- Top negative route examples
- Recent corrections after injection
- Memory types that were useful
- Memory types that polluted prompts

Output:
- New route policy snapshot
- New route examples
- Suppression rules
- Boost rules

Example generated policy snapshot:

```
Route policy snapshot v12:

1. For repo implementation-plan discussions, retrieve:
   - correction memories for this repo
   - user planning/style preferences
   - workflow memories involving PLAN.md, VISION.md, package.json
   Avoid generic context memories unless the user asks for background.

2. For short factual questions, route to no_memory unless
   the user references prior preferences.

3. If the latest user message contains a correction, run feedback
   distillation before context selection.

4. When memories conflict, prefer the newest high-confidence
   correction and omit superseded memories.

5. If the user asks for "deep discussion" or "implementation",
   style preferences are usually relevant.
```

That snapshot becomes part of the next route LLM prompt.

**This is the learned route function:**
```
RouteFn(current turn, active route policy, nearest examples, recent outcomes)
  → RouteDecision
```

Not hardcoded rules. Not RL in the strict sense. But it IS learned because its prompt policy and examples are continuously distilled from outcomes.

---

## Deterministic guardrails

The deterministic code handles:

1. **Validation** — LLM JSON output matches schema
2. **Redaction** — all stored content passes through `redactText()` before persistence
3. **Budget** — hard cap on injection characters and count
4. **Persistence** — SQLite CRUD, transactions, FTS index
5. **Audit** — proof events for every capture, injection, and learning pass
6. **Safety** — `rawTranscriptUpload=true` → fail closed. `allowPromptInjection=false` → no injection.

The deterministic code does NOT:
- Parse natural language
- Decide what a correction means
- Decide what context is relevant
- Rank memories by semantic relevance
- Decide whether a turn needs memory

---

## Feedback distiller prompt

```
You are OpenClawBrain's feedback distiller.

Your job is to identify durable feedback from the current turn.
Durable feedback includes:
- corrections from the user
- preferences from the user
- standing instructions
- successful workflows
- negative outcomes after injected memory
- contradictions with existing memory

Do not invent preferences.
Do not store generic conversational content.
Do not treat assistant claims as user preferences.
Do not store raw transcript text.
Return only JSON matching the schema.

Important distinction:
- User says "use X instead of Y" => correction memory.
- User asks "can you use X?" => not durable unless accepted as instruction.
- Assistant says "I'll remember X" => not durable by itself.
- Tool succeeds after a sequence => possible workflow candidate.
- User corrects assistant after a memory was injected => injection feedback.

Existing similar memories:
{{existingSimilarMemories}}

Recent injected memories:
{{recentInjections}}

Current turn:
{{distilledCurrentTurn}}
```

This is where LLMs are much better than regex. The model can understand:

- "Actually, don't do the npm thing here — this repo is pnpm."
- "same as before: check the plan doc first"
- "no, that's not what I meant; I wanted code, not an essay"

Regex will miss or misclassify these. The LLM can distill them into useful memory atoms.

---

## Hooks (unchanged from prior plan)

```typescript
api.on('before_prompt_build', handleBeforePromptBuild);   // route + capture + inject
api.on('after_tool_call', handleAfterToolCall);           // buffer tool events
api.on('agent_end', handleAgentEnd);                      // post-run feedback distillation
api.on('before_compaction', handleBeforeCompaction);      // snapshot state
api.on('llm_output', handleLlmOutput);                    // buffer assistant response
```

Do NOT register `before_agent_reply`. Correction detection runs in `before_prompt_build`.

---

## SQLite schema (additions)

Adding to the prior schema, three new tables for the route function:

```sql
-- Route decisions (per turn)
CREATE TABLE IF NOT EXISTS route_decisions (
  id TEXT PRIMARY KEY,
  agent_id TEXT NOT NULL,
  session_id TEXT,
  turn_id TEXT,
  route TEXT NOT NULL,
  confidence REAL NOT NULL,
  turn_frame_json TEXT NOT NULL,
  retrieval_plan_json TEXT NOT NULL,
  injection_plan_json TEXT NOT NULL,
  selected_memory_ids_json TEXT NOT NULL DEFAULT '[]',
  omitted_memory_ids_json TEXT NOT NULL DEFAULT '[]',
  model TEXT NOT NULL,
  prompt_version TEXT NOT NULL,
  policy_snapshot_id TEXT,
  outcome TEXT DEFAULT 'pending',
  reward REAL DEFAULT 0,
  created_at TEXT NOT NULL,
  resolved_at TEXT
);

-- Route examples (distilled learning)
CREATE TABLE IF NOT EXISTS route_examples (
  id TEXT PRIMARY KEY,
  agent_id TEXT NOT NULL,
  turn_frame_json TEXT NOT NULL,
  route_decision_json TEXT NOT NULL,
  outcome TEXT NOT NULL,
  reward REAL NOT NULL,
  lesson TEXT NOT NULL,
  tags TEXT NOT NULL DEFAULT '[]',
  created_at TEXT NOT NULL
);

-- Route policy snapshots
CREATE TABLE IF NOT EXISTS route_policy_snapshots (
  id TEXT PRIMARY KEY,
  agent_id TEXT NOT NULL,
  policy_text TEXT NOT NULL,
  examples_json TEXT NOT NULL DEFAULT '[]',
  created_at TEXT NOT NULL,
  active INTEGER NOT NULL DEFAULT 0
);
```

The full schema (memory_nodes, memory_edges, capture_candidates, memory_injections, proof_events, learning_runs, FTS5) is unchanged from the prior plan.

---

## Config additions

```typescript
export interface OpenClawBrainConfig {
  // ... existing fields from prior plan ...

  llm: {
    model: string;           // model for distillers (default: use OpenClaw's model)
    timeoutMs: number;       // per LLM call timeout (default: 10000)
    maxRetries: number;      // JSON validation retries (default: 1)
  };

  route: {
    enabled: boolean;        // use learned route function (default: true)
    policyRefreshIntervalMs: number;  // how often to update policy (default: 3600000 = 1 hour)
    maxExamples: number;     // max route examples in context (default: 10)
  };

  // ... rest unchanged ...
}
```

---

## Build phases (revised)

### PR 1 — SQLite store + LLM infrastructure
- `memory-types.ts`
- `sqlite-driver.ts`
- `memory-store.ts` (with route_decisions, route_examples, route_policy_snapshots tables)
- `llm-client.ts` (abstract interface + test fake)
- `llm-json.ts` (structured JSON calls with validation)
- Storage tests + LLM integration tests

### PR 2 — Turn distiller + route function
- `turn-distiller.ts` (LLM extracts TurnFrame)
- `route-fn.ts` (LLM route decision with policy + examples)
- `injection.ts` (budget enforcement + recording)
- Route decision recording
- Route smoke tests

### PR 3 — Feedback distiller (immediate)
- `feedback-distiller.ts` (LLM distills user corrections/preferences)
- Immediate capture in `before_prompt_build`
- Dedup/merge/contradiction handling
- Memory node creation from distilled feedback
- Capture tests

### PR 4 — Context selector
- `context-selector.ts` (LLM selects/distills candidate memories)
- Two-pass routing integration
- Injection formatting and budget enforcement
- Injection tests

### PR 5 — Post-run feedback + route learning
- `agent_end` feedback distillation
- `route-learning.ts` (background policy improvement)
- Route outcome resolution
- Background service registration
- Learning tests

### PR 6 — OpenClaw memory supplements
- `search.ts` (MemoryCorpusSupplement + MemoryPromptSupplement)
- `/search` route
- No double injection

### PR 7 — Self-regulation + release
- Pruning, node cap, edge caps
- Status, graph, learn routes
- Fresh install test
- ClawHub publish v0.2

---

## Test plan (revised)

### LLM infrastructure tests (8)
1. `callJsonModel` validates against schema
2. `callJsonModel` retries on validation failure
3. `callJsonModel` throws after max retries
4. TestFakeModelClient works for all components
5. Redaction applied before LLM returns memory candidates
6. Schema rejects raw transcript content
7. Timeout handled gracefully
8. Repair prompt generates valid retry

### Turn distiller tests (6)
1. Extracts TurnFrame from simple question
2. Identifies correction turn
3. Identifies coding task type
4. Extracts topic keys
5. Memory need assessment matches expected
6. Capture plan flags feedback detection

### Route function tests (8)
1. Routes direct question to no_memory
2. Routes correction to retrieve_and_distill
3. Uses policy snapshot in decision
4. Uses nearest examples in decision
5. Retrieval plan has correct memory types
6. Injection plan respects budget
7. Conservative mode thresholds correct
8. Route decision recorded in DB

### Feedback distiller tests (10)
1. Detects explicit correction
2. Detects explicit preference
3. Detects standing instruction
4. Does NOT store assistant "I'll remember" alone
5. Detects injection feedback (corrected)
6. Detects workflow from tool sequence
7. Distills raw text into operational memory (not raw transcript)
8. Detects contradiction with existing memory
9. Deduplicates with existing similar memory
10. High-confidence correction promotes immediately

### Context selector tests (8)
1. Selects relevant correction over irrelevant preference
2. Omits superseded memory
3. Respects prompt budget
4. Distills selected memories into compact block
5. Format: must_follow vs prefer vs consider
6. Risk assessment for injection
7. Records selection with reasoning
8. Returns empty if no candidates are relevant

### Route learning tests (6)
1. Resolves route outcomes from injection feedback
2. Updates policy snapshot from examples
3. New policy contains learned rules
4. Nearest examples retrieval works
5. Learning run writes proof event
6. Policy snapshot stored and marked active

### Route tests (6)
1. /status shows memory stats + last route decision
2. /proof returns no raw user text
3. /graph returns redacted nodes
4. /search finds known memory
5. /learn returns route learning stats
6. Routes require gateway auth

### Hook registration tests (6)
1. Registers before_prompt_build
2. Registers agent_end
3. Registers learning service
4. Registers corpus supplement
5. Does NOT register before_agent_reply
6. TestFakeModelClient works end-to-end

### End-to-end test (1)
```
Session 1:
  User: Actually use pnpm, not npm.
  → Feedback distiller detects correction
  → Memory node created
  → Proof event written

Session 2:
  User: Install dependencies for this repo.
  → Turn distiller: coding task, topic=repo, memoryNeed=medium
  → Route fn: retrieve_memory
  → Retrieval: finds pnpm correction
  → Context selector: selects, formats as "Must follow: Use pnpm"
  → Injection recorded
  → Assistant uses pnpm

Session 2 next turn:
  User does not correct.
  → agent_end: resolve injection as accepted/useful
  → importance increases
```

**Total: ~59 test cases.**
