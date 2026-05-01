# OpenClawBrain v0.2 Implementation Feedback

**Date:** 2026-05-01  
**Audience:** OpenClawBrain implementer / maintainer  
**Scope:** Cleaned-up implementation feedback for building the “real” OpenClawBrain v0.2: SQLite memory graph, automatic capture, background learning, adaptive injection, and self-regulation.

---

## 0. Executive Summary

The v0.2 plan is strong. The direction is right: replace flat-file prompt context with a local, inspectable, self-regulating memory graph that automatically captures corrections, preferences, workflows, and useful context, then injects only the few memories that matter for the current turn.

The most important implementation recommendation is this:

> Build OpenClawBrain v0.2 as an **event-driven local memory runtime**, not merely as “context-files.ts replaced by SQLite.”

That runtime should have three separate planes:

1. **Evidence plane** — what happened.
   - Hook observations.
   - Capture candidates.
   - Injection decisions.
   - Outcome signals.
   - Proof events.

2. **Memory plane** — what the system currently believes.
   - Memory nodes.
   - Memory edges.
   - Confidence, importance, freshness, supersession, and redacted content.

3. **Recall plane** — what the model sees or can search.
   - Ranked retrieval.
   - Bounded prompt context augmentation.
   - Native memory corpus supplement.
   - Search route.

This separation matters because auto-capture will make mistakes. If the system stores only final memory nodes, it becomes hard to debug bad behavior. If it stores evidence, candidates, injections, and outcomes separately, you can inspect exactly why a memory exists and why it was injected.

The two highest-value changes I would make to the current plan are:

1. **Use additive OpenClaw memory supplements by default.**
   - Prefer `registerMemoryCorpusSupplement` and `registerMemoryPromptSupplement`.
   - Keep `registerMemoryCapability` as an optional exclusive mode only.
   - The vision says OpenClawBrain is not a replacement for OpenClaw’s memory engine, so defaulting to the additive path better matches the product.

2. **Add a `memory_injections` table immediately.**
   - The learning engine needs injection-level outcomes.
   - `use_count` and `useful_count` on memory nodes are not enough.
   - Each injection should become its own event with query, rank, score, turn/session/run IDs, and eventual outcome.

If those two changes are made, the core v0.2 success case becomes real rather than demo-only:

```text
User: Actually use pnpm, not npm.

System:
1. Detects a high-confidence correction candidate.
2. Redacts and stores it as a memory node.
3. Creates proof evidence.
4. On the next relevant turn, retrieves and injects it.
5. Records that injection.
6. Observes whether the next response succeeds or gets corrected.
7. Boosts, decays, supersedes, or prunes the memory over time.
```

---

## 1. Current Plan Assessment

The existing plan correctly identifies the main v0.2 components:

- `src/memory-store.ts` for SQLite-backed nodes, edges, proof events, and FTS5 search.
- `src/capture.ts` for automatic detection of corrections, preferences, and workflows.
- `src/learning.ts` for scoring, pruning, and link building.
- `src/search.ts` for OpenClaw memory/search integration.
- `src/graph.ts` for graph queries and traversal.
- `src/injection.ts` replacing flat-file context injection.
- Extended `index.ts`, `policy.ts`, `config.ts`, `status.ts`, and `proof-store.ts`.

The product requirements are also right:

- Corrections must stick automatically.
- The agent should learn from outcomes.
- Memory should be a graph, not a flat file.
- Prompts should remain small.
- The system should self-regulate.
- The user should be able to inspect what was captured and why.
- The system should remain local and redact before persistence.

The plan’s biggest risk is not technical feasibility. The risk is **over-capturing and over-injecting**. A memory system that remembers too much or injects too aggressively will become worse than no memory at all. That means the implementation should optimize for conservative capture, conservative injection, auditable proof, and outcome-based learning.

---

## 2. Implementation Principles

### 2.1 Do not store raw transcript text

This should remain a hard invariant:

```ts
rawTranscriptStored: false
rawUserTextStored: false
redactionApplied: true
```

Everything persisted as memory content should pass through redaction first. Proof events should store decisions and hashes, not raw conversation fragments.

### 2.2 Capture candidates before promoting durable memories

Do not let every regex match immediately become a high-authority memory. Store capture candidates separately, then promote them when confidence is high or supporting evidence appears.

A correction like:

```text
Actually use pnpm, not npm.
```

can be promoted immediately because it is explicit and structured.

A softer statement like:

```text
I usually use pnpm for this stuff.
```

should probably become a lower-confidence candidate or preference, not a strong correction.

### 2.3 Treat assistant output as weak evidence

Assistant-generated text such as:

```text
I’ll remember to use pnpm going forward.
```

should not, by itself, create durable memory. It can support a memory candidate that came from the user, but it is not authoritative. The user is the authority for preferences and corrections.

### 2.4 Make learning deterministic first

Do not implement RL for v0.2. The plan’s heuristic scoring is the right baseline.

A good v0.2 learning loop is:

```text
observe injection → observe outcome → update score → decay stale memory → prune weak memory → build/adjust links
```

The logic should be explainable in proof events and route responses.

### 2.5 Keep prompts small and boring

Memory injection should look like concise operational guidance, not a log dump.

Good:

```text
Relevant memory:
- Correction: Use pnpm instead of npm for this repo.
- Workflow: For OpenClawBrain plugin changes, run build, tests, and pack before release.
```

Bad:

```text
Here are 30 prior events, tool traces, dates, confidence scores, and conversations...
```

The model should see only the distilled content. The user/debugger should see proof and scores through routes.

---

## 3. Recommended Architecture

### 3.1 Proposed source tree

```text
packages/openclaw-plugin/src/
  index.ts                 # plugin registration and wiring only
  config.ts                # config schema/types/defaults/resolution
  redact.ts                # redaction, hashing, safe snippets
  policy.ts                # turn classification and injection gating

  memory-types.ts          # shared TS interfaces and enums
  sqlite-driver.ts         # tiny adapter around better-sqlite3 or node:sqlite
  memory-store.ts          # schema, migrations, CRUD, FTS, proof, stats
  capture.ts               # candidate extraction and promotion logic
  injection.ts             # search/rank/format/record injection
  learning.ts              # scoring, outcome resolution, pruning, linking
  graph.ts                 # edge creation, traversal, contradiction logic
  search.ts                # OpenClaw memory supplement integration
  status.ts                # status payloads
  routes.ts                # /status /proof /graph /learn /search
```

The current plan lists five new files. I would add `memory-types.ts`, `sqlite-driver.ts`, and `routes.ts` because they keep the rest of the implementation cleaner.

### 3.2 Module responsibilities

| File | Responsibility | Should not do |
|---|---|---|
| `index.ts` | Register hooks, services, routes, supplements | SQL, ranking, capture regexes |
| `memory-store.ts` | All SQL, migrations, transactions, FTS, persistence | Hook-specific logic |
| `capture.ts` | Extract and normalize candidates from events | Prompt injection |
| `injection.ts` | Select and format memories for the current turn | Capture or learning |
| `learning.ts` | Resolve outcomes, update scores, prune, link | Prompt formatting |
| `graph.ts` | Contradiction/supersession/related-edge logic | SQL details if avoidable |
| `search.ts` | Expose memory as OpenClaw search/prompt supplement | Replace all injection policy |
| `policy.ts` | Decide whether a turn deserves memory lookup | Database access |
| `routes.ts` | HTTP route handlers and safe serialization | Core memory algorithms |

---

## 4. OpenClaw API Integration

### 4.1 Use `before_prompt_build` for prompt mutation

OpenClaw’s agent-loop documentation says `before_prompt_build` runs after session load and can inject `prependContext`, `systemPrompt`, `prependSystemContext`, or `appendSystemContext`. That is the right hook for adaptive memory injection.

Use it for:

- Detecting explicit user memory requests in the latest user turn.
- Resolving delayed outcomes from prior injections.
- Searching memory for the current turn.
- Returning bounded `prependContext` when relevant.

### 4.2 Do not make `before_agent_reply` claim the turn

OpenClaw’s docs describe `before_agent_reply` as a hook that can claim the turn and return a synthetic reply or silence the turn. That is not what OpenClawBrain usually wants.

For v0.2, `before_agent_reply` should either be unused or strictly observational. It should not claim turns unless you add a very explicit future feature.

Recommended approach:

```ts
api.on('before_prompt_build', handleBeforePromptBuild);
api.on('after_tool_call', handleAfterToolCall);
api.on('agent_end', handleAgentEnd);
api.on('before_compaction', handleBeforeCompaction);
api.on('llm_output', handleLlmOutput); // only if this hook is stable in target runtime
```

### 4.3 Use additive memory surfaces by default

OpenClaw’s SDK separates additive memory-adjacent APIs from exclusive memory plugin APIs.

Default path:

```ts
api.registerMemoryCorpusSupplement?.({ ... });
api.registerMemoryPromptSupplement?.({ ... });
```

Optional exclusive mode:

```ts
api.registerMemoryCapability?.({ ... });
```

I would put exclusive memory capability behind config:

```ts
memoryIntegration: 'supplement' | 'exclusive'
```

Default:

```ts
memoryIntegration: 'supplement'
```

Reason: the vision says OpenClawBrain plugs into OpenClaw’s memory capabilities and is not a replacement for OpenClaw’s memory engine.

### 4.4 Use `registerService` for the learning loop

The learning loop belongs in a background service. Keep it stoppable and idempotent.

```ts
api.registerService?.({
  id: 'openclawbrain-learning',
  start: async () => learning.start(),
  stop: async () => learning.stop(),
});
```

Do not let the service keep a transaction open across ticks. Each learning pass should open short transactions and return a report.

### 4.5 Routes must declare auth explicitly

OpenClaw plugin routes require explicit `auth`. Use `auth: 'gateway'` for inspection routes.

Recommended routes:

```text
GET  /plugins/openclawbrain/status
GET  /plugins/openclawbrain/proof?limit=20
GET  /plugins/openclawbrain/graph?agentId=main&limit=50
GET  /plugins/openclawbrain/search?agentId=main&q=pnpm
GET  /plugins/openclawbrain/learn?agentId=main
POST /plugins/openclawbrain/learn/run-once?agentId=main
```

Avoid unauthenticated mutation routes. If future routes allow deleting, editing, pinning, or exporting memory, require a clear admin-level auth story.

---

## 5. SQLite Store Design

### 5.1 Use SQLite, but isolate the driver

The plan chooses `better-sqlite3`, which is reasonable for a local plugin:

- local file
- synchronous operations
- fast reads/writes
- transactions are simple
- easy backup/inspection

However, put it behind a tiny adapter so you can later switch to Node’s built-in `node:sqlite` or another implementation.

```ts
// sqlite-driver.ts
export interface SqliteDriver {
  prepare<T = unknown>(sql: string): PreparedStatement<T>;
  exec(sql: string): void;
  pragma(sql: string, options?: { simple?: boolean }): unknown;
  transaction<TArgs extends unknown[], TResult>(
    fn: (...args: TArgs) => TResult,
  ): (...args: TArgs) => TResult;
  close(): void;
}
```

Important: if using `better-sqlite3`, transaction callbacks must remain synchronous. Do not use `async` transaction functions.

### 5.2 Database pragmas

At open time:

```ts
db.pragma('journal_mode = WAL');
db.pragma('foreign_keys = ON');
db.pragma('busy_timeout = 5000');
```

Also consider:

```ts
db.pragma('synchronous = NORMAL');
```

WAL mode is useful for concurrent reads while writes occur. Keep writes short.

### 5.3 Use integer `rowid` plus public UUID `id`

Do not use UUID strings as the FTS row ID. Use SQLite integer row IDs for FTS and a separate string `id` for public memory IDs.

```sql
rowid INTEGER PRIMARY KEY,
id TEXT NOT NULL UNIQUE
```

FTS5 external-content tables work most cleanly with integer `rowid` joins.

### 5.4 Recommended schema

This expands the current plan’s schema with:

- explicit capture candidates
- injection events
- learning runs
- soft deletes
- supersession
- stable IDs
- FTS-compatible integer row IDs

```sql
CREATE TABLE IF NOT EXISTS memory_nodes (
  rowid INTEGER PRIMARY KEY,
  id TEXT NOT NULL UNIQUE,

  agent_id TEXT NOT NULL,
  type TEXT NOT NULL CHECK (
    type IN ('correction', 'preference', 'workflow', 'context', 'tool_result')
  ),

  content TEXT NOT NULL,
  positive TEXT,
  negative TEXT,

  confidence REAL NOT NULL DEFAULT 0.5,
  importance REAL NOT NULL DEFAULT 0.3,
  freshness REAL NOT NULL DEFAULT 1.0,

  use_count INTEGER NOT NULL DEFAULT 0,
  useful_count INTEGER NOT NULL DEFAULT 0,
  capture_count INTEGER NOT NULL DEFAULT 1,

  tags_json TEXT NOT NULL DEFAULT '[]',
  topic_key TEXT,
  metadata_json TEXT NOT NULL DEFAULT '{}',

  source_hook TEXT,
  source_hash TEXT NOT NULL,
  source_turn_id TEXT,
  source_session_id TEXT,
  source_run_id TEXT,

  origin TEXT NOT NULL DEFAULT 'auto' CHECK (
    origin IN ('auto', 'explicit', 'derived', 'seeded')
  ),

  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  last_seen_at TEXT NOT NULL,
  last_used_at TEXT,

  superseded_by TEXT,
  deleted_at TEXT,

  redaction_applied INTEGER NOT NULL DEFAULT 1,

  UNIQUE(agent_id, type, source_hash)
);

CREATE INDEX IF NOT EXISTS idx_memory_nodes_agent_active
  ON memory_nodes(agent_id, type, importance DESC, freshness DESC)
  WHERE deleted_at IS NULL AND superseded_by IS NULL;

CREATE INDEX IF NOT EXISTS idx_memory_nodes_topic
  ON memory_nodes(agent_id, topic_key)
  WHERE deleted_at IS NULL;

CREATE TABLE IF NOT EXISTS memory_edges (
  id TEXT PRIMARY KEY,
  agent_id TEXT NOT NULL,

  from_id TEXT NOT NULL,
  to_id TEXT NOT NULL,

  relation TEXT NOT NULL CHECK (
    relation IN (
      'related',
      'contradicts',
      'supersedes',
      'supports',
      'extends',
      'used_with',
      'supports_workflow'
    )
  ),

  weight REAL NOT NULL DEFAULT 0.5,
  evidence_count INTEGER NOT NULL DEFAULT 1,

  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,

  UNIQUE(agent_id, from_id, to_id, relation)
);

CREATE INDEX IF NOT EXISTS idx_memory_edges_from
  ON memory_edges(agent_id, from_id, relation, weight DESC);

CREATE INDEX IF NOT EXISTS idx_memory_edges_to
  ON memory_edges(agent_id, to_id, relation, weight DESC);

CREATE TABLE IF NOT EXISTS capture_candidates (
  id TEXT PRIMARY KEY,
  agent_id TEXT NOT NULL,

  type TEXT NOT NULL,
  content TEXT NOT NULL,
  positive TEXT,
  negative TEXT,

  confidence REAL NOT NULL,
  tags_json TEXT NOT NULL DEFAULT '[]',
  topic_key TEXT,
  metadata_json TEXT NOT NULL DEFAULT '{}',

  source_hook TEXT NOT NULL,
  source_hash TEXT NOT NULL,
  source_turn_id TEXT,
  source_session_id TEXT,
  source_run_id TEXT,

  status TEXT NOT NULL DEFAULT 'pending' CHECK (
    status IN ('pending', 'promoted', 'rejected', 'merged')
  ),

  promoted_memory_id TEXT,
  rejection_reason TEXT,

  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_capture_candidates_pending
  ON capture_candidates(agent_id, status, confidence DESC);

CREATE TABLE IF NOT EXISTS memory_injections (
  id TEXT PRIMARY KEY,
  agent_id TEXT NOT NULL,

  memory_id TEXT NOT NULL,
  run_id TEXT,
  turn_id TEXT,
  session_id TEXT,

  query TEXT NOT NULL,
  turn_slice TEXT,
  rank INTEGER NOT NULL,
  score REAL NOT NULL,

  injected_at TEXT NOT NULL,
  resolved_at TEXT,

  outcome TEXT NOT NULL DEFAULT 'pending' CHECK (
    outcome IN (
      'pending',
      'accepted',
      'useful',
      'corrected',
      'ignored',
      'tool_success',
      'tool_failure',
      'unknown'
    )
  ),

  correction_signal TEXT,
  metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_memory_injections_pending
  ON memory_injections(agent_id, outcome, injected_at);

CREATE INDEX IF NOT EXISTS idx_memory_injections_memory
  ON memory_injections(agent_id, memory_id, injected_at DESC);

CREATE TABLE IF NOT EXISTS proof_events (
  id TEXT PRIMARY KEY,
  agent_id TEXT NOT NULL,

  event_type TEXT NOT NULL,
  created_at TEXT NOT NULL,

  source_hook TEXT,
  turn_id TEXT,
  session_id TEXT,
  run_id TEXT,

  memory_id TEXT,
  candidate_id TEXT,
  injection_id TEXT,

  decision TEXT,
  reason TEXT,

  raw_transcript_stored INTEGER NOT NULL DEFAULT 0,
  raw_user_text_stored INTEGER NOT NULL DEFAULT 0,

  payload_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_proof_events_recent
  ON proof_events(agent_id, created_at DESC);

CREATE TABLE IF NOT EXISTS learning_runs (
  id TEXT PRIMARY KEY,
  agent_id TEXT NOT NULL,

  started_at TEXT NOT NULL,
  finished_at TEXT,
  duration_ms INTEGER,

  scores_updated INTEGER NOT NULL DEFAULT 0,
  outcomes_resolved INTEGER NOT NULL DEFAULT 0,
  edges_created INTEGER NOT NULL DEFAULT 0,
  nodes_pruned INTEGER NOT NULL DEFAULT 0,
  errors_json TEXT NOT NULL DEFAULT '[]'
);
```

### 5.5 FTS5 table and triggers

Use an external-content FTS5 table:

```sql
CREATE VIRTUAL TABLE IF NOT EXISTS memory_search
USING fts5(
  content,
  tags,
  topic_key,
  content='memory_nodes',
  content_rowid='rowid',
  tokenize='porter unicode61'
);
```

Maintain it with triggers:

```sql
CREATE TRIGGER IF NOT EXISTS memory_nodes_ai
AFTER INSERT ON memory_nodes
WHEN new.deleted_at IS NULL
BEGIN
  INSERT INTO memory_search(rowid, content, tags, topic_key)
  VALUES (
    new.rowid,
    new.content,
    new.tags_json,
    COALESCE(new.topic_key, '')
  );
END;

CREATE TRIGGER IF NOT EXISTS memory_nodes_ad
AFTER DELETE ON memory_nodes
BEGIN
  INSERT INTO memory_search(memory_search, rowid, content, tags, topic_key)
  VALUES (
    'delete',
    old.rowid,
    old.content,
    old.tags_json,
    COALESCE(old.topic_key, '')
  );
END;

CREATE TRIGGER IF NOT EXISTS memory_nodes_au
AFTER UPDATE ON memory_nodes
BEGIN
  INSERT INTO memory_search(memory_search, rowid, content, tags, topic_key)
  VALUES (
    'delete',
    old.rowid,
    old.content,
    old.tags_json,
    COALESCE(old.topic_key, '')
  );

  INSERT INTO memory_search(rowid, content, tags, topic_key)
  SELECT
    new.rowid,
    new.content,
    new.tags_json,
    COALESCE(new.topic_key, '')
  WHERE new.deleted_at IS NULL AND new.superseded_by IS NULL;
END;
```

SQLite FTS5 external-content tables can become inconsistent if the content table and FTS index are not kept aligned. Triggers are the simplest way to enforce alignment.

### 5.6 Migrations

Use `PRAGMA user_version`.

```ts
export function migrate(db: SqliteDriver): void {
  const version = db.pragma('user_version', { simple: true }) as number;

  if (version < 1) {
    db.transaction(() => {
      db.exec(schemaV1);
      db.pragma('user_version = 1');
    })();
  }
}
```

For every migration:

- Run inside a transaction.
- Be idempotent where possible.
- Add a test that opens an old schema and migrates forward.
- Never wipe memory on migration failure.

---

## 6. Type Contracts

Create `memory-types.ts` so all modules agree on shape.

```ts
export type MemoryType =
  | 'correction'
  | 'preference'
  | 'workflow'
  | 'context'
  | 'tool_result';

export type EdgeRelation =
  | 'related'
  | 'contradicts'
  | 'supersedes'
  | 'supports'
  | 'extends'
  | 'used_with'
  | 'supports_workflow';

export interface MemoryNode {
  id: string;
  rowid?: number;
  agentId: string;
  type: MemoryType;
  content: string;
  positive?: string | null;
  negative?: string | null;
  confidence: number;
  importance: number;
  freshness: number;
  useCount: number;
  usefulCount: number;
  captureCount: number;
  tags: string[];
  topicKey?: string | null;
  metadata: Record<string, unknown>;
  sourceHook?: string | null;
  sourceHash: string;
  sourceTurnId?: string | null;
  sourceSessionId?: string | null;
  sourceRunId?: string | null;
  origin: 'auto' | 'explicit' | 'derived' | 'seeded';
  createdAt: string;
  updatedAt: string;
  lastSeenAt: string;
  lastUsedAt?: string | null;
  supersededBy?: string | null;
  deletedAt?: string | null;
  redactionApplied: boolean;
}

export interface CaptureCandidate {
  id?: string;
  agentId: string;
  type: MemoryType;
  content: string;
  positive?: string;
  negative?: string;
  confidence: number;
  tags: string[];
  topicKey?: string;
  metadata: Record<string, unknown>;
  source: {
    hook: string;
    turnId?: string;
    sessionId?: string;
    runId?: string;
    pattern?: string;
  };
  sourceHash: string;
}

export interface SearchQuery {
  agentId: string;
  query: string;
  tags?: string[];
  types?: MemoryType[];
  limit?: number;
  includeSuperseded?: boolean;
  includeDeleted?: boolean;
}

export interface MemorySearchResult extends MemoryNode {
  textRank: number;
  relevanceScore: number;
  graphBoost: number;
  finalScore: number;
}

export interface InjectionDecision {
  action: 'stay_silent' | 'proof_only' | 'inject';
  reason: string;
  turnSlice: string;
  query: string;
  selected: MemorySearchResult[];
  injectionText: string;
  budgetChars: number;
}
```

---

## 7. Memory Store API

`memory-store.ts` should be the only file that knows SQL.

Recommended interface:

```ts
export interface MemoryStore {
  migrate(): void;
  close(): void;

  insertMemory(input: InsertMemoryInput): MemoryNode;
  upsertMemory(input: InsertMemoryInput): MemoryNode;
  getMemory(agentId: string, id: string): MemoryNode | null;
  listMemories(query: ListMemoriesQuery): MemoryNode[];
  searchMemories(query: SearchQuery): MemorySearchResult[];
  updateMemory(agentId: string, id: string, patch: MemoryPatch): MemoryNode | null;
  softDeleteMemory(agentId: string, id: string, reason: string): void;

  insertCandidate(input: CaptureCandidate): StoredCaptureCandidate;
  promoteCandidate(candidateId: string, memoryId: string): void;
  rejectCandidate(candidateId: string, reason: string): void;
  listPendingCandidates(agentId: string, limit?: number): StoredCaptureCandidate[];

  insertEdge(input: InsertEdgeInput): MemoryEdge;
  upsertEdge(input: InsertEdgeInput): MemoryEdge;
  listEdges(agentId: string, memoryId: string): MemoryEdge[];
  listGraph(agentId: string, limit: number): MemoryGraphSnapshot;

  recordInjection(input: RecordInjectionInput): MemoryInjection;
  resolveInjection(input: ResolveInjectionInput): void;
  listPendingInjections(agentId: string, opts?: PendingInjectionOptions): MemoryInjection[];

  insertProofEvent(input: ProofEventInput): void;
  listProofEvents(agentId: string, limit: number): ProofEvent[];

  recordLearningRun(input: LearningRunInput): void;
  getStats(agentId: string): MemoryStats;
  prune(agentId: string, opts: PruneOptions): PruneResult;
}
```

The key design rule:

> No other file should call `db.prepare()`.

This prevents capture, learning, injection, and routes from growing their own inconsistent database behavior.

---

## 8. Capture Engine

### 8.1 Capture should run in stages

Recommended capture pipeline:

```text
event → extract candidate → redact → hash → dedupe/merge → contradiction check → insert/promote → proof event
```

Pseudocode:

```ts
export async function observeUserTurn(event: PromptBuildEvent, deps: Deps): Promise<void> {
  const agentId = resolveAgentId(event, deps.config);
  const latestUserText = extractLatestUserText(event);
  if (!latestUserText) return;

  const redacted = redactText(latestUserText);

  const candidates = [
    ...detectCorrections(redacted, event),
    ...detectPreferences(redacted, event),
    ...detectExplicitMemoryRequests(redacted, event),
  ];

  for (const candidate of candidates) {
    await storeCaptureCandidateOrPromote(candidate, deps);
  }
}
```

### 8.2 Correction detection

High-confidence correction patterns:

```ts
const correctionPatterns = [
  {
    name: 'use_x_instead_of_y',
    regex: /\b(?:actually,?\s*)?use\s+(.+?)\s+(?:instead of|not)\s+(.+?)(?:[.!?]|$)/i,
    confidence: 0.9,
  },
  {
    name: 'dont_use_y_use_x',
    regex: /\b(?:don't|do not)\s+use\s+(.+?)[,;]\s*use\s+(.+?)(?:[.!?]|$)/i,
    confidence: 0.9,
  },
  {
    name: 'prefer_x_over_y',
    regex: /\bprefer\s+(.+?)\s+(?:over|to)\s+(.+?)(?:[.!?]|$)/i,
    confidence: 0.82,
  },
  {
    name: 'correct_way_is_x',
    regex: /\bthe correct\s+(?:way|approach|command|tool)\s+is\s+(.+?)(?:[.!?]|$)/i,
    confidence: 0.75,
  },
];
```

For the pattern:

```text
use pnpm instead of npm
```

Normalize to:

```ts
{
  type: 'correction',
  content: 'Use pnpm instead of npm.',
  positive: 'pnpm',
  negative: 'npm',
  confidence: 0.9,
  tags: ['package-manager'],
  topicKey: 'package-manager'
}
```

### 8.3 Preference detection

Preference patterns:

```ts
const preferencePatterns = [
  {
    name: 'always_x',
    regex: /\b(?:always|from now on)\s+(.+?)(?:[.!?]|$)/i,
    confidence: 0.72,
  },
  {
    name: 'i_prefer_x',
    regex: /\bI\s+prefer\s+(.+?)(?:[.!?]|$)/i,
    confidence: 0.68,
  },
  {
    name: 'my_timezone_is_x',
    regex: /\bmy\s+timezone\s+is\s+([A-Za-z_\/+-]+)(?:[.!?]|$)/i,
    confidence: 0.85,
  },
];
```

Preferences should start lower than corrections unless explicitly stated.

Recommended base confidence:

```text
explicit correction: 0.85-0.95
explicit preference: 0.65-0.85
explicit remember request: 0.70-0.90
inferred workflow: 0.45-0.70
inferred context: 0.35-0.60
```

### 8.4 Explicit memory requests

Patterns:

```ts
const explicitMemoryPatterns = [
  /\bremember\s+(?:that\s+)?(.+?)(?:[.!?]|$)/i,
  /\bdon't\s+forget\s+(.+?)(?:[.!?]|$)/i,
  /\bnote\s*:\s*(.+?)(?:[.!?]|$)/i,
  /\bfor\s+future\s+reference[:,]?\s*(.+?)(?:[.!?]|$)/i,
];
```

These should create either `preference` or `context` depending on content.

### 8.5 Workflow capture

Workflow capture should primarily come from tool observations plus outcome signals.

Good workflow memory:

```text
For OpenClawBrain plugin changes, run `npm test` after build and before package release.
```

Bad workflow memory:

```text
Tool call returned 2700 lines of logs...
```

Workflow capture source signals:

- `after_tool_call` saw a successful tool result.
- A sequence of tools completed without error.
- `agent_end` produced a successful final reply.
- User confirmed success.
- The same tool pattern recurred across sessions.

Initial v0.2 workflow capture can be conservative:

```ts
if (toolSequence.hasBuild && toolSequence.hasTest && finalReplyLooksSuccessful) {
  createWorkflowCandidate(...);
}
```

Do not persist raw tool output. Store a distilled workflow description, tool names, and outcome metadata.

### 8.6 Candidate promotion

Recommended logic:

```ts
function shouldPromote(candidate: CaptureCandidate): boolean {
  if (candidate.type === 'correction' && candidate.confidence >= 0.8) return true;
  if (candidate.type === 'preference' && candidate.confidence >= 0.75) return true;
  if (candidate.type === 'context' && candidate.confidence >= 0.8) return true;
  return false;
}
```

Lower-confidence candidates can stay in `capture_candidates` until:

- repeated evidence appears,
- user explicitly confirms,
- assistant behavior succeeds repeatedly after candidate injection,
- or a background learning pass rejects/prunes them.

### 8.7 Deduplication

Compute source hash from normalized redacted content, not raw input:

```ts
const sourceHash = hashText([
  candidate.agentId,
  candidate.type,
  normalizeForHash(candidate.content),
].join('\n'));
```

If duplicate:

```ts
capture_count += 1
last_seen_at = now
confidence = min(1, confidence + 0.03)
```

Do not create another node.

---

## 9. Contradiction and Supersession

### 9.1 Explicit corrections are easiest

If old memory is:

```text
Use pnpm instead of npm.
positive = pnpm
negative = npm
```

and new memory is:

```text
Use npm instead of pnpm.
positive = npm
negative = pnpm
```

Then the system can deterministically detect contradiction:

```ts
function isDirectContradiction(a: MemoryNode, b: MemoryNode): boolean {
  return Boolean(
    a.positive &&
    a.negative &&
    b.positive &&
    b.negative &&
    normalize(a.positive) === normalize(b.negative) &&
    normalize(a.negative) === normalize(b.positive)
  );
}
```

### 9.2 Supersede instead of delete

When a new correction supersedes an old one:

```sql
UPDATE memory_nodes
SET
  superseded_by = :newId,
  importance = MIN(importance, 0.1),
  updated_at = :now
WHERE id = :oldId;
```

Then insert an edge:

```sql
INSERT INTO memory_edges (
  id, agent_id, from_id, to_id, relation, weight, evidence_count, created_at, updated_at
)
VALUES (
  :edgeId, :agentId, :newId, :oldId, 'supersedes', 1.0, 1, :now, :now
);
```

Do not hard-delete superseded memories. They are valuable for auditability.

### 9.3 Topic keys help contradiction lookup

Use normalized topic keys for common memory domains:

```text
package-manager
timezone
test-command
build-command
repo-tooling
communication-style
answer-format
release-workflow
```

A correction can search first by `topic_key`, then by FTS.

---

## 10. Injection Engine

### 10.1 Overall algorithm

```text
1. Resolve agent ID and config.
2. Fail closed if disabled or unsafe config.
3. Classify the turn.
4. If the turn does not need memory, stay silent and log proof.
5. Build a redacted query summary.
6. Search FTS for seed memories.
7. Expand graph only for boosts/support, not prompt flooding.
8. Rerank candidates.
9. Apply mode thresholds.
10. Fit selected memories into character budget.
11. Format bounded prompt section.
12. Record each injection in memory_injections.
13. Write proof event.
14. Return prependContext.
```

### 10.2 Turn classification still matters

Do not delete the existing policy idea. The old v0.1 policy is valuable: direct-answer turns should stay silent.

Recommended behavior:

| Turn slice | Memory action |
|---|---|
| `direct-answer` | stay silent |
| `continuation` | low aggression |
| `correction-follow-up` | search corrections first |
| `retrieval-heavy` | search context/preferences |
| `tool-heavy` | search workflows/corrections |
| `stale-memory-conflict` | search corrections and supersession edges |

### 10.3 Candidate search

First-stage FTS query:

```sql
SELECT
  n.*,
  s.rank AS text_rank
FROM memory_search s
JOIN memory_nodes n ON n.rowid = s.rowid
WHERE
  memory_search MATCH :query
  AND n.agent_id = :agentId
  AND n.deleted_at IS NULL
  AND n.superseded_by IS NULL
ORDER BY s.rank ASC
LIMIT :limit;
```

Then rerank in TypeScript.

### 10.4 Ranking formula

Use:

```ts
finalScore =
  relevanceScore *
  importanceFactor *
  freshnessFactor *
  confidenceFactor *
  typeBoost *
  sliceBoost *
  graphBoost *
  safetyPenalty;
```

Where:

```ts
const typeBoost = {
  correction: 1.35,
  preference: 1.0,
  workflow: 0.95,
  context: 0.8,
  tool_result: 0.6,
}[node.type];
```

Corrections should usually outrank preferences because they prevent repeated user frustration.

### 10.5 Freshness formula

Use exponential decay, not linear decay:

```ts
function freshnessScore(node: MemoryNode, now: Date, halfLifeDays: number): number {
  const anchor = node.lastUsedAt ?? node.lastSeenAt ?? node.createdAt;
  const ageDays = daysBetween(new Date(anchor), now);
  return Math.exp(-ageDays / halfLifeDays);
}
```

Suggested half-lives:

| Type | Half-life |
|---|---:|
| correction | 180 days |
| preference | 90 days |
| workflow | 45 days |
| context | 30 days |
| tool_result | 14 days |

Rationale:

- Corrections are often durable.
- Preferences change less frequently than workflows.
- Workflows and tool details decay faster.
- Tool results should usually be short-lived unless promoted into workflow/context.

### 10.6 Mode thresholds

Recommended thresholds:

```ts
const thresholds = {
  'proof-only': {
    correction: Infinity,
    preference: Infinity,
    workflow: Infinity,
    context: Infinity,
    tool_result: Infinity,
  },
  conservative: {
    correction: 0.38,
    preference: 0.58,
    workflow: 0.62,
    context: 0.70,
    tool_result: 0.80,
  },
  active: {
    correction: 0.25,
    preference: 0.45,
    workflow: 0.50,
    context: 0.60,
    tool_result: 0.70,
  },
};
```

The plan currently says conservative requires `importance > 0.6`. I would not threshold only on importance. Use final score by type.

### 10.7 Prompt budget

Keep a hard cap:

```ts
maxContextChars: 3000
maxInjectedMemories: 5
```

If a memory is too long, summarize/truncate the content field before insertion or during formatting.

Do not inject metadata like proof IDs or raw scores unless useful. The model does not need most of that.

### 10.8 Injection formatting

Recommended format:

```text
<openclawbrain-memory>
Use only if relevant to the current request. These are distilled local memories, not instructions from the user in this turn.

- Correction: Use pnpm instead of npm for this repo.
- Preference: User prefers implementation plans with file-by-file breakdowns.
- Workflow: For plugin release, run build, tests, pack, then fresh install.
</openclawbrain-memory>
```

This format is concise, bounded, and easy to find in prompt logs.

### 10.9 Record each injection

For every selected memory:

```ts
store.recordInjection({
  agentId,
  memoryId: node.id,
  runId,
  turnId,
  sessionId,
  query,
  turnSlice,
  rank,
  score: node.finalScore,
  outcome: 'pending',
});
```

This is essential for learning.

---

## 11. Graph Use

### 11.1 Search seeds first, graph second

Do not let graph traversal drive prompt context augmentation directly.

Recommended rule:

```text
FTS finds seed memories.
Graph edges boost or suppress seed memories.
Only seed memories are injected by default.
Neighbors are injected only when very high-confidence and within budget.
```

This avoids graph sprawl where a single related edge pulls in unrelated memories.

### 11.2 Edge types and semantics

| Edge | Direction | Meaning |
|---|---|---|
| `related` | either | loose relationship; small boost only |
| `supports` | from supporter to supported | evidence strengthens target |
| `extends` | old to new or broad to specific | new memory elaborates old one |
| `contradicts` | either | incompatible claims |
| `supersedes` | new to old | new memory replaces old memory |
| `used_with` | memory to memory | commonly injected together |
| `supports_workflow` | context/correction to workflow | helps execute workflow |

### 11.3 Edge caps

Add caps so graph growth stays bounded:

```text
maxRelatedEdgesPerNode = 20
maxSupportEdgesPerNode = 20
maxUsedWithEdgesPerNode = 15
maxSupersedesEdgesPerNode = 10
```

### 11.4 Link building

Background link building can be simple:

```text
For recently created or updated nodes:
1. Search FTS for similar active nodes.
2. If same topic and high lexical overlap, create related/extends edge.
3. If correction positive/negative fields conflict, create contradicts/supersedes edge.
4. If two memories were repeatedly injected together and accepted, create used_with edge.
```

Do not build edges across all 10K nodes every tick. Link recent nodes and sample older ones.

---

## 12. Learning Engine

### 12.1 Learning jobs

`learning.ts` should expose deterministic jobs:

```ts
export interface LearningEngine {
  runOnce(agentId: string, now?: Date): LearningRunResult;
  resolvePendingOutcomes(agentId: string, now?: Date): OutcomeResolutionResult;
  recomputeScores(agentId: string, now?: Date): ScoreUpdateResult;
  buildLinks(agentId: string, now?: Date): LinkBuildResult;
  pruneStaleMemories(agentId: string, now?: Date): PruneResult;
  enforceNodeLimit(agentId: string): PruneResult;
  start(): void;
  stop(): void;
}
```

### 12.2 Delayed outcome resolution

Do not assume `agent_end` is enough to know whether a memory helped.

Often the failure signal appears in the next user message:

```text
Assistant: npm install
User: No, I told you to use pnpm.
```

So each injection should remain `pending` until one of these happens:

- The user explicitly corrects the response.
- A tool workflow succeeds or fails.
- A timeout/window passes with no correction.
- The next turn appears and does not contain a correction signal.

Recommended lifecycle:

```text
pending → useful
pending → accepted
pending → corrected
pending → ignored
pending → tool_success
pending → tool_failure
pending → unknown
```

### 12.3 Outcome heuristics

Signals for `corrected`:

```text
"no"
"wrong"
"actually"
"I said"
"I told you"
"use X, not Y"
"that's not right"
```

Signals for `tool_success`:

- Tool result exit code 0.
- Build/test command succeeds.
- Final answer says done and no immediate correction follows.

Signals for `tool_failure`:

- Tool error.
- Exit code nonzero.
- Assistant retries because prior command failed.

Signals for `accepted`:

- No correction in the next turn.
- Same memory later reinjected and not corrected.
- User says thanks/works/good.

Be conservative. It is better to leave an outcome `unknown` than to mark a bad memory useful.

### 12.4 Importance scoring

Recommended scoring model:

```ts
function baseImportance(type: MemoryType): number {
  switch (type) {
    case 'correction': return 0.55;
    case 'preference': return 0.35;
    case 'workflow': return 0.30;
    case 'context': return 0.22;
    case 'tool_result': return 0.12;
  }
}

function computeImportance(node: MemoryNode, stats: OutcomeStats): number {
  const utility = stats.resolved > 0
    ? (stats.useful + stats.accepted + stats.toolSuccess) / stats.resolved
    : 0;

  const failureRate = stats.resolved > 0
    ? (stats.corrected + stats.toolFailure) / stats.resolved
    : 0;

  const repeatedEvidenceBoost = 0.08 * Math.log1p(node.captureCount);
  const useBoost = 0.05 * Math.log1p(node.useCount);

  return clamp01(
    baseImportance(node.type) +
    repeatedEvidenceBoost +
    useBoost +
    0.30 * utility -
    0.35 * failureRate
  );
}
```

This keeps scores interpretable.

### 12.5 Freshness scoring

Compute freshness separately from importance.

```ts
function computeFreshness(node: MemoryNode, now: Date, config: Config): number {
  const halfLife = halfLifeForType(node.type, config);
  const anchor = node.lastUsedAt ?? node.lastSeenAt ?? node.createdAt;
  const ageDays = daysBetween(new Date(anchor), now);
  return clamp01(Math.exp(-ageDays / halfLife));
}
```

### 12.6 Pruning

Recommended prune rules:

```text
1. Never hard-delete by default; soft-delete with deleted_at.
2. Prune deleted/superseded memories from injection immediately.
3. Soft-delete low-importance, low-confidence, old memories.
4. Hard-delete only through maintenance/export-cleanup if explicitly configured.
```

Prune candidates:

```text
importance < 0.05
confidence < 0.4
age > 30 days
use_count = 0 or useful_count = 0
not correction unless very old and contradicted
```

Superseded memories:

```text
If superseded_by is set and superseding node has survived 7+ days, hide from search/injection.
Keep visible in /graph unless includeDeleted=false.
```

Node cap:

```text
maxMemoryNodes = 10_000
Prune oldest lowest-score nodes first.
Preserve corrections and high-confidence explicit preferences longer than workflows/context.
```

### 12.7 Learning run proof

Every learning pass should write a summary proof event:

```json
{
  "eventType": "learning_run",
  "scoresUpdated": 42,
  "outcomesResolved": 9,
  "edgesCreated": 7,
  "nodesPruned": 2,
  "rawTranscriptStored": false,
  "rawUserTextStored": false
}
```

---

## 13. Search Integration

### 13.1 Corpus supplement

Expose OpenClawBrain memories through an additive memory corpus supplement.

Sketch:

```ts
export function registerMemorySearch(api: OpenClawPluginApi, deps: Deps): void {
  api.registerMemoryCorpusSupplement?.({
    id: 'openclawbrain',
    label: 'OpenClawBrain',

    search: async ({ query, maxResults, agentId }) => {
      const resolvedAgentId = agentId ?? deps.config.defaultAgentId;
      const results = deps.store.searchMemories({
        agentId: resolvedAgentId,
        query,
        limit: maxResults ?? 10,
      });

      return results.map((node) => ({
        corpus: 'openclawbrain',
        id: node.id,
        path: node.id,
        title: node.tags[0] ?? node.type,
        kind: node.type,
        score: node.finalScore,
        snippet: safeSnippet(node.content, 200),
        citation: `openclawbrain:${node.id}`,
      }));
    },

    get: async ({ id, agentId }) => {
      const node = deps.store.getMemory(agentId ?? deps.config.defaultAgentId, id);
      if (!node || node.deletedAt || node.supersededBy) return null;
      return {
        id: node.id,
        path: node.id,
        title: node.tags[0] ?? node.type,
        content: node.content,
        kind: node.type,
      };
    },
  });
}
```

### 13.2 Prompt supplement

Use prompt supplement only if the API expects additive memory sections separate from `before_prompt_build`. Avoid double injection.

Config:

```ts
promptInjectionMode: 'hook' | 'supplement' | 'both-proof-only'
```

Default:

```ts
promptInjectionMode: 'hook'
```

Reason: your existing plugin already injects via `before_prompt_build`. A prompt supplement plus hook injection could duplicate memory unless carefully coordinated.

---

## 14. Config Recommendations

The current manifest defaults to disabled and conservative behavior. Keep that posture.

Recommended config shape:

```ts
export interface OpenClawBrainConfig {
  enabled: boolean;
  mode: 'off' | 'proof-only' | 'conservative' | 'active';

  activationRoot: string;
  proofEvents: boolean;
  proofRetentionEvents: number;
  rawTranscriptUpload: false;

  memory: {
    dbPath?: string;
    maxNodes: number;
    maxEdgesPerNode: number;
    maxInjectedMemories: number;
    maxInjectionChars: number;
    minSearchScore: number;
    integration: 'supplement' | 'exclusive';
  };

  capture: {
    enabled: boolean;
    corrections: boolean;
    preferences: boolean;
    workflows: boolean;
    explicitRequests: boolean;
    minConfidence: number;
    promoteCorrectionsAbove: number;
    promotePreferencesAbove: number;
    promoteWorkflowsAbove: number;
  };

  learning: {
    enabled: boolean;
    intervalMs: number;
    staleOutcomeAfterMs: number;
    pruneAfterDays: number;
    correctionHalfLifeDays: number;
    preferenceHalfLifeDays: number;
    workflowHalfLifeDays: number;
    contextHalfLifeDays: number;
    toolResultHalfLifeDays: number;
    boostOnUseful: number;
    penaltyOnCorrected: number;
  };

  privacy: {
    redactBeforeStore: true;
    storeRawTranscript: false;
  };

  hooks: {
    allowPromptContext: boolean;
    allowConversationAccess: boolean;
    allowToolObservation: boolean;
  };

  scopes: {
    agents: string[];
  };
}
```

Suggested defaults:

```ts
const defaults = {
  enabled: false,
  mode: 'conservative',
  memory: {
    maxNodes: 10_000,
    maxEdgesPerNode: 25,
    maxInjectedMemories: 5,
    maxInjectionChars: 3000,
    minSearchScore: 0.35,
    integration: 'supplement',
  },
  capture: {
    enabled: true,
    corrections: true,
    preferences: true,
    workflows: true,
    explicitRequests: true,
    minConfidence: 0.55,
    promoteCorrectionsAbove: 0.80,
    promotePreferencesAbove: 0.75,
    promoteWorkflowsAbove: 0.70,
  },
  learning: {
    enabled: true,
    intervalMs: 5 * 60 * 1000,
    staleOutcomeAfterMs: 24 * 60 * 60 * 1000,
    pruneAfterDays: 30,
    correctionHalfLifeDays: 180,
    preferenceHalfLifeDays: 90,
    workflowHalfLifeDays: 45,
    contextHalfLifeDays: 30,
    toolResultHalfLifeDays: 14,
    boostOnUseful: 0.10,
    penaltyOnCorrected: 0.15,
  },
  privacy: {
    redactBeforeStore: true,
    storeRawTranscript: false,
  },
};
```

---

## 15. `index.ts` Wiring

Keep `index.ts` thin.

Sketch:

```ts
export default definePluginEntry({
  id: 'openclawbrain',
  name: 'OpenClawBrain',
  version: PLUGIN_VERSION,

  register(api) {
    const config = resolveOpenClawBrainConfig(api.config);
    const stores = new MemoryStoreRegistry(config);

    const deps = {
      api,
      config,
      stores,
      capture: createCaptureEngine(config),
      injection: createInjectionEngine(config),
      learning: createLearningEngine(config, stores),
    };

    registerRoutes(api, deps);
    registerMemorySupplements(api, deps);

    api.on('before_prompt_build', async (event) => {
      return handleBeforePromptBuild(event, deps);
    });

    api.on('after_tool_call', async (event) => {
      return handleAfterToolCall(event, deps);
    });

    api.on('agent_end', async (event) => {
      return handleAgentEnd(event, deps);
    });

    api.on('before_compaction', async (event) => {
      return handleBeforeCompaction(event, deps);
    });

    api.on('llm_output', async (event) => {
      return handleLlmOutput(event, deps);
    });

    api.registerService?.({
      id: 'openclawbrain-learning',
      start: async () => deps.learning.start(),
      stop: async () => deps.learning.stop(),
    });
  },
});
```

### 15.1 Agent-scoped stores

Avoid this pattern:

```ts
config.scopes.agents[0]
```

Use a resolver:

```ts
function resolveAgentId(event: unknown, config: OpenClawBrainConfig): string {
  return (
    getNestedString(event, ['agent', 'id']) ??
    getNestedString(event, ['agentId']) ??
    config.scopes.agents[0] ??
    'main'
  );
}
```

Then:

```ts
const store = stores.forAgent(agentId);
```

Even if the first release mostly uses `main`, the DB schema and store registry should support multiple agents cleanly.

---

## 16. Routes and Inspectability

### 16.1 `/status`

Should show memory health, not just plugin enabled state.

Example:

```json
{
  "ok": true,
  "enabled": true,
  "mode": "conservative",
  "agentId": "main",
  "db": {
    "path": "~/.openclawbrain/activation/main/memory.db",
    "schemaVersion": 1,
    "wal": true
  },
  "memory": {
    "nodes": 128,
    "edges": 241,
    "corrections": 17,
    "preferences": 42,
    "workflows": 31,
    "context": 38,
    "pendingCandidates": 4,
    "pendingOutcomes": 5,
    "deleted": 9
  },
  "learning": {
    "enabled": true,
    "lastRunAt": "2026-05-01T12:00:00.000Z",
    "lastRunMs": 24,
    "scoresUpdated": 12,
    "edgesCreated": 8,
    "nodesPruned": 1
  }
}
```

### 16.2 `/graph`

Return redacted nodes and edges.

```json
{
  "ok": true,
  "agentId": "main",
  "nodes": [
    {
      "id": "mem_...",
      "type": "correction",
      "content": "Use pnpm instead of npm.",
      "confidence": 0.9,
      "importance": 0.62,
      "freshness": 0.99,
      "useCount": 3,
      "usefulCount": 2,
      "tags": ["package-manager"],
      "supersededBy": null,
      "createdAt": "..."
    }
  ],
  "edges": [
    {
      "fromId": "mem_new",
      "toId": "mem_old",
      "relation": "supersedes",
      "weight": 1.0
    }
  ]
}
```

### 16.3 `/proof`

Show recent operations:

```json
{
  "ok": true,
  "events": [
    {
      "eventType": "memory_captured",
      "decision": "promoted_candidate",
      "reason": "correction_pattern:use_x_instead_of_y",
      "rawTranscriptStored": false,
      "rawUserTextStored": false
    }
  ]
}
```

### 16.4 `/learn`

Show learning stats and last run summary.

```json
{
  "ok": true,
  "agentId": "main",
  "pendingOutcomes": 5,
  "lastLearningRun": {
    "startedAt": "...",
    "finishedAt": "...",
    "scoresUpdated": 12,
    "outcomesResolved": 4,
    "edgesCreated": 3,
    "nodesPruned": 1
  },
  "importanceDistribution": {
    "min": 0.05,
    "max": 0.91,
    "mean": 0.38,
    "median": 0.31
  }
}
```

### 16.5 `/search`

Expose memory search for debugging:

```text
GET /plugins/openclawbrain/search?agentId=main&q=pnpm&limit=10
```

Return:

```json
{
  "ok": true,
  "query": "pnpm",
  "results": [
    {
      "id": "mem_...",
      "type": "correction",
      "content": "Use pnpm instead of npm.",
      "score": 0.72,
      "textRank": -1.34,
      "importance": 0.62,
      "freshness": 0.99
    }
  ]
}
```

---

## 17. Proof Store Migration

Keep `proof-store.ts` as a compatibility facade.

Old callers should still be able to do:

```ts
writeProofEvent(...)
readProofEvents(...)
```

Internally:

```ts
export function writeProofEvent(store: MemoryStore, event: ProofEventInput): void {
  store.insertProofEvent(event);
}

export function readRecentProofEvents(
  store: MemoryStore,
  agentId: string,
  limit: number,
): ProofEvent[] {
  return store.listProofEvents(agentId, limit);
}
```

This limits churn in existing tests and routes.

---

## 18. Build Phases

The current five phases are reasonable, but I would restructure them into smaller PRs so each PR proves one loop.

### PR 1 — SQLite store and migration foundation

Deliver:

- `memory-types.ts`
- `sqlite-driver.ts`
- `memory-store.ts`
- schema v1
- FTS5 triggers
- proof store facade
- status stats
- storage tests

Acceptance:

```text
- DB opens and migrates.
- Node insert/search works.
- FTS updates on insert/update/delete.
- Proof event insert/read works.
- Soft delete hides memory from search.
```

### PR 2 — Graph-backed injection using seeded/manual memories

Deliver:

- `injection.ts`
- policy integration
- memory search/ranking
- bounded prompt formatting
- injection recording

Acceptance:

```text
- Manually seeded “Use pnpm instead of npm” memory is retrieved.
- Relevant turn injects pnpm correction.
- Direct-answer turn stays silent.
- Injection event is recorded.
```

### PR 3 — Correction and preference capture

Deliver:

- `capture.ts`
- correction patterns
- preference patterns
- explicit memory request patterns
- dedupe/merge
- proof events
- contradiction/supersession basics

Acceptance:

```text
- User says “Actually use pnpm, not npm.”
- DB has correction node.
- /proof shows capture.
- /graph shows node.
- Next session injects correction.
```

### PR 4 — Delayed outcome learning

Deliver:

- `memory_injections` outcome resolution
- `learning.ts`
- `agent_end` observation
- next-turn correction observation
- score recomputation

Acceptance:

```text
- Useful injected memory gets boosted.
- Corrected injected memory gets penalized.
- Pending outcomes resolve.
- /learn shows run stats.
```

### PR 5 — Workflow capture and graph linking

Deliver:

- tool sequence observations
- workflow candidate creation
- successful workflow promotion
- related/supports/used_with edges

Acceptance:

```text
- Successful build/test tool sequence creates workflow memory.
- Workflow memory is injected on a similar future tool-heavy turn.
- Related edges appear in /graph.
```

### PR 6 — OpenClaw memory supplement integration

Deliver:

- `search.ts`
- `registerMemoryCorpusSupplement`
- optional `registerMemoryPromptSupplement`
- `/search` route parity

Acceptance:

```text
- Native memory search can surface OpenClawBrain memories.
- /search returns matching memories.
- Prompt is not double-injected.
```

### PR 7 — Self-regulation and release polish

Deliver:

- hard 10K node cap
- pruning
- edge caps
- status polish
- docs
- release bump
- fresh install test

Acceptance:

```text
- Low-value stale memories prune.
- Superseded memories stop injecting.
- Node cap enforcement works.
- All routes redact output.
- v0.2 package installs cleanly.
```

---

## 19. Test Plan

### 19.1 Storage tests

```text
1. Creates schema idempotently.
2. Migrates from empty DB.
3. Inserts memory node.
4. Upserts duplicate memory by source hash.
5. Updates FTS after insert.
6. Updates FTS after content update.
7. Removes FTS entry after soft delete.
8. Inserts edge.
9. Upserts duplicate edge and increments evidence_count.
10. Records injection event.
11. Resolves injection outcome.
12. Writes and reads proof events.
13. Computes stats by memory type.
14. Enforces max node cap.
```

### 19.2 Capture tests

```text
1. “Use pnpm, not npm” => correction.
2. “Actually use pnpm instead of npm” => correction.
3. “Do not use npm; use pnpm” => correction.
4. “I prefer concise plans” => preference.
5. “My timezone is America/Chicago” => preference/context.
6. “Remember that this repo uses pnpm” => context/correction.
7. “Thanks” => no candidate.
8. Assistant says “I’ll remember” => no durable memory alone.
9. Duplicate correction merges, not duplicates.
10. Opposite correction supersedes old correction.
```

### 19.3 Injection tests

```text
1. Direct-answer turn stays silent.
2. Correction-follow-up turn searches corrections first.
3. Tool-heavy turn searches workflows.
4. Superseded memory is not injected.
5. Deleted memory is not injected.
6. Low-confidence memory is not injected in conservative mode.
7. Same memory injects in active mode.
8. Prompt budget is respected.
9. Max injected memory count is respected.
10. Injection event recorded for each selected memory.
```

### 19.4 Learning tests

```text
1. Accepted injection increases importance.
2. Useful injection increases useful_count.
3. Corrected injection decreases importance.
4. Tool failure penalizes workflow memory.
5. Freshness decays over time.
6. Correction half-life is longer than workflow half-life.
7. Low-score old context memory prunes.
8. High-confidence correction resists pruning.
9. Superseded memory is hidden from search.
10. Learning run writes proof summary.
```

### 19.5 Route tests

```text
1. /status returns memory stats.
2. /proof returns no raw user text.
3. /graph returns redacted nodes.
4. /search finds known memory.
5. /learn returns last learning run.
6. Routes require gateway auth.
```

### 19.6 Hook registration tests

Use a fake OpenClaw API:

```text
1. Registers before_prompt_build.
2. Registers after_tool_call if tool observation enabled.
3. Registers agent_end.
4. Registers before_compaction.
5. Registers learning service.
6. Registers corpus supplement in supplement mode.
7. Does not register exclusive capability by default.
8. Does not claim before_agent_reply turns.
```

### 19.7 End-to-end success test

```text
Session 1:
  User: Actually use pnpm, not npm.

Expected:
  - correction node created
  - proof event created
  - no raw text stored

Session 2:
  User: Install dependencies for this repo.

Expected:
  - pnpm correction injected
  - memory_injections row created
  - use_count incremented

Session 2, next turn:
  User does not correct the package manager.

Expected:
  - injection outcome resolves accepted/useful
  - useful_count increments
  - importance increases
```

---

## 20. Risk Register

| Risk | Impact | Mitigation |
|---|---|---|
| False-positive capture | Wrong memory persists | Candidate stage, low starting confidence, proof, user inspection, decay |
| Prompt pollution | Model follows irrelevant memory | High thresholds, small budget, direct-answer silence, injection logging |
| Graph sprawl | Irrelevant relationships pollute ranking | Edge caps, shallow traversal, seed-first search |
| SQLite packaging friction | Plugin install problems | Test packed install on target Node/OpenClaw versions; isolate driver |
| Hook instability or coverage gaps | Capture/learning misses events | Hook smoke tests, fallback routes, avoid depending on one hook for critical behavior |
| Bad outcome inference | Scores move incorrectly | Leave uncertain outcomes as `unknown`; require repeated evidence |
| Privacy regression | User loses trust | Redaction before store, no raw transcript storage, proof assertions, route redaction |
| Capability mode conflict | Competes with OpenClaw memory engine | Default to additive supplements; exclusive mode opt-in only |
| Node cap pruning useful memory | Valuable memory lost | Preserve explicit corrections/preferences longer; soft-delete first |
| Long-running learning service | Runtime disruption | Short transactions, stop cleanly, run-once route, record errors |

---

## 21. Concrete Implementation Notes by File

### 21.1 `memory-store.ts`

Must include:

```text
- openMemoryStore(agentId, config)
- migrations
- CRUD for nodes/candidates/edges/injections/proof/learning_runs
- FTS search
- JSON serialization/deserialization helpers
- stats
- prune/enforce limit
```

Avoid:

```text
- regex capture logic
- hook event parsing
- prompt formatting
```

### 21.2 `capture.ts`

Must include:

```text
- detectCorrections
- detectPreferences
- detectExplicitMemoryRequests
- detectWorkflowCandidate
- normalizeCandidate
- promoteCandidate
- mergeDuplicate
- contradiction resolution call
```

Avoid:

```text
- writing SQL directly
- injecting prompt text
```

### 21.3 `learning.ts`

Must include:

```text
- start/stop background timer
- runOnce
- resolvePendingOutcomes
- recomputeScores
- pruneStaleMemories
- buildLinks
- record learning run summary
```

Avoid:

```text
- async better-sqlite3 transaction callbacks
- huge all-node link scans every tick
```

### 21.4 `injection.ts`

Must include:

```text
- buildTurnQuery
- search candidate memories
- rerank
- apply thresholds
- enforce budget
- format memory block
- record injection rows
- proof event
```

Avoid:

```text
- storing raw user text
- injecting superseded or deleted memories
- injecting low-confidence context in conservative mode
```

### 21.5 `graph.ts`

Must include:

```text
- direct contradiction detection
- supersede old memory
- related edge suggestion
- edge caps
- shallow expansion for scoring boosts
```

Avoid:

```text
- deep traversal by default
- prompt formatting
```

### 21.6 `search.ts`

Must include:

```text
- registerMemoryCorpusSupplement
- optional registerMemoryPromptSupplement
- conversion from MemoryNode to OpenClaw search result
```

Avoid:

```text
- defaulting to registerMemoryCapability
- double-injecting with before_prompt_build
```

### 21.7 `policy.ts`

Keep the v0.1 philosophy:

```text
- classify turn
- fail closed
- direct answers stay silent
- mode controls aggressiveness
```

Change from:

```text
static context file selection
```

to:

```text
graph search type filters and thresholds
```

### 21.8 `config.ts`

Add nested config while preserving old fields for compatibility.

Migration strategy:

```text
- Keep maxContextChars as alias for memory.maxInjectionChars.
- Keep activationRoot.
- Keep proofEvents.
- Keep rawTranscriptUpload const false.
- Add memory/capture/learning objects with defaults.
```

### 21.9 `status.ts`

Extend status with:

```text
- DB path
- schema version
- node count
- edge count
- type counts
- pending candidates
- pending outcomes
- last learning run
- last injection summary
```

### 21.10 `proof-store.ts`

Rewrite as SQLite facade, not a separate JSONL system.

---

## 22. Suggested Edits to PLAN.md

I would update `PLAN.md` with these explicit changes:

```text
1. Add memory_injections table.
2. Add capture_candidates table.
3. Add learning_runs table.
4. Use integer rowid plus UUID id for FTS compatibility.
5. Maintain FTS5 external-content index with triggers.
6. Use registerMemoryCorpusSupplement/registerMemoryPromptSupplement by default.
7. Keep registerMemoryCapability only as optional exclusive mode.
8. Treat before_agent_reply as dangerous for this plugin because it can claim turns.
9. Use before_prompt_build for prompt mutation.
10. Add delayed outcome resolution from next user turn.
11. Scope stores by resolved agent_id, not config.scopes.agents[0].
12. Preserve proof-store.ts as a SQLite facade.
13. Add soft-delete and superseded_by behavior.
14. Add edge caps.
15. Add hook smoke tests for target OpenClaw version.
16. Add packed-install test because better-sqlite3 is a native dependency.
```

---

## 23. Minimal Acceptance Criteria for v0.2

A v0.2 release should not ship unless these are true:

```text
1. Plugin starts disabled by default.
2. rawTranscriptUpload=true fails closed.
3. Memory DB initializes and migrates idempotently.
4. Correction capture works for “use pnpm, not npm”.
5. Correction is redacted before storage.
6. Correction appears in /graph.
7. Proof event asserts no raw user text stored.
8. Next relevant session injects the correction.
9. Direct-answer turns stay silent.
10. Injection event is recorded.
11. Learning pass updates at least one score from an outcome.
12. Superseded correction stops injecting.
13. Search route finds memory by keyword.
14. Native corpus supplement exposes memory search.
15. Node cap and prune logic work in tests.
16. Package installs cleanly from packed tarball.
```

---

## 24. Suggested README Pitch for v0.2

```markdown
OpenClawBrain v0.2 gives your OpenClaw agent local, automatic, self-regulating memory.

It watches for corrections, preferences, and successful workflows; stores distilled redacted memories in a local SQLite graph; and injects only the few memories that are relevant to the current turn. It does not upload data. It does not dump chat history into the prompt. Every capture and injection is inspectable through local routes.

Example:

> User: Actually use pnpm, not npm.

Next session, when the agent needs to install dependencies, OpenClawBrain injects:

> Correction: Use pnpm instead of npm.

No manual note file. No prompt bloat. No cloud memory.
```

---

## 25. Source Notes

The recommendations above were informed by these source materials:

- OpenClawBrain `PLAN.md`: implementation plan, current v0.1 state, proposed v0.2 files, phases, schema, and design decisions.  
  https://github.com/jonathangu/openclawbrain/blob/main/PLAN.md

- OpenClawBrain `VISION.md`: product vision and requirements for automatic correction capture, graph memory, background learning, adaptive injection, self-regulation, search integration, inspectability, and safety.  
  https://github.com/jonathangu/openclawbrain/blob/main/VISION.md

- OpenClawBrain package manifest: current version, Node engine, OpenClaw compatibility, and package structure.  
  https://github.com/jonathangu/openclawbrain/blob/main/packages/openclaw-plugin/package.json

- OpenClaw plugin manifest: current config schema, hooks, fail-closed `rawTranscriptUpload`, and v0.1 route/hook posture.  
  https://github.com/jonathangu/openclawbrain/blob/main/packages/openclaw-plugin/openclaw.plugin.json

- OpenClaw Agent Loop docs: hook placement, especially `before_prompt_build`, `before_agent_reply`, `agent_end`, tool hooks, and compaction hooks.  
  https://docs.openclaw.ai/concepts/agent-loop

- OpenClaw Plugin SDK Overview: additive memory supplements, exclusive memory capability APIs, services, routes, hooks, and decision semantics.  
  https://docs.openclaw.ai/plugins/sdk-overview

- OpenClaw Plugin Internals: route registration and explicit auth behavior.  
  https://docs.openclaw.ai/plugins/architecture

- SQLite FTS5 docs: FTS5 virtual tables, external content, `MATCH`, and ranking behavior.  
  https://www.sqlite.org/fts5.html

- `better-sqlite3` API docs: synchronous DB open, transactions, pragmas, and the warning that transaction functions should not be async.  
  https://github.com/WiseLibs/better-sqlite3/blob/master/docs/api.md

---

## 26. Bottom Line

The v0.2 plan is absolutely worth building. The path should be:

```text
SQLite store → graph-backed injection → correction capture → delayed learning → workflow capture → native search supplement → pruning/supersession polish
```

The one-line architecture:

> Capture redacted evidence, promote it into a scored memory graph, inject only the top relevant memories, then learn from whether those injections helped.

The one-line implementation warning:

> Do not let the graph become a prompt dump, and do not let regex matches become unquestioned truth.

The one-line product test:

> User corrects once: “use pnpm, not npm.” Next relevant session: the agent uses pnpm automatically, and `/proof` can explain exactly why.
