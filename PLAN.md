# OpenClawBrain v0.2 — Implementation Plan

*How to actually build the real product. Incorporates implementation feedback.*

---

## Architecture: Three Planes

Build v0.2 as an **event-driven local memory runtime** with three separate planes:

1. **Evidence plane** — what happened
   - Hook observations, capture candidates, injection decisions, outcome signals, proof events

2. **Memory plane** — what the system currently believes
   - Memory nodes, edges, confidence, importance, freshness, supersession, redacted content

3. **Recall plane** — what the model sees or can search
   - Ranked retrieval, bounded prompt injection, native memory corpus supplement, search route

This separation matters because auto-capture will make mistakes. If the system stores only final memory nodes, it becomes hard to debug bad behavior. If it stores evidence, candidates, injections, and outcomes separately, you can inspect exactly why a memory exists and why it was injected.

---

## Source tree

```
packages/openclaw-plugin/src/
  index.ts            # plugin registration and wiring only
  config.ts           # config schema/types/defaults/resolution
  redact.ts           # redaction, hashing, safe snippets
  policy.ts           # turn classification and injection gating
  memory-types.ts     # shared TS interfaces and enums          [NEW]
  sqlite-driver.ts    # tiny adapter around better-sqlite3       [NEW]
  memory-store.ts     # schema, migrations, CRUD, FTS, proof, stats [NEW]
  capture.ts          # candidate extraction and promotion logic [NEW]
  injection.ts        # search/rank/format/record injection      [NEW, replaces context-files.ts]
  learning.ts         # scoring, outcome resolution, pruning, linking [NEW]
  graph.ts            # edge creation, traversal, contradiction logic [NEW]
  search.ts           # OpenClaw memory supplement integration   [NEW]
  status.ts           # status payloads
  routes.ts           # HTTP route handlers and safe serialization [NEW]
```

Three new files beyond the original plan: `memory-types.ts`, `sqlite-driver.ts`, `routes.ts`. They keep the rest of the implementation cleaner.

---

## Module responsibilities

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

**Key design rule:** No file other than `memory-store.ts` should call `db.prepare()`.

---

## OpenClaw API integration

### Hooks

```typescript
api.on('before_prompt_build', handleBeforePromptBuild);   // inject memory
api.on('after_tool_call', handleAfterToolCall);           // capture workflows
api.on('agent_end', handleAgentEnd);                      // observe outcomes
api.on('before_compaction', handleBeforeCompaction);      // snapshot state
api.on('llm_output', handleLlmOutput);                    // detect preferences
```

**Do NOT register `before_agent_reply`.** It can claim turns. For v0.2, `before_agent_reply` should be either unused or strictly observational. Do not claim turns.

Correction detection runs in `before_prompt_build` on the *next* turn (looking back at what was said), not in `before_agent_reply`.

### Memory surfaces — use additive supplements by default

```typescript
api.registerMemoryCorpusSupplement?.({
  id: 'openclawbrain',
  label: 'OpenClawBrain',
  search: async ({ query, maxResults, agentId }) => { ... },
  get: async ({ id, agentId }) => { ... },
});

api.registerMemoryPromptSupplement?.({
  id: 'openclawbrain',
  builder: ({ availableTools }) => {
    // Build prompt sections from high-importance memories
  }
});
```

**Do NOT use `registerMemoryCapability` by default.** Put it behind config:

```typescript
memoryIntegration: 'supplement' | 'exclusive'  // default: 'supplement'
```

Reason: the vision says OpenClawBrain plugs into OpenClaw's memory engine and is not a replacement.

### Background service

```typescript
api.registerService?.({
  id: 'openclawbrain-learning',
  start: async () => learning.start(),
  stop: async () => learning.stop(),
});
```

Each learning pass opens short transactions and returns a report. Never keep a transaction open across ticks.

### Routes — all require `auth: 'gateway'`

```
GET  /plugins/openclawbrain/status
GET  /plugins/openclawbrain/proof?limit=20
GET  /plugins/openclawbrain/graph?agentId=main&limit=50
GET  /plugins/openclawbrain/search?agentId=main&q=pnpm
GET  /plugins/openclawbrain/learn?agentId=main
POST /plugins/openclawbrain/learn/run-once?agentId=main
```

---

## SQLite store design

### Driver adapter — `sqlite-driver.ts`

Isolate SQLite behind a tiny adapter so you can later switch implementations:

```typescript
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

Important: if using `better-sqlite3`, transaction callbacks must remain **synchronous**. Do not use `async` transaction functions.

### Pragmas

```sql
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;
PRAGMA busy_timeout = 5000;
PRAGMA synchronous = NORMAL;
```

### Schema

Use integer `rowid` plus public UUID `id`. FTS5 external-content tables work most cleanly with integer `rowid` joins.

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
      'related', 'contradicts', 'supersedes', 'supports',
      'extends', 'used_with', 'supports_workflow'
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

-- Capture candidates (don't promote immediately)
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

-- Injection events (critical for learning)
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
      'pending', 'accepted', 'useful', 'corrected', 'ignored',
      'tool_success', 'tool_failure', 'unknown'
    )
  ),
  correction_signal TEXT,
  metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_memory_injections_pending
  ON memory_injections(agent_id, outcome, injected_at);
CREATE INDEX IF NOT EXISTS idx_memory_injections_memory
  ON memory_injections(agent_id, memory_id, injected_at DESC);

-- Proof events
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

-- Learning run summaries
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

### FTS5 external-content table with triggers

```sql
CREATE VIRTUAL TABLE IF NOT EXISTS memory_search USING fts5(
  content,
  tags,
  topic_key,
  content='memory_nodes',
  content_rowid='rowid',
  tokenize='porter unicode61'
);

-- Triggers keep FTS aligned with content table
CREATE TRIGGER IF NOT EXISTS memory_nodes_ai AFTER INSERT ON memory_nodes
  WHEN new.deleted_at IS NULL
BEGIN
  INSERT INTO memory_search(rowid, content, tags, topic_key)
  VALUES (new.rowid, new.content, new.tags_json, COALESCE(new.topic_key, ''));
END;

CREATE TRIGGER IF NOT EXISTS memory_nodes_ad AFTER DELETE ON memory_nodes
BEGIN
  INSERT INTO memory_search(memory_search, rowid, content, tags, topic_key)
  VALUES ('delete', old.rowid, old.content, old.tags_json, COALESCE(old.topic_key, ''));
END;

CREATE TRIGGER IF NOT EXISTS memory_nodes_au AFTER UPDATE ON memory_nodes
BEGIN
  INSERT INTO memory_search(memory_search, rowid, content, tags, topic_key)
  VALUES ('delete', old.rowid, old.content, old.tags_json, COALESCE(old.topic_key, ''));
  INSERT INTO memory_search(rowid, content, tags, topic_key)
  SELECT new.rowid, new.content, new.tags_json, COALESCE(new.topic_key, '')
  WHERE new.deleted_at IS NULL AND new.superseded_by IS NULL;
END;
```

### Migrations

Use `PRAGMA user_version`. Each migration runs inside a transaction. Never wipe memory on migration failure.

```typescript
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

---

## Type contracts — `memory-types.ts`

```typescript
export type MemoryType = 'correction' | 'preference' | 'workflow' | 'context' | 'tool_result';

export type EdgeRelation =
  | 'related' | 'contradicts' | 'supersedes' | 'supports'
  | 'extends' | 'used_with' | 'supports_workflow';

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

export interface MemoryInjection {
  id: string;
  agentId: string;
  memoryId: string;
  runId?: string;
  turnId?: string;
  sessionId?: string;
  query: string;
  turnSlice: string;
  rank: number;
  score: number;
  injectedAt: string;
  resolvedAt?: string;
  outcome: 'pending' | 'accepted' | 'useful' | 'corrected' | 'ignored' | 'tool_success' | 'tool_failure' | 'unknown';
  correctionSignal?: string;
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

## Capture engine — `src/capture.ts`

### Pipeline

```
event → extract candidate → redact → hash → dedupe/merge →
contradiction check → insert/promote → proof event
```

### Do not store raw transcript text

Hard invariant: `rawTranscriptStored: false`, `rawUserTextStored: false`, `redactionApplied: true`.

### Capture candidates before promoting

Do not let every regex match immediately become a high-authority memory. Store capture candidates separately, then promote when confidence is high or supporting evidence appears.

### Treat assistant output as weak evidence

Assistant-generated text like "I'll remember to use pnpm going forward" should not, by itself, create durable memory. It can support a candidate that came from the user, but the user is the authority.

### Correction detection patterns

```typescript
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

Normalize "use pnpm instead of npm" to:
```typescript
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

### Preference detection patterns

```typescript
const preferencePatterns = [
  { name: 'always_x', regex: /\b(?:always|from now on)\s+(.+?)(?:[.!?]|$)/i, confidence: 0.72 },
  { name: 'i_prefer_x', regex: /\bI\s+prefer\s+(.+?)(?:[.!?]|$)/i, confidence: 0.68 },
  { name: 'my_timezone_is_x', regex: /\bmy\s+timezone\s+is\s+([A-Za-z_\/+-]+)(?:[.!?]|$)/i, confidence: 0.85 },
];
```

### Explicit memory request patterns

```typescript
const explicitMemoryPatterns = [
  /\bremember\s+(?:that\s+)?(.+?)(?:[.!?]|$)/i,
  /\bdon't\s+forget\s+(.+?)(?:[.!?]|$)/i,
  /\bnote\s*:\s*(.+?)(?:[.!?]|$)/i,
  /\bfor\s+future\s+reference[:,]?\s*(.+?)(?:[.!?]|$)/i,
];
```

### Confidence tiers

```
explicit correction:    0.85–0.95
explicit preference:    0.65–0.85
explicit remember req:  0.70–0.90
inferred workflow:      0.45–0.70
inferred context:       0.35–0.60
```

### Candidate promotion thresholds

```typescript
function shouldPromote(candidate: CaptureCandidate): boolean {
  if (candidate.type === 'correction' && candidate.confidence >= 0.8) return true;
  if (candidate.type === 'preference' && candidate.confidence >= 0.75) return true;
  if (candidate.type === 'context' && candidate.confidence >= 0.8) return true;
  return false;
}
```

Lower-confidence candidates stay in `capture_candidates` until repeated evidence appears, user confirms, or background learning passes.

### Deduplication

```typescript
const sourceHash = hashText([
  candidate.agentId,
  candidate.type,
  normalizeForHash(candidate.content),
].join('\n'));

// If duplicate:
capture_count += 1
last_seen_at = now
confidence = min(1, confidence + 0.03)
```

---

## Contradiction and supersession

### Detection

```typescript
function isDirectContradiction(a: MemoryNode, b: MemoryNode): boolean {
  return Boolean(
    a.positive && a.negative &&
    b.positive && b.negative &&
    normalize(a.positive) === normalize(b.negative) &&
    normalize(a.negative) === normalize(b.positive)
  );
}
```

### Supersede instead of delete

```sql
UPDATE memory_nodes
SET superseded_by = :newId,
    importance = MIN(importance, 0.1),
    updated_at = :now
WHERE id = :oldId;

-- Then insert edge:
INSERT INTO memory_edges (id, agent_id, from_id, to_id, relation, weight, ...)
VALUES (:edgeId, :agentId, :newId, :oldId, 'supersedes', 1.0, ...);
```

Do not hard-delete superseded memories. They are valuable for auditability.

### Topic keys for contradiction lookup

Use normalized topic keys for common domains:

```
package-manager, timezone, test-command, build-command,
repo-tooling, communication-style, answer-format, release-workflow
```

Search by `topic_key` first, then by FTS.

---

## Injection engine — `src/injection.ts`

### Algorithm

```
1. Resolve agent ID and config.
2. Fail closed if disabled or unsafe config.
3. Classify the turn.
4. If turn does not need memory, stay silent and log proof.
5. Build redacted query summary.
6. Search FTS for seed memories.
7. Expand graph only for boosts/support, not prompt flooding.
8. Rerank candidates.
9. Apply mode thresholds.
10. Fit selected memories into character budget.
11. Format bounded prompt section.
12. Record EACH injection in memory_injections.
13. Write proof event.
14. Return prependContext.
```

### Turn classification still matters

| Turn slice | Memory action |
|---|---|
| `direct-answer` | stay silent |
| `continuation` | low aggression |
| `correction-follow-up` | search corrections first |
| `retrieval-heavy` | search context/preferences |
| `tool-heavy` | search workflows/corrections |
| `stale-memory-conflict` | search corrections and supersession edges |

### Ranking formula

```typescript
finalScore = relevanceScore * importanceFactor * freshnessFactor *
             confidenceFactor * typeBoost * sliceBoost *
             graphBoost * safetyPenalty;
```

```typescript
const typeBoost = {
  correction: 1.35,
  preference: 1.0,
  workflow: 0.95,
  context: 0.8,
  tool_result: 0.6,
}[node.type];
```

### Freshness — exponential decay with type-specific half-lives

```typescript
function freshnessScore(node: MemoryNode, now: Date, halfLifeDays: number): number {
  const anchor = node.lastUsedAt ?? node.lastSeenAt ?? node.createdAt;
  const ageDays = daysBetween(new Date(anchor), now);
  return Math.exp(-ageDays / halfLifeDays);
}
```

| Type | Half-life |
|---|---:|
| correction | 180 days |
| preference | 90 days |
| workflow | 45 days |
| context | 30 days |
| tool_result | 14 days |

### Mode thresholds

Use final score by type, not just importance:

```typescript
const thresholds = {
  'proof-only': { correction: Infinity, preference: Infinity, workflow: Infinity, context: Infinity, tool_result: Infinity },
  conservative: { correction: 0.38, preference: 0.58, workflow: 0.62, context: 0.70, tool_result: 0.80 },
  active:       { correction: 0.25, preference: 0.45, workflow: 0.50, context: 0.60, tool_result: 0.70 },
};
```

### Prompt budget

Hard cap: `maxContextChars: 3000`, `maxInjectedMemories: 5`.

### Injection format

```text
<openclawbrain-memory>
Use only if relevant to the current request.
- Correction: Use pnpm instead of npm for this repo.
- Workflow: For plugin release, run build, tests, pack, then fresh install.
</openclawbrain-memory>
```

### Record each injection

For every selected memory, record in `memory_injections`:

```typescript
store.recordInjection({
  agentId, memoryId: node.id, runId, turnId, sessionId,
  query, turnSlice, rank, score: node.finalScore,
  outcome: 'pending',
});
```

This is critical for learning.

---

## Graph rules

### Search seeds first, graph second

FTS finds seed memories. Graph edges boost or suppress. Only seed memories are injected by default. Neighbors injected only when very high-confidence and within budget.

### Edge types

| Edge | Direction | Meaning |
|---|---|---|
| `related` | either | loose relationship; small boost only |
| `supports` | supporter → supported | evidence strengthens target |
| `extends` | old → new or broad → specific | new memory elaborates old |
| `contradicts` | either | incompatible claims |
| `supersedes` | new → old | new memory replaces old |
| `used_with` | memory ↔ memory | commonly injected together |
| `supports_workflow` | context/correction → workflow | helps execute workflow |

### Edge caps

```
maxRelatedEdgesPerNode = 20
maxSupportEdgesPerNode = 20
maxUsedWithEdgesPerNode = 15
maxSupersedesEdgesPerNode = 10
```

---

## Learning engine — `src/learning.ts`

### Jobs

```typescript
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

### Delayed outcome resolution

Do not assume `agent_end` is enough. Failure often appears in the next user turn:

```
Assistant: npm install
User: No, I told you to use pnpm.
```

Each injection remains `pending` until:
- User explicitly corrects the response
- A tool workflow succeeds or fails
- A timeout window passes with no correction
- The next turn appears without correction

Outcome lifecycle:
```
pending → useful
pending → accepted
pending → corrected
pending → ignored
pending → tool_success
pending → tool_failure
pending → unknown
```

### Outcome heuristics

**corrected signals:** "no", "wrong", "actually", "I said", "I told you", "use X, not Y"

**tool_success:** tool result exit code 0, build/test succeeds, final answer says done

**tool_failure:** tool error, exit code nonzero, assistant retries

**accepted:** no correction in next turn, same memory later reinjected without correction, user says thanks/works/good

Be conservative. Leave outcome `unknown` rather than marking a bad memory useful.

### Importance scoring

```typescript
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

### Freshness scoring (separate from importance)

```typescript
function computeFreshness(node: MemoryNode, now: Date, config: Config): number {
  const halfLife = halfLifeForType(node.type, config);
  const anchor = node.lastUsedAt ?? node.lastSeenAt ?? node.createdAt;
  const ageDays = daysBetween(new Date(anchor), now);
  return clamp01(Math.exp(-ageDays / halfLife));
}
```

### Pruning rules

1. Never hard-delete by default; soft-delete with `deleted_at`
2. Prune deleted/superseded memories from injection immediately
3. Soft-delete low-importance, low-confidence, old memories
4. Hard-delete only through maintenance if explicitly configured

Prune candidates:
```
importance < 0.05 AND confidence < 0.4 AND age > 30 days
```

Superseded memories hidden from search/injection when superseding node has survived 7+ days.

Node cap: `maxMemoryNodes = 10,000`. Prune oldest lowest-score first. Preserve corrections and explicit preferences longer.

### Learning run proof

Every pass writes:
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

## `index.ts` wiring

Keep it thin:

```typescript
export default definePluginEntry({
  id: 'openclawbrain',
  name: 'OpenClawBrain',
  version: PLUGIN_VERSION,
  register(api) {
    const config = resolveOpenClawBrainConfig(api.config);
    const stores = new MemoryStoreRegistry(config);
    const deps = {
      api, config, stores,
      capture: createCaptureEngine(config),
      injection: createInjectionEngine(config),
      learning: createLearningEngine(config, stores),
    };

    registerRoutes(api, deps);
    registerMemorySupplements(api, deps);

    api.on('before_prompt_build', async (event) => handleBeforePromptBuild(event, deps));
    api.on('after_tool_call', async (event) => handleAfterToolCall(event, deps));
    api.on('agent_end', async (event) => handleAgentEnd(event, deps));
    api.on('before_compaction', async (event) => handleBeforeCompaction(event, deps));
    api.on('llm_output', async (event) => handleLlmOutput(event, deps));

    api.registerService?.({
      id: 'openclawbrain-learning',
      start: async () => deps.learning.start(),
      stop: async () => deps.learning.stop(),
    });
  },
});
```

### Agent-scoped stores

Resolve agent ID from event, not config:

```typescript
function resolveAgentId(event: unknown, config: OpenClawBrainConfig): string {
  return (
    getNestedString(event, ['agent', 'id']) ??
    getNestedString(event, ['agentId']) ??
    config.scopes.agents[0] ??
    'main'
  );
}

const store = stores.forAgent(agentId);
```

---

## Config

```typescript
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
    allowPromptInjection: boolean;
    allowConversationAccess: boolean;
    allowToolObservation: boolean;
  };
  scopes: { agents: string[] };
}
```

Defaults:
```typescript
{
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
    enabled: true, corrections: true, preferences: true,
    workflows: true, explicitRequests: true,
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
  privacy: { redactBeforeStore: true, storeRawTranscript: false },
}
```

---

## Build phases (7 PRs)

### PR 1 — SQLite store and migration foundation
- `memory-types.ts`
- `sqlite-driver.ts`
- `memory-store.ts` — schema v1, FTS5 triggers, proof store facade
- Storage tests

**Acceptance:** DB opens, node insert/search works, FTS updates on insert/update/delete, soft delete hides from search.

### PR 2 — Graph-backed injection using seeded memories
- `injection.ts`
- Policy integration, memory search/ranking
- Bounded prompt formatting
- Injection recording

**Acceptance:** Manually seeded "Use pnpm" memory is retrieved on relevant turn. Direct-answer stays silent. Injection event recorded.

### PR 3 — Correction and preference capture
- `capture.ts`
- Correction/preference/explicit patterns
- Dedupe/merge
- Proof events
- Contradiction/supersession basics

**Acceptance:** "Actually use pnpm, not npm" creates correction node. `/proof` shows capture. `/graph` shows node. Next session injects correction.

### PR 4 — Delayed outcome learning
- `memory_injections` outcome resolution
- `learning.ts`
- `agent_end` observation
- Next-turn correction observation
- Score recomputation

**Acceptance:** Useful injection gets boosted. Corrected injection gets penalized. Pending outcomes resolve. `/learn` shows run stats.

### PR 5 — Workflow capture and graph linking
- Tool sequence observations
- Workflow candidate creation
- Successful workflow promotion
- Related/supports/used_with edges

**Acceptance:** Successful build/test sequence creates workflow memory. Related edges appear in `/graph`.

### PR 6 — OpenClaw memory supplement integration
- `search.ts`
- `registerMemoryCorpusSupplement`
- Optional `registerMemoryPromptSupplement`
- `/search` route parity

**Acceptance:** Native memory search surfaces OCB memories. `/search` returns matches. No double injection.

### PR 7 — Self-regulation and release polish
- Hard 10K node cap
- Pruning
- Edge caps
- Status polish
- Docs, release, fresh install test

**Acceptance:** Low-value stale memories prune. Superseded memories stop injecting. All routes redact. v0.2 installs cleanly.

---

## Routes — full inventory

| Endpoint | Description |
|---|---|
| `GET /plugins/openclawbrain/status` | Plugin state, memory count, learning stats |
| `GET /plugins/openclawbrain/proof?limit=20` | Recent proof events |
| `GET /plugins/openclawbrain/graph?limit=50` | Memory graph nodes + edges (redacted) |
| `GET /plugins/openclawbrain/search?q=pnpm` | Search memory graph |
| `GET /plugins/openclawbrain/learn` | Learning engine stats, importance distribution |
| `POST /plugins/openclawbrain/learn/run-once` | Trigger a single learning pass |

---

## Proof store migration

Keep `proof-store.ts` as a compatibility facade. Internally backed by SQLite.

```typescript
export function writeProofEvent(store: MemoryStore, event: ProofEventInput): void {
  store.insertProofEvent(event);
}
export function readRecentProofEvents(store: MemoryStore, agentId: string, limit: number): ProofEvent[] {
  return store.listProofEvents(agentId, limit);
}
```

---

## Test plan

### Storage tests (14)
Schema, migrations, CRUD, FTS sync, dedup, stats, node cap.

### Capture tests (10)
Correction detection, preference detection, explicit requests, dedup, contradiction.

### Injection tests (10)
Turn classification, threshold filtering, budget enforcement, superseded exclusion, injection recording.

### Learning tests (10)
Scoring, outcome resolution, pruning, freshness decay, link building.

### Route tests (6)
Status, proof, graph, search, learn — all authenticated, all redacted.

### Hook registration tests (8)
Correct hooks registered, `before_agent_reply` NOT registered, service registered, supplement registered.

### End-to-end test (1)
Correction captured → injected → outcome resolved → score updated.

**Total: ~59 test cases.**
