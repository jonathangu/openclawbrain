# OpenClawBrain v0.2 — Implementation Plan

*How to actually build the real product.*

---

## Current state

The v0.1 plugin at `packages/openclaw-plugin/` has:

```
src/
  config.ts          — config types, defaults, resolution
  context-files.ts   — reads flat files from activation root
  index.ts           — plugin entry, hooks, routes
  policy.ts          — heuristic turn classification and decision
  proof-store.ts     — proof event append/read
  redact.ts          — text redaction, hashing
  status.ts          — status payload builder
```

It hooks `before_prompt_build`, classifies the turn, reads flat files, and injects bounded context. It does not capture anything automatically. It does not learn. It does not have a memory graph.

---

## What changes

### New files

| File | Purpose |
|------|---------|
| `src/memory-store.ts` | SQLite database: nodes, edges, proof events, FTS5 index |
| `src/capture.ts` | Automatic detection of corrections, preferences, workflows |
| `src/learning.ts` | Background scoring, pruning, link building |
| `src/search.ts` | MemoryCorpusSupplement + MemoryPromptSectionBuilder |
| `src/graph.ts` | Graph queries, traversal, related-memory discovery |

### Rewritten files

| File | What changes |
|------|-------------|
| `src/index.ts` | New hooks registered, service registered, memory capabilities registered |
| `src/injection.ts` (replaces `context-files.ts`) | Injection from memory graph instead of flat files |
| `src/policy.ts` | Extended: adaptive scoring replaces static rules |
| `src/config.ts` | New config fields for learning, capture, memory limits |

### Mostly unchanged

| File | Notes |
|------|-------|
| `src/redact.ts` | Same redaction logic, no changes needed |
| `src/status.ts` | Extended with memory stats |
| `src/proof-store.ts` | Moved to SQLite-backed storage |

### Dependency change

- Add `better-sqlite3` for synchronous SQLite (works well in Node.js plugin context)
- Remove dependency on flat file system for memory (flat files still used for activation context as fallback)

---

## Phase 1: Memory store + auto capture (2-3 days)

### 1.1 SQLite memory store — `src/memory-store.ts`

The foundation. All memory lives in SQLite.

```typescript
// Opens/creates the database at ~/.openclawbrain/<agentId>/memory.db
export function openMemoryStore(agentId: string, stateDir: string): MemoryStore

export interface MemoryStore {
  // Node operations
  insertNode(node: MemoryNode): string
  getNode(id: string): MemoryNode | null
  updateNode(id: string, updates: Partial<MemoryNode>): void
  deleteNode(id: string): void
  searchNodes(query: string, opts?: SearchOpts): MemoryNode[]
  listNodes(agentId: string, opts?: ListOpts): MemoryNode[]

  // Edge operations
  insertEdge(edge: MemoryEdge): void
  deleteEdge(sourceId: string, targetId: string): void
  getEdges(nodeId: string): MemoryEdge[]

  // Proof events
  insertProofEvent(event: ProofEvent): void
  getProofEvents(opts?: { limit?: number }): ProofEvent[]

  // Stats
  getStats(agentId: string): MemoryStats
  getImportanceDistribution(agentId: string): { min: number, max: number, mean: number, median: number }

  // Maintenance
  prune(agentId: string, config: PruneConfig): number  // returns count pruned
  close(): void
}
```

Schema (from VISION.md, already designed):

```sql
CREATE TABLE memory_nodes (
  id TEXT PRIMARY KEY,
  agent_id TEXT NOT NULL,
  content TEXT NOT NULL,
  type TEXT NOT NULL,           -- correction | preference | context | workflow | tool-result
  importance REAL DEFAULT 0.3,
  freshness REAL DEFAULT 1.0,
  captured_at TEXT NOT NULL,
  last_used_at TEXT,
  use_count INTEGER DEFAULT 0,
  useful_count INTEGER DEFAULT 0,
  tags TEXT,                    -- JSON array
  source_hash TEXT,
  origin TEXT,                  -- auto | explicit | derived
  superseded_by TEXT,
  redaction_applied INTEGER DEFAULT 1
);

CREATE TABLE memory_edges (
  source_id TEXT NOT NULL,
  target_id TEXT NOT NULL,
  relation TEXT NOT NULL,
  weight REAL DEFAULT 0.5,
  created_at TEXT NOT NULL,
  PRIMARY KEY (source_id, target_id)
);

CREATE TABLE proof_events (
  id TEXT PRIMARY KEY,
  timestamp TEXT NOT NULL,
  event_type TEXT NOT NULL,
  memory_id TEXT,
  decision TEXT,
  reason TEXT,
  raw_transcript_stored INTEGER DEFAULT 0,
  raw_user_text_stored INTEGER DEFAULT 0
);

CREATE VIRTUAL TABLE memory_search USING fts5(
  content,
  tags,
  content='memory_nodes',
  content_rowid='rowid'
);
```

Key operations:
- Insert triggers FTS5 index update
- Importance/freshness are computed fields, updated by learning engine
- Pruning is soft-delete (sets `importance = 0`, can be recovered)

### 1.2 Capture engine — `src/capture.ts`

Hooks into agent lifecycle to detect and store memories automatically.

```typescript
export interface CaptureEngine {
  // Called from before_agent_reply hook
  detectCorrection(event: AgentReplyEvent, store: MemoryStore): CaptureResult | null

  // Called from llm_output hook
  detectPreference(event: LlmOutputEvent, store: MemoryStore): CaptureResult | null

  // Called from after_tool_call hook
  detectWorkflow(event: ToolCallEvent, store: MemoryStore): CaptureResult | null

  // Called from before_prompt_build hook
  detectExplicitRequest(event: PromptEvent, store: MemoryStore): CaptureResult | null
}
```

**Correction detection patterns** (regex + heuristics on assistant response):

```
"actually, [use/do/prefer] X [instead of/not] Y"
"the correct [way/approach] is X"
"I was wrong about X, it should be Y"
"you're right, I should [use/do] X"
"let me correct that: X"
```

When detected:
1. Extract the correction (what was wrong, what's right)
2. Redact the content
3. Compute source hash (for dedup)
4. Check if it contradicts an existing memory → if so, mark old as superseded
5. Insert as `correction` node with `importance = 0.5` (corrections start higher than other types)
6. Write proof event

**Preference detection patterns** (on assistant response):
```
"I'll use X going forward"
"from now on, I'll [use/do/prefer] X"
"let me make a note: [preference]"
```

When detected:
1. Extract preference
2. Redact
3. Insert as `preference` node with `importance = 0.3`

**Explicit request detection** (on user message):
```
"remember that X"
"don't forget X"
"note: X"
"from now on, always X"
```

When detected:
1. Extract the thing to remember
2. Redact
3. Insert as `context` node with `importance = 0.5` (explicit = important)

### 1.3 Injection from memory graph — `src/injection.ts`

Replaces `context-files.ts`. Instead of reading flat files, queries the memory graph.

```typescript
export interface InjectionEngine {
  // Main injection decision (called from before_prompt_build)
  decide(
    turn: ClassifiedTurn,
    store: MemoryStore,
    config: PluginConfig
  ): InjectionDecision
}

export interface InjectionDecision {
  action: 'stay_silent' | 'proof_only' | 'inject'
  memories: SelectedMemory[]     // sorted by relevance
  injectionText: string          // formatted for prompt
  reasoning: string              // why these memories were selected
}
```

Selection algorithm:
1. Get the turn classification (from policy.ts)
2. Search memory graph for relevant nodes:
   - FTS5 search on turn topic
   - Filter by `agent_id`
   - Weight by `importance × freshness × relevance`
3. Apply mode constraints:
   - `off`: action = stay_silent
   - `proof-only`: action = proof_only
   - `conservative`: only include nodes with `importance > 0.6`
   - `active`: only include nodes with `importance > 0.3`
4. Sort by weighted score, take top N within `injectionBudget` characters
5. Format as labeled sections:

```
[correction guidance]
Use pnpm, not npm. (correction, captured 2026-04-30)

[context]
Working on OpenClawBrain v0.2 native plugin. (context, captured 2026-04-28)

[workflow pattern]
Build → test → pack → publish cycle. (workflow, captured 2026-04-25)
```

6. Update `use_count` and `last_used_at` for each injected memory
7. Write proof event

### 1.4 Updated hooks — `src/index.ts`

Register new hooks:

```typescript
// Existing
api.on('before_prompt_build', handleBeforePromptBuild)
api.on('model_call_started', handleModelCallStarted)
api.on('model_call_ended', handleModelCallEnded)
api.on('gateway_start', handleGatewayStart)
api.on('gateway_stop', handleGatewayStop)

// NEW: capture corrections from assistant responses
api.on('before_agent_reply', handleBeforeAgentReply)

// NEW: capture workflows from tool calls
api.on('after_tool_call', handleAfterToolCall)

// NEW: observe outcomes for learning
api.on('agent_end', handleAgentEnd)

// NEW: snapshot before compaction
api.on('before_compaction', handleBeforeCompaction)

// NEW: detect preferences
api.on('llm_output', handleLlmOutput)
```

### 1.5 Gate for Phase 1

- Install the plugin
- Correct the agent once: "Use pnpm, not npm"
- Check `/proof` → correction was captured
- Check `/graph` → correction node exists
- Next session: correction is injected automatically
- No manual file editing required

---

## Phase 2: Learning engine (2-3 days)

### 2.1 Background service — `src/learning.ts`

```typescript
export interface LearningEngine {
  // Runs periodically (configurable, default 5 min)
  learningTick(store: MemoryStore, config: PluginConfig): LearningReport

  // Called after agent_end to update scores
  updateOutcomes(store: MemoryStore, agentId: string, turnResult: TurnResult): void

  // Score a single memory node
  scoreNode(node: MemoryNode, now: Date): ScoredNode

  // Prune low-value memories
  prune(store: MemoryStore, config: PruneConfig): PruneReport

  // Build edges between related memories
  linkRelated(store: MemoryStore, agentId: string): LinkReport
}
```

**Scoring algorithm** (runs periodically for all nodes):

```typescript
function scoreNode(node: MemoryNode, now: Date): { importance: number, freshness: number } {
  const ageDays = (now - new Date(node.captured_at)) / 86400000
  const daysSinceUse = node.last_used_at
    ? (now - new Date(node.last_used_at)) / 86400000
    : ageDays

  // Usefulness ratio
  const usefulness = node.use_count > 0
    ? node.useful_count / node.use_count
    : 0

  // Freshness decays over time since last use
  const freshness = Math.max(0, 1 - (daysSinceUse * config.importanceDecayPerDay))

  // Importance = base + usefulness bonus - age penalty
  const base = node.type === 'correction' ? 0.5 : 0.3  // corrections start higher
  const usefulBonus = usefulness * 0.3
  const agePenalty = Math.min(0.2, ageDays * 0.002)

  const importance = Math.max(0, Math.min(1,
    base + usefulBonus - agePenalty + (node.use_count > 5 ? 0.1 : 0)
  ))

  return { importance, freshness }
}
```

**Outcome observation** (called from `agent_end` hook):

```typescript
function updateOutcomes(store: MemoryStore, agentId: string, turnResult: TurnResult) {
  // If memories were injected this turn and the response was good:
  // → boost useful_count for each injected memory
  // Heuristic for "good response":
  //   - No user correction in next message
  //   - Response was substantive (not "I don't know")
  //   - Tool calls succeeded (if any)

  for (const memoryId of turnResult.injectedMemoryIds) {
    if (turnResult.outcome === 'good') {
      store.updateNode(memoryId, {
        useful_count: { $inc: 1 },
        importance: { $inc: config.importanceBoostOnUseful }
      })
    }
  }
}
```

**Pruning** (runs periodically):

```typescript
function prune(store: MemoryStore, config: PruneConfig): number {
  // 1. Remove nodes where importance < 0.05 AND age > pruneAfterDays
  // 2. Remove nodes where superseded_by is set AND superseding node is > 7 days old
  // 3. If total nodes > maxMemoryNodes, remove oldest low-importance nodes
  // 4. Write proof events for each pruned node
  // Returns count of pruned nodes
}
```

**Link building** (runs periodically):

```typescript
function linkRelated(store: MemoryStore, agentId: string): number {
  // For each new node (captured in last learning interval):
  //   - FTS5 search for similar content
  //   - If similarity > threshold: create 'related' edge
  //   - If content contradicts (one says X, other says not-X): create 'contradicts' edge
  //   - If same topic/domain: create 'extends' edge
  // Returns count of new edges created
}
```

### 2.2 Service registration

```typescript
api.registerService({
  id: 'openclawbrain-learner',
  start: async (ctx) => {
    // Start periodic learning loop
    const interval = setInterval(() => {
      learningEngine.learningTick(store, config)
    }, config.learningIntervalMs)
    // Store interval ref for cleanup
  },
  stop: async (ctx) => {
    // Clear interval, close store
  }
})
```

### 2.3 Gate for Phase 2

- Correct agent 5 times over 3 sessions
- Check `/learn` → importance scores have changed
- Check `/graph` → some memories have higher importance than others
- Check `/graph` → edges exist between related memories
- Pruning works: old unused memories get removed

---

## Phase 3: Memory capabilities + search (1-2 days)

### 3.1 MemoryCorpusSupplement — `src/search.ts`

```typescript
api.registerMemoryCapability({
  corpus: {
    search: async ({ query, maxResults = 10 }) => {
      const results = store.searchNodes(query, {
        agentId: config.scopes.agents[0],
        limit: maxResults
      })
      return results.map(node => ({
        corpus: 'openclawbrain',
        path: node.id,
        title: node.tags?.[0] || node.type,
        kind: node.type,
        score: node.importance * node.freshness,
        snippet: node.content.slice(0, 200),
        id: node.id,
        citation: `openclawbrain:${node.id}`
      }))
    },
    get: async ({ lookup }) => {
      const node = store.getNode(lookup)
      if (!node) return null
      return {
        content: node.content,
        path: node.id,
        title: node.tags?.[0] || node.type
      }
    },
    list: async () => {
      return store.listNodes(config.scopes.agents[0]).map(node => ({
        path: node.id,
        title: node.tags?.[0] || node.type,
        kind: node.type
      }))
    }
  },
  promptBuilder: ({ availableTools }) => {
    // Build prompt sections from high-importance memories
    const memories = store.listNodes(config.scopes.agents[0], {
      minImportance: config.importanceThreshold,
      limit: 10
    })
    return memories.map(node =>
      `[${node.type}] ${node.content}`
    )
  }
})
```

### 3.2 Search route

```typescript
api.registerHttpRoute({
  path: '/plugins/openclawbrain/search',
  auth: 'gateway',
  match: 'prefix',
  handler: async (req, res) => {
    const query = req.query?.q || ''
    const limit = Math.min(50, Number(req.query?.limit) || 10)
    const results = store.searchNodes(query, {
      agentId: config.scopes.agents[0],
      limit
    })
    writeJson(res, { ok: true, query, results })
  }
})
```

### 3.3 Gate for Phase 3

- `curl /plugins/openclawbrain/search?q=pnpm` returns the pnpm correction
- Agent does a memory search → OpenClawBrain results appear
- Memory prompt section builder produces relevant sections

---

## Phase 4: Self-regulation + polish (1-2 days)

### 4.1 Contradiction resolution

When `capture.ts` detects a new correction:
1. FTS5 search for existing memories on the same topic
2. If an existing memory says X and new correction says Y:
   - Mark existing memory as `superseded_by = newMemory.id`
   - Create `superseded_by` edge
   - Reduce existing memory's importance to 0.1
   - New memory gets `importance = 0.6` (superseding = important)

### 4.2 Growth control

```typescript
function enforceMemoryLimit(store: MemoryStore, agentId: string, maxNodes: number) {
  const count = store.countNodes(agentId)
  if (count > maxNodes) {
    const excess = count - maxNodes
    // Remove oldest, lowest-importance nodes
    store.pruneOldestLowest(agentId, excess)
  }
}
```

### 4.3 Graph route

```typescript
api.registerHttpRoute({
  path: '/plugins/openclawbrain/graph',
  auth: 'gateway',
  match: 'prefix',
  handler: async (req, res) => {
    const limit = Math.min(200, Number(req.query?.limit) || 50)
    const nodes = store.listNodes(config.scopes.agents[0], { limit })
    const edges = store.getAllEdges(config.scopes.agents[0])
    writeJson(res, {
      ok: true,
      nodes: nodes.map(redactNode),
      edges,
      stats: store.getStats(config.scopes.agents[0])
    })
  }
})
```

### 4.4 Learn route

```typescript
api.registerHttpRoute({
  path: '/plugins/openclawbrain/learn',
  auth: 'gateway',
  match: 'exact',
  handler: async (req, res) => {
    const stats = store.getStats(config.scopes.agents[0])
    const dist = store.getImportanceDistribution(config.scopes.agents[0])
    writeJson(res, {
      ok: true,
      ...stats,
      importanceDistribution: dist,
      lastLearningRun: learningEngine.lastRunTime(),
      nextLearningRun: learningEngine.nextRunTime()
    })
  }
})
```

### 4.5 Gate for Phase 4

- Memory limit enforced at 10,000 nodes
- Contradicting corrections resolve correctly (new supersedes old)
- `/graph` returns redacted memory graph
- `/learn` returns learning stats
- E2E test: 20-session simulation passes

---

## Phase 5: Release (1 day)

- Bump version to `0.2.0`
- Update `openclaw.plugin.json` and `package.json`
- Update `README.md` with v0.2 description
- Update `VISION.md` as final reference
- Run all gates: check, test, pack, fresh install, live routes
- Publish to ClawHub as `openclawbrain@0.2.0`
- Update openclawbrain.ai
- Commit, tag `openclawbrain-v0.2.0`, push

---

## File-by-file change summary

| File | Action | Lines estimate |
|------|--------|---------------|
| `src/memory-store.ts` | NEW | ~400 |
| `src/capture.ts` | NEW | ~300 |
| `src/learning.ts` | NEW | ~350 |
| `src/search.ts` | NEW | ~150 |
| `src/graph.ts` | NEW | ~200 |
| `src/injection.ts` | NEW (replaces `context-files.ts`) | ~200 |
| `src/index.ts` | REWRITE | ~350 |
| `src/policy.ts` | EXTEND | ~150 → ~250 |
| `src/config.ts` | EXTEND | ~80 → ~120 |
| `src/status.ts` | EXTEND | ~40 → ~80 |
| `src/proof-store.ts` | MIGRATE to SQLite | ~60 → ~80 |
| `src/redact.ts` | UNCHANGED | ~100 |
| `test/index.test.mjs` | REWRITE | ~12 → ~30 tests |
| `openclaw.plugin.json` | UPDATE schema | +5 config fields |
| `package.json` | ADD `better-sqlite3` dep | minor |

**Total new code estimate: ~1,750 lines of TypeScript**
**Total test estimate: ~30 test cases**

---

## Key design decisions

### SQLite over flat files

SQLite gives us:
- FTS5 full-text search for free
- Efficient queries (importance × freshness ranking)
- ACID transactions for concurrent hook access
- Single file = easy backup, inspect, debug

### Synchronous SQLite (`better-sqlite3`)

Plugin hooks are async, but SQLite operations are fast (microseconds). `better-sqlite3` is synchronous and 10x faster than async alternatives. We use it synchronously inside async hooks — no blocking issues for local DB operations.

### Heuristic learning, not RL

The original vision mentioned policy-gradient routing. v0.2 uses heuristic scoring (importance/freshness/usefulness). This is:
- Simpler to implement
- Easier to debug and audit
- Proven sufficient for the eval traces

RL can be added later as an optimization if the heuristic baseline isn't good enough.

### Correction detection heuristics, not models

Detecting corrections from assistant responses uses regex patterns, not a classifier model. This is:
- No model calls = no latency, no cost
- Predictable and debuggable
- Good enough for obvious corrections ("actually, use X")
- Can be improved incrementally

### Memory stays local, forever

No cloud sync. No cross-machine sharing. Each agent's memory is a local SQLite file. This is a safety constraint, not a limitation.

---

## Risks and mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Correction detection misses subtle corrections | Some corrections not captured | Start with high-confidence patterns, expand over time |
| Importance scoring doesn't converge | Some memories stay important when they shouldn't | Pruning catches stale memories. Can tune decay rate. |
| SQLite performance under heavy load | Slow hook responses | SQLite is fast for local ops. Benchmark at 10K nodes. |
| Memory graph grows unbounded | Disk usage | Hard limit at 10K nodes. Pruning keeps it lean. |
| Injection makes prompts too long | Agent performance degrades | Injection budget (default 3000 chars) is hard capped. |
| False positive corrections | Wrong memories stored | Corrections start at importance 0.5, not 1.0. Pruning removes unused ones. |

---

## Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| 1. Memory store + capture | 2-3 days | Auto-capture works, injection from graph |
| 2. Learning engine | 2-3 days | Background scoring, pruning, linking |
| 3. Memory capabilities | 1-2 days | Search integration, prompt sections |
| 4. Self-regulation | 1-2 days | Contradiction resolution, growth control |
| 5. Release | 1 day | v0.2 on ClawHub |
| **Total** | **7-11 days** | **Working product** |
