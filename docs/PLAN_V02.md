# OpenClawBrain v0.2 — Real Learning Memory for OpenClaw

## What this is

A native OpenClaw plugin that automatically captures corrections, builds a local memory graph, learns what helps in the background, and injects relevant context when it matters. No manual file editing. No flat markdown. Real adaptive memory.

This is the product Jonathan originally described: corrections stick, prompts stay small, the agent gets smarter over time, and the system self-regulates.

## What v0.1 got wrong

v0.1 was a static file injector. You manually wrote notes into three files. The plugin sometimes read them back. That's not memory — that's a notepad with a turn classifier.

The real product needs:
1. Automatic capture — corrections are recorded without you doing anything
2. Background learning — observes outcomes, adapts routing
3. Memory graph — organizes context into something useful and searchable
4. Adaptive injection — learns what context actually helps
5. Self-regulation — the system decides what to keep vs. discard

## Architecture

```
┌─────────────────────────────────────────────┐
│                 OpenClaw                     │
│                                              │
│  hooks:                                      │
│    before_prompt_build ──► inject memory     │
│    before_agent_reply ──► capture response   │
│    after_tool_call ──► capture tool result   │
│    agent_end ──► capture outcome             │
│    before_compaction ──► snapshot state      │
│                                              │
│  memory capabilities:                        │
│    MemoryCorpusSupplement ──► searchable     │
│    MemoryPromptSectionBuilder ──► injectable │
│                                              │
│  service:                                    │
│    background learning loop                  │
│                                              │
│  routes:                                     │
│    /status ──► plugin state                  │
│    /graph ──► memory graph (redacted)        │
│    /proof ──► operation log                  │
│    /learn ──► learning stats                 │
└──────────────────┬───────────────────────────┘
                   │
    ┌──────────────▼──────────────┐
    │      OpenClawBrain v0.2     │
    │                             │
    │  ┌─────────┐  ┌──────────┐ │
    │  │ Capture │  │ Learning │ │
    │  │ Engine  │  │ Engine   │ │
    │  └────┬────┘  └────┬─────┘ │
    │       │            │       │
    │  ┌────▼────────────▼─────┐ │
    │  │    Memory Graph Store  │ │
    │  │    (SQLite, local)     │ │
    │  └────┬────────────┬─────┘ │
    │       │            │       │
    │  ┌────▼────┐  ┌───▼─────┐ │
    │  │ Injection│  │ Search  │ │
    │  │ Engine  │  │ Index   │ │
    │  └─────────┘  └─────────┘ │
    └───────────────────────────┘
```

## Components

### 1. Memory Graph Store (SQLite)

Local SQLite database at `~/.openclawbrain/<agentId>/memory.db`.

**Nodes** — each piece of memory:
```sql
CREATE TABLE memory_nodes (
  id TEXT PRIMARY KEY,
  agent_id TEXT NOT NULL,
  content TEXT NOT NULL,           -- redacted, never raw
  type TEXT NOT NULL,              -- correction | preference | context | workflow | tool-result
  importance REAL DEFAULT 0.5,    -- 0-1, learned over time
  freshness REAL DEFAULT 1.0,     -- decays over time, boosted by reuse
  captured_at TEXT NOT NULL,       -- ISO timestamp
  last_used_at TEXT,               -- last time it was injected
  use_count INTEGER DEFAULT 0,    -- times injected
  useful_count INTEGER DEFAULT 0, -- times injected AND outcome was good
  tags TEXT,                       -- JSON array of search tags
  source_hash TEXT,                -- SHA-256 of original content (for dedup)
  origin TEXT,                     -- how it was captured: auto | explicit | derived
  redaction_applied INTEGER DEFAULT 1
);
```

**Edges** — relationships between memories:
```sql
CREATE TABLE memory_edges (
  source_id TEXT NOT NULL,
  target_id TEXT NOT NULL,
  relation TEXT NOT NULL,          -- contradicts | supports | extends | related
  weight REAL DEFAULT 0.5,
  created_at TEXT NOT NULL,
  PRIMARY KEY (source_id, target_id)
);
```

**Proof events** — audit log:
```sql
CREATE TABLE proof_events (
  id TEXT PRIMARY KEY,
  timestamp TEXT NOT NULL,
  event_type TEXT NOT NULL,       -- capture | inject | learn | prune | search
  memory_id TEXT,
  decision TEXT,                   -- what was decided
  reason TEXT,                     -- why
  raw_transcript_stored INTEGER DEFAULT 0,
  raw_user_text_stored INTEGER DEFAULT 0
);
```

### 2. Capture Engine (Automatic)

Hooks into agent lifecycle to automatically capture memories.

**Correction detection** — `before_agent_reply` hook:
- Monitors assistant responses for correction language:
  - "Actually, use X instead of Y"
  - "No, don't do that"
  - "The correct way is..."
  - "I made a mistake, it should be..."
  - Pattern: assistant acknowledges and incorporates a correction
- Extracts the correction, redacts it, stores as a `correction` node
- Auto-tags with the topic/domain

**Workflow capture** — `after_tool_call` hook:
- When a sequence of tool calls produces a successful result
- Captures the workflow pattern (not raw tool output)
- Stores as a `workflow` node with the decision that was made

**Preference capture** — `llm_output` hook:
- Detects preference signals in assistant responses
- "I'll use pnpm going forward" → preference node
- "Let me check the family calendar" → context node

**Explicit capture** — user says:
- "Remember that..." → capture as `context`
- "From now on, always..." → capture as `preference`
- "Don't forget..." → capture as `correction`

**Compaction snapshot** — `before_compaction` hook:
- Before OpenClaw compacts the conversation, snapshot the current state
- Captures any uncommitted memories from the session

### 3. Learning Engine (Background)

A background service (`registerService`) that runs periodically.

**Scoring** — every N turns, for each memory node:
```
usefulness = useful_count / max(use_count, 1)
importance = f(importance, usefulness, recency, use_count)
freshness  = decay since last_used_at
```

If `usefulness < threshold` and `age > max_age`: mark for pruning.
If `usefulness > high_threshold`: boost importance.

**Outcome observation** — `agent_end` hook:
- When an agent run completes, compare:
  - Was context injected? → Was the response better than without?
  - Was a correction injected? → Did the agent actually use it?
- This is heuristic (we can't run both paths), but we can infer:
  - If a correction was injected and the agent followed it → useful
  - If context was injected and the response was detailed/useful → useful
  - If nothing was injected and the response was fine → injection not needed

**Pruning** — periodic:
- Nodes with `importance < 0.1` and `freshness < 0.1` and `age > 30d` → soft delete
- Nodes that are exact duplicates (same `source_hash`) → merge
- Contradicting edges where one node was corrected → keep newer

**Link building** — periodic:
- Compare new nodes against existing for topic overlap
- Create edges for `related`, `supports`, `contradicts`

### 4. Injection Engine

Same hook as v0.1 (`before_prompt_build`) but much smarter.

**Selection algorithm:**
1. Classify the turn (same heuristic as v0.1)
2. Query the memory graph for relevant nodes:
   - Search by turn topic/tags
   - Weight by `importance * freshness`
   - Filter by turn slice (corrections for correction turns, workflows for tool-heavy, etc.)
3. Apply mode constraints:
   - `off`: return nothing
   - `proof-only`: log selection, don't inject
   - `conservative`: only inject if `importance > 0.6` and `usefulness > 0.3`
   - `active`: inject if `importance > 0.3`
4. Build bounded injection text:
   - Sort by relevance, take top N within `maxContextChars`
   - Format as labeled sections
   - Apply redaction (redundant safety check)

### 5. Search (MemoryCorpusSupplement)

Register with OpenClaw's memory system so the agent can search its own memory:

```typescript
api.registerMemoryCapability({
  corpus: {
    search: async ({ query, maxResults }) => {
      // Full-text search over memory_nodes.content
      // Return results ranked by importance * freshness * relevance
    },
    get: async ({ lookup }) => {
      // Get a specific memory by ID or hash
    },
    list: async () => {
      // List all memories (for agent introspection)
    }
  },
  promptBuilder: ({ availableTools }) => {
    // Build prompt sections from high-importance memories
  }
});
```

This means the agent can find memories through OpenClaw's native memory search, not just through injection.

### 6. Self-Regulation

The system manages its own memory lifecycle:

- **Growth control**: new memories are scored low. They only become important through repeated useful injection.
- **Decay**: unused memories naturally lose importance over time.
- **Pruning**: useless memories are automatically removed.
- **Contradiction resolution**: when a new correction contradicts an old one, the old one is marked as superseded.
- **Size limit**: total memory count is bounded (configurable, default 10,000 nodes). New captures beyond the limit replace the lowest-importance nodes.

## Configuration

```json
{
  "plugins.entries.openclawbrain.config": {
    "enabled": true,
    "mode": "conservative",
    "autoCapture": true,
    "backgroundLearning": true,
    "maxMemoryNodes": 10000,
    "importanceThreshold": 0.5,
    "injectionBudget": 3000,
    "learningIntervalMs": 300000,
    "pruneAfterDays": 30,
    "scopes": { "agents": ["main"] },
    "rawTranscriptUpload": false
  }
}
```

## Safety (same rules as v0.1, stricter)

- Local only. No network calls. No data upload.
- All stored content is redacted before storage.
- Proof events assert `rawTranscriptStored: false`, `rawUserTextStored: false`.
- `rawTranscriptUpload: true` → plugin fails closed entirely.
- Memory graph is inspectable via `/graph` route.
- Pruned memories are soft-deleted (recoverable) for 7 days.

## Hooks used

| Hook | Purpose |
|------|---------|
| `before_prompt_build` | Inject relevant memories |
| `before_agent_reply` | Detect corrections in assistant responses |
| `after_tool_call` | Capture successful tool workflows |
| `agent_end` | Observe run outcomes for learning |
| `before_compaction` | Snapshot uncommitted memories |
| `llm_output` | Detect preference signals |
| `gateway_start` | Start background learning loop |
| `gateway_stop` | Stop background learning loop |

## Memory capabilities registered

| Capability | Purpose |
|------------|---------|
| `MemoryCorpusSupplement` | Agent can search its own memory via native OpenClaw memory search |
| `MemoryPromptSectionBuilder` | High-importance memories injected in memory prompt section |

## HTTP routes

| Endpoint | Description |
|----------|-------------|
| `/plugins/openclawbrain/status` | Plugin state, memory count, learning stats |
| `/plugins/openclawbrain/graph?limit=50` | Memory graph nodes (redacted) |
| `/plugins/openclawbrain/proof?limit=20` | Recent operation proof events |
| `/plugins/openclawbrain/learn` | Learning engine stats, importance distribution |

## Build phases

### Phase 1: Memory Store + Auto Capture (2-3 days)
- SQLite schema and store
- Correction detection hook
- Basic injection from memory graph instead of flat files
- Proof events
- Config update

### Phase 2: Learning Engine (2-3 days)
- Background service for scoring
- Outcome observation from agent_end
- Importance/freshness scoring
- Pruning
- Link building

### Phase 3: Memory Capability Registration (1-2 days)
- MemoryCorpusSupplement integration
- MemoryPromptSectionBuilder
- Search index

### Phase 4: Self-Regulation + Polish (1-2 days)
- Contradiction resolution
- Growth control
- Size limits
- E2E testing with production traces

### Phase 5: Release (1 day)
- ClawHub publish as v0.2
- Update website
- Update docs

Total estimate: 7-11 days of focused work.

## What success looks like

1. You correct the agent once ("use pnpm, not npm"). Next session, it uses pnpm. You didn't write anything.
2. You run a successful workflow. The system remembers it. Next time you need something similar, it's available.
3. You stop correcting the same things twice.
4. The memory graph is inspectable — you can see what the agent "knows" and why.
5. Prompts stay small because only relevant memories are injected.
6. Old unused memories naturally decay and get pruned.
