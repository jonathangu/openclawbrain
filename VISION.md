# OpenClawBrain — Vision, Requirements, and Build Plan

*Draft — for Jonathan Gu*

---

## The problem

You use the same OpenClaw agent every day. You correct it. You tell it your preferences. You teach it workflows.

Next session, it forgets everything.

You correct it again. And again. And again.

The agent never gets smarter. It has no memory across sessions. Every conversation starts from zero.

---

## The vision

OpenClawBrain makes an OpenClaw agent remember. Not by dumping your entire chat history into the prompt. By building a local memory graph that learns from experience and brings back only what helps.

Five things that must be true:

### 1. Corrections stick automatically

You correct the agent once. It remembers. You never have to say it again.

Not by writing corrections into a file. Not by manually tagging things. The system watches what happens, detects when you correct the agent, and stores the correction. Automatically.

Examples:
- "Use pnpm, not npm" → stored as a correction. Next session: pnpm.
- "The timezone is America/Los_Angeles" → stored as a preference. Every session: correct timezone.
- "That was wrong, do it this way instead" → stored. The wrong path is marked, the right path is remembered.

### 2. The agent learns from experience

Not just from corrections. From outcomes.

When a workflow succeeds, the system notices. When context helps, the system notices. When injection is irrelevant, the system notices. Over time, the system gets better at knowing what to bring back and what to leave alone.

This is not a static rule engine. It adapts.

### 3. Memory is a graph, not a file

Memories are connected. A correction about "use pnpm" relates to a preference about "work on the OpenClawBrain project" relates to a workflow about "build and test the plugin."

The graph captures these relationships. When searching for relevant context, the system doesn't just keyword-match — it follows connections. "You're working on OpenClawBrain, which uses pnpm, which you corrected on April 30."

### 4. Prompts stay small

The system has thousands of memories. It brings back 2-5 relevant ones, not all of them.

Injection is bounded. Only high-confidence, high-relevance memories make it into the prompt. The rest stay in the graph, available when searched but never dumped.

### 5. The system self-regulates

Memory grows when useful, decays when not, prunes when dead. New corrections start important. Old forgotten preferences fade. Unused workflows get cleaned up.

You don't manage the memory. The memory manages itself.

---

## What this is NOT

- **Not a cloud service.** Everything is local. No data leaves your machine.
- **Not a replacement for OpenClaw's memory engine.** It's a plugin that plugs into OpenClaw's memory capabilities (MemoryCorpusSupplement, MemoryPromptSectionBuilder).
- **Not a general AI.** It's a memory and routing layer. It doesn't reason. It remembers and retrieves.
- **Not a RAG system over your documents.** It's not reading your files and embedding them. It's learning from agent interactions — corrections, outcomes, decisions.
- **Not transparent to the user.** You can inspect the memory graph. You can see what it captured. You can see why it injected what it injected.

---

## Core requirements

### R1: Automatic correction capture

The system MUST detect when a user corrects the agent and store the correction without any user action.

Detection signals:
- User says "no", "wrong", "actually", "use X instead of Y", "don't do that"
- Assistant acknowledges a correction and changes behavior
- User repeats a previously stated preference

Storage:
- Redacted content (never raw user text)
- Topic/domain tags (auto-generated)
- Timestamp
- Confidence score (how certain it was a correction)

### R2: Automatic preference capture

The system MUST detect stated preferences and store them.

Detection signals:
- "I prefer X over Y"
- "Always use X"
- "My timezone is X"
- Repeated behavioral patterns (user consistently does X)

### R3: Automatic workflow capture

The system MUST detect successful tool-use patterns and store them.

Detection signals:
- Sequence of tool calls that produces a successful result
- User confirms a workflow worked
- No errors in the tool chain

Storage:
- Workflow description (not raw tool output)
- Tools used
- Decision points
- Outcome

### R4: Memory graph

The system MUST organize memories as a connected graph, not a flat list.

Graph structure:
- **Nodes**: individual memories (corrections, preferences, workflows, context, tool-results)
- **Edges**: relationships between memories (contradicts, supports, extends, related, superseded_by)
- **Node properties**: importance score, freshness score, use count, useful count, timestamps

Graph capabilities:
- Search by content (keyword matching + similarity)
- Search by topic/tags
- Follow edges to find related memories
- Rank results by importance × freshness × relevance

### R5: Background learning

The system MUST run a background process that learns from outcomes.

Learning loop:
1. Observe: was memory injected this turn?
2. Observe: was the response good/bad?
3. Update: if injected and good → boost importance. If injected and bad → lower importance. If not injected and good → no change.
4. Periodic: recalculate all importance/freshness scores
5. Periodic: prune low-value memories
6. Periodic: build edges between related memories

Learning is NOT online training. It's heuristic scoring based on observed patterns.

### R6: Adaptive injection

The system MUST inject relevant memories based on learned importance, not static rules.

Injection algorithm:
1. Classify the turn (what kind of help is needed)
2. Search memory graph for relevant nodes
3. Rank by: importance × freshness × relevance × turn-slice-match
4. Select top N within budget (character limit)
5. Format as bounded prompt section
6. Log injection decision with full reasoning

Mode affects aggressiveness:
- `off`: never inject
- `proof-only`: log what would be injected, don't inject
- `conservative`: inject only high-confidence matches
- `active`: inject more aggressively

### R7: Self-regulation

The system MUST manage its own memory lifecycle.

Regulation rules:
- **New memories start at importance 0.3** — they must prove useful to become important
- **Importance decays over time** — unused memories lose importance at 0.01/day
- **Importance boosts on useful injection** — +0.1 when a memory demonstrably helps
- **Prune when importance < 0.05 and age > 30 days**
- **Contradiction resolution** — when a new correction contradicts an old one, mark old as superseded
- **Size limit** — max 10,000 nodes per agent. Oldest low-importance nodes pruned first.
- **Duplicate detection** — same content hash → merge, not duplicate

### R8: Memory search integration

The system MUST register as an OpenClaw MemoryCorpusSupplement so the agent can natively search its memory.

This means:
- When the agent does a memory search, OpenClawBrain results appear alongside other memory sources
- The agent can find memories through OpenClaw's native memory search, not just through injection
- Memory results are ranked by the same importance/freshness scoring

### R9: Inspectability

The user MUST be able to see what the system captured and why.

Inspection surfaces:
- `/plugins/openclawbrain/status` — current state, memory count, learning stats
- `/plugins/openclawbrain/graph?limit=50` — memory graph nodes (redacted)
- `/plugins/openclawbrain/proof?limit=20` — recent operations
- `/plugins/openclawbrain/learn` — learning engine stats
- `/plugins/openclawbrain/search?q=...` — search the memory graph

### R10: Safety

- **Local only.** No network calls. No data upload. No cloud service.
- **Redacted storage.** All stored content is redacted before persistence. Emails, tokens, phone numbers, URLs removed.
- **No raw user text.** Proof events assert `rawUserTextStored: false`. Memory nodes assert `redactionApplied: true`.
- **Fail-closed.** If `rawTranscriptUpload` is set to `true`, the plugin shuts down. If `allowPromptInjection` is `false`, no injection happens.
- **Not a single point of failure.** If the plugin errors, the agent runs normally without it.
- **Inspectable.** Every memory is visible. Every injection is logged. Every decision is auditable.

---

## What the eval proved

The v0.1 eval pipeline ran 40 real privacy-scrubbed traces through four backends (none, correction-only, correction+heuristics, full-ocb). Blind judging showed:

- **40/40 traces correctly classified** by turn slice
- **full-ocb consistently won** over none — bringing back relevant corrections and context helped
- **Injection was bounded** — the system didn't dump everything, it selected relevant pieces
- **Direct answers stayed silent** — no overhead when the agent didn't need help
- **All 6 turn slices covered**: direct-answer, continuation, correction-follow-up, retrieval-heavy, tool-heavy, stale-memory-conflict

The eval proves the architecture works. The v0.1 plugin didn't implement it.

---

## How this differs from existing systems

### vs. RAG (Retrieval-Augmented Generation)

RAG embeds your documents and searches them by similarity. OpenClawBrain learns from agent interactions.

| | RAG | OpenClawBrain |
|---|---|---|
| Source | Your documents | Agent interactions |
| Search | Embedding similarity | Importance + freshness + relevance |
| Learning | Static (re-embed when docs change) | Adaptive (learns from outcomes) |
| Self-regulation | None | Yes (decay, prune, grow) |

### vs. OpenClaw Memory

OpenClaw Memory is the platform's built-in memory system. OpenClawBrain plugs into it as a MemoryCorpusSupplement.

| | OpenClaw Memory | OpenClawBrain |
|---|---|---|
| Scope | Platform feature | Plugin |
| Capture | User-driven | Automatic |
| Learning | None (store and retrieve) | Yes (importance/freshness scoring) |
| Graph | Flat corpus | Connected graph with edges |

### vs. Conversation History

Chat history is everything that was said. OpenClawBrain is distilled memories.

| | Chat History | OpenClawBrain |
|---|---|---|
| Size | Grows forever | Bounded (max 10K nodes) |
| Content | Raw messages | Redacted memories |
| Relevance | None (dumped or discarded) | Scored and selected |
| Learning | None | Yes |

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                    OpenClaw                       │
│                                                   │
│  Hooks:                                           │
│    before_prompt_build     → inject memories      │
│    before_agent_reply      → detect corrections   │
│    after_tool_call         → capture workflows    │
│    agent_end               → observe outcomes     │
│    before_compaction       → snapshot state       │
│    llm_output              → detect preferences   │
│    gateway_start           → start learning loop  │
│    gateway_stop            → stop learning loop   │
│                                                   │
│  Memory Capabilities:                             │
│    MemoryCorpusSupplement  → searchable memory    │
│    MemoryPromptSectionBuilder → prompt sections   │
│                                                   │
│  Service:                                         │
│    background learning engine                     │
│                                                   │
│  HTTP Routes:                                     │
│    /status   → plugin state                       │
│    /graph    → memory graph (redacted)            │
│    /proof    → operation audit log                │
│    /learn    → learning stats                     │
│    /search   → search memory graph                │
└───────────────────────┬───────────────────────────┘
                        │
     ┌──────────────────▼──────────────────┐
     │        OpenClawBrain v0.2            │
     │                                      │
     │  ┌────────────┐   ┌──────────────┐  │
     │  │  Capture    │   │  Learning    │  │
     │  │  Engine     │   │  Engine      │  │
     │  │             │   │              │  │
     │  │ correction  │   │ scoring      │  │
     │  │ preference  │   │ pruning      │  │
     │  │ workflow    │   │ linking      │  │
     │  │ tool-result │   │ outcome      │  │
     │  └──────┬──────┘   └──────┬───────┘  │
     │         │                 │          │
     │  ┌──────▼─────────────────▼───────┐  │
     │  │      Memory Graph Store         │  │
     │  │      (SQLite, local)            │  │
     │  │                                 │  │
     │  │  nodes: corrections, prefs,     │  │
     │  │         workflows, context,     │  │
     │  │         tool-results            │  │
     │  │  edges: contradicts, supports,  │  │
     │  │         extends, related,       │  │
     │  │         superseded_by           │  │
     │  └──────┬──────────────┬──────────┘  │
     │         │              │             │
     │  ┌──────▼──────┐ ┌────▼──────────┐  │
     │  │  Injection  │ │  Search       │  │
     │  │  Engine     │ │  Index        │  │
     │  │             │ │  (FTS5)       │  │
     │  │ relevance   │ │               │  │
     │  │ importance  │ │ keyword +     │  │
     │  │ freshness   │ │ similarity    │  │
     │  │ budget      │ │               │  │
     │  └─────────────┘ └───────────────┘  │
     │                                      │
     │  ┌─────────────────────────────────┐  │
     │  │  Proof & Audit Layer            │  │
     │  │  (redacted JSONL + SQLite)      │  │
     │  └─────────────────────────────────┘  │
     └──────────────────────────────────────┘
```

---

## Data model

### Memory node

```sql
CREATE TABLE memory_nodes (
  id TEXT PRIMARY KEY,
  agent_id TEXT NOT NULL,
  content TEXT NOT NULL,             -- redacted, never raw
  type TEXT NOT NULL,                -- correction | preference | context | workflow | tool-result
  importance REAL DEFAULT 0.3,      -- 0-1, learned over time
  freshness REAL DEFAULT 1.0,       -- decays, boosted by reuse
  captured_at TEXT NOT NULL,
  last_used_at TEXT,
  use_count INTEGER DEFAULT 0,
  useful_count INTEGER DEFAULT 0,
  tags TEXT,                         -- JSON array
  source_hash TEXT,                  -- SHA-256 for dedup
  origin TEXT,                       -- auto | explicit | derived
  superseded_by TEXT,                -- id of newer memory that replaces this one
  redaction_applied INTEGER DEFAULT 1
);

CREATE INDEX idx_nodes_agent_importance ON memory_nodes(agent_id, importance DESC);
CREATE INDEX idx_nodes_agent_type ON memory_nodes(agent_id, type);
CREATE INDEX idx_nodes_source_hash ON memory_nodes(source_hash);
```

### Memory edge

```sql
CREATE TABLE memory_edges (
  source_id TEXT NOT NULL,
  target_id TEXT NOT NULL,
  relation TEXT NOT NULL,           -- contradicts | supports | extends | related | superseded_by
  weight REAL DEFAULT 0.5,
  created_at TEXT NOT NULL,
  PRIMARY KEY (source_id, target_id),
  FOREIGN KEY (source_id) REFERENCES memory_nodes(id),
  FOREIGN KEY (target_id) REFERENCES memory_nodes(id)
);
```

### Proof event

```sql
CREATE TABLE proof_events (
  id TEXT PRIMARY KEY,
  timestamp TEXT NOT NULL,
  event_type TEXT NOT NULL,         -- capture | inject | learn | prune | search | link
  memory_id TEXT,
  decision TEXT,
  reason TEXT,
  raw_transcript_stored INTEGER DEFAULT 0,
  raw_user_text_stored INTEGER DEFAULT 0
);
```

### Search index (SQLite FTS5)

```sql
CREATE VIRTUAL TABLE memory_search USING fts5(
  content,
  tags,
  content='memory_nodes',
  content_rowid='rowid'
);
```

---

## Configuration

```json
{
  "enabled": true,
  "mode": "conservative",
  "autoCapture": true,
  "backgroundLearning": true,
  "maxMemoryNodes": 10000,
  "importanceThreshold": 0.5,
  "injectionBudget": 3000,
  "learningIntervalMs": 300000,
  "pruneAfterDays": 30,
  "importanceDecayPerDay": 0.01,
  "importanceBoostOnUseful": 0.1,
  "newMemoryImportance": 0.3,
  "scopes": { "agents": ["main"] },
  "rawTranscriptUpload": false
}
```

---

## Build phases

### Phase 1: Memory store + auto capture (2-3 days)
- SQLite schema and CRUD operations
- FTS5 search index
- Correction detection in `before_agent_reply`
- Preference detection in `llm_output`
- Basic injection from memory graph (replaces flat file injection)
- Proof events
- Config update
- **Gate**: corrections are auto-captured and injected in the next session

### Phase 2: Learning engine (2-3 days)
- Background service (`registerService`)
- Importance/freshness scoring algorithm
- Outcome observation from `agent_end`
- Pruning (importance decay, age-based removal)
- Link building between related memories
- **Gate**: importance scores change based on injection outcomes

### Phase 3: Memory capabilities + search (1-2 days)
- `MemoryCorpusSupplement` registration
- `MemoryPromptSectionBuilder` registration
- FTS5 search integration
- `/search` HTTP route
- **Gate**: agent can natively search its own memory graph

### Phase 4: Self-regulation + polish (1-2 days)
- Contradiction resolution
- Growth control (size limits)
- Workflow capture from `after_tool_call`
- `/graph` and `/learn` HTTP routes
- E2E testing
- **Gate**: memory graph self-manages without user intervention

### Phase 5: Release (1 day)
- ClawHub publish as v0.2 (replaces v0.1)
- Update openclawbrain.ai
- Update docs
- **Gate**: fresh install, enable, agent corrects once, remembers next session

**Total: 7-11 days of focused work.**

---

## Success criteria

The product works when ALL of these are true:

1. **You correct the agent once. It remembers.** No manual file editing. No configuration. The system captures the correction automatically and brings it back next session.

2. **The agent gets smarter over time.** After 20 sessions, the agent makes fewer mistakes than after 2 sessions. Not because of model improvement — because of memory.

3. **Prompts stay small.** The agent has thousands of memories. Only 2-5 relevant ones are injected per turn. Total prompt size doesn't grow unboundedly.

4. **You can see what it knows.** `/graph` shows the memory graph. `/proof` shows what was captured and why. No black box.

5. **Old memories decay.** A preference you stated 3 months ago and never reinforced fades away. A correction you made yesterday is fresh and important.

6. **It stays silent when not needed.** On simple direct answers, nothing is injected. No overhead. No noise.

7. **It's safe.** Local only. Redacted. Inspectable. Fail-closed. No data leaves your machine.

---

## What we're not building (yet)

Things that are in the spirit of the vision but out of scope for v0.2:

- **Cross-agent memory sharing.** Each agent has its own memory graph. Sharing between agents is a future feature.
- **Memory export/import.** You can't move memories between machines yet.
- **Memory visualization.** The `/graph` route returns JSON, not a visual graph explorer.
- **Multi-modal memory.** Only text memories for now. No images, no audio.
- **Memory-driven agent behavior changes.** The system injects context; it doesn't change which tools are available or how the agent behaves.
- **Formal RL (policy gradients).** The original vision mentioned policy-gradient routing. v0.2 uses heuristic scoring, not formal RL. RL is a future optimization once the heuristic baseline is proven.

---

## The honest status

**What exists today:** v0.1 — a static file injection plugin. Safe, published, works. Not the product.

**What the eval proved:** The architecture (classification → selection → injection → proof) works correctly on 40 real traces. The foundation is sound.

**What needs to be built:** Everything in this document. Automatic capture. Memory graph. Background learning. Adaptive injection. Self-regulation.

**What's the risk:** The learning loop is heuristic. We can't formally prove that importance scores converge to optimal. We can observe that the system gets better over time and audit every decision.

**What's the bet:** That a local, heuristic, self-regulating memory graph is more useful than no memory at all. The eval evidence says yes. The question is whether the heuristic learning is good enough in practice, or whether it needs formal RL to work well.

We'll find out by building it and testing it on ourselves first.
