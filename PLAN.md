# OpenClawBrain: Complete Implementation Plan

> Fork of [Martian-Engineering/lossless-claw](https://github.com/Martian-Engineering/lossless-claw) with a paper-faithful reinforcement learning layer for context routing, structural graph mutations, async teacher labeling, and multi-source label harvesting.

---

## 1. What We Are Building

OpenClawBrain is a **learning context engine** for OpenClaw. It extends lossless-claw's DAG-based summarization with a knowledge graph whose routing policy learns from outcomes. The agent's context retrieval improves over time — it never relearns the same lesson twice.

**The paper's core result (Lemma 6.1):**

```
∂/∂ρ v_ρ(s_t) = E[z · Σ_{l=t}^{T} ∂logP_ρ(a_l|s_l)/∂ρ]
```

Full-trajectory REINFORCE with terminal reward, stochastic policy, Monte Carlo rollouts. The update sums log-probability gradients over the **entire trajectory** from state t to terminal state T, weighted by the terminal outcome z. This is not one-step — it assigns credit to every routing decision that led to the final outcome.

**Paper assumptions we must honor:**
1. Finite-time game (traversal has bounded hops)
2. Reward only at terminal state (not intermediate)
3. Stochastic policy P_ρ(a|s) yielding a distribution over actions
4. Monte Carlo rollouts to estimate the expectation
5. The value function v_ρ(s) = E[z | s_t = s, a_{t,...,T} ~ P_ρ]

---

## 2. What lossless-claw Already Provides

We inherit all of this — **do not rebuild any of it**:

| Capability | Key Files |
|---|---|
| SQLite storage, every message persisted | `src/db/connection.ts`, `src/db/migration.ts` |
| DAG-based hierarchical summarization | `src/compaction.ts`, `src/summarize.ts` |
| Context assembly with token budgeting | `src/assembler.ts` (ContextAssembler) |
| Expansion routing policy (heuristic) | `src/expansion-policy.ts` |
| Expansion orchestrator (DAG traversal) | `src/expansion.ts` (ExpansionOrchestrator) |
| Retrieval engine (grep, describe, expand) | `src/retrieval.ts` (RetrievalEngine) |
| Agent tools (lcm_grep/describe/expand/expand_query) | `src/tools/*.ts` |
| Authorization grants for sub-agent delegation | `src/expansion-auth.ts` |
| Integrity checking & metrics | `src/integrity.ts` |
| Large file handling | `src/large-files.ts` |
| Full-text search (FTS5) | `src/store/conversation-store.ts` |
| Crash recovery / bootstrap | `src/engine.ts` (reconcileSessionTail) |
| Go TUI for inspection/repair | `tui/` |
| Dependency injection | `src/types.ts` (LcmDependencies) |
| Plugin registration (registerContextEngine + 4 tools) | `index.ts` |

---

## 3. The MDP: Mapping the Paper to Context Routing

### State

```typescript
interface TraversalState {
  currentNodeId: string | null;   // null at seed phase (t=0)
  queryEmbedding: Float32Array;   // the user's query, embedded
  visited: Set<string>;           // nodes already expanded
  fired: string[];                // nodes selected for final context
  budgetRemaining: number;        // chars/tokens left in context window
  hopCount: number;               // traversal depth so far
  maxHops: number;                // hard cap (Assumption 1: finite game)
}
```

### Actions

```typescript
// At each state, the agent chooses one action:
type Action =
  | { type: "traverse"; targetNodeId: string }  // Follow an edge
  | { type: "stop" };                            // Terminate traversal

// Action set at state s_t:
// A(s_t) = { traverse(n) for n in neighbors(currentNode) } ∪ { STOP }
//
// At the seed phase (t=0, currentNode=null):
// A(s_0) = { traverse(n) for n in topK_seeds(query) } ∪ { STOP }
```

### Terminal Conditions (Assumption 1: game ends in finite time)

1. Agent chooses STOP
2. Budget exhausted: `budgetRemaining <= 0`
3. Max hops reached: `hopCount >= maxHops`
4. No outgoing edges (dead end)

### Terminal Reward (Assumption 2: reward only at end)

```typescript
// z ∈ [-1, +1], signed continuous
// Sources ranked by trust (human corrections outrank everything):
type RewardSource = "human" | "self" | "scanner" | "teacher";

// Human:   z = +1 (confirmed good context) or z = -1 (user corrected/rejected)
// Self:    z = +1 (task succeeded, test passed) or z = -1 (task failed)
// Scanner: z ∈ [-0.5, +0.5] (structural heuristic quality signal)
// Teacher: z ∈ [-1, +1] (off-path LLM judgment, lowest automated trust)
```

### Policy (the learned function P_ρ)

```typescript
// For each candidate action a_j at state s_t:
function score(action: Action, state: TraversalState, graph: BrainGraph): number {
  if (action.type === "stop") {
    // STOP score increases as budget depletes and hops accumulate
    return stopBias + budgetPressure * (1 - state.budgetRemaining / totalBudget)
                    + hopPressure * (state.hopCount / state.maxHops);
  }

  const edge = graph.getEdge(state.currentNodeId, action.targetNodeId);
  const targetNode = graph.getNode(action.targetNodeId);

  // Learned weight · structural prior + query relevance + edge-kind bias
  return edge.weight * edge.prior
       + dotProduct(state.queryEmbedding, targetNode.embedding)
       + edgeKindBias[edge.kind];
}

// Softmax policy (stochastic — NEVER argmax during learning):
// P_ρ(a_j | s_t) = exp(score(a_j) / τ) / Σ_k exp(score(a_k) / τ)
//
// τ = temperature:
//   - Learning mode: τ = 1.0 (explore)
//   - Serving mode:  τ = 0.1 (exploit, nearly deterministic but still stochastic)
```

### Update Rule (Lemma 6.1, paper-faithful)

```typescript
// For a completed episode with trajectory [(s_0,a_0), (s_1,a_1), ..., (s_T,a_T)]
// and terminal reward z:

function reinforce(episode: Episode, learningRate: number, baseline: number): WeightUpdates {
  const updates: Map<string, number> = new Map();
  const advantage = episode.reward - baseline;  // Variance reduction

  for (let l = 0; l < episode.trajectory.length; l++) {
    const step = episode.trajectory[l];
    // ∂logP_ρ(a_l|s_l)/∂ρ for the softmax policy
    // = (1 - P_ρ(a_l|s_l)) for the chosen action's weight
    // = -P_ρ(a_l|s_l) for all other actions' weights
    //
    // The full-trajectory sum Σ_{l=t}^{T} gives credit to EVERY step,
    // not just the last one. This is the paper's key correction over Williams (1992).
    const gradLogP = 1 - step.chosenActionProbability;
    const delta = learningRate * advantage * gradLogP;

    // Update the edge weight for the chosen action
    if (step.chosenAction.type === "traverse") {
      const edgeKey = `${step.state.currentNodeId}→${step.chosenAction.targetNodeId}`;
      updates.set(edgeKey, (updates.get(edgeKey) ?? 0) + delta);
    }
    // STOP weight updated similarly via stopBias parameter
  }

  return updates;
}

// Baseline: running exponential moving average of recent episode rewards
// baseline_{n+1} = α · z_n + (1 - α) · baseline_n, α = 0.1
```

---

## 4. Knowledge Graph: Types and Representation

### Node Kinds

```typescript
type NodeKind =
  | "chunk"            // Document/code fragment from workspace
  | "workflow"         // Multi-step procedure (extracted from numbered lists, runbooks)
  | "correction"       // Human-authored fix: "use X not Y", "always do Z for this error"
  | "toolcard"         // When/how to use a specific tool, what failures to expect
  | "episode_anchor"   // Pointer to a prior successful episode for a similar query
  | "summary_bridge";  // Bridges to LCM's existing summary DAG

interface BrainNode {
  id: string;                          // "bn_" prefix + nanoid
  kind: NodeKind;
  content: string;                     // The actual text content
  embedding: Float32Array | null;      // Embedding vector (null until computed)
  sourceUri: string | null;            // Where this came from (file path, session ID, etc.)
  trust: "human" | "scanner" | "teacher" | "self";
  tags: string[];                      // Free-form tags for filtering
  tokenCount: number;                  // Estimated tokens for budget tracking
  createdAt: number;                   // Unix ms
  updatedAt: number;
}
```

### Edge Kinds

```typescript
type EdgeKind =
  | "sibling"    // Same-document adjacency (prior = 1.0, from document order)
  | "semantic"   // Embedding cosine similarity (prior = cosine score)
  | "learned"    // Created by learning/mutation (prior = 0.5)
  | "inhibitory" // Suppresses traversal (weight < 0, "don't go here from here")
  | "bridge";    // Links brain node to LCM summary (cross-system edge)

interface BrainEdge {
  source: string;             // Source node ID
  target: string;             // Target node ID
  kind: EdgeKind;
  weight: number;             // Learned parameter ρ (signed; negative = suppress)
  prior: number;              // Structural/semantic prior (immutable baseline)
  logits: number;             // Cached score for fast softmax (recomputed per query)
  decayedAt: number;          // Last decay timestamp (Unix ms)
  createdAt: number;
}
```

### How This Connects to LCM's Summary DAG

LCM already has a summary DAG (leaf summaries → condensed → higher). We don't replace it. Instead:

1. **`summary_bridge` nodes** link brain graph nodes to specific LCM summaries
2. During brain init, we create bridge nodes for each LCM summary
3. During traversal, if the policy routes to a bridge node, we use LCM's existing expand() to retrieve the content
4. This means the brain can learn to route through LCM's compression hierarchy

```
Brain Knowledge Graph                    LCM Summary DAG
┌──────────┐    bridge    ┌──────────┐
│ correction├─────────────►│ summary  │
│ "use X"  │              │ (leaf)   │
└─────┬────┘              └────┬─────┘
      │ learned                │ parent
┌─────▼────┐              ┌────▼─────┐
│ chunk    │    bridge    │ summary  │
│ "setup"  ├─────────────►│(condensed)│
└─────┬────┘              └──────────┘
      │ sibling
┌─────▼────┐
│ toolcard │
│ "use gh" │
└──────────┘
```

---

## 5. SQLite Schema Extensions

Added alongside LCM's existing `runLcmMigrations()`:

```sql
-- ═══════════════════════════════════════════
-- Brain Knowledge Graph
-- ═══════════════════════════════════════════

CREATE TABLE IF NOT EXISTS brain_nodes (
  id            TEXT PRIMARY KEY,
  kind          TEXT NOT NULL CHECK (kind IN ('chunk','workflow','correction','toolcard','episode_anchor','summary_bridge')),
  content       TEXT NOT NULL,
  embedding     BLOB,                   -- Float32Array as raw bytes
  source_uri    TEXT,
  trust         TEXT NOT NULL DEFAULT 'scanner' CHECK (trust IN ('human','scanner','teacher','self')),
  tags          TEXT NOT NULL DEFAULT '[]',  -- JSON array
  token_count   INTEGER NOT NULL DEFAULT 0,
  created_at    INTEGER NOT NULL,
  updated_at    INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS brain_edges (
  source        TEXT NOT NULL REFERENCES brain_nodes(id) ON DELETE CASCADE,
  target        TEXT NOT NULL REFERENCES brain_nodes(id) ON DELETE CASCADE,
  kind          TEXT NOT NULL CHECK (kind IN ('sibling','semantic','learned','inhibitory','bridge')),
  weight        REAL NOT NULL DEFAULT 0.5,   -- The learned ρ
  prior         REAL NOT NULL DEFAULT 0.5,   -- Immutable structural baseline
  decayed_at    INTEGER NOT NULL,
  created_at    INTEGER NOT NULL,
  PRIMARY KEY (source, target, kind)
);

CREATE INDEX IF NOT EXISTS brain_edges_source_idx ON brain_edges(source);
CREATE INDEX IF NOT EXISTS brain_edges_target_idx ON brain_edges(target);
CREATE INDEX IF NOT EXISTS brain_nodes_kind_idx ON brain_nodes(kind);

-- ═══════════════════════════════════════════
-- Episodes (full traversal records)
-- ═══════════════════════════════════════════

CREATE TABLE IF NOT EXISTS brain_episodes (
  id                TEXT PRIMARY KEY,
  conversation_id   INTEGER,
  query_text        TEXT,
  query_embedding   BLOB,
  trajectory        TEXT NOT NULL,      -- JSON: TrajectoryStep[]
  fired_nodes       TEXT NOT NULL,      -- JSON: string[]
  vetoed_nodes      TEXT NOT NULL DEFAULT '[]',  -- JSON: string[]
  context_chars     INTEGER NOT NULL DEFAULT 0,
  reward            REAL,               -- null until labeled
  reward_source     TEXT,               -- 'human'|'self'|'scanner'|'teacher'
  pack_version      INTEGER,
  created_at        INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS brain_episodes_reward_idx ON brain_episodes(reward);
CREATE INDEX IF NOT EXISTS brain_episodes_created_idx ON brain_episodes(created_at);

-- ═══════════════════════════════════════════
-- Labels (pending reward signals)
-- ═══════════════════════════════════════════

CREATE TABLE IF NOT EXISTS brain_labels (
  id            TEXT PRIMARY KEY,
  episode_id    TEXT NOT NULL REFERENCES brain_episodes(id) ON DELETE CASCADE,
  source        TEXT NOT NULL CHECK (source IN ('human','self','scanner','teacher')),
  value         REAL NOT NULL CHECK (value >= -1.0 AND value <= 1.0),
  confidence    REAL NOT NULL DEFAULT 1.0 CHECK (confidence >= 0.0 AND confidence <= 1.0),
  reason        TEXT,                   -- Why this label was assigned
  applied       INTEGER NOT NULL DEFAULT 0,  -- 0=pending, 1=applied to weights
  created_at    INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS brain_labels_episode_idx ON brain_labels(episode_id);
CREATE INDEX IF NOT EXISTS brain_labels_applied_idx ON brain_labels(applied);

-- ═══════════════════════════════════════════
-- Packs (immutable serving snapshots)
-- ═══════════════════════════════════════════

CREATE TABLE IF NOT EXISTS brain_packs (
  version       INTEGER PRIMARY KEY AUTOINCREMENT,
  node_count    INTEGER NOT NULL,
  edge_count    INTEGER NOT NULL,
  health_json   TEXT NOT NULL,           -- JSON: HealthMetrics
  promoted_at   INTEGER,                 -- null = candidate, non-null = promoted
  rolled_back   INTEGER NOT NULL DEFAULT 0,
  created_at    INTEGER NOT NULL
);

-- ═══════════════════════════════════════════
-- Mutation Proposals
-- ═══════════════════════════════════════════

CREATE TABLE IF NOT EXISTS brain_mutations (
  id            TEXT PRIMARY KEY,
  kind          TEXT NOT NULL CHECK (kind IN ('split','merge','prune','connect','inject')),
  proposal      TEXT NOT NULL,           -- JSON: full proposal details
  evidence      TEXT,                    -- JSON: episode IDs and signals that motivated this
  expected_gain REAL,                    -- Estimated improvement
  status        TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','validated','promoted','rejected')),
  created_at    INTEGER NOT NULL,
  resolved_at   INTEGER
);

-- ═══════════════════════════════════════════
-- Training State
-- ═══════════════════════════════════════════

CREATE TABLE IF NOT EXISTS brain_training_state (
  key           TEXT PRIMARY KEY,
  value         TEXT NOT NULL
);
-- Keys: 'baseline_reward', 'total_episodes', 'last_update_at', 'learning_rate'

-- ═══════════════════════════════════════════
-- Decision Traces (for instant feedback)
-- ═══════════════════════════════════════════

CREATE TABLE IF NOT EXISTS brain_traces (
  id            TEXT PRIMARY KEY,
  episode_id    TEXT REFERENCES brain_episodes(id) ON DELETE SET NULL,
  pack_version  INTEGER,
  query_text    TEXT,
  seed_scores   TEXT NOT NULL,           -- JSON: {nodeId, score}[]
  trajectory    TEXT NOT NULL,           -- JSON: {state, candidates: {action, probability}[], chosen, stopProb}[]
  fired_nodes   TEXT NOT NULL,           -- JSON: string[]
  vetoed_nodes  TEXT NOT NULL DEFAULT '[]',
  context_chars INTEGER NOT NULL,
  footer        TEXT NOT NULL,           -- "Brain v3 · 4 seeds · 3 hops · 5 fired · 1 veto · 2048 chars"
  created_at    INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS brain_traces_created_idx ON brain_traces(created_at DESC);
```

---

## 6. Files to Create

All new code goes under `src/brain/`:

```
src/brain/
├── types.ts              # All interfaces above + TrajectoryStep, Episode, HealthMetrics
├── store.ts              # SQLite CRUD for all brain_* tables
├── graph.ts              # In-memory graph: adjacency lists, neighbor queries, seed selection
├── embed.ts              # Embedding: transformers.js (local) with API fallback
├── policy.ts             # Softmax policy: score(), softmax(), sample(), logProb()
├── traverse.ts           # Full traversal loop: seed → expand → sample → fire → STOP
├── update.ts             # REINFORCE: full-trajectory Δρ with baseline, weight application
├── decay.ts              # Exponential decay: weight → prior over time
├── health.ts             # Graph health: fired/query, dormant%, cross-edges, orphans, churn
├── episode.ts            # Episode recording, trajectory serialization, replay
├── trace.ts              # Decision trace recording, footer generation, trace queries
├── ingest.ts             # Source discovery, chunking (mdast), seed graph building
├── harvester.ts          # Label collection from message ingestion (human/self/scanner)
├── teacher.ts            # Off-path async teacher: LLM evaluation of routing decisions
├── trainer.ts            # Batch update loop: poll episodes → apply REINFORCE → decay → health
├── mutator.ts            # Structural mutations: split/merge/prune/connect/inject proposals
├── pack.ts               # Immutable pack: build candidate, replay-gate, promote/rollback
├── tools.ts              # Agent tools: brain_teach, brain_status, brain_trace
├── service.ts            # Background learner service (registered with OpenClaw)
└── index.ts              # Brain API: init(), query(), teach(), status(), export single surface
```

---

## 7. Integration Points with LCM (Precise File Locations)

### A. Schema Migration

**File:** `src/db/migration.ts`
**Location:** End of `runLcmMigrations()` function (after line 561)
**Change:** Call `runBrainMigrations(db)` to create brain_* tables

### B. Context Assembly — Inject Brain Nodes

**File:** `src/assembler.ts`
**Location:** `ContextAssembler.assemble()` method, after context items are resolved (line ~581)
**Change:** After resolving LCM context items, prepend brain-selected context:

```typescript
// After: const resolved = await this.resolveItems(contextItems);
// Add:
const brainContext = await this.brain?.query(queryEmbedding, {
  budgetChars: Math.floor(tokenBudget * 0.3),  // Reserve 30% for brain
  maxHops: 8,
  temperature: 0.1,  // Low temp for serving
});
if (brainContext?.fired.length) {
  // Inject brain nodes as high-priority resolved items
  // Corrections first, then evidence, then toolcards
  const brainItems = brainContext.fired.map(toBrainResolvedItem);
  resolved.unshift(...brainItems);  // Prepend = highest priority
}
```

### C. Context Assembly — System Prompt Addition

**File:** `src/assembler.ts`
**Location:** `buildSystemPromptAddition()` function (line ~51)
**Change:** After LCM recall section, add brain context section:

```typescript
// If brain nodes were injected:
sections.push(`## Brain Context
The following context was retrieved by the brain's learned routing policy.
- Correction cards override other context when they conflict.
- Use brain_teach to teach the brain new corrections or patterns.
- Use brain_trace to see why specific context was selected.`);
```

### D. Message Ingestion — Label Harvesting

**File:** `src/engine.ts`
**Location:** `ingestSingle()` method, after `createMessageParts()` (line ~1173)
**Change:** Harvest labels from the ingested message:

```typescript
// After: await conversationStore.createMessageParts(msgRecord.messageId, parts);
// Add:
if (this.brain) {
  await this.brain.harvestLabelsFromMessage({
    messageId: msgRecord.messageId,
    conversationId,
    role: stored.role,
    content: stored.content,
    parts,
  });
}
```

**What gets harvested:**
- **Human labels:** User says "no", "wrong", "use X instead", "that's not right" → negative reward on recent episodes. User says "perfect", "exactly", explicit praise → positive reward.
- **Self labels:** Tool result contains "error", "failed", test failure signals → negative. Tool success, "passed", "deployed" → positive.
- **Scanner labels:** Message contains numbered steps → extract workflow node. Message references specific file paths → strengthen chunk→file edges. Repeated tool use patterns → create/strengthen toolcard.

### E. Expansion Policy — Brain-Aware Routing

**File:** `src/expansion-policy.ts`
**Location:** `decideLcmExpansionRouting()` function (the main decision)
**Change:** Before the heuristic routing, check if brain has a learned route:

```typescript
export function decideLcmExpansionRouting(
  input: LcmExpansionRoutingInput,
  brain?: BrainRouter,  // NEW optional parameter
): LcmExpansionRoutingDecision {
  // If brain is available and has relevant nodes, use brain routing
  if (brain) {
    const brainDecision = brain.shouldRoute(input.query, input.candidateSummaryCount);
    if (brainDecision.confident) {
      // Brain handles this query — it has learned routing for this pattern
      // The actual traversal happens in the assembler, not here
      // But we record that brain routing was used for trace/feedback
      return { ...defaultDecision, brainRouted: true, ...brainDecision.overrides };
    }
  }

  // Fall through to existing heuristic routing
  // ... existing code unchanged ...
}
```

### F. Plugin Registration — Add Brain Tools + Service

**File:** `index.ts`
**Location:** Inside `register(api)` method (after tool registrations, line ~1310)
**Change:** Register brain tools and background learner:

```typescript
// After existing tool registrations:

// Brain tools
api.registerTool((ctx) =>
  createBrainTeachTool({ brain, sessionKey: ctx.sessionKey }),
);
api.registerTool((ctx) =>
  createBrainStatusTool({ brain, sessionKey: ctx.sessionKey }),
);
api.registerTool((ctx) =>
  createBrainTraceTool({ brain, sessionKey: ctx.sessionKey }),
);

// Background learner service
api.registerService({
  id: "brain-learner",
  start: () => brain.startLearner(),
  stop: () => brain.stopLearner(),
});
```

---

## 8. The Three New Agent Tools

### brain_teach

```typescript
// "Remember that for deployment errors, always check the CI logs first"
// "Use gh pr create, not hub"
// "When user asks about auth, the answer is in src/auth/README.md"
{
  name: "brain_teach",
  description: "Teach the brain a correction, pattern, or preference. Creates a high-trust node.",
  parameters: {
    instruction: Type.String({ description: "What to remember or correct" }),
    kind: Type.Optional(Type.String({
      enum: ["correction", "toolcard", "workflow"],
      description: "Node kind. Default: correction"
    })),
    tags: Type.Optional(Type.Array(Type.String(), { description: "Tags for filtering" })),
  },
  execute: async (toolCallId, params) => {
    // Creates a brain_node with trust="human"
    // Creates learned edges to relevant existing nodes (by embedding similarity)
    // Assigns z=+1 label to any recent episode this corrects
    // Returns confirmation with node ID
  }
}
```

### brain_status

```typescript
{
  name: "brain_status",
  description: "Show brain health: node/edge counts, pack version, learning stats, recent traces.",
  parameters: {},
  execute: async () => {
    // Returns:
    // - Pack version, promoted_at
    // - Node counts by kind
    // - Edge counts by kind
    // - Health metrics (fired/query avg, dormant%, orphan count)
    // - Recent episode count, avg reward
    // - Last trainer run timestamp
    // - Pending labels count
  }
}
```

### brain_trace

```typescript
{
  name: "brain_trace",
  description: "Show the detailed decision trace for the most recent (or specified) brain query.",
  parameters: {
    traceId: Type.Optional(Type.String({ description: "Specific trace ID. Default: most recent." })),
  },
  execute: async (toolCallId, params) => {
    // Returns full trace:
    // - Query text and embedding summary
    // - Seed ranking (top 10 with scores)
    // - Each traversal step: state, candidate probabilities, chosen action, STOP probability
    // - Fired nodes (content snippets)
    // - Vetoed nodes (which inhibitory edges suppressed them)
    // - Total context chars
    // - Footer: "Brain v3 · 4 seeds · 3 hops · 5 fired · 1 veto · 2048 chars · trace abc123"
    // - Labels applied (if any)
    // - Reward (if resolved)
  }
}
```

---

## 9. Brain Initiation (Cold Start)

When a user first enables the brain (`openclawbrain init` or on first plugin load):

### Step 1: Discover Sources

```typescript
async function discoverSources(workspaceRoot: string): Promise<Source[]> {
  // 1. Markdown files: **/*.md (docs, READMEs, runbooks)
  // 2. Config files: package.json, tsconfig.json, .env.example, Dockerfile
  // 3. OpenClaw session transcripts: ~/.openclaw/sessions/*.jsonl
  // 4. Existing LCM summaries: SELECT * FROM summaries (bridge to DAG)
  // 5. Tool descriptions from OpenClaw plugin registry (if available)
  // Returns array of { uri, content, type } objects
}
```

### Step 2: Structure-Aware Chunking

```typescript
async function chunkSources(sources: Source[]): Promise<Chunk[]> {
  // Markdown: Use remark/mdast to parse AST
  //   - Split on heading boundaries (## / ### / ####)
  //   - Keep code blocks intact (never split mid-block)
  //   - Preserve list items as atomic units
  //   - Target: 200-500 tokens per chunk
  //
  // Code files: Split on function/class boundaries
  //   - Use simple regex heuristics (export function, class, const ... = () =>)
  //   - Keep imports with their function
  //
  // Config: One chunk per file (usually small enough)
  //
  // Session transcripts: Extract key episodes
  //   - Tool successes with context
  //   - User corrections
  //   - Workflow completions
}
```

### Step 3: Create Nodes + Compute Embeddings

```typescript
async function createNodesFromChunks(chunks: Chunk[]): Promise<BrainNode[]> {
  // For each chunk:
  //   1. Determine kind: chunk, workflow (if numbered steps), toolcard (if tool reference)
  //   2. Compute embedding via embed()
  //   3. Estimate token count
  //   4. Insert into brain_nodes
}

// Embedding strategy:
// Primary: transformers.js with all-MiniLM-L6-v2 (local, no API key)
// Fallback: OpenAI text-embedding-3-small via API (if configured)
// The embedding model is configurable via plugin config
```

### Step 4: Create Cold-Start Edges

```typescript
async function createColdStartEdges(nodes: BrainNode[]): Promise<void> {
  // 1. Sibling edges: nodes from same document, adjacent chunks
  //    weight = 0.8, prior = 1.0, kind = "sibling"
  //
  // 2. Semantic edges: for each node, find top-3 most similar (cosine > 0.7)
  //    weight = cosine_score, prior = cosine_score, kind = "semantic"
  //
  // 3. Bridge edges: link summary_bridge nodes to their LCM summary IDs
  //    weight = 0.5, prior = 0.5, kind = "bridge"
  //
  // 4. Toolcard edges: if a chunk references a tool name, link to toolcard
  //    weight = 0.6, prior = 0.6, kind = "learned"
}
```

### Step 5: Build Pack v0 + Health Check

```typescript
async function buildInitialPack(): Promise<number> {
  // 1. Compute health metrics on the seed graph
  // 2. Run smoke test: auto-generate 5 queries from document headings
  //    - For each: run traversal, check that >= 1 relevant node fires
  //    - Log results but don't block on failures
  // 3. Insert brain_packs record with health_json
  // 4. Set promoted_at = now (this is the initial serving pack)
  // 5. Return pack version number
}
```

### Step 6: Show Immediate Feedback

```
Brain initialized:
  Nodes: 142 (87 chunks, 23 workflows, 12 toolcards, 20 bridges)
  Edges: 489 (134 sibling, 267 semantic, 68 learned, 20 bridge)
  Health: 3.2 fired/query avg · 0% dormant · 0 orphans
  Pack: v1 (promoted)
  Smoke test: 5/5 queries returned relevant context

  Use brain_teach to add corrections. Use brain_trace to inspect decisions.
```

---

## 10. The Harvester: Label Collection

### Human Labels (Highest Trust)

Detected during message ingestion in `engine.ts`:

```typescript
// Patterns that signal negative human feedback:
const NEGATIVE_PATTERNS = [
  /\bno[,.]?\s+(that'?s?\s+)?(not|wrong|incorrect)/i,
  /\bdon'?t\s+(use|do|try)/i,
  /\binstead\s+(use|do|try)/i,
  /\bactually[,]?\s+(it'?s|the|you)/i,
  /\bthat'?s\s+not\s+(right|correct|what)/i,
];

// Patterns that signal positive human feedback:
const POSITIVE_PATTERNS = [
  /\b(perfect|exactly|correct|right|yes[!.])\b/i,
  /\bthat('?s| is)\s+(exactly\s+)?(right|correct|what\s+i)/i,
  /\bthank(s| you)/i,  // Weak positive
];

// Explicit teach commands are always z=+1 for the created node
// and z=-1 for whatever the brain previously suggested (if correction)
```

### Self Labels (Outcome-Based)

Detected from tool results and task completion:

```typescript
// Tool success signals (from message parts with partType="tool"):
// - toolStatus === "success" or tool_output contains no error → z = +0.5
// - toolStatus === "error" or tool_output contains stack trace → z = -0.5
// - Test runner output: "X passed, Y failed" → z based on pass rate
// - Git operations: commit/push succeeded → z = +0.3
// - Build output: "Build succeeded" → z = +0.3

// Session completion signals:
// - User ends session after successful task → z = +0.3 for all episodes in session
// - User abandons mid-task (no positive closure) → z = -0.1 (weak negative)
```

### Scanner Labels (Structural Heuristics)

Run periodically by the trainer, not on the hot path:

```typescript
// 1. Workflow extraction: numbered steps in a message → create workflow node
//    If similar workflow already exists, strengthen edges between them
//
// 2. Repeated patterns: same tool sequence used 3+ times → create/strengthen toolcard
//    z = +0.3 for episodes that used this pattern
//
// 3. File reference patterns: message mentions file path that appears in chunks
//    Strengthen edges between the chunk and the conversation context
//    z = +0.2 for episodes that correctly routed to that chunk
//
// 4. Dormancy detection: nodes that haven't fired in 50+ episodes
//    Flag for potential pruning (don't label, just track)
```

### Teacher Labels (Off-Path LLM)

Run asynchronously by the background learner:

```typescript
async function teacherLabel(episode: Episode, deps: LcmDependencies): Promise<Label> {
  // CRITICAL: Teacher sees ONLY what the router saw. No cheating.
  const prompt = `You are evaluating a context routing decision.

Query: "${episode.queryText}"

Candidate nodes the router could have chosen:
${episode.trajectory.map(step =>
  step.candidates.map(c => `- ${c.nodeId}: ${c.snippet}`).join('\n')
).join('\n\n')}

Nodes actually selected (fired):
${episode.firedNodes.map(id => `- ${id}: ${getNodeSnippet(id)}`).join('\n')}

Nodes suppressed (vetoed):
${episode.vetoedNodes.map(id => `- ${id}: ${getNodeSnippet(id)}`).join('\n')}

Was this the right context for the query? Score from -1.0 (terrible selection) to +1.0 (perfect selection).
Consider: relevance, completeness, conciseness, whether corrections were respected.

Return ONLY a JSON object: {"score": <number>, "reason": "<brief explanation>"}`;

  const result = await deps.complete({
    model: teacherModel,
    messages: [{ role: "user", content: prompt }],
    maxTokens: 200,
    temperature: 0.1,
  });

  // Parse score, clamp to [-1, 1]
  // Return as Label with source="teacher"
}
```

---

## 11. The Trainer: Batch Update Loop

Runs as a background service registered with OpenClaw:

```typescript
class BrainTrainer {
  private interval: NodeJS.Timeout | null = null;
  private running = false;

  start(intervalMs = 30_000) {
    this.interval = setInterval(() => this.tick(), intervalMs);
  }

  stop() {
    if (this.interval) clearInterval(this.interval);
  }

  async tick() {
    if (this.running) return;  // Skip if previous tick still running
    this.running = true;
    try {
      await this.processLabels();
      await this.runTeacher();
      await this.applyUpdates();
      await this.runDecay();
      await this.proposeMutations();
      await this.checkPromotion();
    } finally {
      this.running = false;
    }
  }

  // Step 1: Process pending labels → assign rewards to episodes
  async processLabels() {
    const pending = await store.getPendingLabels();
    for (const label of pending) {
      const episode = await store.getEpisode(label.episodeId);
      if (!episode) continue;

      // If episode already has a higher-trust reward, skip
      if (episode.reward !== null && trustRank(episode.rewardSource) >= trustRank(label.source)) {
        await store.markLabelApplied(label.id);
        continue;
      }

      // Apply label as episode reward
      await store.setEpisodeReward(episode.id, label.value, label.source);
      await store.markLabelApplied(label.id);
    }
  }

  // Step 2: Run teacher on episodes without labels (async, off-path)
  async runTeacher() {
    const unlabeled = await store.getUnlabeledEpisodes({ limit: 5 });
    for (const episode of unlabeled) {
      const label = await teacherLabel(episode, deps);
      await store.insertLabel({
        episodeId: episode.id,
        source: "teacher",
        value: label.score,
        reason: label.reason,
      });
    }
  }

  // Step 3: Apply REINFORCE updates for episodes with rewards
  async applyUpdates() {
    const episodes = await store.getEpisodesForUpdate({ limit: 20 });
    if (episodes.length === 0) return;

    const baseline = await store.getTrainingState("baseline_reward") ?? 0;
    let newBaseline = baseline;

    for (const episode of episodes) {
      const updates = reinforce(episode, LEARNING_RATE, baseline);
      await store.applyWeightUpdates(updates);
      await store.markEpisodeUpdated(episode.id);

      // Update running baseline
      newBaseline = 0.9 * newBaseline + 0.1 * episode.reward;
    }

    await store.setTrainingState("baseline_reward", newBaseline);
    await store.setTrainingState("last_update_at", Date.now());
  }

  // Step 4: Decay all edge weights toward their priors
  async runDecay() {
    const DECAY_RATE = 0.995;  // Per tick
    const DECAY_INTERVAL = 60_000;  // Only decay every 60s
    const lastDecay = await store.getTrainingState("last_decay_at") ?? 0;
    if (Date.now() - lastDecay < DECAY_INTERVAL) return;

    await store.decayAllWeights(DECAY_RATE);
    await store.setTrainingState("last_decay_at", Date.now());
  }

  // Step 5: Propose structural mutations based on episode patterns
  async proposeMutations() { /* see section 12 */ }

  // Step 6: Check if candidate pack should be promoted
  async checkPromotion() { /* see section 13 */ }
}
```

---

## 12. Structural Mutations

### Split

When a node is semantically mixed — only part of its content fires:

```typescript
// Evidence: node X fires in episodes where query relates to topic A,
// but node X also contains content about topic B that never fires.
// Signal: High firing rate but low average reward → mixed content.
// Action: Split node X into X_a (topic A content) and X_b (topic B content).
// Edges from X are duplicated to both children, then the trainer learns which to strengthen.
```

### Merge

When near-duplicate nodes always co-fire:

```typescript
// Evidence: nodes X and Y fire together in >80% of episodes where either fires.
// Cosine similarity between X and Y embeddings > 0.9.
// Action: Merge into single node Z with combined content (deduplicated).
// Edges from both X and Y are redirected to Z.
```

### Connect

When successful episodes repeatedly bridge two graph regions:

```typescript
// Evidence: episodes with reward > 0.5 frequently fire node X then node Y,
// but X and Y have no direct edge.
// Action: Create learned edge X→Y with initial weight = average reward of bridging episodes.
```

### Prune

When edges stay dormant across decay windows:

```typescript
// Evidence: edge weight has decayed to within 0.01 of 0 (or prior) for 100+ episodes.
// No episode in last 200 has traversed this edge.
// Action: Remove edge. (Node stays unless it becomes an orphan.)
```

### Inject

When a human correction or successful episode deserves its own node:

```typescript
// Evidence: User used brain_teach to create a correction.
// Or: an episode with reward = +1 from human contains a novel pattern not in any existing node.
// Action: Create new node, compute embedding, create edges to nearest neighbors.
```

### Validation Gate

Every mutation is a **proposal**, not a live change:

```typescript
interface MutationProposal {
  id: string;
  kind: "split" | "merge" | "prune" | "connect" | "inject";
  affectedNodes: string[];
  affectedEdges: string[];
  evidence: { episodeIds: string[]; avgReward: number; frequency: number };
  expectedGain: number;
}

async function validateMutation(proposal: MutationProposal): Promise<boolean> {
  // 1. Apply mutation to a temporary in-memory copy of the graph
  // 2. Replay last 50 episodes against the mutated graph
  // 3. Compare: does average reward improve? Does health hold?
  // 4. If yes: promote mutation. If no: reject.
  // 5. Record result in brain_mutations table.
  return replayImproves && healthHolds;
}
```

---

## 13. Pack Promotion

```typescript
async function checkPromotion(): Promise<void> {
  const currentPack = await store.getCurrentPack();
  const health = await computeHealth();

  // Build candidate pack
  const candidate = await buildCandidatePack(health);

  // Replay gate: run last 100 episodes against candidate
  const replayResult = await replayAgainstPack(candidate, { limit: 100 });

  // Promote only if:
  // 1. Average replay reward >= current pack's average (no regression)
  // 2. Health metrics within bounds (fired/query >= 1.0, dormant% < 30%, orphans < 10%)
  // 3. No human-labeled episodes regressed
  if (replayResult.avgReward >= currentPack.avgReward
      && health.firedPerQuery >= 1.0
      && health.dormantPercent < 0.3
      && health.orphanCount < 10
      && replayResult.humanEpisodesRegressed === 0) {
    await store.promotePack(candidate.version);
    log.info(`[brain] Pack v${candidate.version} promoted. Reward: ${replayResult.avgReward.toFixed(3)}`);
  } else {
    log.info(`[brain] Pack v${candidate.version} rejected. Current: ${currentPack.avgReward.toFixed(3)}, Candidate: ${replayResult.avgReward.toFixed(3)}`);
  }
}
```

---

## 14. Health Metrics

```typescript
interface HealthMetrics {
  nodeCount: number;
  edgeCount: number;
  nodesByKind: Record<NodeKind, number>;
  edgesByKind: Record<EdgeKind, number>;
  firedPerQuery: number;         // Avg nodes fired per episode (target: 2-8)
  dormantPercent: number;        // % of nodes that haven't fired in 100 episodes (target: < 30%)
  inhibitoryPercent: number;     // % of edges with weight < 0 (informational)
  orphanCount: number;           // Nodes with no edges (target: 0)
  avgPathLength: number;         // Avg hops per episode (target: 2-5)
  avgReward: number;             // Running average episode reward (target: > 0)
  crossFileEdgePercent: number;  // % of edges connecting nodes from different sources
  churn: number;                 // Weight changes per tick (stability indicator)
  packVersion: number;
  lastUpdateAt: number;
  totalEpisodes: number;
}
```

---

## 15. Decision Trace Format

Every brain query emits a trace for instant feedback:

```typescript
interface DecisionTrace {
  id: string;           // "bt_" + nanoid
  episodeId: string;
  packVersion: number;
  queryText: string;
  seeds: Array<{ nodeId: string; kind: NodeKind; score: number; snippet: string }>;
  trajectory: Array<{
    hop: number;
    currentNode: string;
    candidates: Array<{
      action: string;      // "traverse:bn_xxx" or "stop"
      score: number;
      probability: number;
    }>;
    chosen: string;
    stopProbability: number;
  }>;
  fired: Array<{ nodeId: string; kind: NodeKind; snippet: string; tokenCount: number }>;
  vetoed: Array<{ nodeId: string; reason: string }>;  // Which inhibitory edge blocked it
  contextChars: number;
  footer: string;   // "Brain v3 · 4 seeds · 3 hops · 5 fired · 1 veto · 2048 chars · trace bt_abc123"
  createdAt: number;
}
```

The footer is appended to every brain context injection so the user always knows what happened:

```
[Brain v3 · 4 seeds · 3 hops · 5 fired · 1 veto · 2048 chars · trace bt_abc123]
```

---

## 16. Logging

Use LCM's existing `deps.log` interface (injected via LcmDependencies):

```typescript
// Structured log levels:
deps.log.info(`[brain] Pack v${version} promoted. Nodes: ${count}, Reward: ${avg.toFixed(3)}`);
deps.log.warn(`[brain] High dormancy: ${dormant}% of nodes haven't fired in 100 episodes`);
deps.log.error(`[brain] Trainer tick failed: ${error.message}`);
deps.log.debug(`[brain] Traversal: ${hops} hops, ${fired} fired, STOP prob: ${stopProb.toFixed(3)}`);

// All brain operations logged with [brain] prefix for easy filtering
// Decision traces stored in brain_traces table for deep inspection
// Health metrics logged every trainer tick
```

---

## 17. Configuration

Added to `openclaw.plugin.json` configSchema and environment variables:

```typescript
interface BrainConfig {
  // Core
  enabled: boolean;                    // default: true
  embeddingModel: string;              // default: "all-MiniLM-L6-v2" (local transformers.js)
  embeddingProvider: string;           // default: "local" | "openai"

  // Traversal
  maxHops: number;                     // default: 8
  servingTemperature: number;          // default: 0.1
  learningTemperature: number;         // default: 1.0
  budgetFraction: number;              // default: 0.3 (30% of context window for brain)
  maxSeeds: number;                    // default: 10
  semanticThreshold: number;           // default: 0.7 (cosine cutoff for semantic edges)

  // Learning
  learningRate: number;                // default: 0.01
  baselineAlpha: number;               // default: 0.1 (EMA smoothing)
  decayRate: number;                   // default: 0.995
  trainerIntervalMs: number;           // default: 30000 (30s)
  teacherModel: string;                // default: same as LCM summary model
  teacherEnabled: boolean;             // default: true

  // Promotion
  replayEpisodeCount: number;          // default: 100
  minFiredPerQuery: number;            // default: 1.0
  maxDormantPercent: number;           // default: 0.3
  maxOrphanCount: number;              // default: 10

  // Mutation
  mutationsEnabled: boolean;           // default: true
  splitThreshold: number;              // default: 0.3 (reward variance threshold)
  mergeThreshold: number;              // default: 0.9 (cosine similarity threshold)
  pruneAfterEpisodes: number;          // default: 200
  connectMinFrequency: number;         // default: 3 (co-firing count)
}

// Environment variable overrides (all prefixed BRAIN_):
// BRAIN_ENABLED, BRAIN_EMBEDDING_MODEL, BRAIN_MAX_HOPS,
// BRAIN_LEARNING_RATE, BRAIN_TEACHER_MODEL, BRAIN_TEACHER_ENABLED,
// BRAIN_MUTATIONS_ENABLED, BRAIN_TRAINER_INTERVAL_MS
```

---

## 18. Build Order (Implementation Phases)

### Phase 1: Foundation (src/brain/types.ts, store.ts, graph.ts)
- All TypeScript interfaces
- SQLite migration for brain_* tables
- CRUD operations for nodes, edges, episodes, labels, packs, mutations, traces, training state
- In-memory graph: load from SQLite, adjacency list, neighbor queries, seed selection by embedding similarity

### Phase 2: Core Algorithm (policy.ts, traverse.ts, update.ts, decay.ts, health.ts)
- Softmax policy with temperature
- Full traversal loop: seed → sample → fire → STOP
- Episode recording with full trajectory
- REINFORCE update with baseline
- Weight decay
- Health metrics computation
- **Tests proving:** stochastic sampling, positive/negative reward effects, full-trajectory credit, inhibitory edges, STOP pressure, decay convergence

### Phase 3: Embedding + Ingest (embed.ts, ingest.ts)
- Embedding via transformers.js (local) with API fallback
- Source discovery (markdown, code, sessions, LCM summaries)
- Structure-aware chunking (mdast for markdown)
- Cold-start graph building (nodes + sibling/semantic/bridge edges)
- Pack v0 creation with health check
- Smoke test

### Phase 4: Integration (modifications to assembler.ts, engine.ts, expansion-policy.ts, index.ts)
- Brain context injection in assembler
- Label harvesting in message ingestion
- Brain-aware expansion routing
- Tool registration (brain_teach, brain_status, brain_trace)
- Decision trace recording
- Footer in every brain context block

### Phase 5: Harvester + Teacher + Trainer (harvester.ts, teacher.ts, trainer.ts, service.ts)
- Human label detection (positive/negative patterns in messages)
- Self label detection (tool success/failure)
- Scanner labels (periodic structural analysis)
- Teacher labels (off-path LLM evaluation)
- Trainer batch loop (process labels → REINFORCE → decay → health)
- Background service registration

### Phase 6: Mutations + Packs (mutator.ts, pack.ts, episode.ts)
- Split/merge/prune/connect/inject proposals
- Mutation validation via replay gate
- Pack building and promotion
- Pack rollback
- Episode replay harness

### Phase 7: Polish
- Update README.md
- Update openclaw.plugin.json with brain config
- Update package.json metadata
- Ensure all existing LCM tests still pass
- Write brain-specific tests
- Push to GitHub

---

## 19. Testing Strategy

### Existing Tests (MUST NOT BREAK)
All tests in `test/*.test.ts` must continue to pass. The brain is additive.

### New Brain Tests

```
test/brain/
├── policy.test.ts        # Softmax is stochastic, temperature works, STOP pressure
├── traverse.test.ts      # Full traversal produces valid episodes, respects budget/maxHops
├── update.test.ts        # REINFORCE: positive z strengthens, negative weakens, baseline reduces variance
├── decay.test.ts         # Weights converge toward prior over time
├── health.test.ts        # Metrics correctly computed
├── graph.test.ts         # CRUD, neighbor queries, seed selection
├── store.test.ts         # Round-trip persistence for all brain tables
├── harvester.test.ts     # Pattern detection for human/self/scanner labels
├── mutator.test.ts       # Split/merge/prune/connect/inject proposals
├── pack.test.ts          # Build, promote, rollback, replay gate
├── ingest.test.ts        # Source discovery, chunking, seed graph
├── integration.test.ts   # Full pipeline: query → episode → label → update → improved query
└── trace.test.ts         # Trace recording, footer generation
```

### Key Test Patterns (matching LCM style)

```typescript
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

// Temp dir pattern (same as engine.test.ts)
const tempDirs: string[] = [];
function createTestDb() {
  const dir = mkdtempSync(join(tmpdir(), "brain-test-"));
  tempDirs.push(dir);
  return join(dir, "test.db");
}
afterEach(() => {
  for (const dir of tempDirs) rmSync(dir, { recursive: true, force: true });
  tempDirs.length = 0;
});

// Mock dependencies (same as LCM test pattern)
function makeBrainDeps(overrides?: Partial<LcmDependencies>): LcmDependencies {
  return {
    config: { enabled: true, databasePath: createTestDb(), /* ... */ },
    complete: vi.fn(async () => ({ content: [{ type: "text", text: '{"score": 0.5}' }] })),
    log: { info: vi.fn(), warn: vi.fn(), error: vi.fn(), debug: vi.fn() },
    ...overrides,
  } as LcmDependencies;
}
```

---

## 20. Dependencies

### New (to add to package.json)

```json
{
  "dependencies": {
    "@xenova/transformers": "^3.0.0"  // Local embeddings (transformers.js)
  }
}
```

Everything else uses what LCM already has:
- `node:sqlite` (DatabaseSync) — already used
- `node:crypto` (randomUUID) — already used
- `@sinclair/typebox` — already used for tool schemas
- `vitest` — already used for tests

### Optional (deferred)
- `pino` — if structured logging beyond deps.log is needed later
- `@lancedb/lancedb` — if vector index needed at scale (SQLite linear scan fine for v1)

---

## 21. What NOT to Build

- No separate daemon process (learner runs as OpenClaw background service)
- No LanceDB (SQLite linear scan is fine for < 10K nodes)
- No dashboard UI (brain_trace tool + TUI is enough for v1)
- No multi-workspace support (single brain per OpenClaw instance)
- No `registerContextEngine` replacement (use `before_prompt_build`-style injection via assembler modification)
- No launchd/systemd service management
- No separate npm packages / monorepo (everything stays in the fork)

---

## 22. Recurrence Gate

Not every query should go through brain traversal. Static documentation lookups, one-off reads, and simple questions don't benefit from learned routing.

```typescript
function shouldUseBrain(query: string, recentEpisodes: Episode[]): boolean {
  // Skip brain if:
  // 1. Query is a simple file read ("read src/foo.ts") → no
  // 2. Query has zero embedding similarity to any brain node → no
  // 3. Brain has < 10 nodes (not initialized) → no
  //
  // Use brain if:
  // 1. Query matches patterns from prior successful episodes → yes
  // 2. Query relates to a domain with corrections/toolcards → yes
  // 3. Default for structured/workflow queries → yes
  //
  // Fall through: use brain (default on)
  return true;
}
```

---

## Summary

This plan transforms lossless-claw from a **lossless memory system** (never lose context) into a **learning memory system** (get better at what context to surface). The paper's full-trajectory REINFORCE update is the engine. The summary DAG is the foundation. The brain graph is the learned overlay. The teacher, harvester, and mutator make it improve autonomously.

Every existing LCM capability is preserved. The brain is purely additive.
