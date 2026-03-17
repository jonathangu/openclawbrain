# Architecture

This document explains the architecture that OpenClawBrain is built on.

Important framing:
- the **transcript-memory substrate** is inherited from lossless-claw / LCM
- the **learned routing layer** is the OpenClawBrain-specific addition on top
- this doc covers both, so deeper sections will still talk in substrate terms where that is the real implementation boundary

## Architecture in one minute

OpenClawBrain has two cooperating layers:

1. **LCM substrate**
   - persists messages and message parts
   - compacts old history into a summary DAG
   - assembles summaries plus fresh raw turns back into model context
   - supports expansion back into compressed detail

2. **Learned routing layer**
   - decides whether to use learned retrieval, shadow the route, or skip with an explicit reason
   - retrieves from immutable promoted packs only
   - records episodes and traces on the live path
   - trains in the background from structured evidence and replay-gated promotion
   - treats LCM summaries as a routing prior rather than canonical truth
   - commits explicit user corrections through a provenance-grounded teach path

The repo identity is OpenClawBrain. The substrate history is still real, but it should not be confused with the whole product story.

A useful mental model is:

- **LCM summary DAG = search/value abstraction over long history**
- **raw transcript expansion = precision path back to source**
- **typed brain memories = durable current-truth overlay**

That means summary nodes help the runtime decide *where to look* and *when to expand*, while explicit user-grounded correction memories decide *what should currently win* when older transcript abstractions conflict.

Related deep dives:
- `docs/routing-prior.md`
- `docs/corrections.md`

## Data model

### Conversations and messages

Every OpenClaw session maps to a **conversation**. The first time a session ingests a message, LCM creates a conversation record keyed by the runtime session ID.

Messages are stored with:
- **seq** — monotonically increasing sequence number within the conversation
- **role** — `user`, `assistant`, `system`, or `tool`
- **content** — plain-text extraction of the message
- **tokenCount** — estimated token count (~4 chars/token)
- **createdAt** — insertion timestamp

Each message also has **message_parts** — structured content blocks that preserve the original shape (text blocks, tool calls, tool results, reasoning, file content, and so on). That lets the assembler reconstruct rich content instead of only flattened text.

### The summary DAG

Summaries form a directed acyclic graph with two main node types:

**Leaf summaries** (depth 0, kind `"leaf"`)
- created from a chunk of raw messages
- linked to source messages via `summary_messages`
- contain a narrative summary with timestamps
- typically 800–1200 tokens

**Condensed summaries** (depth 1+, kind `"condensed"`)
- created from a chunk of summaries at the same depth
- linked to parent summaries via `summary_parents`
- use progressively more abstract prompts as depth increases
- typically 1500–2000 tokens

Every summary carries:
- **summaryId** — `sum_` + 16 hex chars
- **conversationId** — which conversation it belongs to
- **depth** — hierarchy position
- **earliestAt / latestAt** — source time range
- **descendantCount** — total transitive ancestor coverage
- **fileIds** — referenced large files
- **tokenCount** — estimated tokens

### Context items

The `context_items` table maintains the ordered list of what the model sees for each conversation. Each entry is either a message reference or a summary reference.

When compaction creates a summary from a range of messages or summaries, the source items are replaced by a single summary item. That keeps the active context compact while preserving ordering.

## Compaction lifecycle

### Ingestion

When OpenClaw processes a turn, it calls the context engine lifecycle hooks:

1. **bootstrap** — reconciles the JSONL session file with the LCM database for crash recovery
2. **ingest / ingestBatch** — persists new messages and appends them to `context_items`
3. **afterTurn** — ingests new messages, then evaluates whether compaction should run

### Leaf compaction

The leaf pass converts raw messages into leaf summaries:
1. identify the oldest contiguous chunk of raw messages outside the protected fresh tail
2. cap the chunk at `leafChunkTokens`
3. concatenate message content with timestamps
4. pass prior summary context for continuity when available
5. summarize with the leaf prompt
6. normalize provider output into plain text
7. fall back deterministically if normalization is empty or poor
8. persist the summary and replace the source range in `context_items`

### Condensation

The condensed pass merges summaries at the same depth into a higher-level summary:
1. find the shallowest eligible depth with enough contiguous same-depth summaries
2. concatenate content with time-range headers
3. summarize with the depth-appropriate prompt
4. use the same escalation path (normal → aggressive → deterministic fallback)
5. persist the new summary and replace the source range in `context_items`

### Compaction modes

**Incremental**
- runs after turns when raw tokens outside the fresh tail exceed the configured threshold
- may continue with condensation passes depending on `incrementalMaxDepth`
- failures are best-effort and should not break the conversation

**Full sweep**
- repeatedly runs leaf passes, then condensation passes, until no further savings are found
- used for manual `/compact` and overflow recovery

**Budget-targeted**
- runs bounded rounds of full sweeps until context falls under a target budget

### Three-level summarization escalation

Every summarization attempt follows the same escalation:
1. **normal** — standard prompt
2. **aggressive** — tighter prompt with lower token target
3. **fallback** — deterministic truncation

That guarantees compaction still makes progress when model output is weak or malformed.

## Context assembly

Before each model turn, the assembler builds the message array from summaries plus the fresh raw tail:

```text
[summary₁, summary₂, ..., summaryₙ, message₁, message₂, ..., messageₘ]
 ├── budget-constrained ──┤  ├──── fresh tail (always included) ────┤
```

Steps:
1. fetch `context_items` in order
2. resolve each item into either reconstructed rich messages or XML-wrapped summaries
3. split into evictable prefix and protected fresh tail
4. always include the fresh tail, even if it is expensive
5. backfill the remaining budget from the newest evictable items
6. normalize assistant content blocks and sanitize tool pairing

## XML summary format

Summaries are shown to the model as user messages with XML wrappers, for example:

```xml
<summary id="sum_abc123" kind="leaf" depth="0" descendant_count="0"
         earliest_at="2026-02-17T07:37:00" latest_at="2026-02-17T08:23:00">
  <content>
    ...summary text...

    Expand for details about: exact error messages, full config diff, intermediate debugging steps
  </content>
</summary>
```

That metadata gives the model temporal scope, hierarchy, and a hint about what can be expanded for detail.

## Expansion system

When summaries are too compressed for a task, agents use `lcm_expand_query` to recover detail.

High-level flow:
1. agent calls `lcm_expand_query` with a prompt and either `summaryIds` or a search query
2. matching summaries are located if needed
3. a bounded delegated expansion grant is created
4. a sub-agent walks the DAG, source messages, and referenced files
5. the sub-agent returns a focused answer with cited summary IDs
6. the grant is revoked and cleaned up

This is the right tool family for compacted-memory recall. It is separate from the learned-routing tools like `brain_teach`, `brain_status`, and `brain_trace`.

## Large-file handling

Large file blocks are intercepted at ingestion when they exceed the configured threshold:
1. parse file blocks from message content
2. store oversized file content separately on disk
3. generate a lightweight exploration summary
4. insert a `large_files` record with metadata
5. replace the in-message payload with a compact reference

This keeps huge file pastes from consuming the entire active context while preserving access to the content.

## Session reconciliation

LCM handles crash recovery through bootstrap reconciliation:
1. read the JSONL session file
2. compare it to the LCM database
3. find the newest shared anchor message
4. import any later JSONL messages missing from the database

## Operation serialization

Mutating operations are serialized per session using a promise queue. That prevents races between concurrent ingest/compact activity for the same conversation without blocking different conversations.

## Learned-layer overlay

On top of the substrate above, OpenClawBrain adds:
- runtime decisioning (`use_brain`, `shadow`, explicit skip modes)
- correction-first learned retrieval from immutable promoted packs
- immediate `brain_teach` updates when embeddings are configured
- episode/trace recording
- structured evidence harvesting
- replay-gated promotion
- supervised child-worker learning boundary

That overlay is what turns the inherited transcript-memory system into OpenClawBrain as a product.
