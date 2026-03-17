# Routing Prior

This note explains one of the most important architectural shifts in OpenClawBrain:

> **The upgrade is not “better retrieval.” It is decomposing retrieval into routing, evidence, and truth resolution.**

Or, more compactly:

> **Separate “where should I look?” from “what should I believe?”**

## Problem with flat retrieval

A long-running conversation contains several overlapping structures at once:

- chronology
- topic shifts
- stale facts that were later corrected
- exact raw details that matter for some questions but not others
- broad recap material that can be safely compressed

A flat retrieval policy has to do too many jobs at once:

1. choose the relevant region of history
2. choose the right detail level
3. decide whether summary is enough
4. decide whether to expand back to source
5. resolve truth when newer corrections conflict with older evidence

That is how you get aliasing, stale-answer, and chronology bugs.
The failure mode is usually not that the model is dumb. It is that the system chose the wrong historical region too early and then asked the wrong evidence layer to settle truth.

## Architectural principle: route vs believe

OpenClawBrain should treat these as different questions:

- **Where should I look?**
- **What should I believe?**

That gives a simple policy stack:

- **Routing prior** — choose the relevant historical region(s) from summaries
- **Expansion policy** — decide whether summary-level evidence is sufficient or whether raw-source expansion is required
- **Truth policy** — when conflict exists, prefer explicit correction memory over stale summary or older transcript-derived evidence

## Three-layer model

OpenClawBrain uses three different layers for three different jobs:

- **LCM summaries** — cheap hierarchical map over history
- **raw expansion** — exact detail path back to source
- **typed brain memory** — durable current-truth overlay

The key idea is:

> **Summaries are navigation, not truth.**

Or, in product language:

> **It turns summaries from a lossy answer source into a cheap navigation layer.**

## Query-class policy

### Recap queries
Broad recap questions may stay at summary level.

Examples:
- “What have we been working on this week?”
- “What changed in the last conversation?”

### Precision queries
Detail-sensitive questions should expand toward source before asserting specifics.

Examples:
- “What exact command failed?”
- “Which commit changed the worker launch path?”
- “What did the user say verbatim?”

### Current-truth / conflict queries
Current-state or conflict-sensitive questions should consult typed correction memory first, then expand if needed for proof.

Examples:
- “What is the codeword now?”
- “Which instruction currently wins?”
- “Did the user change their preference?”

## Conflict-resolution rules

The hard invariant is:

> **For current-state queries, explicit user corrections outrank summaries and prior transcript-derived memories. Summaries may supply scope, but not override authoritative corrections.**

When multiple evidence layers are in play, the runtime should prefer:

1. explicit typed user correction memory
2. recent raw user turns
3. expanded raw source turns
4. leaf summaries
5. condensed summaries

Operationally:

- if summary and correction agree, answer directly
- if summary and correction conflict, correction wins
- if no correction exists and the query is exact or chronological, expand before answering
- if expansion evidence is mixed, answer with uncertainty and provenance

## Product and latency implications

This architecture improves more than raw retrieval quality.

### Better chronology handling
The system can route toward the recent region, the older region, or a broad recap across both instead of collapsing them too early.

### Better current-truth handling
Correction memory does not need to overpower all transcript noise globally. It only needs to win at the truth layer after the relevant region is on the table.

### Better latency discipline
Broad recap can stay cheap. Exactness pays for source expansion only when needed.

### Better inspectability
The system becomes easier to explain:

- summaries chose the region
- expansion recovered the details when needed
- typed correction memory won the conflict when current truth mattered

That is much more legible than “the graph happened to fire these nodes.”

## Next implementation steps

The current routing-prior architecture is the right direction, but the next steps should make it even more explicit in runtime behavior:

- feed summary priors more directly into seed selection and traversal
- make the query-class decision visible in traces
- make summary/expansion/correction conflict resolution explicit in runtime logs
- learn over whole retrieval paths rather than isolated local hops

A small caution on the 2016-paper analogy: this is best treated as an architectural rhyme, not a literal mapping of the paper’s machinery.

## One-line summary

> **Use LCM summaries as a search abstraction over history; use typed brain memory as the current-truth layer.**
