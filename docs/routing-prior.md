# Routing Prior

This note explains one of the most important architectural choices in OpenClawBrain:

> **Separate “where should I look?” from “what should I believe?”**

That split is the real idea.
Everything else here is a consequence of it.

Another good way to say the same thing:

> **LCM summaries are a routing prior, not the final truth layer.**

## Core policy

OpenClawBrain works best when this is stated as an explicit stack rather than implied across sections:

- **Routing prior** — choose the relevant historical region(s) from summaries
- **Expansion policy** — decide whether summary-only is sufficient or whether raw-source expansion is required
- **Truth policy** — when conflict exists, prefer explicit correction memory over stale summary or older transcript evidence

That is the architecture in one block.

## The old failure mode

A long-running OpenClaw conversation contains multiple overlapping structures at once:

- chronology
- topic shifts
- stale facts that were later corrected
- exact raw details that matter for some questions but not others
- broad recap material that can be safely compressed

The older shape overloaded one mechanism with too many jobs at once:

1. choose the relevant region of history
2. choose the right detail level
3. decide whether summary is enough
4. decide whether to expand back to source
5. resolve truth when newer corrections conflict with older transcript-derived evidence

That is how you get aliasing, stale-answer, and chronology bugs.
The failure mode is usually not “the model is dumb.”
It is that the system chose the wrong historical region too early, and then asked the wrong evidence layer to settle truth.

## The split

OpenClawBrain now uses a cleaner decomposition:

- **LCM summary DAG** → cheap hierarchical map over history
- **raw expansion** → exact detail path back to source
- **typed brain memory** → durable current-truth overlay

So the runtime asks two different questions instead of one confused one.

### 1. Where should I look?

This is the routing-prior question.

Summaries are good at:

- telling the system which part of history is even relevant
- cheaply indicating whether a discussion is broad or local
- carrying time range, depth, and descendant coverage metadata
- suggesting what should be expanded for details

This is why summaries belong in the search geometry of the system.
They are a map, not the ground.

### 2. What should I believe?

This is the truth/precedence question.

Summaries are lossy, so they should not be treated as the canonical truth layer.
For current truth, exact detail, and conflict-sensitive questions, the runtime should defer to:

- explicit typed corrections
- recent raw user turns
- expanded source material

A useful way to phrase the design consequence is:

> **It turns summaries from a lossy answer source into a cheap navigation layer.**

## Query classes

The routing layer becomes much clearer when the main query classes are named explicitly:

### Recap queries

These are broad recap questions.
Summary-level evidence may be sufficient.

Examples:
- “What have we been working on this week?”
- “What changed in the last conversation?”

### Precision queries

These are detail-sensitive questions.
They should expand toward source before asserting specifics.

Examples:
- “What exact command failed?”
- “Which commit changed the worker launch path?”
- “What did the user say verbatim?”

### Current-truth / conflict queries

These are the most important class to get right.
They should consult typed correction memory first, then expand if needed for proof or provenance.

Examples:
- “What is the codeword now?”
- “Which instruction currently wins?”
- “Did the user change their preference?”

## Current-truth rule

The invariant should be hard, not implied:

> **For current-state queries, explicit user corrections outrank summaries and prior transcript-derived memories. Summaries may supply scope, but not override authoritative corrections.**

This is the practical role of correction memory.
Correction nodes do not need to overpower all transcript noise globally.
They only need to win at the truth layer after routing has already put the right region on the table.

That is much more stable.

## Practical precedence order

When multiple evidence layers are in play, the runtime should prefer:

1. explicit typed user correction memory
2. recent raw user turns
3. expanded raw source turns
4. leaf summaries
5. condensed summaries

This keeps summaries in the role they are actually good at: navigation and cheap scope.

## Why the summary DAG is the right prior

LCM already maintains a hierarchical summary structure with useful metadata:

- summary id
- leaf vs condensed kind
- depth
- time range
- descendant count
- parent/child references
- explicit “expand for details” guidance

That makes the summary DAG a natural search abstraction over history.

It is the cheap approximation layer that answers:

> “Which neighborhood is worth inspecting next?”

A prior does not need to be perfect.
It needs to be cheap and directionally useful.

## Why this architecture is better

### Better chronology handling
A query can be semantically similar to multiple historical regions. The summary DAG helps decide whether the user probably means the recent region, the older region, or a broad recap across both.

### Better conflict handling
A summary may describe an older state of the world. A typed correction memory may describe the newer one. The routing prior brings the right conflict set on-stage, then the truth policy decides the winner.

### Better latency/quality tradeoff
Not every question should force raw expansion. Broad recap can often stay at summary level. Precision should pay for source recovery only when needed.

### Better inspectability
The behavior becomes more legible:

- summaries chose the region
- expansion recovered the details when needed
- typed correction memory won the conflict when current truth mattered

That is much easier to inspect than “the graph happened to fire these nodes.”

## Failure handling

The policy should also say what happens when signals disagree:

- if summary and correction agree, answer directly
- if summary and correction conflict, correction wins
- if no correction exists and the query is exact or chronological, expand before answering
- if expansion evidence is mixed, answer with uncertainty and provenance

That makes the design operational rather than purely conceptual.

## A small note on the 2016-paper analogy

If this note gestures at the 2016 paper logic, it should stay modest.
This is **not** a literal mapping to the paper’s machinery.
It is better described as an architectural rhyme:

> hierarchical approximation first, selective expansion second, policy improvement over whole retrieval paths rather than isolated local hops.

That is the useful intuition.
It should not be overstated.

## Practical example

Suppose the history contains:

- an old answer: “the codeword is hippo”
- a later explicit user correction: “wrong, the codeword is giraffe”
- summaries that may still mention the earlier state

A good system should do this:

- summaries identify the relevant region of history
- the query “what’s the codeword?” is treated as current-truth sensitive
- typed correction memory is preferred over summary recap
- raw expansion remains available for auditability back to source

The summary helps the system find the conflict.
It does not get to settle it.

## One-line summary

> **Use LCM summaries as a search abstraction over history; use typed brain memory as the current-truth layer.**
