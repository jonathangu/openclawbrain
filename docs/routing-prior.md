# Routing Prior

This note explains one of the most important architectural choices in OpenClawBrain:

> **LCM summaries are a routing prior, not the final truth layer.**

That sounds subtle, but it changes the system shape in an important way.

## The problem with flat retrieval

A long-running OpenClaw conversation contains multiple overlapping structures at once:

- chronology
- topic shifts
- stale facts that were later corrected
- exact raw details that matter for some questions but not others
- broad recap material that can be safely compressed

A single flat retrieval policy has to solve too many things at once:

1. which region of history matters
2. which node should seed retrieval
3. whether a summary is enough
4. whether the system should expand back to raw source
5. whether a newer typed correction should override older transcript abstractions

If you ask embedding similarity and local graph traversal to answer all of those at once, the system works reasonably on short/simple histories and degrades on long, conflict-heavy ones.

The failure mode is usually not “the model is dumb.”
It is that the system chose the wrong historical region too early.

## The routing-prior split

OpenClawBrain now makes a cleaner separation:

- **LCM summary DAG** → cheap hierarchical map over history
- **raw expansion** → exact detail path back to source
- **typed brain memory** → durable current-truth overlay

So the system now asks two different questions instead of one confused one:

### 1. Where should I look?
This is the routing-prior question.

Summaries are good at:
- telling the system which part of history is even relevant
- cheaply indicating whether a discussion is broad or local
- carrying time range, depth, and descendant coverage metadata
- suggesting what should be expanded for details

### 2. What should I believe?
This is the truth/precedence question.

Summaries are **not** the canonical truth layer because they are lossy.
For current truth, exact detail, and conflicts, the system should defer to:
- explicit typed corrections
- recent raw user turns
- expanded source material

## Why the summary DAG is the right prior

LCM already maintains a hierarchical summary structure with useful metadata:

- summary id
- leaf vs condensed kind
- depth
- time range
- descendant count
- parent/child references
- explicit “expand for details” guidance

That makes the summary DAG a natural *search geometry* for history.

It is the cheap approximation layer that answers:

> “Which neighborhood is worth inspecting next?”

That is exactly what a prior should do.
A prior does not need to be perfect; it needs to be cheap and directionally useful.

## What changed in OpenClawBrain

The summary-aware routing wave added a serve-time policy layer that can distinguish between a few important query classes:

- `summary_suffices`
- `expand_to_source`
- `prefer_typed_memory`
- `ignore`

The assembler now exposes summary metadata to the runtime layer, and the runtime can inject guidance such as:

- broad recap questions can stay summary-level initially
- precision/conflict-sensitive questions should expand toward source before asserting specifics
- current-truth questions should prefer correction cards and typed memory over stale summary recap

This is still a modest first cut. The current implementation adds **policy guidance at assembly time**, not a full learned summary-seed traversal policy yet. But it is already a better architecture than treating summaries as just more transcript text.

## Why this is better than the older shape

### Better chronology handling
A query can be semantically similar to multiple historical regions. The summary DAG helps identify whether the user probably means the recent region, the older region, or a broad recap across both.

### Better conflict handling
A summary may describe an older state of the world. A typed correction memory may describe the newer one. The routing prior makes it easier to bring the right conflict set on-stage and then let precedence rules decide the winner.

### Better latency/quality tradeoff
Not every question should force raw expansion. Summary-level answers are often enough for broad recap. The routing prior lets the system stay cheap when it can and pay for raw expansion only when precision matters.

### Better explainability
The behavior becomes more legible:
- summaries chose the region
- expansion recovered the details
- typed correction memory won the conflict

That is easier to reason about than “the graph happened to fire these nodes.”

## The precedence ladder

The routing prior is only half the story. It works best when combined with an explicit precedence ladder:

1. explicit typed user correction memory
2. recent raw user turns
3. expanded raw source turns
4. leaf summaries
5. condensed summaries

This keeps summaries in the role they are actually good at: navigation and cheap scope.

## Practical example

Suppose the history contains:
- an old answer: “the codeword is hippo”
- a later explicit user correction: “wrong, the codeword is giraffe”
- summaries that may still mention the earlier state

A good system should do this:
- summaries indicate the relevant region of history
- the query “what’s the codeword?” is current-truth sensitive
- typed correction memory is preferred over summary recap
- if needed, raw expansion can provide auditability back to source

That is much better than treating the summary text itself as the winner.

## What comes next

The current routing-prior implementation is intentionally conservative.
Future work can push it further by:

- feeding summary priors directly into seed scoring/traversal
- logging the retrieval path more explicitly for inspection
- learning which summary→expand→memory paths actually correlate with better downstream outcomes
- making summary/expansion/correction conflict resolution even more explicit in runtime traces

## One-line summary

> **Use LCM summaries as a search abstraction over history; use typed brain memory as the current-truth layer.**
