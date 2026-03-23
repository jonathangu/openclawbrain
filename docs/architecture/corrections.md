# Corrections

OpenClawBrain now treats explicit user corrections as a first-class runtime path rather than a loose side effect of generic retrieval.

The guiding rule is simple:

> **Summaries provide context. The explicit user quote provides authority. Typed correction memory provides durable current truth.**

## Why corrections deserve their own path

There is a huge difference between:

- broad historical recall
- general negative feedback
- a user making an explicit correction that should persist

If the system blurs those together, it either:
- stores too much vague noise as memory, or
- fails to promote the exact correction that should override older context

OpenClawBrain now separates those cases more carefully.

## What counts as a correction

The correction path is meant for **explicit durable updates**, such as:

- factual corrections
- stable preferences
- workflow rules
- “use X, not Y” style operator instructions

Examples:
- “wrong, the codeword is giraffe”
- “use `gh pr create`, not `hub`”
- “for this repo, use `npm run release:verify` before publishing”

It is **not** meant for generic dissatisfaction, retries, or vague disagreement like:
- “no”
- “try again”
- “not what I meant”

Those may be useful signals for learning, but they are not durable typed corrections by themselves.

## The correction architecture

The correction system has two complementary lanes.

### 1. Fast deterministic lane
This lane runs at ingest time and tries to catch obvious explicit corrections immediately.

Current behavior:
- inspect the new user turn plus minimal recent context
- detect clear correction patterns
- canonicalize them into a typed instruction when confidence is high
- commit through `BrainService.teachUserCorrection(...)`

This gives the product the most important behavior:

> if the user clearly corrects something, the correction can win on the very next turn

### 2. Async proposal lane
Some corrections are real but less easily canonicalized by rules alone.

For those cases, OpenClawBrain can build a richer observation with:
- the user quote
- recent raw messages
- recent LCM summaries
- episode context when available

An off-path model can then propose either:
- `noop`
- `explicit_correction`

If the proposal is high-confidence enough, the runtime commits the correction through the same typed teach path.

This keeps richer semantic interpretation available **without blocking the hot path**.

## Provenance matters

The correction path is deliberately provenance-grounded.

When OpenClawBrain commits an explicit correction, it records metadata such as:
- source authority (`user_explicit`)
- source quote
- source message id
- via / proposal lane metadata

That matters because a good correction memory is not just “some text the system liked.”
It should be traceable back to the exact user turn that justified it.

## Correction precedence

Typed corrections are meant to outrank stale abstractions when the question is current-truth or conflict-sensitive.

A useful precedence ladder is:

1. explicit typed user correction memory
2. recent raw user turns
3. expanded raw source turns
4. leaf summaries
5. condensed summaries

That means the summary DAG can still help the system find the relevant region of history, while the correction memory decides what currently wins.

## Why this is better than summary-only correction behavior

A lossy summary may tell the system:
- “some correction happened in this region”

That is useful, but it is not enough.

The actual winner should be the typed correction memory grounded in the explicit user turn, because:
- summaries compress and omit detail
- summaries may still reflect older states
- corrections are often conflict-sensitive by definition

So the healthy split is:
- summaries nominate the region
- expansion can audit the source
- typed correction memory persists the operator’s current instruction

## What the new runtime path improves

### Faster live behavior
A correction no longer has to wait for slow, accidental retrieval luck. The explicit correction can be committed right away.

### Better override behavior
Older summary/text retrieval no longer has to be “lucky enough” to lose. The correction path gives the newer truth a stronger, typed object in the graph.

### Better trust boundaries
Generic human feedback, scanner signals, and teacher outputs are useful — but they do not all mean the same thing. The correction path reserves the strongest durable authority for explicit user-grounded updates.

### Better auditability
Because the correction memory is provenance-grounded, it is easier to inspect why the system believes the newer instruction should win.

## What corrections are not

Corrections are not the entire learning system.

They do **not** replace:
- the transcript memory substrate
- the learned routing policy
- replay-gated mutation learning
- broader evidence harvesting

They are a focused path for a specific class of high-value update:

> **explicit user instructions that should become durable current truth quickly and safely**

## One-line summary

> **Use summaries to interpret a correction in context; use a provenance-grounded typed memory to persist the correction itself.**
