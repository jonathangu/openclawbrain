# Teacher v3

Status: **target-state architecture/spec**.

This is the canonical design note for Teacher v3 in OpenClawBrain. It describes the intended contract for the next teacher layer, not a claim that every piece is already shipped.

> **Teacher v3 is an off-path compiler of graph structure and compiled artifacts, not an arbiter of current truth.**

## What problem Teacher v3 solves

Teacher v2-era language can blur two different jobs:

- deciding **where to look** in history
- deciding **what is currently true**

Teacher v3 keeps those jobs separate.

It is responsible for off-path knowledge work:

- compiling navigable graph structure
- linting the compiled graph and derived artifacts
- proposing merges, splits, edge changes, demotions, archives, tombstones, and retention changes
- producing replayable proposal envelopes with provenance and lineage
- generating evidence-carrying compiled surfaces such as concept pages, topic indexes, and neighborhood summaries

It is **not** responsible for live truth arbitration on the response path.

## Current-truth authority

Current truth stays grounded in the authoritative substrate:

1. explicit typed user corrections
2. recent raw user/source turns, files, and traces
3. expanded raw source turns when detail is needed
4. explicit operator policy where applicable

Teacher outputs are derived surfaces. They can guide navigation, synthesis, and structure, but they do not outrank explicit correction memory or raw authority.

## Layers

Teacher v3 uses a clear layer split:

### Raw authority layer
The source substrate:

- turns and transcripts
- files, repos, PDFs, images, and traces
- explicit correction memories
- operator policy when relevant

### Compiled artifact layer
Derived knowledge surfaces such as:

- concept pages
- workflow pages
- topic indexes
- map-of-territory pages
- neighborhood summaries
- stale-fact reports
- contradiction reports
- provenance-gap reports

### Candidate graph layer
Off-path structural proposals such as:

- add node
- merge nodes
- split node
- add / strengthen / weaken / inhibit edge
- demote node
- archive node
- tombstone node

### Promoted pack layer
Only promoted results affect live serving. Candidate graph changes stay inert until they pass replay, attribution, boundedness, and rollback checks.

## Operating contract

Teacher v3 work happens off the hot path:

- after turn export
- during candidate compilation
- during scheduled linting or maintenance
- during replay / eval runs
- during explicit background learning cycles

That means:

- no live per-hop teacher calls
- no “ask the teacher what to do next” on the serve path
- no direct overwrite of current truth in live memory
- no unbounded compile or materialization cost
- no loss of rollbackability

## Proposal envelope

All Teacher v3 outputs should serialize into a common proposal envelope so they can be reviewed, replayed, and promoted consistently.

Minimum fields in the target-state contract:

- `proposalId`
- `lane` (`compiler`, `lint`, `mutation`, `forgetting`)
- `lineage` information
- `subjectIds`
- `evidence` and optional `counterevidence`
- `artifacts` and/or `mutations`
- `expectedEffect`
- `confidence`
- `expiresAt` when relevant
- `replaySuites`
- `rollbackKey`

The envelope is a traceability contract first and a convenience format second.

## Truth and derivation hygiene

A recurring failure mode in architecture docs is to make teacher output sound more authoritative than it is.

Teacher v3 should keep the split explicit:

- **authority** = what is allowed to settle current truth
- **derivation** = what was synthesized, summarized, or proposed from authority

That makes it safe to use the teacher for structure without accidentally promoting a summary into a fact source.

## Shipped-state vs target-state hygiene

This file intentionally mixes two kinds of statements:

- **shipped-state**: what current OpenClawBrain already does
- **target-state**: what Teacher v3 is meant to formalize next

Keep those distinct in every related doc and PR.

### Shipped-state today

The repo already supports the key posture this spec builds on:

- learning happens off the response path
- the runtime serves only promoted packs
- explicit user corrections have durable runtime value
- fail-open behavior keeps OpenClaw running when the memory layer cannot safely add context

### Target-state in this spec

Teacher v3 extends that posture with:

- canonical compiler/lint/proposal lanes
- explicit proposal envelopes and lineage tracking
- replay-gated structural mutation proposals
- forgetting proposals that prefer compress/demote/archive over deletion
- a retention state machine (`retained` → `demoted` → `archived` → `tombstoned` → `deleted`) with teacher-driven hard-delete guardrails that never delete `user_explicit` correction memory
- canonical claims hygiene for current-truth vs derived surfaces

### Not claimed here

This spec does **not** claim:

- live self-editing of truth on the serve path
- teacher supremacy over explicit user corrections
- deletion of `user_explicit` correction memory by the teacher
- broad mutation promotion without replay and rollback proof

## Non-goals

Teacher v3 is not trying to become:

- a general truth oracle
- a live per-turn agent controller
- a replacement for the correction path
- a substitute for raw evidence
- a vague “smarter labeler” with no audit trail

## Relation to the rest of the docs

Use this page as the canonical target-state spec for Teacher v3.

Use these docs for shipped/runtime truth:

- [Overview](overview.md)
- [Learning pipeline](learning-pipeline.md)
- [Corrections](corrections.md)
- [Fail-open design](fail-open.md)
- [Routing prior](routing-prior.md)

If this page and a shipped claims surface disagree, the claims surface wins for what is currently true; this page wins only for Teacher v3 target-state intent.
