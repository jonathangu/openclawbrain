# Compiled artifact substrate

Compiled artifacts are **derived, off-path knowledge products**. They help OpenClawBrain organize evidence, synthesize navigable pages, and prepare proposal payloads, but they do **not** become current truth by themselves.

This doc defines the substrate for Wave 1 compiled-artifact work:

- storage and file layout
- markdown + sidecar metadata shape
- evidence / provenance fields
- proposal-envelope integration

Related docs:
- [Architecture overview](overview.md)
- [Learning pipeline](learning-pipeline.md)
- [Corrections](corrections.md)
- [Graphify bridge contract](graphify-bridge.md)

## Core rules

1. **Derived, not authoritative**
   - compiled artifacts are created from raw authority and explicit corrections
   - they may summarize or reorganize evidence
   - they do not outrank raw source, explicit correction memory, or operator policy

2. **Off-path only**
   - compilation runs after export / in background learning work
   - the live serve path only sees promoted results
   - if compilation fails, runtime should fail open rather than injecting partial state

3. **Immutable by version**
   - an artifact version is write-once
   - edits produce a new version / new proposal
   - old versions remain addressable for audit and rollback

4. **Evidence is first-class**
   - every claim in a compiled artifact should be traceable back to one or more evidence refs
   - provenance must describe both source authority and the transformation path

## Artifact classes

The substrate is class-agnostic, but the first supported kinds should include:

- `concept_page`
- `workflow_page`
- `topic_index`
- `map_of_territory`
- `neighborhood_summary`
- `cross_source_synthesis`
- `stale_fact_watch`
- `contradiction_report`
- `provenance_gap_report`

The `kind` field should be stable and machine-readable so later compiler passes can route by class without inventing new storage conventions.

## Runtime storage layout

The authoritative runtime store should live under the activation root, not in the source repo.

Recommended target shape:

```text
<activation-root>/compiled/
  packs/
    <pack-id>/
      pack.manifest.json
      artifacts/
        <artifact-id>/
          artifact.md
          artifact.meta.json
          evidence/
            <evidence-id>.json
      proposals/
        <proposal-id>.json
      indexes/
        by-kind.json
        by-subject.json
        by-status.json
```

### What each directory means

- `packs/<pack-id>/`
  - one promoted or candidate compiled-artifact pack
  - may mirror a promoted brain pack or a candidate compilation run

- `artifacts/<artifact-id>/`
  - one immutable artifact version
  - the markdown body and sidecar metadata are stored together

- `proposals/<proposal-id>.json`
  - the proposal envelope that produced or promoted this artifact set
  - used as the join key for replay / rollback / review

- `indexes/`
  - optional denormalized lookup files for fast inspection
  - these are derived caches, not source-of-truth records

### Proof/export bundle layout

For operator reports and review bundles, snapshot the same structure into the existing evidence/artifact tree, for example:

```text
artifacts/<YYYY-MM-DD>/<git-sha>/compiled-artifacts/
  manifest.json
  <artifact-id>.md
  <artifact-id>.meta.json
```

That export is a read-only snapshot for inspection and should not be treated as live state.

## Markdown + sidecar shape

Each compiled artifact should be stored as a pair:

- `artifact.md` — human-readable body
- `artifact.meta.json` — canonical machine-readable metadata

### Markdown body

Use a small YAML frontmatter block plus structured sections.

Example:

```md
---
artifact_id: ca_concept_01h...
kind: concept_page
status: proposed
title: "Compiled artifact substrate"
proposal_id: prop_01h...
pack_id: pack_01h...
subject_ids:
  - topic:compiled-artifacts
confidence: 0.91
created_at: 2026-04-03T18:26:00Z
updated_at: 2026-04-03T18:26:00Z
content_hash: sha256:...
---

## Summary

Short operator-facing summary.

## Evidence

Bullet list of the evidence refs used to compile the page.

## Provenance

Describe the authority sources and the transformation path.

## Open questions

Anything still uncertain or intentionally unresolved.

## Promotion notes

What replay or rollout gate should check next.
```

The markdown body should remain readable on its own, but the sidecar is the canonical record for machine use.

### Sidecar JSON

The sidecar should be the authoritative metadata source. Suggested top-level fields:

```ts
type CompiledArtifactMetaV1 = {
  schemaVersion: 1;
  artifactId: string;
  kind:
    | "concept_page"
    | "workflow_page"
    | "topic_index"
    | "map_of_territory"
    | "neighborhood_summary"
    | "cross_source_synthesis"
    | "stale_fact_watch"
    | "contradiction_report"
    | "provenance_gap_report";
  title: string;
  status: "draft" | "proposed" | "validated" | "promotable" | "promoted" | "rejected" | "expired" | "superseded";
  packId: string;
  proposalId: string;
  proposalLane: "compiler";
  subjectIds: string[];
  evidence: EvidenceRefV1[];
  counterevidence?: EvidenceRefV1[];
  provenance: ProvenanceV1;
  contentHash: string;
  markdownPath: string;
  metaPath: string;
  createdAt: string;
  updatedAt: string;
  expiresAt?: string;
  confidence: number;
  claims?: ClaimRefV1[];
  promotion?: PromotionMetaV1;
  supersedesArtifactId?: string;
};
```

## Evidence ref shape

Evidence refs must identify both the source and the authority behind it.

```ts
type EvidenceRefV1 = {
  evidenceId: string;
  sourceKind: "user_turn" | "tool_trace" | "file" | "repo" | "summary" | "correction";
  sourceId: string;
  authority: "user_explicit" | "raw_source" | "operator_policy";
  derivation?:
    | "summary_navigation"
    | "teacher_inference"
    | "teacher_compilation"
    | "teacher_lint"
    | "teacher_mutation_proposal"
    | "teacher_forgetting_proposal";
  span?: { start: number; end: number };
  quote?: string;
  digest?: string;
  capturedAt?: string;
  retrievedAt?: string;
};
```

### Evidence rules

- `authority` describes what kind of source supplied the fact
- `derivation` describes how the compiler used the source
- `summary` can be evidence, but only as a derived navigation aid
- explicit corrections remain first-class authority and should be represented distinctly from ordinary summaries
- every nontrivial claim should point to one or more `evidenceId` values

## Provenance fields

The provenance block should explain where the artifact came from and how it was produced.

```ts
type ProvenanceV1 = {
  producer: "teacher-v3" | string;
  producerVersion: string;
  promptHash?: string;
  runId?: string;
  basePackId?: string;
  baseGraphHash?: string;
  scope: string;
  idempotencyKey: string;
  sourceRoots?: string[];
  transformChain?: string[];
};
```

### Provenance rules

- `producer` identifies the system component that wrote the artifact
- `producerVersion` must make replay behavior auditable
- `basePackId` / `baseGraphHash` tie the artifact back to the substrate it compiled from
- `promptHash` is useful when the compiler is model-driven and the prompt template matters for replay
- `idempotencyKey` should prevent duplicate materialization for the same input set
- `transformChain` should record major steps such as `extract -> cluster -> synthesize -> validate`

## Claim granularity

Artifacts should not flatten all evidence into one blob when the claims differ materially.

```ts
type ClaimRefV1 = {
  claimId: string;
  text: string;
  evidenceIds: string[];
  confidence: number;
  status: "supported" | "partial" | "uncertain";
};

type PromotionMetaV1 = {
  promotedAt?: string;
  promotedPackId?: string;
  rejectedAt?: string;
  rejectedReason?: string;
  replaySuites?: string[];
  rollbackKey?: string;
};
```

That lets the UI or downstream proposal logic show which claims are strongly grounded and which are only tentative.

## Proposal-envelope integration

Every compiled artifact should be emitted through a proposal envelope, even if the initial output is only for review.

### Required join fields

The proposal should include:

- `proposalId`
- `lane: "compiler"`
- `lineage`
- `subjectIds`
- `evidence`
- `artifacts`
- `confidence`
- `replaySuites`
- `rollbackKey`

### Recommended envelope shape

```ts
type TeacherProposalV1 = {
  proposalId: string;
  lane: "compiler" | "lint" | "mutation" | "forgetting" | "correction";
  lineage: {
    basePackId?: string;
    baseGraphHash?: string;
    producerVersion: string;
    promptHash?: string;
    scope: string;
    profile?: string;
    idempotencyKey: string;
  };
  subjectIds: string[];
  evidence: EvidenceRefV1[];
  counterevidence?: EvidenceRefV1[];
  artifacts?: { artifactId: string; kind: string; contentHash: string }[];
  expectedEffect?: {
    retrieval?: "better" | "same" | "uncertain";
    truthRisk?: "low" | "medium" | "high";
    tokenBudget?: "lower" | "same" | "higher";
  };
  confidence: number;
  expiresAt?: string;
  replaySuites: string[];
  rollbackKey: string;
  replayGate?: TeacherProposalReplayGateV1;
};
```

### Integration contract

1. the compiler writes the markdown + sidecar pair
2. it writes or updates the matching proposal envelope
3. the proposal gate validates evidence, boundedness, and freshness
4. accepted artifacts become promotable or promoted
5. the sidecar status is updated, but the body stays immutable

This means the proposal envelope is the control plane and the artifact pair is the data plane.

## Lifecycle

A practical artifact lifecycle should be:

- `draft` — created but not yet validated
- `proposed` — emitted with a proposal envelope
- `validated` — structural checks passed
- `promotable` — eligible for replay / operator review
- `promoted` — allowed into the live promoted pack
- `rejected` — not accepted by replay or operator review
- `expired` — stale or superseded before promotion
- `superseded` — replaced by a newer version

### Write path

1. materialize artifact body
2. write sidecar metadata
3. hash both files and store the hashes in the sidecar
4. attach proposal envelope metadata
5. mark status according to validation outcome

### Read path

1. resolve pack and artifact IDs
2. load sidecar first
3. verify content hash against markdown body
4. use the markdown body for human presentation
5. use sidecar + proposal envelope for replay / audit / promotion

## Non-goals for Wave 1

- no live graph mutation from compiled artifacts
- no hard deletion of source authority
- no claim that compiled artifacts are truth on their own
- no promise that every artifact class exists immediately

Wave 1 should establish the substrate so later compiler, lint, and promotion work has a stable place to land.
