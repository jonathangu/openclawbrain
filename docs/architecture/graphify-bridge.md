# Graphify bridge contract

Status: repo-native bridge note.

This document defines the Wave 1 bridge between OpenClawBrain's existing runtime / proof / docs truth and Graphify-derived surfaces.

The core rule is simple:

> **Artifact-first, repo-native, proof-first.**

Graphify is a derived compiler and maintenance surface. It is not a live truth layer, it does not sit on the response path, and it does not outrank stronger OpenClawBrain truth layers.

## 1) Truth hierarchy

Graphify-produced surfaces must sit below the already-shipped truth layers:

1. **runtime truth** — what the live runtime actually reports
2. **proof truth** — what exercised proof bundles capture
3. **docs truth** — what the repo publicly claims and freezes
4. **Graphify proposal truth** — derived bundles, compiled artifacts, and lint findings

If a Graphify output disagrees with a stronger layer, it must say so explicitly and route the disagreement to review or lint. It must never pretend to settle the conflict.

### Correction precedence

Explicit typed corrections outrank every Graphify-derived surface.

That means Graphify outputs must defer to:

- explicit correction memory
- recent raw user / source turns
- raw proof and runtime evidence
- frozen docs truth

A Graphify artifact may report a correction conflict risk, but it may not override the correction itself.

## 2) No Graphify on the hot path

Graphify stays completely off:

- `before_prompt_build`
- live traversal / route scoring
- runtime prompt assembly
- any other serve-path decision point

The live runtime keeps its existing fail-open behavior. If Graphify tooling, export work, or off-path compilation cannot run safely, the runtime still continues without injected Graphify context.

This bridge therefore does **not** add a second live memory layer or a parallel serve path.

## 3) Dual export lane

Wave 1 uses two exports from the same underlying source set.

### A. Canonical machine export

This is the authoritative repo-native export.

It should carry the stable, machine-readable data needed for replay, lineage, and review:

- normalized events or records
- proof / status snapshots
- stable identifiers
- provenance refs
- hashes for every included source

This export is canonical. If the machine export and a projection disagree, the machine export wins.

### B. Graphify projection export

This is the Graphify-facing view of the same source set.

It may include:

- session markdown projections
- proof summary projections
- workspace-note projections
- curated docs / code mirrors

The projection export is disposable and rebuildable. It exists to make Graphify runs easier to inspect, not to redefine truth.

## 4) Artifact-first rule

The bridge must keep the lifecycle in the normal OpenClawBrain order:

**export -> candidate / compiled artifact -> promotion**

Graphify may help produce compiled artifacts, lints, or candidate imports, but it does not invent a separate lifecycle beside that chain.

Wave 1 specifically follows this order:

1. export a canonical machine bundle
2. emit a Graphify projection bundle from the same source set
3. build Graphify-derived compiled artifacts
4. only then consider any bounded import slice

Import is therefore a later consumer of compiled artifacts, not a replacement for them.

## 5) Output classes A-E

Wave 1 has five bridge output classes.

| Class | Name | Purpose | Authority level |
| --- | --- | --- | --- |
| A | Graphify source bundle | reproducible corpus bundle built from current repo / workspace / proof surfaces | canonical machine export is authoritative |
| B | Graphify run bundle | reproducible record of one Graphify compiler run | derived, replayable, diffable |
| C | OCB compiled artifact pack | markdown + sidecar artifact pack for repo-native derived surfaces | derived, proposal-backed |
| D | Conservative import slice | tiny EXTRACTED-only prior slice for later import evaluation | candidate-only, never current truth |
| E | Lint / maintenance bundle | deterministic + semantic maintenance report for drift and provenance review | review-only, never live mutation |

### A. Graphify source bundle

Required contents:

- `corpus-manifest.json`
- canonical machine export(s)
- Graphify projection export(s)
- proof / status snapshots
- provenance and hashes for every included source

Rules:

- canonical machine export is the source of truth for the bundle
- secrets stay out
- bundle hash must be stable and recorded

### B. Graphify run bundle

Required contents:

- `graphify-command.json`
- `graph.json`
- `graph.html`
- `GRAPH_REPORT.md`
- `graphify-summary.json`
- `labels.json`
- `benchmark.json`
- Graphify version / config metadata

Rules:

- every run must point back to an exact source-bundle hash
- run outputs must be diffable between runs
- failures stay confined to repo / workspace artifacts

### C. OCB compiled artifact pack

Required first-artifact kinds:

- `map_of_territory`
- `concept_page`
- `neighborhood_summary`
- `provenance_gap_report`

Rules:

- markdown body + sidecar metadata
- explicit evidence refs
- proposal envelope attached
- derived, not truth

This class should match the existing compiled-artifacts substrate shape.

### D. Conservative import slice

Allowed in Wave 1:

- EXTRACTED hub priors
- EXTRACTED neighborhood priors
- source-grounded evidence / rationale pointers

Forbidden in Wave 1:

- anything that behaves like current truth
- correction-like durable memories
- INFERRED live-eligible edges
- silent host mutations

### E. Lint / maintenance bundle

Required classes:

- `missing_from_pack`
- `stale_vs_current_source`
- `unsupported_by_provenance`
- `correction_conflict_risk`
- `new_current_source_hubs`
- `split_merge_review_hint`
- `provenance_gap`

This bundle is a maintenance surface, not a graph writer.

## 6) EXTRACTED / INFERRED / AMBIGUOUS handling

Graphify-derived content must label trust class explicitly.

| Class | Meaning | Allowed treatment |
| --- | --- | --- |
| EXTRACTED | directly grounded in source or export evidence | may feed canonical machine export, compiled artifacts, and later bounded import candidates |
| INFERRED | synthesized or generalized from source evidence | review-only / candidate-only; may inform compiled artifacts and lint findings, but not live truth |
| AMBIGUOUS | unresolved, conflicting, or underdetermined | diagnostic-only; should surface uncertainty, not certainty |

### Treatment rules

- **EXTRACTED** is the only class that can be considered for any later live-eligible import slice.
- **INFERRED** stays on the candidate / review side.
- **AMBIGUOUS** stays diagnostic and should usually land in a provenance gap, contradiction, or correction-conflict report.

## 7) Promotion and rollback discipline

Graphify outputs never promote themselves.

They must flow through the existing OpenClawBrain promotion model:

- compile or export off-path
- validate against stronger truth layers
- replay or review
- promote only if the candidate passes
- keep rollback binding explicit

If a Graphify-derived bundle is rejected, the live runtime stays unchanged.

## 8) What this bridge is not

This bridge does **not**:

- make Graphify a live truth layer
- replace runtime, proof, or docs truth
- create a new serve-path lifecycle
- bypass explicit correction precedence
- widen import beyond EXTRACTED in Wave 1
- imply that Graphify outputs are current truth by themselves

## 9) Related repo seams

This bridge should stay aligned with the existing docs contract surfaces:

- [Architecture overview](overview.md)
- [Compiled artifact substrate](compiled-artifacts.md)
- [Teacher v3 lint families](teacher-v3-lints.md)
- [Fail-open design](fail-open.md)
- [Corrections](corrections.md)
- [Teacher v3 proposal reporting / proof surfaces](teacher-v3-proof.md)

The key boundary is unchanged: Graphify can help produce derived structure, but current truth still comes from runtime, proof, docs, and explicit corrections.
