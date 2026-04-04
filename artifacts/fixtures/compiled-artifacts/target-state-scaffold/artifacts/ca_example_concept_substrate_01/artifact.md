---
artifact_id: ca_example_concept_substrate_01
kind: concept_page
status: proposed
title: "Compiled artifact substrate examples"
proposal_id: prop_compiled_artifacts_example_01
pack_id: pack_compiled_artifacts_example_01
subject_ids:
  - topic:compiled-artifacts
  - topic:teacher-v3
confidence: 0.95
created_at: 2026-04-03T00:00:00Z
updated_at: 2026-04-03T00:00:00Z
---

## Summary

This concept page shows the canonical pair shape for a compiled artifact: a human-readable markdown body plus a canonical machine-readable sidecar.

## Evidence

- `ev-compiled-artifacts-core-rules` — `docs/architecture/compiled-artifacts.md#core-rules`
- `ev-compiled-artifacts-sidecar-shape` — `docs/architecture/compiled-artifacts.md#markdown--sidecar-shape`
- `ev-teacher-v3-off-path` — `docs/architecture/teacher-v3.md#what-problem-teacher-v3-solves`

## Provenance

This example is derived from the target-state docs only. It is intentionally marked as scaffolding and should not be treated as live truth.

## Open questions

- Should future compiler code materialize packs under the activation root only, or also export read-only snapshots into evidence bundles?
- Which helper should own content-hash verification: the compiler writer, the pack reader, or both?

## Promotion notes

- Use this example when wiring the first compiler writer and artifact verifier.
- Keep the sidecar authoritative and the body immutable once written.
