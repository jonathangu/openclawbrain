---
artifact_id: ca_example_map_of_territory_01
kind: map_of_territory
status: proposed
title: "Teacher v3 map of territory"
proposal_id: prop_compiled_artifacts_example_02
pack_id: pack_compiled_artifacts_example_01
subject_ids:
  - topic:teacher-v3
  - topic:compiled-artifacts
  - topic:promotion-gates
confidence: 0.92
created_at: 2026-04-03T00:00:00Z
updated_at: 2026-04-03T00:00:00Z
---

## Summary

This map keeps the intended split clear: raw authority, compiled artifacts, candidate graph proposals, and promoted live state are different layers with different jobs.

## Evidence

- `ev-teacher-v3-layers` — `docs/architecture/teacher-v3.md#layers`
- `ev-compiled-artifacts-storage-layout` — `docs/architecture/compiled-artifacts.md#runtime-storage-layout`
- `ev-teacher-v3-proof-surface-hierarchy` — `docs/architecture/teacher-v3-proof.md#surface-hierarchy`

## Provenance

This example is derived from the target-state docs only. It is intentionally marked as scaffolding and should not be treated as live truth.

## Open questions

- Which runtime surface should render the target-state map first: a docs page, a proof bundle, or a CLI status overlay?
- How should future compiler output indicate the boundary between shipped truth and target-state intent?

## Promotion notes

- Use this example to guide later UI, docs, and proof-surface wiring.
- Keep promoted-only live semantics explicit when the first runtime reader lands.
