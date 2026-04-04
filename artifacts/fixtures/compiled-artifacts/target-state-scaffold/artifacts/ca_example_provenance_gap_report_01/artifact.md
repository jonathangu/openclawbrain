---
artifact_id: ca_example_provenance_gap_report_01
kind: provenance_gap_report
status: proposed
title: "Compiled artifact provenance gaps"
proposal_id: prop_compiled_artifacts_example_03
pack_id: pack_compiled_artifacts_example_01
subject_ids:
  - topic:provenance
  - topic:compiled-artifacts
  - topic:teacher-v3-lints
confidence: 0.87
created_at: 2026-04-03T00:00:00Z
updated_at: 2026-04-03T00:00:00Z
---

## Summary

This report example lists the implementation gaps that a future compiler and lint pass should close before promoting compiled artifacts.

## Evidence

- `ev-compiled-artifacts-provenance` — `docs/architecture/compiled-artifacts.md#provenance-fields`
- `ev-teacher-v3-lints-ci-first` — `docs/architecture/teacher-v3-lints.md#1-ci-first-deterministic-lint-family`
- `ev-teacher-v3-lints-release-drift` — `docs/architecture/teacher-v3-lints.md#3-release-drift-motivating-case`

## Provenance

This example is derived from the target-state docs only. It is intentionally marked as scaffolding and should not be treated as live truth.

## Open questions

- Should the first implementation write manifests only, or manifest plus artifact pairs in one shot?
- What exact hash check should fail closed if a markdown body changes after sidecar generation?

## Promotion notes

- Treat this as a report example only; it is not proof of any shipped implementation.
- Use it to seed later lint findings and review UX.
