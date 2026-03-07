# Contracts v1

Canonical contract scaffolding lives under `contracts/` and is implemented in TypeScript by `@openclawbrain/contracts`.

Primary contract roots:

- `contracts/runtime_compile/v1`
- `contracts/interaction_events/v1`
- `contracts/feedback_events/v1`
- `contracts/artifact_manifest/v1`

## Scope boundary

For this landing:

- the TypeScript package is the public implementation surface
- the JSON schemas and golden fixtures under `contracts/` are synchronized documentation artifacts
- compiler, pack-format, activation, and learner consume these shapes directly

## Runtime compile v1

`runtime_compile.v1` now documents the actual TS-first compile boundary:

- max-block and max-character context budgets
- explicit native compaction mode selection
- token-aware selected-context blocks
- deterministic pack-backed selection diagnostics and selection digests

## Artifact manifest v1

`artifact_manifest.v1` documents the immutable pack boundary used by activation and compilation:

- graph/vector/router artifact paths
- pack payload checksums
- workspace and event-export provenance
- graph dynamics including structural ops
