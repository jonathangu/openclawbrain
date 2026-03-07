# Contracts v1

Canonical contract scaffolding lives under `contracts/` and is implemented in TypeScript by `@openclawbrain/contracts`.

Primary contract roots:

- `contracts/runtime_compile/v1`
- `contracts/interaction_events/v1`
- `contracts/feedback_events/v1`
- `contracts/artifact_manifest/v1`

## Scope boundary

For the current public landing:

- the published packages plus the JSON schemas and golden fixtures under `contracts/` are the supported public contract surface
- the rest of the workspace is public documentation and proof machinery, not a second contract API
- compiler, pack-format, activation, learner, and OpenClaw bridge helpers consume these shapes directly

## Runtime compile v1

`runtime_compile.v1` documents the served compile boundary:

- max-block and max-character context budgets
- explicit native compaction mode selection
- token-aware selected-context blocks
- deterministic pack-backed selection diagnostics and selection digests
- learned-route evidence through `usedLearnedRouteFn` and `routerIdentity`

## Artifact manifest v1

`artifact_manifest.v1` documents the immutable pack boundary used by learner, activation, and compiler:

- graph/vector/router artifact paths
- pack payload checksums
- workspace and event-export provenance
- route policy plus promoted router identity
- graph dynamics including structural ops
