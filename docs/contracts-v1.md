# Contracts v1

Canonical contract scaffolding lives under `contracts/` and is implemented in TypeScript by `@openclawbrain/contracts`.

Primary contract roots:

- `contracts/runtime_compile/v1`
- `contracts/interaction_events/v1`
- `contracts/feedback_events/v1`
- `contracts/artifact_manifest/v1`

Scope boundary for this landing:

- schemas and golden fixtures are normative
- the TypeScript validators and event builders are the public implementation surface
- compiler, pack-format, and learner packages consume these shapes directly
