# Contracts v1

Phase 1 contract scaffolding for the canonical OpenClaw/OpenClawBrain
rearchitecture lives under [contracts/README.md](../contracts/README.md).

Primary contract roots:

- [contracts/runtime_compile/v1](../contracts/runtime_compile/v1)
- [contracts/interaction_events/v1](../contracts/interaction_events/v1)
- [contracts/feedback_events/v1](../contracts/feedback_events/v1)
- [contracts/artifact_manifest/v1](../contracts/artifact_manifest/v1)

OpenClawBrain-side validation helpers and the legacy feedback bridge live in
[openclawbrain/contracts/v1.py](../openclawbrain/contracts/v1.py).

Scope boundary for this landing:

- schemas and golden fixtures are normative
- tests enforce explicit learned-routing fields where the canonical plan
  requires them
- no runtime or daemon cutover is implied by these files
