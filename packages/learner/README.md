# `@openclawbrain/learner`

Candidate-pack assembly helpers for OpenClawBrain.

This package stays on the artifact side of the boundary: it ingests normalized OpenClaw event exports, emits deterministic candidate pack payloads, and materializes pack directories for downstream validation and activation.

## Install

```bash
pnpm add @openclawbrain/learner
```

## Includes

- normalized event-export ingestion for learner-side pack assembly
- deterministic candidate-pack manifest, graph, vector, and router payload generation
- embedded workspace snapshot provenance inside emitted manifests
- on-disk materialization for coherent downstream activation and eval steps
- summary metadata for pack provenance and event-export coverage
