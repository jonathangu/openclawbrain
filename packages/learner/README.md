# `@openclawbrain/learner`

Candidate-pack assembly helpers for always-on OpenClawBrain learning.

This package stays on the artifact side of the boundary: it ingests normalized OpenClaw event exports, emits deterministic candidate pack payloads, and materializes pack directories for downstream validation and activation.

## Install

```bash
pnpm add @openclawbrain/learner
```

## Includes

- deterministic fast-boot candidate packs with passive background-learning defaults
- human/self label-harvest surfaces embedded into graph blocks, vectors, and manifest summaries
- structural graph learning metadata spanning Hebbian reinforcement, decay, and split/merge/prune/connect ops
- embedded workspace snapshot provenance inside emitted manifests
- on-disk materialization for coherent downstream activation and eval steps
