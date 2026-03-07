# `@openclawbrain/learner`

Candidate-pack and learned `route_fn` assembly helpers for always-on OpenClawBrain learning.

This package stays on the artifact side of the boundary: it ingests normalized event exports, emits deterministic candidate pack payloads, and materializes pack directories for downstream validation and activation.

## Install

```bash
pnpm add @openclawbrain/learner
```

## Includes

- deterministic fast-boot candidate packs with live-first/background-backfill defaults
- learned routing artifacts with stable `routerIdentity` values such as `pack-id:route_fn`
- human/self label-harvest surfaces embedded into graph blocks, vectors, and manifest summaries
- structural graph learning metadata spanning Hebbian reinforcement, decay, and split/merge/prune/connect ops
- embedded workspace snapshot provenance inside emitted manifests
- on-disk materialization for coherent downstream activation and evaluation steps
- bridge-slice and bridge-bundle materialization helpers for continuous learner refreshes
- canonical teacher-supervision artifact builders with dedup and freshness metadata
- teacher-supervision-aware candidate packs that carry fresh operator guidance into future graph/vector payloads
