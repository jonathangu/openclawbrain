# OpenClawBrain 0.4.15 / 0.4.4 split-package release notes

Published packages:

- `@openclawbrain/openclaw@0.4.4`
- `@openclawbrain/cli@0.4.15`

This release publishes the runtime-fix work that made the live OpenClawBrain activation root materially stronger: the promoted pack now carries live numeric embeddings again, learned-route replay scoring no longer hides real wins as ties, and live `learn` no longer dies while persisting traced-learning bridge state.

## Why this release exists

The prior split release kept the public operator lane coherent, but the runtime still had three truth gaps that mattered in real use:

1. promoted candidate packs could skip the canonical embedder reindex path and surface `stored=0/...` embeddings even after successful learning
2. the recorded-session replay summary scorer could flatten learned-route phrase-coverage wins into ties against simpler retrieval baselines
3. live `learn` could recurse traced-learning bridge state until persistence failed during JSON serialization

This release closes those runtime gaps on the canonical public lane.

## What changed

### `@openclawbrain/cli@0.4.15`

- learned-route replay proof now preserves aggregate phrase coverage, making truthful learned-route wins visible instead of flattening them into ties
- traced-learning bridge persistence now stores a flattened bridged-runtime summary instead of recursively nesting prior bridge state
- teacher/runtime truth surfaces now report fresh no-op teacher cycles cleanly instead of conflating them with stale/broken state

### `@openclawbrain/openclaw@0.4.4`

- candidate-pack embedder reindexing now reuses the canonical learner helper, so promoted packs keep live numeric embeddings truthfully
- first-promotion learned-route selection now carries seed cue blocks forward, improving proof-context selection without changing the bounded runtime budget contract
- the published runtime payload reflects the same verified route-selection and embedding-materialization logic that now drives the exercised host surface

## Proof coverage

Focused package-facing proof added or exercised in-repo:

- `packages/cli/dist/test/teacher-status-truth.test.js`
- `packages/cli/dist/test/cli-embedder-reindex.test.js`
- `packages/cli/dist/test/learned-route-seed-carry-forward.test.js`
- `packages/cli/dist/test/replay-score-resolution.test.js`
- `packages/cli/dist/test/traced-learning-bridge.test.js`
- `packages/openclaw/dist/test/teacher-status-truth.test.js`
- `packages/openclaw/dist/test/cli-embedder-reindex.test.js`

## Verification summary

Executed before publish:

```bash
npm test
npm run release:verify:proof
npm run release:verify:openclaw
npm run release:verify:cli
```

Live local verification on the exercised host also promoted a new pack and confirmed:

- serving pack `pack-ddaa6a1f`
- learned `route_fn` available and used on serve
- embeddings `live=yes` with `stored=415/427`
- teacher status `healthy=yes`, `stale=no`

Publish order remains:

```bash
npm publish ./packages/openclaw
npm publish ./packages/cli
```
