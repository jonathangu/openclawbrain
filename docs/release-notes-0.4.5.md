# OpenClawBrain CLI 0.4.5

## What shipped

- vendored the learner bundle into `@openclawbrain/cli` so the published CLI carries the native policy-gradient fix from tracked repo source
- fixed the native V2 policy-gradient route-update seam where serve-time decisions used served-pack block ids but candidate learning reconstructed trajectories against candidate-pack ids
- added a focused regression test proving realistic cross-surface feedback now yields native V2 nonzero updates without falling back to trace-based V1 updates

## Operator lane

```bash
npx @openclawbrain/cli@0.4.5 install --openclaw-home ~/.openclaw
npx @openclawbrain/cli@0.4.5 status --openclaw-home ~/.openclaw --detailed
```

## Native PG proof target

The strict target for this release is that V2 policy gradient itself works end to end: related-interaction feedback binds to serve-time decisions, reconstructed trajectories land on candidate-pack block ids, and the learned router emits nonzero native V2 updates without needing the temporary V1 fallback path.
