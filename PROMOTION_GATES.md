# Promotion Gates

Candidate packs may be promoted only when all gates pass.

## Replay Gates

- No regression on human-positive episodes.
- No increased failure rate on self-negative replay bundles.
- Replayed routing remains stable for protected episodes.

## Health Gates

- `firedPerQuery >= minFiredPerQuery`
- `dormantPercent <= maxDormantPercent`
- `orphanCount <= maxOrphanCount`
- structural churn stays under the per-promotion cap

## Structural Gates

- Candidate mutations are applied to a cloned graph, not the live mutable graph.
- Only bounded proposal batches may be evaluated per promotion.
- `split` and `merge` remain gated until replay evidence is strong enough.

## Evidence Gates

- Trace output must identify pack version, chosen seed, route decisions, and skip reasons.
- Promotion metadata must record pass/fail reason and source evidence.

## Release Gates

- Toy-graph mechanism tests pass.
- Recorded-session replay benchmark passes.
- OpenClaw integration validation passes.
