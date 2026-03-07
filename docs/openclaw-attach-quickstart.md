# OpenClaw Attach Quickstart

This is the operator-facing setup contract for attaching OpenClawBrain to OpenClaw.

## Attach posture

The goal is fast time-to-first-value:

- attach quickly
- materialize a fast-boot pack quickly
- promote quickly
- learn fresh live events immediately
- replay older history passively in the background
- keep candidate-pack refresh always on

Do not block initial value on a full historical replay.

## Top invariant

The attach path should serve only from the promoted pack.

If the promoted manifest requires learned routing, the served compile must prove that the learned `route_fn` ran by exposing `usedLearnedRouteFn=true` plus the active `routerIdentity`.

## Install and prove

This public repo is the package surface and proof harness for that attach boundary.

```bash
corepack enable
pnpm install --frozen-lockfile
pnpm check
pnpm lifecycle:smoke
pnpm observability:smoke
```

For a package-first attach path inside OpenClaw, the narrow install lane is:

```bash
pnpm add @openclawbrain/contracts @openclawbrain/events @openclawbrain/event-export @openclawbrain/learner @openclawbrain/activation @openclawbrain/compiler
```

Add `@openclawbrain/openclaw` when you want the typed bridge for promoted-pack compile consumption and normalized event emission.

That install set maps to the attach flow like this:

- `@openclawbrain/events` and `@openclawbrain/event-export` normalize live and backfill learning inputs
- `@openclawbrain/learner` materializes fast-boot and fresher candidate packs, including learned `route_fn` artifacts when needed
- `@openclawbrain/activation` stages, promotes, and inspects pack slots
- `@openclawbrain/compiler` compiles from the promoted pack and emits route/fallback diagnostics

## Operator flow

For an attached deployment, the intended flow is:

1. Install the OpenClawBrain packages or the OpenClaw bridge.
2. Point the learner/event-export surface at current workspace state plus recent events.
3. Materialize a fast-boot pack immediately.
4. Promote that pack so compile can start right away.
5. Keep live event ingestion running continuously.
6. Keep passive backfill and candidate-pack refresh running continuously.
7. Promote fresher activation-ready packs as they become available.

## Healthy steady state

Once attached, healthy steady state looks like this:

1. First value appears from the fast-boot pack before a full replay finishes.
2. Fresh live events are learned first while older history catches up in the background.
3. Promotions move freshness forward through newer snapshots and export digests.
4. Learned-route diagnostics stay visible whenever the promoted pack requires learned routing.
5. Fallback stays explicit instead of silently hiding selection behavior.

## Non-goals for first setup

The first setup should not require:

- a full archive replay before first use
- a multi-hour import as a hard gate
- separate competing brain services on the production host
- manual babysitting just to keep passive learning alive

## Proof boundary

The concrete proof path in this repo is:

```bash
pnpm check
pnpm lifecycle:smoke
pnpm observability:smoke
```

Those proofs cover normalized events, deterministic export, learner pack materialization, activation promotion, promoted-pack compilation, and operator diagnostics over health, freshness, learned-route evidence, and fallback.

The detailed diagnostics contract lives in [`docs/operator-observability.md`](docs/operator-observability.md).
The repo-wide convergence statement lives in [`docs/learning-first-convergence.md`](docs/learning-first-convergence.md).
