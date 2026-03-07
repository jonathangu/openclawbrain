# OpenClaw Integration

OpenClawBrain integrates behind an OpenClaw-owned runtime boundary.

This document defines that boundary and the current public proof surface. It does not mean this repo ships a live OpenClaw runtime, a production rollout lane, or the full comparative benchmark harness.

## Boundary

### OpenClaw owns
- session and channel orchestration
- fail-open behavior
- prompt assembly and hot-path serving
- deployment, routing, and rollback controls

### OpenClawBrain owns
- contracts and typed event schemas
- event normalization and export boundaries
- workspace metadata and provenance
- immutable pack format and activation helpers
- deterministic compilation and learner behavior
- optional OpenClaw-facing runtime bridge helpers via `@openclawbrain/openclaw`

## Minimal package surface

Start with the narrow package set:

```bash
pnpm add @openclawbrain/contracts @openclawbrain/events @openclawbrain/event-export @openclawbrain/learner @openclawbrain/activation @openclawbrain/compiler
```

Add these when needed:
- `@openclawbrain/pack-format`
- `@openclawbrain/workspace-metadata`
- `@openclawbrain/provenance`
- `@openclawbrain/openclaw` for the runtime-owned bridge layer itself

This GitHub repo is public. The supported integration boundary is the published `@openclawbrain/*` packages plus versioned fixtures under `contracts/`; workspace scripts, smoke lanes, and repo layout are proof/release machinery, not a separate runtime API.

## Bring-up sequence

From the repo root:

```bash
corepack enable
pnpm install --frozen-lockfile
pnpm check
pnpm release:pack
```

## Proofs available in this repo today

### Mechanism proof
Use the built-in smoke lanes:

```bash
pnpm lifecycle:smoke
pnpm observability:smoke
```

These prove:
- pack materialization
- activation staging/promotion
- runtime compilation against promoted packs
- operator-facing observability and freshness diagnostics

They run against the public package surface in this repo and temporary activation state. They do not stand up a deployed OpenClaw service.

### Comparative benchmark proof
The larger comparative benchmark/proof harness currently lives in the sibling public repo:
- `https://github.com/jonathangu/brain-ground-zero`

Use that repo for the published recorded-session and sparse-feedback benchmark families. It is separate from the supported package surface in this repo, and this repo does not currently ship those proof families directly.

## Failure semantics

Integration stays fail-open:
- OpenClaw continues serving if learning or artifact refresh is delayed
- OpenClaw can fall back to core runtime behavior if brain artifacts are stale or unavailable
- learning, harvesting, and graph updates stay off the hot path

## Related docs
- [openclaw-attach-quickstart.md](openclaw-attach-quickstart.md)
- [operator-observability.md](operator-observability.md)
- [reproduce-eval.md](reproduce-eval.md)
- [typescript-first-convergence.md](typescript-first-convergence.md)
