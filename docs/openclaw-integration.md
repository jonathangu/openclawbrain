# OpenClaw Integration

OpenClawBrain integrates with OpenClaw through a narrow promoted-pack boundary.

## Integration contract

OpenClaw keeps the hot path: session flow, prompt assembly, response delivery, and fail-open serving.

OpenClawBrain supplies the learning side of the boundary:

- normalized event contracts
- deterministic event export and provenance
- immutable pack materialization
- activation staging/promotion
- promoted-pack compilation with learned-route diagnostics

## Top invariant

The promoted pack is the only supported artifact OpenClaw should compile from.

If that pack's manifest says learned routing is required, compilation must use the pack's learned `route_fn`, expose `routerIdentity`, and report `usedLearnedRouteFn=true`.

## Minimal package surface

Start with the narrow package set:

```bash
pnpm add @openclawbrain/contracts @openclawbrain/events @openclawbrain/event-export @openclawbrain/learner @openclawbrain/activation @openclawbrain/compiler
```

Add these when needed:

- `@openclawbrain/pack-format`
- `@openclawbrain/workspace-metadata`
- `@openclawbrain/provenance`
- `@openclawbrain/openclaw` for the typed OpenClaw bridge package itself

The supported public integration boundary is the published `@openclawbrain/*` packages plus versioned fixtures under `contracts/`.

## Bring-up sequence

From the repo root:

```bash
corepack enable
pnpm install --frozen-lockfile
pnpm check
pnpm release:pack
```

## Proofs available in this repo today

Use the built-in smoke lanes:

```bash
pnpm lifecycle:smoke
pnpm observability:smoke
```

These prove:

- pack materialization from normalized inputs
- activation staging and promotion
- compilation against promoted packs
- learned `route_fn` evidence and explicit fallback notes
- operator-facing health and freshness diagnostics

They run against the public package surface in this repo and temporary activation state.

Broader comparative benchmark families live in the sibling public repo `brain-ground-zero`.

## Failure semantics

Integration stays fail-open:

- OpenClaw continues serving if learning or artifact refresh is delayed
- compile can fall back deterministically when token selection misses
- learning, harvesting, and pack refresh stay off the hot path

## Related docs

- [openclaw-attach-quickstart.md](openclaw-attach-quickstart.md)
- [operator-observability.md](operator-observability.md)
- [reproduce-eval.md](reproduce-eval.md)
- [learning-first-convergence.md](learning-first-convergence.md)
