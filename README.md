# OpenClawBrain

OpenClawBrain is the TypeScript-first workspace for deterministic, pack-backed context products.

The repo’s public story is now centered on four guarantees:

- larger-context tolerance through explicit compile budgets and token-aware pack blocks
- native structural compaction over pack-backed context instead of ad hoc prompt-side truncation
- deterministic context selection from immutable packs, vectors, and manifest-gated routing artifacts
- a clean repo/package boundary where the monorepo is private and the scoped TypeScript packages are the supported public surface

Phase 2 is now explicit at the workspace root: the repo carries a deterministic lifecycle smoke that proves one true lifecycle across normalized events, event export, learner pack materialization, activation staging/promotion, and compiler runtime compilation against the promoted pack.

## Boundary

The product boundary is intentionally narrow:

- OpenClaw owns runtime orchestration, prompt assembly, diagnostics, sessions, and fail-open behavior.
- OpenClawBrain owns contracts, normalized event flows, workspace and provenance metadata, immutable pack artifacts, activation helpers, native structural compaction, deterministic compilation, and learner-side candidate-pack assembly.

There is no Python daemon, socket, hook, or wheel-release lane in the supported surface of this repo.

## Public packages

The supported packages live under `packages/`:

- `@openclawbrain/contracts`
- `@openclawbrain/events`
- `@openclawbrain/event-export`
- `@openclawbrain/workspace-metadata`
- `@openclawbrain/provenance`
- `@openclawbrain/pack-format`
- `@openclawbrain/activation`
- `@openclawbrain/compiler`
- `@openclawbrain/learner`

## Context story

- `@openclawbrain/contracts` defines `runtime_compile.v1`, immutable pack payloads, token-aware block metadata, and native compaction semantics.
- `@openclawbrain/learner` emits deterministic candidate packs with structural-summary blocks that make large event exports serveable.
- `@openclawbrain/pack-format` preserves the immutable graph/vector/router boundary that compilation reads from disk.
- `@openclawbrain/compiler` performs deterministic pack-backed ranking, enforces learned-routing policy, and applies native structural compaction under character budgets.
- `@openclawbrain/activation` promotes only activation-ready packs into active runtime slots.

## Package flow

- `@openclawbrain/events` builds normalized interaction and feedback events.
- `@openclawbrain/event-export` turns those events into deterministic export ranges and provenance.
- `@openclawbrain/workspace-metadata` and `@openclawbrain/provenance` stamp workspace and build provenance into immutable artifacts.
- `@openclawbrain/learner` assembles candidate packs with structural summaries and deterministic ids.
- `@openclawbrain/activation` stages, promotes, inspects, and rolls back activation state.
- `@openclawbrain/compiler` consumes coherent pack artifacts for runtime-side context selection.

## Workspace

Requires Node 20+ and `pnpm` 10+.

```bash
corepack enable
pnpm install --frozen-lockfile
pnpm check
pnpm lifecycle:smoke
pnpm release:pack
```

`pnpm check` builds the workspace, runs the package tests, and executes the root lifecycle smoke lane.

`pnpm lifecycle:smoke` rebuilds the workspace and runs the Phase-2 lifecycle proof on a temp directory using the existing public package APIs on disk:

- `@openclawbrain/events` creates normalized interaction and feedback events
- `@openclawbrain/event-export` derives deterministic event export range and provenance
- `@openclawbrain/learner` materializes active and candidate packs from those exports
- `@openclawbrain/activation` stages and promotes the candidate pack into the active slot
- `@openclawbrain/compiler` compiles runtime context from the promoted pack

`pnpm release:pack` creates package tarballs in `.release/` for the full public package surface.

For a release-candidate pass, run:

```bash
pnpm release:check
```

That command cleans the workspace, rebuilds it, reruns tests, and produces publishable tarballs for all public `@openclawbrain/*` packages.

## Docs

- `docs/typescript-first-convergence.md`
- `docs/contracts-v1.md`
- `scripts/lifecycle-smoke.mjs`
- `docs/release.md`
- `contracts/README.md`
- `packages/contracts/README.md`
- `packages/compiler/README.md`
- `packages/learner/README.md`
- `packages/pack-format/README.md`
