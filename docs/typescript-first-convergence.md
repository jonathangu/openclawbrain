# TypeScript-First Convergence

This repo now carries the real TypeScript-first OpenClawBrain public surface.

## Current workspace

The supported public package wave is:

- `@openclawbrain/contracts`
- `@openclawbrain/events`
- `@openclawbrain/event-export`
- `@openclawbrain/workspace-metadata`
- `@openclawbrain/provenance`
- `@openclawbrain/pack-format`
- `@openclawbrain/activation`
- `@openclawbrain/compiler`
- `@openclawbrain/learner`

All packages build, test, pack, and publish from the root `pnpm` workspace.

## Boundary

The repo is intentionally narrow:

- OpenClaw owns runtime orchestration, prompt assembly, diagnostics, sessions, and fail-open behavior.
- OpenClawBrain owns public contracts, event normalization, workspace metadata, artifact provenance, immutable pack artifacts, structural compaction, activation helpers, deterministic compilation, and learner-side candidate-pack assembly.

There is no Python runtime-overlap lane in this repo.

## Context product story

This pass makes the repo/package story match the intended end state:

- larger-context tolerance is explicit in `runtime_compile.v1` through max-block and max-character budgets
- native structural compaction is represented in pack blocks and compile outputs instead of hidden prompt munging
- deterministic pack-backed context selection is the compiler contract, using pack graph order as the final tiebreaker
- learner outputs now include structural-summary blocks so large normalized event exports stay usable without a runtime-side database

## Package flow

- `@openclawbrain/events` normalizes runtime interaction and feedback payloads.
- `@openclawbrain/event-export` derives deterministic export ranges and export provenance.
- `@openclawbrain/workspace-metadata` and `@openclawbrain/provenance` produce stable workspace and build provenance.
- `@openclawbrain/learner` turns event exports into candidate pack artifacts with structural summaries.
- `@openclawbrain/activation` promotes coherent pack artifacts into active or rollback-ready slots.
- `@openclawbrain/compiler` loads coherent pack artifacts and selects runtime context deterministically.

## Workspace commands

```bash
pnpm install
pnpm check
pnpm release:pack
pnpm release:check
```

`pnpm release:pack` writes package tarballs to `.release/`, and `pnpm release:check` runs the clean release-candidate pass documented in `docs/release.md`.

## Repo stance

The repo documents and ships the TypeScript workspace only.
