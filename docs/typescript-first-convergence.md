# TypeScript-First Convergence

This repo now carries the real public TypeScript-first OpenClawBrain surface.

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
- OpenClawBrain owns public contracts, event and export normalization, workspace metadata, artifact provenance, immutable pack artifacts, activation helpers, deterministic compilation, and learner-side candidate-pack assembly from normalized OpenClaw event exports.

There is no Python runtime-overlap lane in this repo.

## Learner scope

The learner package in this pass covers:

- normalized event-export ingestion
- workspace-metadata-aware pack provenance
- deterministic candidate-pack assembly
- on-disk candidate-pack materialization for downstream validation and activation work

It stays on the artifact side of the boundary. It does not own runtime activation or serve-path behavior.

## Package flow

- `@openclawbrain/events` normalizes runtime interaction and feedback payloads.
- `@openclawbrain/event-export` derives deterministic export ranges and export provenance.
- `@openclawbrain/workspace-metadata` and `@openclawbrain/provenance` produce stable workspace and build provenance.
- `@openclawbrain/learner` turns event exports into candidate pack artifacts.
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
