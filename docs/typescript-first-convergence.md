# TypeScript-First Convergence

This repo now carries the real public TypeScript-first OpenClawBrain surface.

## Current workspace

The imported package wave is:

- `@openclawbrain/contracts`
- `@openclawbrain/events`
- `@openclawbrain/event-export`
- `@openclawbrain/workspace-metadata`
- `@openclawbrain/provenance`
- `@openclawbrain/pack-format`
- `@openclawbrain/activation`
- `@openclawbrain/compiler`
- `@openclawbrain/learner`

These packages build from the root `pnpm` workspace.

## Boundary

The branch is intentionally narrow:

- OpenClaw owns runtime orchestration, prompt assembly, diagnostics, sessions, and fail-open behavior.
- OpenClawBrain owns public contracts, event and export normalization, workspace metadata, artifact provenance, immutable pack artifacts, activation helpers, deterministic compilation, and learner-side candidate-pack assembly from normalized OpenClaw event exports.

The deleted public surface from this branch is the old Python runtime-overlap model:

- daemon and socket lifecycle
- OpenClaw hook-pack injection
- runtime-facing OpenClaw adapter CLIs

## Learner scope

The learner package in this pass covers:

- normalized event-export ingestion
- workspace-metadata-aware pack provenance
- deterministic candidate-pack assembly
- on-disk candidate-pack materialization for downstream validation and activation work

It stays on the artifact side of the boundary. It does not own runtime activation or serve-path behavior.

## Workspace commands

```bash
pnpm install
pnpm check
pnpm release:pack
pnpm release:check
```

`pnpm release:pack` writes package tarballs to `.release/`, and `pnpm release:check` runs the clean release-candidate pass documented in `docs/release.md`.

## Remaining legacy material

The repo still contains legacy Python graph, replay, benchmark, and offline research code. Those pieces are deletion-target or offline residue only, not part of the future runtime story.
