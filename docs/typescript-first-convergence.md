# TypeScript-First Convergence

This repo now carries the public TypeScript OpenClawBrain surface.

## Workspace

The supported package set is:

- `@openclawbrain/contracts`
- `@openclawbrain/pack-format`
- `@openclawbrain/compiler`
- `@openclawbrain/learner`

These packages build from the root `pnpm` workspace.

## Boundary

- OpenClaw owns runtime orchestration, prompt assembly, diagnostics, sessions, and fail-open behavior.
- OpenClawBrain owns public contracts, immutable pack artifacts, deterministic compilation, and learner-side candidate-pack assembly from normalized event exports.

## Repository rules

- The public repo surface is package-first.
- Artifact contracts and pack layouts are the integration boundary.
- Package docs, contract fixtures, and release metadata are the only public-facing support material kept in-tree.

## Commands

```bash
pnpm install
pnpm check
```
