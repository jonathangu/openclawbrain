# Public Surface

This repo ships OpenClawBrain as a TypeScript package workspace.

## Supported packages

The supported package wave is:

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

## Runtime boundary

The boundary is intentionally narrow:

- OpenClaw owns runtime orchestration, prompt assembly, diagnostics, sessions, and fail-open behavior.
- OpenClawBrain owns contracts, event normalization, workspace/build provenance, immutable pack artifacts, activation helpers, deterministic compilation, and learner-side candidate-pack assembly.

## Runtime and learning behavior

- Fast boot reads existing active-pack files on disk, so runtime answers are available immediately under explicit compile budgets.
- The learner keeps always-on background learning active by materializing candidate packs from normalized event exports.
- Teacher logic stays off the runtime hot path; runtime compilation consumes promoted pack artifacts and never waits on background labeling passes.
- Native structural compaction and deterministic pack-backed selection are enforced at compile time through `runtime_compile.v1`.

## Proof boundary

The root workspace keeps an honest end-to-end proof lane:

- `pnpm check` rebuilds the workspace, runs package tests, and executes the lifecycle smoke.
- `pnpm lifecycle:smoke` reruns the same lifecycle directly in a temp directory using shipped package APIs on disk:
  - normalized events
  - deterministic export
  - learner pack materialization
  - activation staging/promotion
  - runtime compilation against the promoted pack

## Repo stance

The repo documents and ships this package surface only.
