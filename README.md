# OpenClawBrain

OpenClawBrain is now a TypeScript-first `pnpm` workspace.

The repo's forward surface ships the full public package lane for contracts, normalized event flows, artifact provenance, immutable pack artifacts, activation helpers, deterministic compilation, and learner-side candidate-pack assembly.

## Boundary

The product boundary is intentionally narrow:

- OpenClaw owns runtime orchestration, prompt assembly, diagnostics, sessions, and fail-open behavior.
- OpenClawBrain owns contracts, event and export normalization, workspace and provenance metadata, immutable pack artifacts, activation helpers, deterministic compilation, and learner-side candidate-pack assembly from normalized OpenClaw event exports.

There is no Python daemon, socket, hook, or wheel-release lane in the supported surface of this repo.

## Public surface

The supported public packages live under [`packages/`](packages):

- `@openclawbrain/contracts`
- `@openclawbrain/events`
- `@openclawbrain/event-export`
- `@openclawbrain/workspace-metadata`
- `@openclawbrain/provenance`
- `@openclawbrain/pack-format`
- `@openclawbrain/activation`
- `@openclawbrain/compiler`
- `@openclawbrain/learner`

- `@openclawbrain/contracts` defines canonical public payloads, validators, and checksum helpers.
- `@openclawbrain/events` exposes normalized interaction and feedback event builders.
- `@openclawbrain/event-export` derives deterministic event-export ranges and provenance.
- `@openclawbrain/workspace-metadata` normalizes declared workspace snapshot metadata.
- `@openclawbrain/provenance` builds pack provenance from workspace and event-export inputs.
- `@openclawbrain/pack-format` handles immutable pack layout, validation, and activation-pointer helpers.
- `@openclawbrain/activation` provides package-first activation inspection, staging, promotion, and rollback helpers.
- `@openclawbrain/compiler` provides deterministic runtime compilation over a pack boundary.
- `@openclawbrain/learner` assembles candidate packs from normalized event exports.

## Package flow

- `@openclawbrain/events` builds normalized interaction and feedback events.
- `@openclawbrain/event-export` turns those events into deterministic export ranges and provenance.
- `@openclawbrain/workspace-metadata` and `@openclawbrain/provenance` stamp artifact-side workspace and build provenance.
- `@openclawbrain/learner` assembles deterministic candidate packs from event exports and workspace provenance.
- `@openclawbrain/activation` stages, promotes, inspects, and rolls back pack activation state.
- `@openclawbrain/compiler` consumes coherent pack artifacts for runtime-side context selection.

## Workspace

Requires Node 20+ and `pnpm` 10+.

```bash
corepack enable
pnpm install --frozen-lockfile
pnpm check
pnpm release:pack
```

`pnpm check` builds the workspace and runs the package tests.

`pnpm release:pack` creates package tarballs in `.release/` for the full public package surface.

For a full release-candidate pass, run:

```bash
pnpm release:check
```

That command cleans the workspace, rebuilds it, reruns tests, and produces publishable tarballs for all public `@openclawbrain/*` packages.

## Docs

- TypeScript-first workspace: [docs/typescript-first-convergence.md](docs/typescript-first-convergence.md)
- Contracts overview: [contracts/README.md](contracts/README.md)
- Release checklist: [docs/release.md](docs/release.md)
- Public TypeScript packages: [packages/contracts/README.md](packages/contracts/README.md), [packages/events/README.md](packages/events/README.md), [packages/event-export/README.md](packages/event-export/README.md), [packages/workspace-metadata/README.md](packages/workspace-metadata/README.md), [packages/provenance/README.md](packages/provenance/README.md), [packages/pack-format/README.md](packages/pack-format/README.md), [packages/activation/README.md](packages/activation/README.md), [packages/compiler/README.md](packages/compiler/README.md), [packages/learner/README.md](packages/learner/README.md)
