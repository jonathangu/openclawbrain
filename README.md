# OpenClawBrain

This branch now centers the public TypeScript-first convergence workspace.

OpenClawBrain's forward public surface is a `pnpm` monorepo that ships contracts, immutable pack artifacts, deterministic compilation, and learner-side candidate-pack assembly.

## Status

This convergence branch intentionally narrows the product boundary:

- OpenClaw owns runtime orchestration, prompt assembly, diagnostics, sessions, and fail-open behavior.
- OpenClawBrain owns contracts, event and export normalization, workspace and provenance metadata, immutable pack artifacts, activation helpers, deterministic compilation, and learner-side candidate-pack assembly from normalized OpenClaw event exports.

The old Python daemon/socket/hook integration path has been deleted from the forward story on this branch. Remaining Python code in-tree is deletion-target or offline residue, not part of the future public interface this branch is preparing to publish.

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

## Workspace

Requires Node 20+ and `pnpm` 10+.

```bash
corepack enable
pnpm install --frozen-lockfile
pnpm check
pnpm release:pack
```

`pnpm check` builds the TypeScript workspace and runs the package tests for the imported public package wave.

`pnpm release:pack` creates package tarballs in `.release/`.

For a full release-candidate pass, run:

```bash
pnpm release:check
```

That command cleans the workspace, rebuilds it, reruns tests, and produces publishable tarballs for the current public package set.

## Docs

- TypeScript-first convergence: [docs/typescript-first-convergence.md](docs/typescript-first-convergence.md)
- Contracts overview: [contracts/README.md](contracts/README.md)
- Release checklist: [docs/release.md](docs/release.md)
- Public TypeScript packages: [packages/contracts/README.md](packages/contracts/README.md), [packages/events/README.md](packages/events/README.md), [packages/event-export/README.md](packages/event-export/README.md), [packages/workspace-metadata/README.md](packages/workspace-metadata/README.md), [packages/provenance/README.md](packages/provenance/README.md), [packages/pack-format/README.md](packages/pack-format/README.md), [packages/activation/README.md](packages/activation/README.md), [packages/compiler/README.md](packages/compiler/README.md), [packages/learner/README.md](packages/learner/README.md)

## Legacy note

Some Python research, graph, replay, benchmark, and contract-validation code still remains in-tree as legacy offline material. It is no longer the recommended runtime integration or release surface for this branch.
