# OpenClawBrain

OpenClawBrain is now a TypeScript-first workspace for contracts, immutable pack artifacts, deterministic runtime compilation, and learner-side candidate-pack assembly.

## Public packages

The supported public packages live under [`packages/`](packages):

- `@openclawbrain/contracts`
- `@openclawbrain/pack-format`
- `@openclawbrain/compiler`
- `@openclawbrain/learner`

## Product boundary

- OpenClaw owns runtime orchestration, prompt assembly, diagnostics, sessions, and fail-open behavior.
- OpenClawBrain owns public contracts, immutable pack artifacts, deterministic compilation, and learner-side candidate-pack assembly from normalized OpenClaw event exports.

## Workspace

```bash
pnpm install
pnpm check
```

`pnpm check` builds the workspace and runs the package tests.

## Docs

- TypeScript-first overview: `docs/typescript-first-convergence.md`
- Contract scaffolding: `contracts/README.md`
- Contracts package: `packages/contracts/README.md`
- Pack format package: `packages/pack-format/README.md`
- Compiler package: `packages/compiler/README.md`
- Learner package: `packages/learner/README.md`
- Release flow: `docs/release.md`
