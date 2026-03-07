# Reproduce the workspace

This repo is reproducible with a Node 20 + `pnpm` toolchain only.

## Prerequisites

- Node.js `>=20`
- `pnpm` `10.30.3`

## Fresh clone validation

```bash
git clone <repo-url>
cd <repo>
pnpm install --frozen-lockfile
pnpm check
pnpm lifecycle:smoke
pnpm release:pack
```

## Package-scoped validation

```bash
pnpm --filter @openclawbrain/<package-name> build
pnpm --filter @openclawbrain/<package-name> test
```

Published package names:

- `contracts`
- `events`
- `event-export`
- `workspace-metadata`
- `provenance`
- `pack-format`
- `activation`
- `compiler`
- `learner`

## What `pnpm check` covers

- TypeScript project-reference build for all public packages
- Node test runs for each package
- Package fixture validation across the published workspace
- Root Phase-2 lifecycle smoke across events, event export, learner pack materialization, activation promotion, and compiler runtime compilation

## Direct lifecycle proof

```bash
pnpm lifecycle:smoke
```

That command rebuilds the workspace and then runs the same end-to-end lifecycle proof as `pnpm check`: it creates normalized events, exports them, materializes active/candidate packs, stages/promotes activation state, and compiles against the promoted pack inside a temp directory.

## What `pnpm release:pack` covers

- Tarball generation for every published `@openclawbrain/*` package
- Packaged-file validation against each package's `files`, `exports`, and `prepack` hooks
