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
```

## Package-scoped validation

```bash
pnpm --filter @openclawbrain/contracts test
pnpm --filter @openclawbrain/pack-format test
pnpm --filter @openclawbrain/compiler test
pnpm --filter @openclawbrain/learner test
```

## What `pnpm check` covers

- TypeScript project-reference build for all public packages
- Node test runs for each package
- Contract, pack-format, compiler, and learner fixture validation
