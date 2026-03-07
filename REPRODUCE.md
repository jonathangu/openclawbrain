# Reproduce the workspace

This repo is reproducible with Node 20 + `pnpm` only.

## Prerequisites

- Node.js `>=20`
- `pnpm` `10.30.3`

## Fresh clone validation

```bash
git clone <repo-url>
cd <repo>
corepack enable
pnpm install --frozen-lockfile
pnpm check
pnpm lifecycle:smoke
pnpm observability:smoke
pnpm release:pack
```

## What `pnpm check` covers

- TypeScript project-reference build for all public packages
- Node test runs for each package
- package fixture validation across the published workspace
- lifecycle proof across normalized events, event export, learner pack materialization, activation promotion, and promoted-pack compilation
- observability proof across activation health, promotion safety, freshness, learned `route_fn` evidence, and explicit fallback diagnostics

## Fresh npm consumer smoke

For a true outside-consumer proof, run the checked-in npm example from a brand-new temp directory:

```bash
tmpdir="$(mktemp -d)"
cp examples/npm-consumer/package.json "$tmpdir/package.json"
cp examples/npm-consumer/smoke.mjs "$tmpdir/smoke.mjs"
cd "$tmpdir"
npm install
npm run smoke
```

That path installs the published registry packages, imports the public ESM entrypoints with plain Node, validates a `runtime_compile.v1` request, and builds a deterministic normalized event export from the split `contracts` + `events` + `event-export` surface.

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
- `openclaw`

## Direct lifecycle proof

```bash
pnpm lifecycle:smoke
```

That command rebuilds the workspace and then runs the same end-to-end learning proof as `pnpm check`: it creates normalized events, exports them, materializes active/candidate packs, stages/promotes activation state, and compiles against the promoted pack inside a temp directory.

## Direct observability proof

```bash
pnpm observability:smoke
```

That command rebuilds the workspace and proves the operator-facing diagnostics surface on a temp directory: it inspects active/candidate/previous state, verifies promotion and rollback readiness, reads the staged/promoted/rolled-back freshness targets, detects an explicit async-teacher duplicate/no-op, and compiles a request that must surface learned-route evidence and deterministic fallback notes.

## What `pnpm release:pack` covers

- tarball generation for every published `@openclawbrain/*` package
- packaged-file validation against each package's `files`, `exports`, and `prepack` hooks

## Comparative benchmark note

Broader comparative benchmark families live in the separate public proof repo `brain-ground-zero`. Follow that repo's own instructions for benchmark reproduction; this repo's supported reproduction surface is the Node-only package/proof flow above.
