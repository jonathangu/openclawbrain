# Contributing

## Development setup

1. Clone the repository and enter it.
2. Install Node.js 20 or newer.
3. Install workspace dependencies:
   - `pnpm install`
4. Validate the workspace:
   - `pnpm check`

## Working in the workspace

- Keep changes focused on the TypeScript packages, canonical contracts, and supporting docs.
- Prefer package-scoped validation while iterating:
  - `pnpm --filter @openclawbrain/contracts test`
  - `pnpm --filter @openclawbrain/pack-format test`
  - `pnpm --filter @openclawbrain/compiler test`
  - `pnpm --filter @openclawbrain/learner test`
- Run `pnpm check` before handing off a branch.

## Release process

1. Bump versions in the affected package manifests under `packages/*/package.json`.
2. Update `CHANGELOG.md` for the release.
3. Run `pnpm install` if dependency metadata changed.
4. Run `pnpm check`.
5. Create and push a release tag that matches `release-vX.Y.Z`.
