# Contributing

## Development setup

1. Clone the repository and enter it:
   - `git clone <repo-url>`
   - `cd <repo>`
2. Enable Corepack and install the TypeScript workspace:
   - `corepack enable`
   - `pnpm install --frozen-lockfile`
3. Run the workspace checks:
   - `pnpm check`

## Release-readiness commands

- `pnpm build` builds the public workspace packages.
- `pnpm check` rebuilds the workspace and runs the package test suites.
- `pnpm release:pack` creates package tarballs in `.release/` for all public workspace packages.
- `pnpm release:check` performs a clean rebuild, reruns tests, and generates release tarballs.

## Code style

- No dedicated linter or formatter is currently required by the TypeScript workspace.
- Keep public interfaces focused, deterministic, and backward-compatible where possible.
- Match the existing style of the package you touch.

## Test requirements

- All workspace checks should pass before opening a PR.
- Run `pnpm check` for changes that touch the TypeScript workspace surface.
- Add or update tests when package behavior changes.

## Releases

The TypeScript package release flow lives in `docs/release.md`.
