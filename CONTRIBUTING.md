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

- `pnpm clean` removes generated `dist/` outputs and `.release/` tarballs.
- `pnpm build` builds the public workspace packages.
- `pnpm check` typechecks and builds the workspace, runs the package test suites, and executes the root lifecycle smoke lane.
- `pnpm lifecycle:smoke` runs the standalone end-to-end lifecycle proof from normalized events through activation promotion and runtime compilation.
- `pnpm release:pack` creates package tarballs in `.release/` for all public workspace packages.
- `pnpm release:check` performs a clean rebuild, reruns tests, and generates release tarballs.
- `pnpm release:publish:dry-run` exercises the publish path without uploading packages.

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
