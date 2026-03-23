# Contributing to OpenClawBrain

## Quick orientation

OpenClawBrain has two public package surfaces:

- `packages/openclaw` is the OpenClaw plugin and runtime payload
- `packages/cli` is the operator CLI

The repo also contains the transcript-memory substrate, learning pipeline, and operator docs that describe the public boundary.

## Setup

```bash
npm install
npm test
npm run release:verify
```

## Before you open a PR

1. Keep the public install story consistent. If you touch install or status behavior, update `README.md`, `docs/README.md`, `docs/getting-started/quick-start.md`, and any affected package README in the same pass.
2. Update [CHANGELOG.md](CHANGELOG.md) and [CLAIMS.md](CLAIMS.md) when a user-visible capability or boundary changes.
3. Add or update tests for runtime, CLI, or learning-pipeline changes.
4. Run the package-specific verification command when you touch a published surface:

```bash
npm --prefix packages/openclaw run release:verify
npm --prefix packages/cli run release:verify
```

## Scope

OpenClawBrain is a memory layer for OpenClaw. It does not manage the OpenClaw gateway lifecycle for the operator.

Do not add docs or code that imply the plugin:

- restarts OpenClaw automatically
- edits LaunchAgent files or gateway environment files
- turns same-gateway multi-profile or shared-root concurrency into a claimed public lane without proof and claims-boundary updates

## Documentation expectations

- Write for evaluators, operators, and contributors separately.
- Keep front-door install copy unpinned.
- Put version-specific precision in the changelog or release notes.
- Prefer plain language over internal notes voice.

## Architecture reading list

- [docs/architecture/overview.md](docs/architecture/overview.md)
- [docs/architecture/learning-pipeline.md](docs/architecture/learning-pipeline.md)
- [docs/architecture/fail-open.md](docs/architecture/fail-open.md)
- [docs/architecture/deep-dive.md](docs/architecture/deep-dive.md)
