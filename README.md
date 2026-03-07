# OpenClawBrain

OpenClawBrain is the public learning-first package surface for building, promoting, and compiling deterministic context packs.

## Top invariant

The promoted pack is the only supported learning/serve boundary in this repo.

- learner output becomes an immutable pack with graph, vector, provenance, and optional router artifacts
- activation decides which pack is active
- compiler serves only from the promoted pack
- if a pack requires learned routing, compile must use the pack's learned `route_fn`, surface `routerIdentity`, and report `usedLearnedRouteFn=true`

That is the public story this repo now documents, tests, and publishes.

## Learning-first attach posture

Attach for fast time-to-first-value:

- bootstrap from current workspace state and recent normalized events
- promote a fast-boot pack quickly
- learn fresh live events first while older history catches up later
- let OpenClaw compile useful context immediately after attach
- keep passive historical replay/backfill running in the background at all times
- keep real-time event scanning, supervision harvest, and candidate-pack refresh running continuously
- keep OpenClaw as the sole runtime owner and fail open only for non-learned-required serve-path gaps

## Public surface

The supported public integration surface is:

- published `@openclawbrain/*` packages under `packages/`
- versioned schemas and fixtures under `contracts/`

Everything else in the repo is public proof, documentation, or release machinery for that surface.

For the narrow attach lane, start with:

```bash
pnpm add @openclawbrain/contracts @openclawbrain/events @openclawbrain/event-export @openclawbrain/learner @openclawbrain/activation @openclawbrain/compiler
```

Add `@openclawbrain/pack-format`, `@openclawbrain/workspace-metadata`, and `@openclawbrain/provenance` when you need direct artifact inspection or materialization.

Add `@openclawbrain/openclaw` when you want the typed OpenClaw bridge for promoted-pack compile consumption and normalized event emission.

## What this repo proves

The workspace root carries two deterministic proof lanes:

- `pnpm lifecycle:smoke` proves the learning lifecycle from normalized events to promoted-pack compilation
- `pnpm observability:smoke` proves activation health, promotion freshness, learned `route_fn` evidence, graph-dynamics freshness, supervision freshness, teacher freshness, and explicit fallback usage
- `pnpm observability:report` prints the repo-local JSON proof surface for those observability claims

- OpenClaw owns runtime orchestration, prompt assembly, diagnostics, sessions, and guarded fail-open behavior.
- OpenClawBrain owns contracts, normalized event flows, workspace and provenance metadata, immutable pack artifacts, activation helpers, native structural compaction, deterministic compilation, and learner-side candidate-pack assembly.

This GitHub repo is public. The supported public integration surface is narrower: the published `@openclawbrain/*` packages plus the versioned fixtures under `contracts/`. Workspace layout, root scripts, smoke lanes, and release plumbing are public proof-and-build machinery, not a second semver-stable API.

There is no supported Python daemon, socket, hook, or wheel-release lane in that public integration surface.

## Proof boundary

Those lanes prove the package/mechanism boundary implemented here today.

- This repo directly proves the TypeScript package surface, lifecycle mechanics, and operator-observability APIs.
- The broader comparative benchmark families live in the separate public proof repo: `https://github.com/jonathangu/brain-ground-zero`.
- The deleted `docs/brains-dashboard/` route was a documentation placeholder, not a shipped dashboard application.

## Public packages

The supported packages live under `packages/`:

- `@openclawbrain/contracts`
- `@openclawbrain/events`
- `@openclawbrain/event-export`
- `@openclawbrain/workspace-metadata`
- `@openclawbrain/provenance`
- `@openclawbrain/pack-format`
- `@openclawbrain/activation`
- `@openclawbrain/compiler`
- `@openclawbrain/learner`
- `@openclawbrain/openclaw`

The root workspace package remains `private` in `package.json` so the monorepo itself is not published to npm; that does not mean the GitHub repo is private.

## Context story

- `@openclawbrain/contracts` defines `runtime_compile.v1`, immutable pack payloads, token-aware block metadata, and native compaction semantics.
- `@openclawbrain/learner` emits deterministic candidate packs with structural-summary blocks that make large event exports serveable.
- `@openclawbrain/pack-format` preserves the immutable graph/vector/router boundary that compilation reads from disk.
- `@openclawbrain/compiler` performs deterministic pack-backed ranking, enforces learned-routing policy, and applies native structural compaction under character budgets.
- `@openclawbrain/activation` promotes only activation-ready packs into active runtime slots.

## Package flow

- `@openclawbrain/events` builds normalized interaction and feedback events.
- `@openclawbrain/event-export` derives deterministic export ranges and provenance.
- `@openclawbrain/workspace-metadata` and `@openclawbrain/provenance` stamp workspace and build provenance into immutable artifacts.
- `@openclawbrain/learner` materializes fast-boot and candidate packs, including learned `route_fn` artifacts when required.
- `@openclawbrain/pack-format` preserves the immutable on-disk graph/vector/router boundary.
- `@openclawbrain/activation` stages, promotes, inspects, and rolls back active/candidate packs.
- `@openclawbrain/compiler` compiles from the promoted pack, enforces learned-routing policy, and applies native structural compaction.
- `@openclawbrain/openclaw` wraps the OpenClaw-owned runtime boundary: activation-aware compile diagnostics, learned-route hard-fail enforcement, prompt-context formatting, and normalized runtime event export handoff.

## Workspace

Requires Node 20+ and `pnpm` 10+.

```bash
corepack enable
pnpm install --frozen-lockfile
pnpm check
pnpm lifecycle:smoke
pnpm observability:smoke
pnpm observability:report
pnpm release:pack
```

`pnpm check` builds the workspace, runs package tests, and executes the lifecycle plus observability smoke lanes.

`pnpm observability:report` proves only local artifact/export state inside this repo's temporary fixture lane; it does not claim live production supervision latency or external telemetry coverage.

`pnpm release:pack` creates tarballs in `.release/` for every published `@openclawbrain/*` package.

For a clean outside-consumer proof that installs published packages outside this workspace, use `examples/npm-consumer/README.md`.

## Versioning

The current public npm lane is `0.1.x` across the workspace marker and the published `@openclawbrain/*` packages.

Historical repository tags such as `v12.x` remain in git history, but they are not the current package line, changelog lane, or release trigger.

Use `CHANGELOG.md` for release notes and `docs/release.md` for the release checklist.

## Docs

- `docs/openclaw-attach-quickstart.md`
- `docs/openclaw-integration.md`
- `docs/operator-observability.md`
- `docs/reproduce-eval.md`
- `docs/learning-first-convergence.md`
- `docs/contracts-v1.md`
- `docs/worked-example.md`
- `docs/setup-guide.md`
- `docs/glossary.md`
- `docs/release.md`
- `contracts/README.md`
- `packages/contracts/README.md`
- `packages/compiler/README.md`
- `packages/learner/README.md`
- `packages/openclaw/README.md`
- `packages/pack-format/README.md`
