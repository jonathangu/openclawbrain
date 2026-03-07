# OpenClawBrain

OpenClawBrain is the TypeScript-first workspace for deterministic, pack-backed context products.

The repo’s public story is centered on five concrete claims:

- larger-context tolerance through explicit compile budgets and token-aware pack blocks
- native structural compaction over pack-backed context instead of ad hoc prompt-side truncation
- deterministic context selection from immutable packs, vectors, and manifest-gated routing artifacts
- operator-visible diagnostics that prove activation health, promotion safety, freshness, and runtime fallback
- a clean public-repo/package boundary where this GitHub repo is public, but the supported public surface is the scoped TypeScript packages plus versioned fixtures under `contracts/`

The workspace root now carries two deterministic proof lanes:

- `pnpm lifecycle:smoke` proves one true lifecycle across normalized events, event export, learner pack materialization, activation staging/promotion, and compiler runtime compilation against the promoted pack.
- `pnpm observability:smoke` proves the operator-facing diagnostics story for health, promotions, freshness, and priority fallback.

Those lanes prove the package and mechanism boundary implemented in this repo today. They do not claim a live OpenClaw production rollout, a bundled browser dashboard, or the full comparative benchmark harness inside this repo.

## Versioning

The current public npm/package lane is `0.1.x` across the workspace marker and every published `@openclawbrain/*` package in `packages/`.

Historical repository tags such as `v12.x` belong to earlier legacy milestones. They remain in git history, but they are not the current TypeScript package line, changelog lane, or release automation trigger.

Use `CHANGELOG.md` for package-release notes and `docs/release.md` for the active public release checklist.

## OpenClaw attach quickstart

The attach/install story must optimize for **time-to-first-value**, not for full historical completeness before first use.

That means:

- do not block initial activation on a full history scan
- bootstrap from current workspace state and recent normalized events
- learn fresh live events first while older history catches up later
- let OpenClaw compile useful context immediately after attach
- keep passive historical replay/backfill running in the background at all times
- keep real-time event scanning and supervision harvest running continuously
- keep OpenClaw as the sole runtime owner and fail open only for non-learned-required serve-path gaps

The operator-facing setup contract lives in [`docs/openclaw-attach-quickstart.md`](docs/openclaw-attach-quickstart.md), the diagnostics contract lives in [`docs/operator-observability.md`](docs/operator-observability.md), and the repo-wide convergence statement lives in [`docs/typescript-first-convergence.md`](docs/typescript-first-convergence.md).

For a package-first attach lane, start with:

```bash
pnpm add @openclawbrain/contracts @openclawbrain/events @openclawbrain/event-export @openclawbrain/learner @openclawbrain/activation @openclawbrain/compiler
```

For a copy-paste fresh-consumer proof that installs the published packages outside this workspace, use `examples/npm-consumer/README.md`.

Add `@openclawbrain/pack-format`, `@openclawbrain/workspace-metadata`, and `@openclawbrain/provenance` only when you need to inspect or materialize the immutable artifact boundary directly.

Add `@openclawbrain/openclaw` when you are wiring the OpenClaw-owned runtime integration layer itself.

## Boundary

The product boundary is intentionally narrow:

- OpenClaw owns runtime orchestration, prompt assembly, diagnostics, sessions, and guarded fail-open behavior.
- OpenClawBrain owns contracts, normalized event flows, workspace and provenance metadata, immutable pack artifacts, activation helpers, native structural compaction, deterministic compilation, and learner-side candidate-pack assembly.

This GitHub repo is public. The supported public integration surface is narrower: the published `@openclawbrain/*` packages plus the versioned fixtures under `contracts/`. Workspace layout, root scripts, smoke lanes, and release plumbing are public proof-and-build machinery, not a second semver-stable API.

There is no supported Python daemon, socket, hook, or wheel-release lane in that public integration surface.

## Proof boundary

- This repo directly proves the TypeScript package surface, lifecycle mechanics, and operator-observability APIs.
- The broader comparative benchmark families live in the separate public proof repo: `https://github.com/jonathangu/brain-ground-zero`.
- The `docs/brains-dashboard/` route in this repo is a documentation placeholder, not a shipped dashboard application.

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
- `@openclawbrain/event-export` turns those events into deterministic export ranges and provenance.
- `@openclawbrain/workspace-metadata` and `@openclawbrain/provenance` stamp workspace and build provenance into immutable artifacts.
- `@openclawbrain/learner` assembles candidate packs with structural summaries and deterministic ids.
- `@openclawbrain/activation` stages, promotes, inspects, and rolls back activation state.
- `@openclawbrain/compiler` consumes coherent pack artifacts for runtime-side context selection.
- `@openclawbrain/openclaw` wraps the OpenClaw-owned runtime boundary: activation-aware compile diagnostics, learned-route hard-fail enforcement, prompt-context formatting, and normalized runtime event export handoff.

## Workspace

Requires Node 20+ and `pnpm` 10+.

```bash
corepack enable
pnpm install --frozen-lockfile
pnpm check
pnpm lifecycle:smoke
pnpm observability:smoke
pnpm release:pack
```

`pnpm check` builds the workspace, runs the package tests, and executes the root lifecycle and observability smoke lanes in this repo.

`pnpm lifecycle:smoke` rebuilds the workspace and runs the current lifecycle proof on a temp directory using the existing public package APIs on disk:

- `@openclawbrain/events` creates normalized interaction and feedback events
- `@openclawbrain/event-export` derives deterministic event export range and provenance
- `@openclawbrain/learner` materializes active and candidate packs from those exports
- `@openclawbrain/activation` stages and promotes the candidate pack into the active slot
- `@openclawbrain/compiler` compiles runtime context from the promoted pack

`pnpm observability:smoke` rebuilds the workspace and proves the operator-facing diagnostics contract on a temp directory using the same public APIs on disk:

- `@openclawbrain/activation.inspectActivationState()` proves active/candidate health and promotion or rollback readiness
- `@openclawbrain/activation.describeActivationTarget()` proves freshness through pack id, workspace snapshot, event range, export digest, and built-at timestamps
- `@openclawbrain/compiler.compileRuntimeFromActivation()` proves runtime fallback and selection state through stable compile diagnostics

`pnpm release:pack` creates package tarballs in `.release/` for the full public package surface.

For a release-candidate pass, run:

```bash
pnpm release:check
```

That command cleans the workspace, rebuilds it, reruns tests, and produces publishable tarballs for all public `@openclawbrain/*` packages.

## Docs

- `docs/openclaw-attach-quickstart.md`
- `docs/openclaw-integration.md`
- `docs/operator-observability.md`
- `docs/reproduce-eval.md`
- `docs/typescript-first-convergence.md`
- `docs/contracts-v1.md`
- `scripts/lifecycle-smoke.mjs`
- `scripts/observability-smoke.mjs`
- `docs/release.md`
- `contracts/README.md`
- `packages/contracts/README.md`
- `packages/compiler/README.md`
- `packages/learner/README.md`
- `packages/openclaw/README.md`
- `packages/pack-format/README.md`
