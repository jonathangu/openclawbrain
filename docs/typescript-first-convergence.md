# TypeScript-First Convergence

This is the repo-wide convergence statement for OpenClawBrain's supported end state.

The project is converging on one narrow, public, package-first surface:

- TypeScript packages under `packages/` are the supported implementation boundary
- immutable pack artifacts are the runtime and release boundary
- OpenClaw remains the runtime owner for sessions, prompts, fail-open behavior, and operator UX
- OpenClawBrain owns contracts, normalized event flows, learner materialization, activation safety, and deterministic compilation

## What converges here

The end state is not a grab bag of side paths.
It is one coherent flow:

1. normalize runtime interactions and feedback into stable contracts
2. derive deterministic event-export ranges and provenance
3. materialize immutable fast-boot and candidate packs from workspace plus event state
4. stage and promote only activation-ready packs
5. compile runtime context from the promoted pack with explicit diagnostics

That single path is what the workspace tests, smoke lanes, docs, and publishable packages should describe.

## Structural graph learning stays first-class

OpenClawBrain is not converging toward a prompt-only truncation layer.
The structural graph remains part of the product surface:

- structural operations like `split`, `merge`, `prune`, and `connect` stay explicit in pack provenance
- Hebbian reinforcement and decay stay attached to pack blocks instead of disappearing into ad hoc runtime state
- learner-produced structural summaries make large normalized event exports serveable without abandoning determinism
- native compaction remains a pack-native operation rather than a hidden prompt-side fallback

This is why the pack graph, vector payloads, manifest provenance, and compiler compaction path all stay public and testable.

## Package-first public surface

The supported public packages are:

- `@openclawbrain/contracts`
- `@openclawbrain/events`
- `@openclawbrain/event-export`
- `@openclawbrain/workspace-metadata`
- `@openclawbrain/provenance`
- `@openclawbrain/pack-format`
- `@openclawbrain/activation`
- `@openclawbrain/compiler`
- `@openclawbrain/learner`

Everything else in the repo exists to document, test, or release that surface.

## Repo rules implied by convergence

The converged path carries a few rules that should keep showing up everywhere:

- docs should point at current repo artifacts, not stale private plans or roadmap notes
- proofs should run through the public packages on disk, not hidden internal hooks
- activation diagnostics should expose health, promotion safety, freshness, and fallback clearly
- fast time-to-first-value should win over blocking on full historical replay
- passive background learning should continue improving candidate packs after attach

## Proof lanes

The workspace already proves this convergence through two deterministic lanes:

- `pnpm lifecycle:smoke` validates the end-to-end pack lifecycle from normalized events to promoted compilation
- `pnpm observability:smoke` validates the operator contract for activation health, freshness, and compile fallback diagnostics

If a change does not fit those lanes or weakens the package-first boundary, it is probably moving away from the intended end state.
