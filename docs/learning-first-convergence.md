# Learning-First Convergence

This is the repo-wide convergence statement for OpenClawBrain's supported public shape today.

## Top invariant

The promoted pack is the only supported learning/serve boundary.

- normalized events and workspace state feed learner materialization
- learner emits immutable packs with graph, vector, provenance, and optional router artifacts
- activation chooses the active pack
- compiler serves only from that promoted pack
- when the manifest requires learned routing, the pack's learned `route_fn` must be the routing source of truth

The public package surface, fixtures, smoke lanes, and docs should all describe that same path.

## What converges here

The end state is one coherent learning-first flow:

1. normalize interactions and feedback into stable contracts
2. derive deterministic event-export ranges and provenance
3. materialize fast-boot and candidate packs from workspace plus event state
4. train or refresh learned routing artifacts as part of pack materialization
5. stage and promote only activation-ready packs
6. compile from the promoted pack with explicit route/fallback diagnostics

## Why `route_fn` sits at the center

OpenClawBrain is not just a prompt-truncation helper.
Its public artifact boundary includes learned routing as a first-class output:

- learned `route_fn` artifacts live in the pack manifest/runtime assets
- compile diagnostics must prove whether learned routing was actually used
- `routerIdentity` ties a served compile back to the promoted artifact
- fallback stays explicit instead of silently bypassing the learned route surface

This is the clearest public invariant for the repo: once a pack says learned routing is required, serving must prove it.

## Public surface

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
- `@openclawbrain/openclaw`

Versioned schemas and fixtures under `contracts/` are part of the same supported surface.

Everything else in the repo is public documentation, proof machinery, or release plumbing for that surface.

## Repo rules implied by convergence

- docs should describe the promoted-pack learning path, not stale side stories
- proofs should run through public packages on disk, not hidden hooks
- activation diagnostics should expose health, promotion safety, freshness, and learned-route evidence clearly
- fast time-to-first-value should win over blocking on full historical replay
- live events should be learned first while older history catches up passively in the background

## Proof lanes

The workspace already proves this convergence through two deterministic lanes:

- `pnpm lifecycle:smoke` validates the end-to-end path from normalized events to promoted-pack compilation
- `pnpm observability:smoke` validates activation health, freshness, learned `route_fn` evidence, and explicit fallback diagnostics

If a change weakens the promoted-pack + learned-route boundary, it is moving away from the intended public shape.
