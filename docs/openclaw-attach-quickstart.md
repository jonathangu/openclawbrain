# OpenClaw Attach Quickstart

This is the operator-facing setup contract for attaching OpenClawBrain to OpenClaw.

The goal is not “perfect historical completeness before first use.”
The goal is **fast time-to-first-value**:

- attach the brain quickly
- turn it on quickly
- get useful context gains immediately
- let deeper learning continue in the background without blocking runtime

The first attach must follow four rules:

- **no full history replay gate before first value**
- **live events are learned first once attach is on**
- **passive background learning stays on continuously**
- **diagnostics prove health, promotions, freshness, and fallback**

## What the attach path must guarantee

1. **Fast startup wins over full replay**
   - A new operator should not need to wait for a full history scan before the brain becomes useful.
   - Initial activation should be able to start from current workspace state, recent normalized events, and whatever prior artifacts already exist.

2. **Background history learning is non-blocking**
   - Historical replay/backfill should start after attach and continue passively.
   - Old history should improve the brain over time, but it must not delay initial activation.

3. **Ongoing event harvesting is always on**
   - New runtime interactions, human feedback, self-feedback, and harvested/derived supervision should keep flowing into normalized event exports continuously.
   - The brain should prioritize fresh live events while historical backfill catches up behind the scenes.

4. **OpenClaw remains the runtime owner**
   - OpenClaw owns sessions, channels, diagnostics, fail-open behavior, and prompt assembly.
   - OpenClawBrain stays on the memory/compiler/learning side of the boundary.

5. **Fail open, never block messaging**
   - If the brain is cold, missing, or mid-refresh, OpenClaw must still answer with its standard runtime path.

6. **Diagnostics are part of the product contract**
   - Operators must be able to prove activation health, promotion safety, freshness, and fallback from public APIs.
   - The repo-level proof lane for that contract is `pnpm observability:smoke`.

## Install and prove

This public repo is the current TypeScript package surface and proof harness for OpenClaw attach/install work.

```bash
corepack enable
pnpm install --frozen-lockfile
pnpm check
pnpm lifecycle:smoke
pnpm observability:smoke
```

For a package-first attach path inside OpenClaw, the narrow install lane is:

```bash
pnpm add @openclawbrain/contracts @openclawbrain/events @openclawbrain/event-export @openclawbrain/learner @openclawbrain/activation @openclawbrain/compiler
```

Add `@openclawbrain/openclaw` when you are wiring the OpenClaw-owned runtime integration layer itself; the narrow attach lane above remains the minimal install surface.

That package lane is the supported public attach surface. The workspace root, smoke lanes, and docs in this repo exist to build, prove, and release that lane; they are not a separate runtime product.

That install set maps to the attach flow like this:

- `@openclawbrain/events` and `@openclawbrain/event-export` normalize and bridge the live/backfill learning surface
- `@openclawbrain/learner` materializes fast-boot and fresher candidate packs
- `@openclawbrain/activation` stages, promotes, and inspects runtime slots
- `@openclawbrain/compiler` compiles from the promoted pack and emits operator-visible diagnostics

`pnpm lifecycle:smoke` proves that the attach path can materialize, stage, promote, and compile from a useful pack immediately.

`pnpm observability:smoke` proves that the operator diagnostics surface can answer four questions without private runtime plumbing:

- is the active slot healthy now?
- can the candidate be promoted safely now?
- what exact snapshot/export/build is runtime serving now?
- did runtime serve token-matched context or deterministic fallback now?

## Operator expectation

The setup experience should feel like this:

1. Install the OpenClawBrain packages or attach the OpenClaw integration.
2. Point OpenClawBrain at the existing OpenClaw event/workspace surface.
3. Materialize a fast-boot pack immediately from current state.
4. Promote that pack so runtime compilation can start right away.
5. Keep passive background learning running on:
   - historical replay/backfill
   - ongoing runtime events
   - human corrections/teachings
   - self-detected failures/successes
   - harvested/derived supervision

## Product behavior after attach

Once attached, the default lifecycle should be:

1. **Attach** OpenClawBrain to OpenClaw.
2. **Bootstrap fast** from current state and recent events.
3. **Serve immediately** with deterministic pack-backed compilation.
4. **Replay older history in the background** without blocking runtime.
5. **Continuously harvest ongoing events** from the live OpenClaw stream.
6. **Continuously materialize better candidate packs** from the growing event/export surface.
7. **Promote safely** when activation-ready artifacts are available.
8. **Inspect continuously** so health, freshness, and fallback stay visible to operators.

## Non-goals for first setup

The first setup should **not** require:

- a full archive replay before first use
- a multi-hour history import as a hard gate
- separate competing runtime services on the production host
- manual operator babysitting just to keep passive learning alive

## Current repo stance

Today this repository exposes the documented package-first attach surface used here: contracts, normalized events, event export, learner, activation, compiler, and the optional `@openclawbrain/openclaw` bridge package.

The concrete proof path in the repo today is the lifecycle plus observability smoke:

```bash
pnpm check
pnpm lifecycle:smoke
pnpm observability:smoke
```

Those proofs cover:

- normalized events
- deterministic event export
- learner pack materialization
- activation staging/promotion
- runtime compilation from the promoted pack
- activation health and promotion readiness inspection
- freshness inspection over workspace snapshot, event range, export digest, and build time
- deterministic fallback diagnostics when token matching does not hit

Any future tighter single-path OpenClaw attach/install UX should preserve the same invariants while keeping the supported package surface narrow.

The detailed diagnostics contract lives in [`docs/operator-observability.md`](docs/operator-observability.md).
The repo-wide convergence statement lives in [`docs/typescript-first-convergence.md`](docs/typescript-first-convergence.md).

## Required doc promise

Public docs and install guidance should keep repeating these points clearly:

- **do not wait for a full history scan to get started**
- **turn the brain on quickly and let it learn in the background**
- **passive history learning should stay on continuously**
- **real-time event scanning and harvesting should stay on continuously**
- **fresh live events should be learned first while old history catches up**
- **diagnostics should prove health, promotions, freshness, and fallback**
