# OpenClaw Attach Quickstart

This is the operator-facing setup contract for attaching OpenClawBrain to OpenClaw.

The goal is not “perfect historical completeness before first use.”
The goal is **fast time-to-first-value**:

- attach the brain quickly
- turn it on quickly
- get useful context gains immediately
- let deeper learning continue in the background without blocking runtime

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

## Non-goals for first setup

The first setup should **not** require:

- a full archive replay before first use
- a multi-hour history import as a hard gate
- separate competing runtime services on the production host
- manual operator babysitting just to keep passive learning alive

## Current repo stance

Today this repository provides the TypeScript-first contracts, normalized events, event export, learner, activation, and compiler surface that the attach flow will use.

The concrete proof path in the repo today is the lifecycle smoke:

```bash
pnpm check
pnpm lifecycle:smoke
```

That smoke proves the core artifact lifecycle:

- normalized events
- deterministic event export
- learner pack materialization
- activation staging/promotion
- runtime compilation from the promoted pack

The eventual single-path OpenClaw attach/install flow should preserve the same invariants while optimizing for immediate usefulness and always-on passive learning.

## Required doc promise

Public docs and install guidance should keep repeating these points clearly:

- **do not wait for a full history scan to get started**
- **turn the brain on quickly and let it learn in the background**
- **passive history learning should stay on continuously**
- **real-time event scanning and harvesting should stay on continuously**
- **fresh live events should be learned first while old history catches up**
