# Worked Example: One OpenClaw Turn End to End

This page shows an illustrative but generic OpenClaw turn wired through the current OpenClawBrain package surface.

If you want the smallest runnable consumer proof instead of the architectural walkthrough below, use `examples/npm-consumer/README.md`.

## Boundary

- OpenClaw keeps the hot path, prompt assembly, response delivery, and fail-open serving.
- OpenClawBrain supplies normalized events, packs, activation helpers, deterministic compilation, and asynchronous learner refresh.

## Example turn shape

### 1) OpenClaw receives a user turn

OpenClaw receives a user message and keeps control of the hot path.

### 2) Compile from the promoted pack

OpenClaw resolves the active pack and compiles bounded context from it.

If the promoted manifest requires learned routing, the compile must use that pack's learned `route_fn` and expose `routerIdentity` in diagnostics.

Relevant package surface:

- `@openclawbrain/activation`
- `@openclawbrain/compiler`
- optional `@openclawbrain/openclaw`

### 3) Prompt assembly stays in OpenClaw

OpenClaw assembles the final prompt from the compiled context and serves the model call.

### 4) Delivery stays fail-open

OpenClaw sends the response and can keep serving even if learner refresh is stale or unavailable.

### 5) Normalized events are exported off the hot path

The turn is written into normalized interaction and feedback events for later learning.

Relevant package surface:

- `@openclawbrain/events`
- `@openclawbrain/event-export`
- `@openclawbrain/openclaw`

### 6) Learner and activation update asynchronously

OpenClawBrain materializes candidate packs, refreshes learned routing artifacts, and stages/promotes them behind the hot path.

Relevant package surface:

- `@openclawbrain/learner`
- `@openclawbrain/activation`
- `@openclawbrain/pack-format`

## Related docs

- [openclaw-integration.md](openclaw-integration.md)
- [reproduce-eval.md](reproduce-eval.md)
- [operator-observability.md](operator-observability.md)

## Claim boundary

This worked example proves the intended package boundary and learning-first operator story.
It does not by itself prove comparative benchmark performance, full benchmark coverage in this repo, or live production answer quality.
