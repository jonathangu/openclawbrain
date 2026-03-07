# Worked Example: One OpenClaw Turn End to End

This page shows an illustrative but generic OpenClaw turn wired through the current OpenClawBrain package surface.

It is an operator example, not a benchmark, a dashboard demo, or a captured production trace.

## Stack boundary

- OpenClaw owns the live runtime, fail-open behavior, prompt assembly, and response delivery.
- OpenClawBrain provides typed contracts, event normalization, pack artifacts, activation helpers, deterministic compilation, and learner-side updates.

## Example turn shape

### 1) Runtime receives a user turn
OpenClaw receives a user message and keeps control of the hot path.

### 2) Compile from promoted pack
OpenClaw resolves the active pack and compiles bounded context from it.

Relevant package surface:
- `@openclawbrain/activation`
- `@openclawbrain/compiler`
- optional `@openclawbrain/openclaw`

### 3) Prompt assembly stays in OpenClaw
OpenClaw assembles the final prompt from the compiled context and serves the model call.

### 4) Delivery happens in OpenClaw
OpenClaw sends the response and remains fail-open if the brain side is stale or unavailable.

### 5) Normalized events are exported off the hot path
The turn can be written into normalized interaction/feedback events for later learning.

Relevant package surface:
- `@openclawbrain/events`
- `@openclawbrain/event-export`
- `@openclawbrain/openclaw`

### 6) Learner and activation update asynchronously
OpenClawBrain materializes candidate packs, evaluates them, and stages/promotes them behind the runtime boundary.

Relevant package surface:
- `@openclawbrain/learner`
- `@openclawbrain/activation`
- `@openclawbrain/pack-format`

## Related docs
- [openclaw-integration.md](openclaw-integration.md)
- [reproduce-eval.md](reproduce-eval.md)
- [operator-observability.md](operator-observability.md)

## Claim boundary

This worked example proves the intended package/runtime boundary and the operator story.
It does not by itself prove comparative benchmark performance, full benchmark coverage in this repo, or live production answer quality.
