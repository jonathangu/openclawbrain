# OpenClawBrain 0.4.45

Canonical install lane:

```bash
openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
openclawbrain status --openclaw-home ~/.openclaw --detailed
openclawbrain proof --openclaw-home ~/.openclaw
```

Internal published packages:

- `@openclawbrain/openclaw@0.4.45`
- `@openclawbrain/cli@0.4.45`

## Why this release exists

`0.4.45` exists to ship the operator-surface cleanup that landed after `0.4.44`.

The important change is not a new broad capability. It is that the plain human `openclawbrain status` path now behaves more like an operator expects on a real host: fast local truth by default, deeper probing only when explicitly requested.

## What changed

- stops plain summary status from eagerly loading detailed-only teacher snapshot and event-export supervision surfaces
- keeps summary status on cheap local truth when reporting graph materialization and passive-learning summaries
- skips active-pack embedding inspection and synchronous Ollama probing on plain status, while preserving the richer `--detailed` lane for live proof
- keeps the lighter summary-path selection internal so the public operator API stays narrow and stable

## Operator truth

This release does **not** change the supported front door.

The public lane is still:

- run `openclawbrain install --openclaw-home ...`
- restart the gateway
- verify with `status --detailed`
- capture durable evidence with `proof`

What changed is the operator experience around routine checks:

- plain `status` is now lighter and less noisy
- `status --detailed` remains the place for heavyweight local proof

## Honest boundary

This release improves the status hot path and operator trust surfaces.
It does **not** claim:

- new learning capability
- new proof breadth
- live teacher mutation

## What success looks like

After upgrading, a healthy host should still converge through the same no-drama install / restart / detailed-status / proof lane.

The difference is that ordinary summary status should no longer spend time on heavyweight detailed-only surfaces unless you explicitly ask for them.

## Focused verification

- `node --test packages/cli/dist/test/operator-status-regressions.test.js packages/cli/dist/test/status-single-snapshot.test.js packages/cli/dist/test/teacher-status-truth.test.js`
- `npm run release:plan`
- `npm run release:verify`

## Upgrade

```bash
openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
openclawbrain status --openclaw-home ~/.openclaw --detailed
openclawbrain proof --openclaw-home ~/.openclaw
```

This should behave like the same stable OCB lane as before, but with a lighter plain-status operator path in the shipped package surface.
