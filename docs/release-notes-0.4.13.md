# OpenClawBrain 0.4.13 / 0.4.2 split-package release notes

Published packages:

- `@openclawbrain/compiler@0.3.5`
- `@openclawbrain/openclaw@0.4.2`
- `@openclawbrain/cli@0.4.13`

This release carries the learned `STOP_LOCAL` fix through the canonical split-package install lane.

## Why this release exists

Earlier on 2026-03-24, the compatibility package `@jonathangu/openclawbrain@0.3.6` shipped the unified learned local branching fix for older combined-package installs. That closed the architecture seam for the legacy package, but it did **not** update the live split-package dogfood surface used by the local host.

This release closes that gap on the canonical public lane:

```bash
openclaw plugins install @openclawbrain/openclaw
npx @openclawbrain/cli install --openclaw-home ~/.openclaw
openclaw gateway restart
npx @openclawbrain/cli status --openclaw-home ~/.openclaw --detailed
npx @openclawbrain/cli proof --openclaw-home ~/.openclaw --skip-install --skip-restart
```

## What changed

### `@openclawbrain/compiler@0.3.5`
- graph-walk runtime compilation now honors learned source-specific `STOP_LOCAL` updates
- when a learned stop action outranks the next traversal candidate for a source block, graph-walk expansion stops there instead of always consuming the next edge

### `@openclawbrain/cli@0.4.13`
- native V2 route updates now emit source-specific learned `STOP_LOCAL` updates keyed by source block
- `STOP_LOCAL` is no longer treated as an always-virtual no-op during trajectory policy-gradient aggregation

### `@openclawbrain/openclaw@0.4.2`
- runtime package now pins the STOP-aware compiler patch so plugin/runtime installs pick up the learned stop behavior truthfully

## Proof coverage

Focused proof added in-repo:

- `packages/cli/dist/test/stop-local-policy-update.test.js`
  - proves native V2 learning emits a nonzero source-specific `STOP_LOCAL` update
- `packages/openclaw/dist/test/graph-walk-stop-policy.test.js`
  - proves graph-walk expansion halts when learned `STOP_LOCAL` outranks traversal

## Verification summary

Executed from the repo before publish:

- `node --test packages/cli/dist/test/native-pg-v2-route-update.test.js packages/cli/dist/test/stop-local-policy-update.test.js packages/openclaw/dist/test/graph-walk-stop-policy.test.js`
  - passed (`4` tests)
- `npm --prefix packages/openclaw pack --dry-run`
  - passed
- `npm --prefix packages/cli pack --dry-run`
  - passed

After publish, the intended dogfood lane is an in-place local upgrade on the live host, not a destructive reinstall.
