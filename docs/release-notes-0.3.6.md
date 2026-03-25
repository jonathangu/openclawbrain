# OpenClawBrain 0.3.6 release notes

Published package: `@jonathangu/openclawbrain@0.3.6`

This is a post-split compatibility release for older combined-package installs. The canonical public install story remains the split package lane (`@openclawbrain/openclaw` + `@openclawbrain/cli`), but this patch keeps the legacy combined package truthful for hosts that still depend on it.

## What changed

- Promoted the local branching policy to one unified learned action set:
  - `{ traverse(edge_1), traverse(edge_2), ..., STOP_LOCAL }`
- `STOP_LOCAL` now has a real learned per-source parameter surface instead of a trace-only seam.
- REINFORCE now updates chosen `STOP_LOCAL` actions truthfully from expansion/substep trajectories.
- Forced `STOP_LOCAL` actions with probability `1.0` still emit no learned update, which preserves truthful learning semantics under hard caps.
- Learned stop-local weights now persist through:
  - SQLite state
  - pack snapshots
  - runtime graph reloads
  - CLI load/promote hydration paths
- The release also carries a narrow CLI hydration fix so manual CLI load/promote flows do not silently drop learned seed or stop-local weights.

## Why it matters

Before this patch, the combined-package traversal scaffold could record truthful `stop_local` choices but still failed to learn them as first-class local policy actions. That left a real seam between the intended branching policy and the shipped compatibility package.

`0.3.6` closes that seam without reopening frontier learning: local branching is learned, while frontier scheduling stays deterministic/FIFO for now.

## Verification summary

Executed from the repo before publish:

- `npm run release:verify`
  - passed
- root Vitest suite:
  - `43` files passed
  - `384` tests passed
- split-package tarball verification also passed as part of release verify:
  - `@openclawbrain/openclaw@0.4.1`
  - `@openclawbrain/cli@0.4.12`

Focused proof inside the task closeout also passed:

- `./node_modules/.bin/vitest run test/brain-core/policy.test.ts test/brain-core/update.test.ts test/brain-runtime/graph-io.test.ts test/brain-worker/worker.test.ts test/brain-runtime/promotion-story.test.ts`
  - `5` files / `34` tests passed
- `./node_modules/.bin/tsc --noEmit`
  - passed
