# OpenClawBrain 0.4.19 / 0.4.6 split-package release notes

Published packages:

- `@openclawbrain/cli@0.4.19`
- plugin/runtime remains `@openclawbrain/openclaw@0.4.6`

## Why this release exists

This is the honest canonical follow-up release after the merged repo tranche on `main`.

It carries forward two operator-surface truths through the public split-package lane:

- large status/proof reads must stay bounded on oversized learning-spine logs
- proof health snapshots must read live runtime serve truth instead of stale legacy worker-only status

## What changed

### `@openclawbrain/cli@0.4.19`

- `openclawbrain status` and related proof/status surfaces keep oversized learning-spine reads bounded
- traced-learning status payloads stay JSON-serializable instead of recursively inflating bridged provenance
- proof-cron health snapshots now probe the operator/runtime status surface and report live serve-state truth
- health snapshots now surface:
  - runtime healthy
  - serve state
  - active pack
  - learned-route active
  - load proof

### `@openclawbrain/openclaw@0.4.6`

- no runtime/plugin version bump in this release
- the shipped changes are in the operator CLI surface, not the plugin payload

## Verification

- `npm exec -- tsc -p tsconfig.json --noEmit` passed
- `npm test` passed (`54` files / `472` tests)
- `npm run release:plan -- --json` passed on `main`
- `npm run release:verify` passed

## Upgrade

```bash
npm install -g @openclawbrain/cli@0.4.19
openclawbrain status --openclaw-home ~/.openclaw --detailed
```

If you are already on the canonical split-package lane, you do **not** need a new plugin/runtime install for this specific release.
