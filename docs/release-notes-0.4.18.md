# OpenClawBrain 0.4.18 / 0.4.6 split-package release notes

Published packages:

- `@openclawbrain/cli@0.4.18`
- plugin/runtime remains `@openclawbrain/openclaw@0.4.6`

## Why this release exists

This is a focused operator-surface fix for a real published-install failure:

- `openclawbrain status` could crash or OOM when it whole-read an oversized learning-spine JSONL after an otherwise successful install/update

The fix changes status-oriented learning-spine reads to bounded tail reads instead of unconditional whole-file reads.

## What changed

### `@openclawbrain/cli@0.4.18`

- `openclawbrain status` now uses bounded/tail reads for oversized serve-time learning-spine logs
- route freshness status/proof surfaces stop whole-reading large learning-spine logs
- last-learning-update status surfaces stop whole-reading large learning-spine logs
- added regression coverage for large-log status behavior

### `@openclawbrain/openclaw@0.4.6`

- no plugin/runtime version bump in this release
- the bugfix is in the CLI/operator package, not the runtime payload

## Verification

- focused regression tests passed for bounded large-log status behavior
- `npm --prefix packages/cli run release:verify` passed
- live published-CLI verification succeeded on the reference host with:

```bash
openclawbrain status --openclaw-home ~/.openclaw --detailed
```

## Upgrade

```bash
npm install -g @openclawbrain/cli@0.4.18
openclawbrain status --openclaw-home ~/.openclaw --detailed
```

If you are already on the canonical split-package lane, you do **not** need a new plugin/runtime install for this specific fix.
