# OpenClawBrain 0.4.17 / 0.4.6 split-package release notes

Published packages:

- `@openclawbrain/openclaw@0.4.6`
- `@openclawbrain/cli@0.4.17`

This release carries the newest bounded-anytime serving truth into the canonical split-package lane after the repo-root engine tranche landed on `main`.

## Why this release exists

The repo already had the new engine behavior:
- stronger provenance / attribution
- better learned-route quality
- operator-visible feedback / coverage truth
- bounded-anytime serving interruption accounting

But the canonical split packages still needed to expose the bounded-serving truth through their own published runtime and CLI surfaces. This release closes that gap so the public package lane stays aligned with the real runtime behavior.

## What changed

### `@openclawbrain/openclaw@0.4.6`

- packaged runtime context now forwards bounded-anytime interruption truth when present
- packaged serve-time decision logs now persist explicit interruption fields:
  - `queryInterrupted`
  - `interruptionStage`
  - `interruptionReason`
  - `servedPartial`
  - `interruptionAccounting`
- package-local runtime-budget / bounded-reader tests were updated to pin that behavior

### `@openclawbrain/cli@0.4.17`

- traced-learning status bridge now derives a compact interruption summary from `brain_training_state.last_assembly_decision_json`
- persisted traced-learning bridge state no longer hides newer interruption truth from the latest assembly decision
- CLI status/detail surfaces now expose a compact summary of:
  - interruption reason / stage
  - partial vs empty serve
  - dropped frontier count
  - dropped proposal count
  - budget utilization
- focused package regression tests were added for the new status-bridge truth

## Verification summary

Executed before publish:

```bash
npm run release:prepare:packages
npm run release:verify
node --test packages/openclaw/dist/test/plugin-package-surface.test.js packages/openclaw/dist/test/runtime-budget-forwarding.test.js packages/openclaw/dist/test/bounded-jsonl-reader.test.js packages/openclaw/dist/test/teacher-status-truth.test.js packages/openclaw/dist/test/teacher-decision-match.test.js
node --test packages/cli/dist/test/traced-learning-bridge.test.js
node scripts/verify-openclaw-package-tarball.mjs
node scripts/verify-openclaw-cli-package-tarball.mjs
```

Key results:

- root verification passed: `47` test files / `454` tests
- proof smoke gate passed
- `@openclawbrain/openclaw@0.4.6` package verification passed
- `@openclawbrain/cli@0.4.17` package verification passed
- full split-package `release:verify` passed after hydrating package-local dependencies

## Operator truth after publish

Canonical lane:

```bash
openclaw plugins install @openclawbrain/openclaw
npx @openclawbrain/cli install --openclaw-home ~/.openclaw
openclaw gateway restart
npx @openclawbrain/cli status --openclaw-home ~/.openclaw --detailed
```

After this release, the canonical split-package lane truthfully carries the latest bounded-anytime interruption surface instead of only exposing it in the repo-root runtime path.
