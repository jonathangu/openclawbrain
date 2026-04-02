# OpenClawBrain 0.3.8 release notes

Published package: `@jonathangu/openclawbrain@0.3.8`

This is a post-split compatibility release for older combined-package installs. The canonical public install story remains the split package lane (`@openclawbrain/openclaw` + `@openclawbrain/cli`), but this release keeps the combined package truthful after the large replay/proof/eval tranche landed on `main`.

## What changed

This compatibility release carries the repo-side runtime and proof work that accumulated after `0.3.7`, including:

- stronger teacher-truth and attribution visibility on the combined-package live path
- deeper bounded-anytime runtime truth around interruption, clipping, fit/drop accounting, and fail-open behavior
- deterministic recorded-session replay proof bundle generation
- a canonical frozen 20-trace replay manifest and eval lane
- explicit replay/eval proof surfaces for:
  - `no_brain`
  - `vector_only`
  - `graph_prior_only`
  - `learned_route`
- a calibrated frozen-eval economics gate that keeps the prompt-cost proxy visible but does not treat per-call cheapness as the default blocker
- stronger operator proof/reporting surfaces and compatibility fixes accumulated in the combined-package code path since `0.3.7`

## Why it matters

Before this patch, the combined compatibility package was missing the newer replay/proof/eval truth surfaces and the long-run-economics framing that now exists in the repo.

`0.3.8` makes the legacy combined-package path more honest for maintainers and older installs:
- replay proof bundles are reproducible
- the frozen replay gate is explicit and deterministic
- quality remains hard-gated
- prompt-cost proxy remains reported
- the gate no longer overclaims that `learned_route` must be cheaper on every individual call

## Important boundary

This release does **not** change the canonical new-user split-package versions.

The truthful split-package public lane remains:
- `@openclawbrain/openclaw@0.4.6`
- `@openclawbrain/cli@0.4.20`

Those package payloads did not change in this tranche, so bumping them here would have been version churn without new package code.

Also preserved explicitly:
- the frozen replay set is equivalent-only, not verified first-party real production traces
- the replay gate still does not directly prove long-run task-level economics or raw LLM/API call reduction by itself
- the prompt-cost figure remains an observational proxy unless an explicit threshold is configured

## Verification summary

Executed from the repo before publish:

- `npm run release:verify`
- `npx vitest run test/frozen-recorded-session-eval-gate.test.ts test/frozen-recorded-session-eval-gate-economics.test.ts`
- `npm run proof:frozen-eval-gate -- --output-dir /tmp/ocb-release-0.3.8-frozen-gate`

Expected truthful outcome before publish:
- root compatibility package verification passes
- frozen recorded-session eval gate passes on canonical truth
- split-package tarball verification still passes as part of root release verification

## Release boundary

This release updates the **combined compatibility package** only.

It is for older combined-package installs and maintainer workflows that still depend on `@jonathangu/openclawbrain`. The canonical operator/public install story remains the split package lane.
