# OpenClawBrain 0.4.20 / 0.4.6 split-package release notes

Published packages:

- `@openclawbrain/cli@0.4.20`
- plugin/runtime remains `@openclawbrain/openclaw@0.4.6`

## Why this release exists

This is the honest public follow-up after the savings-instrumentation + truth-sync + teacher/watch hardening swarm landed on `main`.

It improves three user-facing things without changing the plugin/runtime package version:

- better public truth surfaces about what OpenClawBrain is already proving vs what is still only proxied
- better cheap proof metrics for prompt/context, hop/correction, and estimated cost rollups
- a real compatibility hardening fix in the local session-tail path for legacy `custom_message` records

## What changed

### `@openclawbrain/cli@0.4.20`

- proof surfaces now expose deterministic prompt-side savings proxies from replay truth:
  - selected context chars
  - selected context blocks
  - estimated prompt tokens
- proof surfaces now expose deterministic hop/correction proxy metrics from replay/trace truth:
  - retrieval/tool-hop proxy from selection-digest counts
  - non-approval feedback proxy for correction recurrence
- proof surfaces now support a versioned estimated-cost path:
  - checked-in pricing table at `scripts/pricing-table.v1.json`
  - estimated prompt / completion / total USD rollups when the needed signals exist
- public truth wording is tighter and more honest:
  - replay is already better than `no_brain` on checked real traces
  - replay is still small and mixed
  - `learned_route` has real wins but is not the universal winner
  - direct spend savings are not yet proven
  - the hot path stays bounded and does not call a live LLM on every traversal hop
- session-tail parsing now accepts legacy `custom_message` records instead of treating them as unknown and skipping the path

### `@openclawbrain/openclaw@0.4.6`

- no runtime/plugin version bump in this release
- the published package delta is in the operator CLI surface, proof rollups, and session-tail compatibility logic carried by the CLI package

## Important caveats

- completion-side production counts remain `null` until the replay surfaces expose completion text
- hop and correction metrics are still deterministic proxies, not perfect literal counters
- lower direct spend is still proxied/mechanistic, not broadly proven

## Verification

- `npm run release:plan -- --json` passed on `main`
- `npm run release:verify` passed
- `npm test -- --run test/proof-cron.test.ts` passed
- targeted session-store/session-tail regression tests passed in both `packages/cli` and `packages/openclaw`
- `git diff --check` passed for the public repo/site truth-sync edits

## Upgrade

```bash
npm install -g @openclawbrain/cli@0.4.20
openclawbrain status --openclaw-home ~/.openclaw --detailed
```

If you are already on the canonical split-package lane, you do **not** need a plugin/runtime reinstall for this specific release.
