# Changelog

This changelog is intentionally conservative.

It should help an operator understand two different truths:
- what was actually published to npm
- what has landed on `main` since that publish

## Unreleased (current trunk after 0.3.2)

The current repo may move ahead of the published `0.3.2` package again. If it does, this section should describe that drift plainly instead of pretending the last npm release says everything.

At the moment, trunk is aligned with the published `0.3.2` package for the correction/routing release slice.

## 0.3.2

Published package: `@jonathangu/openclawbrain@0.3.2`

Git tag: `v0.3.2`

Deep release note:
- [`docs/release-notes-0.3.2.md`](docs/release-notes-0.3.2.md)

### Published notes

- ships the new **summary-aware routing prior** so LCM summaries act as a search/routing abstraction rather than the durable truth layer
- ships the new **explicit user-correction commit path**, including a real `BrainService.teachUserCorrection()` API
- adds both a **fast deterministic correction lane** and an **off-path async proposal lane**
- exposes assembled `summaryMetadata` so runtime policy can distinguish between summary-suffices, expand-to-source, and prefer-typed-memory situations
- adds polished architecture notes for both:
  - `docs/routing-prior.md`
  - `docs/corrections.md`
- restores repo-wide `tsc --noEmit` cleanliness by reconciling the stale type/test surface drift
- hardens the release path by adding a Brain DB `busy_timeout`, fixing the flaky `database is locked` failure exposed during release verification
- passes:
  - full test suite
  - `npm pack --dry-run`
  - repo-wide `tsc --noEmit`

### Why this release matters

0.3.2 is the point where the public npm package catches up to the repo’s new correction/routing architecture:
- summaries help the system decide where to look
- explicit typed correction memory helps the system decide what currently wins
- release/docs/package truth now reflects that split clearly

## 0.3.0

Published package: `@jonathangu/openclawbrain@0.3.0`

### Published notes

- `f1dfa5c`: catch up the release notes for work merged after `0.2.8`
- adds Anthropic OAuth setup-token support in the TUI
- resolves SecretRef-backed auth-profile credentials and provider-level custom provider configuration during summarization
- formats LCM tool timestamps in the local timezone instead of UTC

### Important historical note
These published notes are accurate for the package that went out, but they are now incomplete relative to the later `0.3.2` package and current repo state.
