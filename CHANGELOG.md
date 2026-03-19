# Changelog

This changelog is intentionally conservative.

It should help an operator understand two different truths:
- what was actually published to npm
- what has landed on `main` since that publish

## Unreleased (current trunk after 0.4.0)

The current repo may move ahead of the published `0.4.0` split packages again. If it does, this section should describe that drift plainly instead of pretending the last npm release says everything.

At the moment, trunk is intended to align with the published `0.4.0` split-package release and the now-proven public-registry operator flow.

## 0.4.0

Published packages:
- `@openclawbrain/openclaw@0.4.0`
- `@openclawbrain/cli@0.4.0`

Split landing commit on `main`: `b3ada81`

Deep release note:
- [`docs/release-notes-0.4.0.md`](docs/release-notes-0.4.0.md)

### Published notes

- publishes the package split as real public surface rather than staged repo work:
  - `@openclawbrain/openclaw@0.4.0` is the plugin/runtime payload
  - `@openclawbrain/cli@0.4.0` is the operator CLI
- makes the native plugin install plus CLI attach flow the canonical public lane:
  - `openclaw plugins install @openclawbrain/openclaw@0.4.0`
  - `npx @openclawbrain/cli@0.4.0 openclawbrain install --openclaw-home ~/.openclaw`
  - `openclaw gateway restart`
  - `npx @openclawbrain/cli@0.4.0 openclawbrain status --openclaw-home ~/.openclaw --detailed`
- records that the exact public-registry flow already passed on the real host `redogfood`
- keeps the remaining host/plugin warning visible: some hosts still report a plugin id mismatch because the manifest uses `openclawbrain` while the package/entry hint uses `openclaw`; the warning is currently cosmetic, not evidence that install failed
- leaves `@jonathangu/openclawbrain@0.3.5` in place as a compatibility holdover for older installs rather than the primary operator story

### Why this release matters

0.4.0 is the point where the split package story becomes the honest public story.
Outside operators can now follow a single public-registry flow that has already passed on a real host, while the docs stay explicit about the one remaining host/plugin warning instead of pretending the seam is cleaner than it is.

## 0.3.5

Published package: `@jonathangu/openclawbrain@0.3.5`

Git tag: `v0.3.5`

Deep release note:
- [`docs/release-notes-0.3.5.md`](docs/release-notes-0.3.5.md)

### Published notes

- hardens the prompt-assembly compatibility bridge so usable `event.prompt` text still flows when `before_prompt_build` arrives with an empty or non-text message envelope
- adds focused teacher-status truth coverage so a fresh watch heartbeat with `no_teacher_artifacts` does not get mislabeled as stale/unhealthy
- recovers the `packages/openclaw` front-door package tree into the public repo so the shipped install surface matches the package that owns the installed runtime guard
- preserves the single extra-LLM design: local Ollama teacher remains `qwen3.5:9b`; no extra model roles were added
- passes:
  - focused regression coverage for prompt fallback and teacher-status truth
  - live host verification after reinstall/relink, including clean runtime-guard prompt probes and `teacher healthy=yes stale=no`

### Why this release matters

0.3.5 turns a local runtime repair into a truthful public ship.
The install surface that actually owns the generated runtime hook is now present in the public repo, the hook handles prompt-envelope edge cases more gracefully, and teacher health reporting is more honest when the latest cycle is a genuine no-op.

## 0.3.4

Published package: `@jonathangu/openclawbrain@0.3.4`

Git tag: `v0.3.4`

Deep release note:
- [`docs/release-notes-0.3.4.md`](docs/release-notes-0.3.4.md)

### Published notes

- stops heartbeat prompts, startup/reset scaffolding, and metadata wrapper text from being misclassified as human supervision evidence
- adds a dedicated system-message filter at the evidence-detection boundary
- preserves genuine human correction and teaching signals while excluding operational scaffolding
- adds focused unit and integration tests for both exclusion and inclusion behavior
- passes:
  - full test suite
  - `npm pack --dry-run`
  - explicit exclusion/inclusion regression coverage for teacher-pollution cases

### Why this release matters

0.3.4 closes a real learning-integrity bug.
The system should learn from actual human supervision, not from runtime scaffolding that happens to contain imperative language.
This release makes the passive-learning boundary more honest and reduces fake supervision entering the route-learning substrate.

## 0.3.3

Published package: `@jonathangu/openclawbrain@0.3.3`

Git tag: `v0.3.3`

Deep release note:
- [`docs/release-notes-0.3.3.md`](docs/release-notes-0.3.3.md)

### Published notes

- fixes the launchd-served child-worker boot regression where Node resolved `tsx/esm` from `/` and crash-looped operator installs
- resolves the child worker loader to an absolute `file://` import and launches the child from the plugin root so module resolution no longer depends on service cwd
- restores truthful `brainWorkerMode=child` operation on Eagle and other launchd-style operator installs using linked local plugin paths
- adds a focused runtime test that reproduces the exact `cwd=/` worker-launch seam and proves the child boots without `Cannot find package 'tsx'` failures
- passes:
  - Eagle live child-worker validation (`workerHealthy=true` after restart on the real Eagle profile)
  - full test suite
  - `npm pack --dry-run`
  - repo-wide release verification via `npm run release:verify`

### Why this release matters

0.3.3 closes a real operator-facing reliability bug in the supervised child-worker boundary.
The plugin already worked in local/dev contexts, but launchd-served installs could silently fall back to a crash loop because the worker loader depended on cwd-sensitive `tsx` resolution.
This release makes the child-worker launch path deterministic again, which is the honest production boundary for OpenClawBrain.

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
