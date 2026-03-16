# Changelog

This changelog is intentionally conservative.

It should help an operator understand two different truths:
- what was actually published to npm
- what has landed on `main` since that publish

## Unreleased (current trunk after 0.3.0)

The current repo is materially ahead of the published `0.3.0` package.

### Runtime / proof / operator work landed on trunk
- hook-based compatibility fallback for hosts where `api.registerContextEngine` is gone
- sterile harness cleanup so the dead `plugins.slots.contextEngine` seam is no longer treated as the stable install boundary
- deterministic session-bound `brain_teach` proof
- deterministic runtime proof for immediate teach retrieval and serve-from-last-promoted-pack after worker failure
- child-worker supervision hardening: explicit protocol messages, restart accounting, reload acknowledgements, stale-lease handling, and stronger status/doctor truth
- structured evidence pipeline improvements: richer raw evidence metadata, multi-signal preservation, and stronger scanner/self-evidence handling
- docs truth refresh across the front door, proof ladder, configuration guide, maintainer/reference docs, and TUI/reference surfaces

### Still not frozen on trunk
- the full sterile host-surface harness is still not frozen end to end because it currently stalls during `openclawbrain init`
- bundle-level mutation replay is still not the final contract
- CI/release gates are still looser than the intended operator-grade standard
- full-repo `npx tsc --noEmit` is still not green

### Why this section exists
The published `0.3.0` release notes do not reflect the full Phase 1 / 2 / 3 hardening now present on `main`. This section keeps the release narrative honest until a later tagged release catches up.

## 0.3.0

Published package: `@jonathangu/openclawbrain@0.3.0`

### Published notes

- `f1dfa5c`: catch up the release notes for work merged after `0.2.8`
- adds Anthropic OAuth setup-token support in the TUI
- resolves SecretRef-backed auth-profile credentials and provider-level custom provider configuration during summarization
- formats LCM tool timestamps in the local timezone instead of UTC

### Important historical note
These published notes are accurate for the package that went out, but they are now incomplete relative to the current trunk. Read the **Unreleased** section above for the present repo state.
