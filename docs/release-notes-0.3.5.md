# OpenClawBrain 0.3.5 release notes

Published package: `@jonathangu/openclawbrain@0.3.5`

## What changed

- Hardened prompt handling around `before_prompt_build` edge cases so usable `event.prompt` text still flows when `messages` is empty or the last message has no text content.
- Added focused teacher-status truth coverage so a fresh watch heartbeat with `no_teacher_artifacts` is treated as idle/healthy rather than stale/unhealthy.
- Recovered the `packages/openclaw` front-door package tree into the public repo so the package that actually owns the generated runtime guard is visible in version control.
- Preserved the single-extra-LLM design: the async teacher lane remains the local Ollama teacher (`qwen3.5:9b`), with no new model-role sprawl.

## Why it matters

This release closes the gap between:

1. what the live install was actually doing, and
2. what the public repo/package surface could truthfully ship.

It makes the generated runtime hook more graceful on partial prompt envelopes, keeps teacher health reporting honest during real no-op cycles, and reduces confusion about which package owns the install/runtime path.

## Verification summary

- focused prompt-fallback regression tests passed
- focused teacher-status truth tests passed
- live host reinstall/relink verification passed
- direct runtime-guard probe passed for:
  - `messages: []` + usable `prompt`
  - non-text last message + usable `prompt`
- live status after reinstall/relink showed:
  - `teacher healthy=yes`
  - `stale=no`
  - learned `route_fn` still serving the active pack
