# Claims boundary

Last updated: 2026-03-24

This file defines what OpenClawBrain claims to do today. Items under "Claimed" are exercised on real OpenClaw profiles on macOS. Items under "Not yet claimed" are intentionally outside the public claims boundary until they are proven and documented.

## Claimed

- The public install lane works with `@openclawbrain/openclaw`, `@openclawbrain/cli`, a gateway restart, `status --detailed`, and the first-class `proof` bundle command.
- Install, attach, status, proof capture, rollback, detach, and uninstall work on real OpenClaw profiles.
- The runtime can fail open: if the memory layer cannot safely compile context, OpenClaw continues without injected brain context.
- The learning pipeline exports turns, builds candidate packs off the response path, and only serves promoted packs.
- Explicit user corrections can be stored as durable memory and used at retrieval time.
- Dedicated-brain separation across two real profiles is proven.
- The native V2 metadata surfaced by the current detailed-status surfaces is accurate for real promoted packs.
- The operator surfaces expose both human-readable and machine-readable status output, plus durable proof bundles for install/runtime reporting.

## Not yet claimed

- Same-gateway multi-profile attachment as a general public lane
- Shared-root concurrent write safety
- Broad cross-platform support beyond the macOS hosts exercised so far
- Automatic gateway lifecycle management or service orchestration
- Manual slot rewriting, LaunchAgent editing, or environment-file management as supported operator flows
- Universal dated citations on every recalled memory surface
- Exact evidence-backed attribution on every learning and supervision path
- Broad live answer-quality gains beyond the current operator/runtime proof surfaces

## How to extend this boundary

- Update this file when a release turns an unclaimed item into a claimed one.
- Link the supporting release in [CHANGELOG.md](CHANGELOG.md) or the relevant release note.
- If a change weakens a claimed behavior, call it out in the changelog and contributor docs before release.
