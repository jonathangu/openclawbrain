# OpenClawBrain 0.3.7 release notes

Published package: `@jonathangu/openclawbrain@0.3.7`

This is a post-split compatibility release for older combined-package installs. The canonical public install story remains the split package lane (`@openclawbrain/openclaw` + `@openclawbrain/cli`), but this patch keeps the legacy combined package truthful for hosts that still depend on it.

## What changed

- added soft compile-deadline support through `brainMaxCompileMs`
- added structured bounded-runtime metadata on the compatibility-package live path:
  - `compileElapsedMs`
  - `compileDeadlineMs`
  - `compileDeadlineHit`
  - `brainDropReason`
  - `brainDropStage`
- added truthful fail-open phase-boundary bailouts for:
  - deadline before query
  - deadline after query
  - deadline before injection
- decoupled retrieval/query budget from the final `maxContextChars` injection cap
- preserved durable clip/deadline attribution through trace, observation, teacher materialization, and status surfaces
- added focused regression coverage for bounded-runtime behavior and budget/cap separation

## Why it matters

Before this patch, the combined compatibility package had bounded-context clipping truth, but it still lacked a real compile-deadline surface and still overloaded `maxContextChars` as both retrieval budget and final injection cap.

`0.3.7` closes that bounded-runtime seam for older combined-package installs without pretending to implement true mid-traversal interruption. The new behavior is a **soft phase-boundary deadline** with honest fail-open outcomes.

## Verification summary

Executed from the repo before publish:

- `npm run release:verify`
  - passed
- root Vitest suite:
  - `43` files passed
  - `395` tests passed
- split-package tarball verification also passed as part of release verify:
  - `@openclawbrain/openclaw@0.4.2`
  - `@openclawbrain/cli@0.4.13`

Focused bounded-runtime proof also passed before release:

- `npx vitest run test/brain-runtime/assembler-extension.test.ts test/brain-runtime/service.test.ts test/brain-runtime/observation.test.ts test/brain-core/teacher.test.ts test/config.test.ts test/engine.test.ts`
  - `6` files / `101` tests passed

## Release boundary

This release updates the **combined compatibility package** only.

The split-package public lane (`@openclawbrain/openclaw` / `@openclawbrain/cli`) was **not** version-bumped in this release because the bounded-runtime code changes did not land in those package payloads during this lane. Bumping them here would have been version churn without new code.