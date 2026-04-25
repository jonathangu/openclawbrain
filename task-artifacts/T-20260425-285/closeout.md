# T-20260425-285 — OpenClawBrain harvest-all-good-work sweep

## Result

Harvested all low-risk valuable residue into canonical `main`, then pruned local branch clutter after audit.

## Harvested commits

- `bd8a4b57` — Harvest package truth surfaces
  - Restored/updated package type truth for bounded serving interruption accounting and context-feedback surfaces.
  - Added package surface tests.
- `80b4f48d` — Map canonical package release seam for T-20260331-077
  - Preserved package seam audit artifacts.
- `008a89ac` — Update task artifacts with commit SHA
  - Preserved corrected lane artifact metadata.
- `249a3470` — Add sanitized real-trace replay corpus
  - Preserved sanitized real-trace replay corpus and exporter.

## Branch cleanup

- Wrote full pre-prune branch manifest: `task-artifacts/T-20260425-285/branch-manifest-before-prune.txt`.
- Deleted `122` merged or patch-equivalent local branches.
- Deleted `70` stale/superseded conflict branches after audit.
- Remaining local branch: `main` only.
- Remaining worktree: primary checkout only.

## Audit verdict

Subagent `ocb-harvest-branch-audit` found no additional low-risk cherry-picks worth integrating. Remaining branches were classified as duplicate, semantically harvested already, superseded by richer current-main implementations, or stale target-state variants that would conflict/regress current package/proof surfaces.

Examples:

- `t217/*`, `t218/*`: already harvested / duplicate.
- `t129-gate*`: already harvested or superseded Teacher v3 proof/canary surfaces.
- `task/T-20260325-031-wave*`: old wave work already present in main or superseded.
- `task/T-20260402-105-*`: duplicate / release-only stale.
- issue fix branches: conceptually already handled by current runtime/proof/status surfaces or stale package-era shapes.

## Verification

- `git diff --check` — passed.
- `npm run release:verify:docs-drift` — passed.
- `npm run release:verify` — passed:
  - dependency policy clean
  - root Vitest: `113` files / `743` tests passed
  - proof smoke passed
  - root `npm pack --dry-run` completed for `@jonathangu/openclawbrain@0.4.47`
  - `@openclawbrain/openclaw@0.4.47` verify: `85` node tests passed + tarball verified
  - `@openclawbrain/cli@0.4.47` verify: `184` node tests passed + tarball verified
- Registry check before publish: npm still shows `@openclawbrain/cli@0.4.46` and `@openclawbrain/openclaw@0.4.46`; local packages are `0.4.47`.

## Publish readiness

No code/test blocker remains before publishing `0.4.47`.

Recommended final sequence before npm publish:

1. Commit this closeout state.
2. Push `main` to GitHub.
3. Publish `@openclawbrain/openclaw@0.4.47` and `@openclawbrain/cli@0.4.47` using package `prepublishOnly` gates.
4. Create/push release tag / GitHub release if following the existing release process.
5. After publish, optionally install/converge local OpenClawBrain `0.4.47` with explicit approval before any gateway restart.

Boundary: no npm publish, GitHub push, local install, or gateway restart was performed in this sweep.
