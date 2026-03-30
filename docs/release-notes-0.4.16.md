# OpenClawBrain 0.4.16 / 0.4.5 split-package release notes

Published packages:

- `@openclawbrain/openclaw@0.4.5`
- `@openclawbrain/cli@0.4.16`

This release carries the harvested post-0.4.15 fixes through the canonical split-package lane: the published CLI now classifies fresh-state proof/reinstall outcomes truthfully, the learned-route replay fix is part of the public package, teacher no-op status is clearer, and the release contract itself is tighter and more reproducible.

## Why this release exists

The previous split release (`@openclawbrain/openclaw@0.4.4` / `@openclawbrain/cli@0.4.15`) proved that the public registry install path works, but the published operator CLI still had three high-signal truth gaps:

1. `proof` treated fresh-seed-state `STATUS warn` as a blocking failure even when stronger runtime proofs already showed `runtime=proven`, `loadProof=status_probe_ready`, `serve=serving_active_pack`, and `routeFn available=yes`
2. `proof` could not target a sterile foreground gateway explicitly, so dogfood on a host with another live gateway could probe the wrong runtime
3. reinstall/repair could still exit `manual_action_required` even when runtime load was already proven

This release closes those published-CLI gaps and ships the rest of the harvested swarm fixes behind the same canonical split-package surface.

## What changed

### `@openclawbrain/cli@0.4.16`

- `proof` now accepts explicit gateway probe overrides so sterile/operator-real proof runs can target the intended foreground gateway truthfully
- `proof` now treats fresh-state `STATUS warn` as a warning instead of a blocking proof failure when stronger runtime truths already prove live load
- reinstall/repair now trusts runtime-proven state instead of requiring literal displayed `STATUS ok`
- learned-route replay no longer duplicates older non-seed runtime-turn feedback into held-out eval context; the former `real-trace-live-proof-story` tie now reruns with `learned_route=100` vs `70` for `graph_prior_only` and `vector_only`
- teacher/no-op status now separates benign idle cycles from cycles that likely missed teachable material
- release preflight now derives tag/title/release-notes truth from one checked-in plan helper and verifies package-local dependencies before split tarball checks

### `@openclawbrain/openclaw@0.4.5`

- synchronized split-package runtime release that keeps the public plugin payload aligned with the new CLI/operator release contract
- verified again through the tightened split-package tarball checks and publish preflight on the canonical repo release path
- no broader new runtime-quality claim is introduced beyond alignment with the verified operator lane and current release contract

## Verification summary

Executed before publish:

```bash
npm test
npx vitest run test/release-plan.test.ts
npm run release:verify:proof
npm run release:verify:openclaw
npm run release:verify:cli
npm run release:plan -- --json
```

Key results:

- root verification passed: `46` test files / `438` tests
- release-plan verification passed: `6` tests
- proof smoke passed against the checked-in frozen bundle gate
- `@openclawbrain/openclaw@0.4.5` tarball verification passed
- `@openclawbrain/cli@0.4.16` tarball verification passed

## Operator truth after publish

Canonical lane:

```bash
openclaw plugins install @openclawbrain/openclaw
npx @openclawbrain/cli install --openclaw-home ~/.openclaw
openclaw gateway restart
npx @openclawbrain/cli status --openclaw-home ~/.openclaw --detailed
npx @openclawbrain/cli proof --openclaw-home ~/.openclaw --skip-install --skip-restart
```

Published dogfood expectation after this release:

- fresh sterile install still works on the split packages
- detailed status still proves runtime load
- proof/reinstall verdicts now align with that runtime truth instead of failing on the old fresh-state classification gap
