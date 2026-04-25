# T-20260425-287 — OpenClawBrain 0.4.48 release readiness

Generated: 2026-04-25 07:55 PDT
Branch: `t287-ocb-048-integration`
Current integrated head: `2adcf724` (`Integrate OpenClawBrain 0.4.48 swarm lanes`)
Target version: `0.4.48`

## Decision

Pre-publish readiness is **mostly green locally**, with one intentional external blocker remaining: the release commit is not yet on `origin/main`, so `npm run release:plan` correctly refuses a publish plan.

No npm publish, tag/GitHub release, local install, or gateway restart has been performed.

## Integrated lane commits

Cherry-picked into the integration branch in order:

1. `4e178a15` cold-start lane → integrated as `ce47ce35`
2. `4420f8c3` learned-route lane → integrated as `83d06a38`
3. `c4ede0ca` Teacher graph-maintenance lane → integrated as `4726792f`
4. `2adcf724` integration merge commit preserving `f3eb405c` release contract

## What is allowed to claim

- `0.4.48` strengthens cold-start candidate-artifact replay selection: it avoids the old single-block under-selection path and matches `graph_prior_only` on the frozen scorecard while using less context.
- `0.4.48` adds learned-route activation-usefulness accounting: beneficial wins, harmful activations, neutral ties, missed opportunities, correct abstentions, and proxy prompt/context deltas.
- `0.4.48` adds one narrow durable Teacher v3 graph-maintenance lifecycle for shadow-only `add_edge` proposals with replay and rollback evidence.

## What is forbidden to claim

- published/shipped status before npm publish, tag/release, install, status, and proof evidence exist
- broad memory solved
- broad online answer-quality improvement
- fresh-home equivalence to trained homes
- cold-start beats the served learned router
- Graphify or Teacher graph maintenance is live runtime truth authority
- graph mutations promote or edit live graph truth automatically
- wins from ties/no-regression evidence

## Evidence and gates already run

### Integration focused gates

From `task-artifacts/T-20260425-287/integration-merge-report.md`:

- cold-start focused set: `4` files / `24` tests passed
- learned-route/replay focused set: `3` files / `19` tests passed
- Teacher graph-maintenance focused set: `4` files / `10` tests passed
- extended Teacher v3 proof/replay set: `8` files / `30` tests passed
- `git diff --check` passed

### Additional pre-publish gates run after review

```bash
npm run test:learned-route-mission
```

Result: passed — `8` files / `45` tests.

```bash
npx vitest run --dir test test/brain-core/cold-start-router-approved-export-loader.test.ts test/brain-core/cold-start-router-replay-gate.test.ts test/brain-core/cold-start-router-runtime-selection.test.ts test/eval/cold-start-scorecard.test.ts test/brain-core/cold-start-router-runtime.test.ts test/brain-core/cold-start-router-trainer.test.ts --reporter=dot
```

Result: passed — `6` files / `32` tests.

```bash
npx vitest run --dir test test/brain-core/graphify-training-bridge.test.ts test/graphify-final-replay-proof.test.ts test/brain-core/teacher-v3-graph-maintenance.test.ts test/brain-store/teacher-v3-proposals.test.ts test/brain-core/teacher-v3-shadow-replay.test.ts test/brain-core/teacher-v3-replay.test.ts --reporter=dot
```

Result: passed — `6` files / `13` tests.

```bash
npm test
```

Result: passed — `116` files / `748` tests.

```bash
git diff --check
```

Result: passed.

```bash
npm run release:verify:docs-drift
```

Result: passed — release/docs drift lint clean for `0.4.48`.

```bash
npm run release:plan
```

Result: blocked as expected before external mainline publication:

- `release_ref_not_on_mainline`: HEAD `2adcf7242dcb74da84839fb3c92c92ec54c24613` is not reachable from `origin/main`; publish the merged release commit on main.

### Typecheck note

```bash
npx tsc --noEmit --pretty false
```

Still reports pre-existing repo-wide fixture/type drift outside this release lane. `npm test` and all focused release gates pass, so typecheck remains a known non-publish gate for this tranche unless separately cleaned up.

## Version/docs alignment

Aligned locally on the integration branch:

- root package version: `0.4.48`
- `packages/openclaw/package.json`: `0.4.48`
- `packages/cli/package.json`: `0.4.48`
- `package-lock.json`: `0.4.48` package surfaces
- `README.md`: current version `0.4.48`
- `docs/README.md`: current release notes link `0.4.48`
- `docs/END_STATE.md`: split-package versions `0.4.48`
- `docs/architecture/teacher-v3-lints.md`: release-surface example versions `0.4.48`
- `CHANGELOG.md`: added `0.4.48` entry
- `docs/release-notes-0.4.48.md`: added bounded release notes

## Remaining steps before npm publish

1. Commit this release-prep/readiness update.
2. Merge/push the release commit to `origin/main` only when Jonathan approves that external write.
3. Rerun `npm run release:plan` from the mainline commit; it should clear once HEAD is reachable from `origin/main`.
4. Only then proceed to npm publish/tag/GitHub release/install/restart/proof, with the normal explicit approval boundary.

## Final pre-publish verification after package-surface fix

A full release verification pass initially caught a real package-surface drift:

- `packages/openclaw/openclaw.plugin.json` still reported `0.4.47` while `packages/openclaw/package.json` reported `0.4.48`.

Fixed before publishing by aligning:

- `packages/openclaw/openclaw.plugin.json` → `0.4.48`
- `docs/getting-started/quick-start.md` CLI examples → `0.4.48`
- `docs/lifecycle.md` CLI examples → `0.4.48`

Then reran:

```bash
npm run release:verify
```

Result: passed.

Included gates:

- dependency policy clean
- root `npm test`: `116` files / `748` tests passed
- proof smoke ok
- root `npm pack --dry-run` for `@jonathangu/openclawbrain@0.4.48`
- OpenClaw package verify: `85` node tests passed + tarball verified for `@openclawbrain/openclaw@0.4.48`
- CLI package verify: `184` node tests passed + tarball verified for `@openclawbrain/cli@0.4.48`
