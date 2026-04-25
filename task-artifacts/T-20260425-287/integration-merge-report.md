# T-20260425-287 — OpenClawBrain 0.4.48 integration merge report

Generated: 2026-04-25 07:32 PDT
Branch: `t287-ocb-048-integration`
Base/integration contract preserved: `f3eb405c` (`Add 0.4.48 integration contract`)

## Commits merged

Cherry-picked in the recommended order:

1. Cold-start lane: `4e178a15` → `ce47ce35` (`ocb: add cold-start prior scorecard`)
2. Learned-route lane: `4420f8c3` → `83d06a38` (`Add learned-route activation usefulness labels`)
3. Teacher graph-maintenance lane: `c4ede0ca` → `4726792f` (`Add Teacher graph maintenance proposal lifecycle`)

The harmless untracked `node_modules` symlink in this worktree was removed before integration. A no-save dependency repair was used only to run local tests after symlink removal:

```bash
npm install --package-lock=false --no-save --ignore-scripts --fund=false --audit=false
```

No npm publish, GitHub push, local OpenClaw install, or gateway restart was performed.

## Conflicts / resolutions

- Cold-start cherry-pick: clean.
- Learned-route cherry-pick: `src/replay-proof-lane.ts` auto-merged cleanly by git; no manual conflict markers or dropped tests/artifacts.
- Teacher graph-maintenance cherry-pick: clean.

## Focused verification run on integrated branch

```bash
npx vitest run test/brain-core/cold-start-router-runtime-selection.test.ts test/eval/cold-start-scorecard.test.ts test/brain-core/cold-start-router-runtime.test.ts test/brain-core/cold-start-router-trainer.test.ts --reporter=dot
```

Result: pass — 4 files / 24 tests.

```bash
npx vitest run --dir test test/replay-proof-lane.test.ts test/eval/openclawbrain-explainable-scorecard.test.ts test/eval/comparative-eval-runner.test.ts --reporter=dot
```

Result: pass — 3 files / 19 tests.

```bash
npx vitest run --dir test test/brain-core/teacher-v3-graph-maintenance.test.ts test/brain-store/teacher-v3-proposals.test.ts test/brain-core/teacher-v3-shadow-replay.test.ts test/brain-core/teacher-v3-replay.test.ts --reporter=dot
```

Result: pass — 4 files / 10 tests.

```bash
npx vitest run --dir test test/brain-core/teacher-v3-contracts.test.ts test/teacher-v3-proposal-artifact.test.ts test/teacher-v3-proof-bundle.test.ts test/teacher-v3-replay-outcomes.test.ts test/teacher-v3-promotable-examples.test.ts test/teacher-v3-shadow-worked-examples.test.ts test/brain-core/teacher-v3-graph-maintenance.test.ts test/brain-store/teacher-v3-proposals.test.ts --reporter=dot
```

Result: pass — 8 files / 30 tests.

```bash
git diff --check
```

Result: pass.

## Blockers

None for the requested merge integration. The combined focused lane tests passed on the integrated branch.

Known boundary inherited from lane reports: repo-wide `npx tsc --noEmit` was not used as a release gate here because lane reports document existing repo-wide TypeScript errors outside the touched lane surfaces. This merge report does not claim full repo typecheck or full `npm test` coverage.

## Next gates before any 0.4.48 release claim

- Assemble/update release-readiness evidence and checklist completion against `task-artifacts/T-20260425-287/release-checklist.v1.json`.
- Align version/docs/release surfaces intentionally to `0.4.48` if release-prep proceeds.
- Run `npm run release:verify:docs-drift` after docs/version alignment.
- Run `npm run release:plan` after package/release-note/changelog alignment.
- Run full `npm test` or record a precise pre-existing blocker/focused-subset rationale before publish.
- Keep public claims bounded: cold-start prior improvement, learned-route centrality/attribution, and off-path Teacher/Graphify graph-maintenance proposal lifecycle only.
