# T-20260425-287 Lane B route report

## Baseline inspected
- `src/replay-proof-lane.ts` already had pairwise optimize-over labels for `learned_route` vs `graph_prior_only`, plus activation precision/proxy summaries.
- The T-20260425-283 proof report explicitly left a blocker: replay surfaces still lacked an independent beneficial-opportunity oracle and true activation/usefulness accounting beyond `activationTaken` precision/proxy.
- Current route substrates inspected: route decision/outcome events, cold-start trainer/replay gate, policy supervision rows, comparative/explainable eval surfaces, and replay-lane tests.

## Change
- Added explicit per-turn `activationUsefulness` labels in replay summary tables:
  - `didLearnedRoutingFire` from emitted `activationTaken` on learned_route turns.
  - `shouldHaveFired` from deterministic pairwise outcome (`learned_route` better than `graph_prior_only`).
  - `usefulness`: `beneficial`, `harmful`, `neutral`, `missed_beneficial_opportunity`, `correct_abstention`, or `unobserved`.
  - prompt/context cost deltas against `graph_prior_only` where estimates exist.
- Added scorecard-level `activationUsefulness` summary with unique beneficial wins, harmful activations, neutral/no-op ties, missed opportunities, correct abstentions, and fired prompt/context cost deltas.
- Surfaced the new activation-usefulness summary in replay lane README and closeout summary markdown.

## Measured deterministic result
On the existing deterministic replay-lane fixture (`trace-comparative-replay` + `trace-score-resolution`):
- observed activation labels: 5/5 comparable turns
- fired learned routing: 5 turns
- should-have-fired opportunities: 1 turn
- unique beneficial learned-route wins: 1 turn (`plan-turn-3`)
- harmful activations: 0
- neutral activation ties / no-op ties: 4
- missed beneficial opportunities: 0
- fired prompt-token delta vs graph prior: +93 estimated tokens

This is an incremental learned-route win in the deterministic fixture, but it is intentionally not claimed as a broad live-runtime win.

## Tests run
- `npm install --package-lock=false --no-save --ignore-scripts --fund=false --audit=false` (local worktree dependency repair only)
- `npx vitest run --dir test test/replay-proof-lane.test.ts --reporter=dot` — pass, 6 tests
- `npx vitest run --dir test test/replay-proof-lane.test.ts test/eval/openclawbrain-explainable-scorecard.test.ts test/eval/comparative-eval-runner.test.ts --reporter=dot` — pass, 19 tests
- `git diff --check` — pass
- `npx tsc --noEmit --pretty false` — blocked by existing repo-wide type errors outside this lane; one new local reducer type issue found by this gate was fixed. Remaining replay-proof-lane TypeScript errors shown by the global gate pre-existed this lane (`src/replay-proof-lane.ts:890`, `:977`, `:1268`).

## Honest boundary
- `shouldHaveFired` is still derived from deterministic replay quality against `graph_prior_only`, not a separate human oracle.
- `didLearnedRoutingFire` is only treated as true when replay emits `activationTaken=true`; proxy divergence is not counted as true activation in the new usefulness metric.
- Cost deltas remain prompt/context proxies, not full system cost.
