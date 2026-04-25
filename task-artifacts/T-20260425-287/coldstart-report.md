# T-20260425-287 — cold-start scorecard and selection improvement

## Baseline inspected

Inspected the current cold-start router/runtime/eval lane:

- `src/brain-core/cold-start-router-runtime.ts`
- `src/brain-core/cold-start-router-trainer.ts`
- `src/replay-proof-lane.ts`
- `src/eval/comparative-eval-runner.ts`
- checked-in score/eval surfaces under `evals/recorded-session-replay/canonical-frozen-20/`
- existing approved/candidate artifacts under `artifacts/cold-start-router-approved-export/` and `artifacts/activation-first-gating-retune/`
- recent task artifacts under `task-artifacts/T-20260425-285/`

The meaningful baseline gap was in the cold-start candidate-artifact replay override path: it selected only one block per turn. On the frozen 20 replayable trace set that under-selected context:

- `cold_start_prior_single` (old single-select shape): mean quality `92.05`, phrase hits `64/74`, selected blocks `45`, selected chars `12,501`.
- `graph_prior_only`: mean quality `92.8`, phrase hits `65/74`, selected blocks `135`, selected chars `26,125`.
- served/replay-trained `learned_route`: mean quality `97.3`, phrase hits `71/74`, selected blocks `135`, selected chars `27,592`.

## Change made

Small runtime selection change, not a broad rewrite:

- `selectColdStartRouteCandidateIdsFromArtifactBundleV1` now accepts optional `maxCandidateIds` and `multiSelectScoreWindow`.
- Default behavior remains legacy single-select for existing runtime callers.
- Replay candidate-artifact override passes the replay turn block budget, but only selects close-score candidates within the default `0.35` score window.
- This fixes the one-block under-selection failure while bounding overfire: it cannot exceed the caller's max block budget and does not add low-score tail candidates.

Added deterministic scorecard harness:

- `src/eval/cold-start-scorecard.ts`
- `scripts/eval/run-cold-start-scorecard.ts`
- scorecard compares: `no_brain`, `graph_prior_only`, `cold_start_prior_single`, `cold_start_prior`, and `learned_route`.
- `cold_start_prior` maps to the candidate-artifact `learned_route` replay override because the existing proof-bundle mode order is fixed.

## Measured result

Frozen scorecard artifact:

- `task-artifacts/T-20260425-287/coldstart-scorecard/scorecard.json`
- `task-artifacts/T-20260425-287/coldstart-scorecard/summary.md`

Scorecard run:

```bash
node --experimental-transform-types scripts/eval/run-cold-start-scorecard.ts \
  --output-dir task-artifacts/T-20260425-287/coldstart-scorecard \
  --scratch-root-dir scratch/T-20260425-287-coldstart-scorecard \
  --generated-at 2026-04-25T14:30:00.000Z
```

Mode summary:

| mode | mean quality | phrase hits | selected blocks | selected chars | prompt tokens | warnings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no_brain | 0 | 0/74 | 0 | 0 | 0 | 0 |
| graph_prior_only | 92.8 | 65/74 | 135 | 26,125 | 6,539 | 0 |
| cold_start_prior_single | 92.05 | 64/74 | 45 | 12,501 | 3,129 | 20 |
| cold_start_prior | 92.8 | 65/74 | 48 | 13,101 | 3,279 | 20 |
| learned_route | 97.3 | 71/74 | 135 | 27,592 | 6,906 | 0 |

Honest useful-context win:

- `cold_start_prior` vs `cold_start_prior_single`: `+0.75` mean quality, `+1` required phrase hit, with only `+3` blocks / `+600` chars.
- `cold_start_prior` vs `graph_prior_only`: equal quality and phrase hits, while selecting `87` fewer blocks and `13,024` fewer chars.
- No overfire vs graph prior on this scorecard: `cold_start_prior` does not select more blocks/chars than `graph_prior_only` and does not lose phrase recall.

## Exact focused tests

Run after implementation:

```bash
./node_modules/.bin/vitest run test/brain-core/cold-start-router-runtime-selection.test.ts test/eval/cold-start-scorecard.test.ts test/brain-core/cold-start-router-runtime.test.ts test/brain-core/cold-start-router-trainer.test.ts
```

Result: passed — `4` files / `24` tests.

## Honest boundary

This does not claim the cold-start prior beats a fully available learned router. It does not: served/replay-trained `learned_route` remains stronger on this frozen scorecard (`71/74` phrase hits vs `65/74`).

The real claim is narrower and now evidenced: the cold-start candidate-artifact prior no longer under-selects a single block by default in replay override use, recovers one required context hit on the frozen scorecard, and matches graph-prior recall with much less selected context. No npm publish, GitHub push, local OpenClaw install, or gateway restart was performed.
