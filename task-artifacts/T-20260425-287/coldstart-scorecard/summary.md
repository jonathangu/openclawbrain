# Cold-start scorecard

Generated: 2026-04-25T14:30:00.000Z
Manifest: `/Users/guclaw/.openclaw/workspace/worktrees/ocb-048-coldstart/evals/recorded-session-replay/canonical-frozen-20/manifest.json`
Candidate artifact: `/Users/guclaw/.openclaw/workspace/worktrees/ocb-048-coldstart/artifacts/activation-first-gating-retune/T-20260419-269/candidate-artifact`
Trace count: 20

## Mode summary

| mode | mean quality | phrase hits | selected blocks | selected chars | prompt tokens | warnings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no_brain | 0 | 0/74 | 0 | 0 | 0 | 0 |
| graph_prior_only | 92.8 | 65/74 | 135 | 26125 | 6539 | 0 |
| cold_start_prior_single | 92.05 | 64/74 | 45 | 12501 | 3129 | 20 |
| cold_start_prior | 92.8 | 65/74 | 48 | 13101 | 3279 | 20 |
| learned_route | 97.3 | 71/74 | 135 | 27592 | 6906 | 0 |

## Deltas

Positive quality/phrase deltas mean the right-hand mode improved; negative context deltas mean it used less context.

| comparison | quality Δ | phrase-hit Δ | block Δ | char Δ | token Δ |
| --- | ---: | ---: | ---: | ---: | ---: |
| cold_start_prior vs no_brain | 92.8 | 65 | 48 | 13101 | 3279 |
| cold_start_prior vs cold_start_prior_single | 0.75 | 1 | 3 | 600 | 150 |
| cold_start_prior vs graph_prior_only | 0 | 0 | -87 | -13024 | -3260 |
| learned_route vs cold_start_prior | 4.5 | 6 | 87 | 14491 | 3627 |

## Verdict

- Useful-context win: `true`
- No overfire vs graph_prior_only: `true`
- Summary: cold_start_prior recovers useful context versus the single-block prior and matches graph_prior_only recall without selecting more context (-13024 chars).

## Honest boundary

This is a frozen replay scorecard over checked-in sanitized/replayable traces. `cold_start_prior` maps to the candidate-artifact `learned_route` replay override, not to the served learned-route hot path. The served `learned_route` baseline remains stronger where a learned router is already available; this lane improves the cold-start prior/selection fallback and keeps the served/publish boundary unchanged.
