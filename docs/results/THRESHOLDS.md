# Product Thresholds

This is the canonical V5 threshold contract. Thresholds are fixed before production judging and must not be tuned on current evaluation results.

## Evidence Warning

Smoke, fixture, synthetic, repo-derived, or adversarial validation data must display:

```text
NOT PRODUCT EVIDENCE
SYNTHETIC PIPELINE VALIDATION ONLY
```

Synthetic threshold runs may validate decision mechanics only. They cannot determine product direction.

## Decision Metric

Product decisions use mean `net_task_utility` across all traces in the fixed primary priority slices:

- `correction-follow-up`
- `continuation`
- `stale-memory-conflict`

They must not use fire-conditioned activation utility alone.

## Required Slice Classes

Primary priority slices are fixed:

1. `correction-follow-up`
2. `continuation`
3. `stale-memory-conflict`

Secondary slices are fixed:

4. `retrieval-heavy`
5. `tool-heavy`
6. `direct-answer`

Product thresholds use primary priority slices unless a threshold explicitly names secondary slices.

## Full OCB Remains Flagship

Full OCB remains flagship only if all conditions are true:

1. `full-ocb` beats `correction-only` by at least 25% mean net task utility across primary priority slices.
2. `correction-only` captures less than 75% of `full-ocb` mean net task utility across primary priority slices.
3. `full-ocb` wins in at least 2 of 3 primary priority slices.
4. `full-ocb` does not increase false-fire harm by more than 5 percentage points versus `correction-only`.
5. `full-ocb` has positive mean net task utility in stale-memory-conflict tasks.
6. `full-ocb` does not regress correction-follow-up net task utility.
7. `full-ocb` cost per utility point is not worse than `correction-only` by more than 25% without corresponding gain.

## Correction-Only Becomes Default

Correction-only becomes default if any condition is true:

1. `correction-only` captures at least 75% of `full-ocb` mean net task utility across primary priority slices.
2. `correction-only` wins or ties `full-ocb` in both correction-follow-up and stale-memory-conflict.
3. `full-ocb` introduces material stale-memory or false-fire harm.
4. `full-ocb` gains are concentrated only in secondary or low-volume slices.

## Correction+Heuristics Becomes Default

Correction+heuristics becomes default if all conditions are true:

1. `correction+heuristics` materially beats `correction-only` in mean net task utility.
2. `correction+heuristics` captures at least 85% of `full-ocb` mean net task utility across primary priority slices.
3. `correction+heuristics` has lower false-fire and stale-memory harm than `full-ocb`.
4. `correction+heuristics` is easier to explain and adopt than `full-ocb`.

## Hybrid Outcome

A hybrid outcome is allowed when correction+heuristics is default and full OCB is enabled only for slices where it beats baselines without harm.

Use this when `full-ocb` wins in retrieval-heavy or tool-heavy tasks but loses or ties in primary slices.

## Pause Conditions

Pause general-memory-runtime claims if any condition is true:

- no backend shows positive net task utility in at least 4 slices
- utility-sign judge disagreement exceeds 30%
- traces are too weak, too clean, or too repo-adjacent to prove product value

## Conflict Resolution

If multiple product outcomes match, the decision generator must not choose post hoc by preference. It must apply `docs/results/PRODUCT_DECISION_TREE.md` and either select exactly one deterministic outcome or emit a declared tie/blocker.

## Evidence Completeness Gate

No threshold can produce product evidence unless Evidence E2E is complete:

- 40 admitted real redacted traces
- required slice counts satisfied
- provenance and privacy metadata present
- all four backends run against all admitted traces
- blind and labeled judging complete
- judged ledger exists
- generated results are derived from the judged ledger
- `docs/results/30_DAY_DECISION.md` is produced

If this gate is not satisfied, the generated decision must state `evidence_e2e_complete=false` and must not choose a final product direction as product evidence.
