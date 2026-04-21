# Correction-persistence second distribution v1

Status: locked second distribution for `T-20260420-274`

## Artifact

- cases: `cases/correction-recurrence-second-distribution-v1.json`
- case count: `50`

## Slice mix

- `correction-follow-up`: 14
- `tool-heavy`: 11
- `stale-memory-conflict`: 15
- `direct-answer`: 10

## Intent

This is the second distribution for the correction-persistence utility tranche.
It keeps the same task logic as the primary distribution but shifts the surface forms and specific preferences so the first run does not become the product story by itself.

What it is trying to measure:
- whether the primary direction holds on a fresh distribution without retuning
- whether stale-memory-conflict strength survives the distribution shift
- whether heavier OCB-on variants stay worth their intervention cost
- whether direct-answer negative controls still stay safe

## Lock rules

- this suite is created after the primary run and before any retuning
- no edits after the first second-distribution run begins
- do not justify threshold changes from outcomes on this suite before the run completes

## Expected comparisons

Run the same agent across:
- `none`
- `correction-only`
- `correction-plus-heuristics`
- `full-ocb`

The readout remains:
- pass rate
- fire-conditioned utility
- false-fire harm
- abstention regret
- tokens per pass
