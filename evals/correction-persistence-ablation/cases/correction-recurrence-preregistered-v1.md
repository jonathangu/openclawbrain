# Correction-persistence preregistered suite v1

Status: locked primary distribution for `T-20260420-274`

## Artifact

- cases: `cases/correction-recurrence-preregistered-v1.json`
- case count: `50`

## Slice mix

- `correction-follow-up`: 14
- `tool-heavy`: 11
- `stale-memory-conflict`: 15
- `direct-answer`: 10

## Intent

This is the first locked `N >= 50` distribution for the correction-persistence utility tranche.
It is deliberately narrow.

What it is trying to measure:
- whether explicit durable corrections and preferences improve later task success
- whether later explicit corrections beat stale earlier instructions
- whether command and workflow preferences persist on later turns
- whether memory stays quiet on direct-answer negative controls

What it is not trying to prove:
- broad general memory-runtime superiority
- repo-shaped replay wins
- threshold-tuned headline performance

## Lock rules

Before the first primary ablation run on this file:
- wording cleanups are allowed only if they do not change task intent or grading direction
- no case may be added or removed without writing a new suite version

After the first primary ablation run on this file:
- no edits to this file
- no edits to grading criteria in this suite
- no threshold tuning justified by outcomes on this suite
- any revisions must go to a new versioned suite artifact

## Expected comparisons

Run the same agent across:
- `none`
- `correction-only`
- `correction-plus-heuristics`
- `full-ocb`

The primary readout remains:
- pass rate
- fire-conditioned utility
- false-fire harm
- abstention regret
- tokens per pass

## Known seam

The current `full-ocb` adapter uses the real isolated `BrainService` teach/query path, but the present `BrainService.query(...)` surface does not expose a stable per-turn gate scalar or threshold. For now, the eval ledger should treat `gate_score` and `gate_threshold` as nullable for this suite rather than backfilling fake values.
