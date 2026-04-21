# Correction-persistence utility ablation

This in-tree harness exists to answer the question the current operator-proof lane does not answer on its own:

> does OpenClawBrain materially improve downstream task outcomes on a preregistered correction-persistence suite?

## Scope

This harness is intentionally narrow.

- primary wedge: correction persistence
- comparison ladder: `none`, `correction-only`, `correction-plus-heuristics`, `full-ocb`
- objective grading: deterministic pass/fail only, no LLM-as-judge
- cost accounting: injected tokens, response tokens, tokens per pass
- harm accounting: abstention regret and false-fire harm versus the `none` baseline

## Truth boundary

- The starter suite in `cases/correction-recurrence.json` is only a seed, not publishable proof.
- The first real tranche needs a preregistered suite with `N >= 50` cases and at least one second distribution.
- `full-ocb` now routes through a real isolated `BrainService` teach/query path for synthetic eval cases.
- The current `BrainService.query(...)` surface still does not expose a stable per-turn gate scalar or threshold, so `full-ocb` currently records `gate_score = null` and `gate_threshold = null` instead of inventing them.
- `better-sqlite3` is used for the durable decision/outcome ledger and may need native compilation on the target host.

## Commands

```bash
pnpm install
pnpm typecheck
pnpm selftest
pnpm demo-results
pnpm ablation
OCB_CASES=./cases/correction-recurrence-preregistered-v1.json pnpm ablation
OCB_CASES=./cases/correction-recurrence-preregistered-v1.json pnpm ablation:full
OCB_RUN_ID=my-primary-run OCB_CASES=./cases/correction-recurrence-preregistered-v1.json pnpm ablation:full
OCB_RUN_ID=my-primary-run OCB_CASES=./cases/correction-recurrence-preregistered-v1.json pnpm ablation:supervise
OCB_RUN_ID=my-primary-run OCB_CASES=./cases/correction-recurrence-preregistered-v1.json pnpm ablation:checkpoint
```

Resume note:

- Set `OCB_RUN_ID` to reuse a run id after a crash or host interruption.
- The harness now skips already-recorded `(run_id, case_id, backend)` outcomes and clears dangling decision rows without outcomes before retrying.
- `pnpm ablation:supervise` wraps the same run id in a small restart supervisor that relaunches the child run after interruptions or stalled pending cells.
- `pnpm ablation:checkpoint` renders an honest partial-results snapshot from the durable ledger while a long run is still in flight.

## Layout

- `src/` core harness logic
- `scripts/` entrypoints and adapter stub
- `cases/` preregistered or seed suites
- `results/` generated output

## Immediate hardening already applied

The vendored harness lands with the first-round fixes already applied:

- multiline-safe regex grading
- partial matrix typing instead of a false full-record cast
- SQLite foreign keys enabled
- uniqueness guard on `(run_id, case_id, backend)`
- heuristic retrieval de-duplicated against deterministic correction retrieval
- dynamic slice derivation for the results page
- durable `finally` close in the main runner
- `results/` creation in the demo renderer
