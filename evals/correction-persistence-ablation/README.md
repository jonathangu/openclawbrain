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
- `full-ocb` stays disabled by default until `scripts/ocb-adapter.ts` is wired to the real runtime.
- `better-sqlite3` is used for the durable decision/outcome ledger and may need native compilation on the target host.

## Commands

```bash
pnpm install
pnpm typecheck
pnpm selftest
pnpm demo-results
pnpm ablation
# wire scripts/ocb-adapter.ts first, then:
pnpm ablation:full
```

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
