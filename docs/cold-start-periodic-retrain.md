# Cold-start periodic retrain lane

This lane is the bounded periodic retrain path for the same-family `route_fn` policy.

## What it does
- loads a governed train export and an eval-only export
- builds a **split registry** that partitions rows into:
  - `train`
  - `eval_only`
  - `quarantine`
- retrains a candidate router artifact only from the train slice
- replay-gates the candidate against the eval slice
- emits a bounded promotion package with rollback binding

## Contracts
- `cold_start_router_route_split_registry.v1`
- `cold_start_router_replay_eval_report.v1`
- `cold_start_router_promotion_package.v1`

## Entry point
- `scripts/periodic-cold-start-router-retrain.ts`

## Default fixture flow
- train export: `artifacts/cold-start-router-approved-export/real-approved-router-export.hotpotqa-musique.v3.json`
- eval export: `artifacts/cold-start-router-approved-export/real-disjoint-eval-only-router-export.hotpotqa-musique.v1.json`
- prior base artifact: `artifacts/cold-start-router-approved-export/real-approved-router-train.hotpotqa-musique.v3`
- candidate/report outputs default to `scratch/cold-start-router-periodic-retrain/`

## Promotion rule
The candidate only becomes promotable when:
- the split registry has no train/eval overlap
- train replay passes
- eval replay passes
- the promotion package can bind a rollback key to the prior base prior

This keeps the retrain lane replay-gated and narrow instead of speculative.
