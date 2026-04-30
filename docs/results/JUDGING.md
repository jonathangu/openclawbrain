# Blind Judging Contract

OpenClawBrain V5 uses blind packets before judged ledger import.

## Smoke mode

Smoke judgments may be synthetic and must keep this warning attached:

```text
NOT PRODUCT EVIDENCE
SYNTHETIC PIPELINE VALIDATION ONLY
```

Smoke import writes `eval/results/<run-id>/ledger-judged.synthetic.jsonl` and forces `counts_as_product_evidence=false`.

## Production mode

Production import fails closed if any candidate is missing a judgment or if a synthetic judgment is supplied. Production evidence still requires admitted real privacy-scrubbed traces and completed threshold checks.

## Commands

- `pnpm ocb:eval:make-blind-packets -- --run-id <run-id>`
- `pnpm ocb:judgments:import -- --mode smoke --run-id <run-id> --judgments <file-or-dir>`
