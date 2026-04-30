# Product Decision Tree

This is the canonical V5 day-30 product decision contract.

## Evidence Warning

Smoke, fixture, synthetic, repo-derived, or adversarial validation data must display:

```text
NOT PRODUCT EVIDENCE
SYNTHETIC PIPELINE VALIDATION ONLY
```

Synthetic decision memos may validate generation mechanics only. They must not choose a final product direction as evidence.

## Allowed Outcomes

At day 30, the decision generator must choose exactly one of:

A. Full OCB remains the product  
B. Correction-sticky product becomes default  
C. Correction+heuristics product becomes default  
D. Hybrid default + slice-gated full OCB  
E. Runtime health / verification layer only  
F. Pause until better traces exist

No other product outcome is allowed in V5.

## Precondition Gate

Before choosing A-D as product direction, all must be true:

- `evidence_e2e_complete=true`
- 40 admitted real redacted traces exist
- required slice counts are satisfied
- provenance and privacy metadata are present
- all four backends ran against all admitted traces
- blind and labeled judging are complete
- judged ledger exists
- `/results` regenerated from judged ledger
- judge disagreement is within threshold
- thresholds in `docs/results/THRESHOLDS.md` are applied

If any item is false, the decision must choose E or F, or emit a declared blocker if implementation cannot safely choose between them.

## Deterministic Decision Order

Apply this order after the precondition gate:

1. If any pause condition in `docs/results/THRESHOLDS.md` is true, choose F.
2. If no backend shows reliable positive net task utility in product-relevant slices but memory/runtime health signals are useful for verification, choose E.
3. If all full-OCB flagship conditions are true and no correction-only/correction+heuristics default condition supersedes it, choose A.
4. If correction+heuristics default conditions are true, choose C.
5. If correction-only default conditions are true, choose B.
6. If full OCB wins only in retrieval-heavy or tool-heavy secondary slices without harm while correction+heuristics is safer in primary slices, choose D.
7. If conditions conflict or ties cannot be resolved by fixed thresholds, choose F with explicit threshold-conflict blockers.

## No Post Hoc Decisions

The decision generator must not:

- invent product evidence
- use synthetic smoke rows for product direction
- hand-edit generated outcomes
- cherry-pick slices after seeing results
- change priority slice classes after seeing results
- tune thresholds on current evaluation results
- select a preferred product direction when thresholds conflict

## Required Decision Memo Fields

A generated decision memo must include:

- run ID
- evidence completion status
- engineering completion status
- trace counts by slice
- backend summary by primary priority slice
- backend summary by secondary slice
- threshold pass/fail table
- judge disagreement summary
- blocker list when applicable
- exactly one allowed outcome or a declared blocker state

Smoke decision memos must be named distinctly, such as `docs/results/30_DAY_DECISION.synthetic.md`, and must display the smoke warning.
