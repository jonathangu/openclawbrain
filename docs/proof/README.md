# Proof packaging

This page maps the repo-side proof surfaces for the current OpenClawBrain agenda.

Current shipped proof is about operator truth and bounded selective-intervention lanes. Future proof surfaces should extend that ladder deliberately; unique wins and ties are not interchangeable.

## Shipped proof surfaces

| Surface | What it proves today | Boundary |
| --- | --- | --- |
| `openclawbrain status --openclaw-home <path> --detailed` | live runtime truth for one selected OpenClaw home | runtime / load / status truth only |
| `openclawbrain proof --openclaw-home <path>` | durable operator proof bundle for that home | install / runtime / reporting truth, not a blanket quality claim |
| `docs/evidence/YYYY-MM-DD/<git-sha>/` | frozen checked bundles | only the exact claims named in each bundle summary |
| `artifacts/activation-first-gating-retune/T-20260419-269/scorecard.json` | bounded activation-first scorecard with unique wins, ties, restraint counts, and regressions | not a broad live answer-quality claim |
| `artifacts/activation-first-gating-retune/T-20260419-269/broad-live-comparative-eval/summary.md` | guardrail replay bundle for the checked broad-live lane | ties and no-regression counts are guardrails, not product wins |
| `scripts/verify-proof-smoke.mjs` | proof-freshness gate for the frozen public lane | smoke boundary, not a full rerun of every proof lane |

## Next proof surfaces to add

- a later-preference current-choice fidelity bundle on the real runtime path
- a second restraint or concrete-specificity bundle that pairs a positive recovery with a must-not-fire keep
- a tool-capability choice bundle only after the first two lanes are frozen

## Derived and target surfaces

- Teacher v3 proposal bundles remain derived review surfaces, not shipped authority.
- Docs-only scaffolds and examples are packaging aids, not proof.
- If a target bundle is seeded from another surface because persistence is incomplete, that seam must be explicit in the bundle summary.

## Honest boundary

Shipped truth comes from runtime status, operator proof bundles, frozen evidence bundles, and named checked scorecards. Anything else should stay labeled as target, review-only, or example.
