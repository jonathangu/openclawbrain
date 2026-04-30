# Baseline Definitions

This is the canonical V5 ablation baseline contract.

## Evidence Warning

Smoke, fixture, synthetic, repo-derived, or adversarial validation data must display:

```text
NOT PRODUCT EVIDENCE
SYNTHETIC PIPELINE VALIDATION ONLY
```

Synthetic baseline runs may validate harness behavior only. They cannot establish backend superiority.

## Ablation Ladder

Every admitted trace must be run through the same four backends:

1. `none`
2. `correction-only`
3. `correction+heuristics`
4. `full-ocb`

All backends must use the same trace input, model ID unless explicitly testing model variation, prompt harness, tool fixtures, evaluation harness commit, and memory snapshot where applicable.

## `none`

`none` means:

- no OpenClawBrain memory activation
- normal in-session context remains available
- normal user-provided prompt context remains available
- normal tool availability remains available
- no extra hidden memory/context injection

`none` must not be crippled. It is the fair no-memory baseline.

## `correction-only`

Eligible memory/context inputs:

- explicit user correction
- explicit stable preference
- explicit instruction to remember
- explicit override of prior behavior

Not eligible:

- inferred preference
- weak semantic association
- graph neighbor
- teacher guess
- workflow hint unless explicitly corrected or preferred by user

## `correction+heuristics`

Eligible memory/context inputs:

- all `correction-only` items
- deterministic recency/authority rules
- exact project/person/task match
- stale-memory avoidance
- conservative STOP behavior
- small approved workflow-hint whitelist

Heuristics must be deterministic and documented before production runs.

## `full-ocb`

Eligible memory/context inputs:

- current full learned route path
- graph/context pack selection
- workflow/tool hints
- STOP_LOCAL
- attribution

`full-ocb` must not receive runtime architecture changes during V5 scoreboard construction.

## Fairness Rules

- Do not degrade a baseline to make another backend win.
- Do not change prompt, tool, or model conditions across backends unless the experiment explicitly tests that variable.
- Do not tune memory thresholds on current evaluation results.
- Do not use generated results as hand-edited evidence.
- Backend labels must be hidden during blind quality judging.
