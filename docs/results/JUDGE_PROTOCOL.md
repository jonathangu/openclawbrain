# Judge Protocol

This is the canonical judging contract for OpenClawBrain V5 results.

## Evidence Warning

Smoke, fixture, synthetic, repo-derived, or adversarial validation data must display:

```text
NOT PRODUCT EVIDENCE
SYNTHETIC PIPELINE VALIDATION ONLY
```

Synthetic judging may validate packet generation and import only. It cannot produce product evidence, backend wins, or final product decisions.

## Required Backends

Every admitted trace must be run through the same ablation ladder:

1. `none`
2. `correction-only`
3. `correction+heuristics`
4. `full-ocb`

All backends use the same trace input, model ID unless explicitly testing model variation, prompt harness, memory snapshot where applicable, tool fixtures, and evaluation harness commit.

## Blind Judging Flow

Blind randomized judging is required wherever feasible.

For each trace:

1. Run all four backends.
2. Remove backend labels from outputs.
3. Randomize output order per trace.
4. Present outputs as `Output A`, `Output B`, `Output C`, and `Output D`.
5. Judges score quality without backend identity.
6. Conduct labeled audit scoring for memory harm and firing behavior.
7. Reveal backend identity only during summary generation.

## Judging Modes

| Mode | Purpose |
|---|---|
| `blind_quality` | Score user-visible quality deltas without backend identity. |
| `labeled_harm_audit` | Score memory firing, false fire, stale-memory conflict, and memory-related harm. |
| `cost_audit` | Validate cost penalty against raw cost fields. |
| `synthetic_smoke` | Validate pipeline mechanics only; not product evidence. |

## Judge Inputs

Judge packets must include only the fields needed for the assigned mode. Blind quality packets must not reveal backend identity, activation labels, memory IDs, or other backend-identifying metadata.

Labeled audit packets may include backend identity and memory behavior only after blind quality scoring is complete or through a separately versioned audit packet.

## Required Scores

Judges must provide the fields required by `docs/results/RUBRIC.md` and `docs/results/LEDGER_SCHEMA.md`, including:

- `correctness_delta`
- `usefulness_delta`
- `specificity_delta`
- `harm_delta`
- `cost_penalty`
- `abstention_regret`
- `judge_mode`
- `judge_notes`
- `judge_id`

Derived fields such as `activation_utility` and `net_task_utility` must be computed by the results pipeline, not manually trusted.

## Disagreement Threshold

Pause or re-judge if judges disagree on utility sign in more than 30% of judged backend outputs.

Utility sign categories are:

- positive
- neutral
- negative

Reports must include:

- mean absolute utility disagreement
- per-slice disagreement rate
- examples of high-disagreement traces

If the threshold is exceeded, product decisions must not be finalized. The decision output must choose a blocker or pause outcome.

## Non-Fabrication Rules

Judges and agents must not invent:

- real user traces
- provenance
- privacy approval
- judge scores
- cost measurements
- model IDs
- memory snapshots
- backend wins
- product evidence
- final product decisions from nonexistent data

Missing production judgments must fail closed and produce blocker artifacts.
