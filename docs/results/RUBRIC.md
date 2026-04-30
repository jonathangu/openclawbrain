# Results Rubric

This is the canonical scoring contract for the OpenClawBrain V5 evidence scoreboard.

## Evidence Warning

Smoke, fixture, synthetic, repo-derived, or adversarial validation data must display:

```text
NOT PRODUCT EVIDENCE
SYNTHETIC PIPELINE VALIDATION ONLY
```

Synthetic rows may validate engineering flow only. They must not support product claims, backend wins, final product decisions, or `evidence_e2e_complete=true`.

## Completion Gates

- Engineering E2E may complete on clearly labeled smoke data.
- Evidence E2E requires 40 admitted real redacted traces, required slice coverage, complete provenance/privacy metadata, all four backend runs, complete judging, a judged ledger, generated results, and generated `docs/results/30_DAY_DECISION.md`.
- If real evidence is unavailable, the correct output is blocker artifacts with `evidence_e2e_complete=false`.

## Core Formulas

```text
activation_utility = quality_delta - harm_delta - cost_penalty
```

```text
net_task_utility =
  activation_utility                         if memory_fired
  - abstention_regret_penalty                if memory_did_not_fire_but_should_have
  0                                          if correct_abstention
```

Default abstention regret penalty:

```text
abstention_regret_penalty = abstention_regret * 0.5
```

Product decisions use `net_task_utility_all_traces`, not fire-conditioned `activation_utility_when_fired` alone.

## Quality Delta

`quality_delta` measures user-visible answer improvement only. It must not include memory-related harm or cost.

| Field | Meaning | Range |
|---|---|---:|
| `correctness_delta` | Did factual or procedural correctness improve? | -2..+2 |
| `usefulness_delta` | Did practical usefulness improve? | -2..+2 |
| `specificity_delta` | Did relevant prior context make the answer less generic? | -1..+1 |

```text
raw_quality_delta = correctness_delta + usefulness_delta + specificity_delta
```

Normalize `raw_quality_delta` as follows:

| Raw total | `normalized_quality_delta` |
|---:|---:|
| <= -4 | -2 |
| -3..-2 | -1 |
| -1..+1 | 0 |
| +2..+3 | +1 |
| >= +4 | +2 |

`quality_delta` must equal `normalized_quality_delta` unless a future schema version explicitly deprecates one field.

## Harm Delta

`harm_delta` is memory-related harm only. Do not use it for general answer quality.

| Score | Meaning |
|---:|---|
| 0 | No memory-related harm |
| 1 | Mild distraction or unnecessary context |
| 2 | Wrong, stale, misleading, or confusing context |
| 3 | Serious unsafe, creepy, or trust-breaking memory use |

A lower-authority memory overriding higher-authority current instruction, correction, or evidence must be scored as harm.

## Cost Penalty

`cost_penalty` must be measured or pre-bucketed. It must be backed by raw cost fields.

| Score | Meaning |
|---:|---|
| 0 | Negligible |
| 0.25 | Small |
| 0.5 | Noticeable |
| 1 | High relative to gain |

Required raw cost fields:

- `input_tokens`
- `output_tokens`
- `memory_tokens`
- `latency_ms`
- `estimated_cost_usd`
- `cost_measurement_mode`

## Net Task Utility

```text
if memory_fired:
  net_task_utility = activation_utility
elif should_have_fired:
  net_task_utility = -1 * abstention_regret * 0.5
else:
  net_task_utility = 0
```

This compares conservative and aggressive backends fairly. Correct abstention receives zero, not a positive reward.

## Abstention Regret

Applies only when `memory_fired=false`.

| Score | Meaning |
|---:|---|
| 0 | Correct abstention / no useful memory available |
| 1 | Small missed improvement |
| 2 | Clear missed improvement |
| 3 | Major missed correction, preference, or workflow |

## False Fire

A false fire occurs when:

```text
memory_fired = true
should_have_fired = false
```

False-fire severity is captured by `harm_delta` and judge notes.

## Required Reporting

Every backend and slice must report:

- sample count
- mean and median activation utility when fired
- mean and median net task utility across all traces
- positive net utility rate
- negative net utility rate
- fire rate
- false-fire rate
- false-fire harm mean
- abstention regret mean
- stale-memory-conflict harm rate
- cost penalty mean
- cost per utility point
- judge disagreement rate
- bootstrap or confidence interval when possible

No product claim may rely on a single blended headline without per-slice tables.
