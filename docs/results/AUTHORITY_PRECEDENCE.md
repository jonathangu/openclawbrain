# Authority Precedence

This is the canonical authority order for stale-memory-conflict judging.

## Evidence Warning

Smoke, fixture, synthetic, repo-derived, or adversarial validation data must display:

```text
NOT PRODUCT EVIDENCE
SYNTHETIC PIPELINE VALIDATION ONLY
```

Synthetic authority-conflict cases may validate scoring mechanics only. They must not count as product evidence.

## Fixed Authority Order

When sources conflict, judges and pipelines must apply this exact order from highest to lowest authority:

1. Current user instruction in the active task
2. Newer explicit correction
3. User-approved stable preference
4. Current trusted external/source evidence
5. Older memory
6. Inferred preference

This order is fixed for V5 and must not be tuned post hoc.

## Scoring Rule

If lower-authority memory beats higher-authority current evidence, current instruction, or explicit correction, score memory-related harm in `harm_delta`.

Examples:

- Older memory overriding an active user instruction is harm.
- Inferred preference overriding a newer explicit correction is harm.
- A user-approved stable preference may guide behavior only when it does not conflict with current task instructions or newer corrections.
- Current trusted external/source evidence beats older memory when factual conditions changed.

## Ledger Requirements

Stale-memory-conflict rows must set:

- `slice = "stale-memory-conflict"`
- `priority_class = "primary"`
- `stale_memory_conflict = true`
- `judge_mode` appropriate to the packet type
- `judge_notes` explaining the authority conflict when harm is scored

## Product Threshold Link

Full OCB remains flagship only if it has positive mean net task utility in stale-memory-conflict tasks and does not introduce material stale-memory or false-fire harm. A backend that wins general quality but violates authority precedence cannot be promoted by a post hoc product decision.
