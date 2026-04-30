# Trace Admission

This is the canonical V5 trace admission contract.

## Evidence Warning

Smoke, fixture, synthetic, repo-derived, or adversarial validation data must display:

```text
NOT PRODUCT EVIDENCE
SYNTHETIC PIPELINE VALIDATION ONLY
```

Synthetic traces may validate Engineering E2E only. They must not count toward the 40 real-trace Evidence E2E requirement.

## Production Trace Requirement

Evidence E2E requires at least 40 admitted real redacted traces.

Required production slice minimums:

| Slice | Minimum traces | Priority class |
|---|---:|---|
| `direct-answer` | 6 | secondary |
| `continuation` | 6 | primary |
| `correction-follow-up` | 8 | primary |
| `retrieval-heavy` | 6 | secondary |
| `tool-heavy` | 6 | secondary |
| `stale-memory-conflict` | 8 | primary |
| Total | 40 | mixed |

These counts are fixed for V5 and must not be changed after seeing results.

## Admission Criteria

A trace may be admitted for production only if all are true:

- provenance metadata exists
- trace is privacy-scrubbed or explicit approval exists
- user-visible task is identifiable
- slice label is assigned before backend runs
- expected memory opportunity or non-opportunity is labeled before backend runs
- trace is not authored purely to favor OCB
- reproducibility metadata can be recorded
- tool-heavy traces are fixture-backed or read-only

## Anti-Gaming Constraints

The admitted trace set must satisfy:

- at least 60% real user/agent behavior not authored by the OCB team
- at least 25% ambiguity, correction, contradiction, or partial failure
- no trace admitted without provenance
- no over-sampling repo/design-doc-adjacent traces
- no excluding ugly traces because OCB may fail them

## Provenance Fields

The manifest must record enough metadata to verify:

- source system
- provenance type
- collection date or time window
- redaction status
- privacy approval when redaction is insufficient
- whether the trace was authored by the OCB team
- slice label and label timestamp
- memory opportunity label and label timestamp
- fixture/read-only status for tool-heavy traces

## Smoke Traces

Smoke mode uses 4–8 explicitly synthetic traces to prove the pipeline works.

Smoke trace metadata must include:

```json
{
  "provenance_type": "synthetic",
  "mode": "smoke",
  "counts_as_product_evidence": false
}
```

Smoke output may set `engineering_e2e_complete=true` but must keep `evidence_e2e_complete=false`.

## Fail-Closed Production Rules

Production mode must fail closed and produce blockers if:

- fewer than 40 admitted real traces exist
- any slice minimum is unmet
- provenance metadata is missing
- `privacy_scrubbed=false` without explicit approval
- tool-heavy traces lack fixtures/read-only mode
- reproducibility metadata is missing
- raw cost fields are missing without `cost_measurement_mode`
- memory snapshot metadata is missing
- blind packets are missing
- judge scores are missing
- judge disagreement threshold is exceeded

Blocker artifacts are honest completion state, not failure.
