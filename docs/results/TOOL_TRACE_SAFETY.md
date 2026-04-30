# Tool Trace Safety

This is the canonical V5 safety contract for tool-heavy evaluation traces.

## Evidence Warning

Smoke, fixture, synthetic, repo-derived, or adversarial validation data must display:

```text
NOT PRODUCT EVIDENCE
SYNTHETIC PIPELINE VALIDATION ONLY
```

Synthetic tool transcripts may validate pipeline mechanics only. They must not count as product evidence.

## Required Safety Mode

Tool-heavy traces must be fixture-backed or read-only.

Allowed modes:

- mocked tool fixtures
- recorded/replayed tool outputs
- local read-only inspection
- synthetic tool transcripts for smoke mode

## Prohibited Effects

No evaluation run may:

- send email
- change calendars
- modify repos
- charge money
- mutate external state
- post messages
- delete files
- write to production systems

## Admission Requirements

A tool-heavy production trace must record:

- fixture or read-only mode
- tool fixture version when applicable
- source system
- provenance type
- privacy/redaction status
- expected memory opportunity label
- reproducibility metadata

Tool-heavy traces lacking fixtures or read-only mode must fail closed and appear in blocker artifacts.

## Backend Constraints

All backends must receive the same tool fixtures and read-only outputs. A backend must not get privileged tool results, hidden state, or mutable side effects.

## Generated Results

Generated `/results` must identify tool-heavy counts, tool-heavy backend metrics, and any tool fixture blockers. Unsafe tool traces cannot support product proof.
