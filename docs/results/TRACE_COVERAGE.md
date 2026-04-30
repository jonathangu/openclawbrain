# Trace Coverage — PR3 Smoke Validator

Status: engineering smoke fixture only.

```text
NOT PRODUCT EVIDENCE
SYNTHETIC PIPELINE VALIDATION ONLY
```

## Command surface

- `pnpm ocb:traces:validate` validates the smoke trace manifest.
- `pnpm ocb:traces:validate:smoke` is the explicit smoke-mode equivalent.
- `pnpm ocb:traces:validate:production` is expected to fail until at least 40 admitted real product-evidence traces exist.
- Direct usage: `node scripts/traces/validate.mjs --mode smoke|production --manifest eval/traces/manifest.json`.

## Smoke trace inventory

| Trace ID | V5 slice | Priority | Product evidence? |
|---|---|---|---|
| `trace-001-direct-answer` | `direct-answer` | secondary | No |
| `trace-002-continuation` | `continuation` | primary | No |
| `trace-003-correction-follow-up` | `correction-follow-up` | primary | No |
| `trace-004-retrieval-heavy` | `retrieval-heavy` | secondary | No |
| `trace-005-tool-heavy` | `tool-heavy` | secondary | No |
| `trace-006-stale-memory-conflict` | `stale-memory-conflict` | primary | No |

## Production gate

Production validation fails closed unless the manifest contains at least 40 admitted real privacy-scrubbed product-evidence traces, with V5 slice minimums:

- `direct-answer`: 6
- `continuation`: 6
- `correction-follow-up`: 8
- `retrieval-heavy`: 6
- `tool-heavy`: 6
- `stale-memory-conflict`: 8

The current PR3 manifest intentionally contains 0 admitted real traces, so production validation must fail and `evidence_e2e_complete` remains `false`.
