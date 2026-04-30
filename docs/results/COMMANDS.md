# Results Command Surface

PR4 adds the eval harness command surface used by later scoreboard lanes.

| Canonical command | Status | Notes |
|---|---:|---|
| `pnpm ocb:eval:run` | implemented | Runs all four eval backends against fixture-backed traces and writes `ledger-draft.jsonl`. |
| `pnpm ocb:eval:make-blind-packets` | implemented | Generates label-hidden blind judge packets from an eval run. |

Smoke fixtures in this lane are labeled:

```text
NOT PRODUCT EVIDENCE
SYNTHETIC PIPELINE VALIDATION ONLY
```
