# OpenClawBrain V5 Completion Map

Source plan: `docs/EXECUTION_PLAN_V5.md`
Date adopted: 2026-04-29
Owner: GUCLAW principal orchestrator

## Prime directive

Build the scoreboard before the cathedral.

OpenClawBrain 1.0 is complete only when we can honestly answer whether memory activation improves user-visible outcomes more than it harms, distracts, or costs.

## Non-fabrication invariant

The engineering pipeline must finish, but evidence must never be faked.

Agents must not invent real traces, provenance, privacy approval, judge scores, cost measurements, model IDs, memory snapshots, backend wins, product evidence, or product decisions.

## Completion gates

### Gate 1 — Engineering E2E
Complete when smoke-mode machinery works end to end:

- `docs/results/*` contracts exist
- ledger schema validates rows
- trace manifest validates coverage/provenance
- all four backends run uniformly
- blind judge packets are generated
- judged ledger can be imported
- `/results` is generated from ledger data
- threshold-derived decision memo is generated
- `RUN_STATE.json` is written
- tests pass
- `pnpm ocb:e2e:smoke` completes

Smoke outputs must be labeled:

```text
NOT PRODUCT EVIDENCE
SYNTHETIC PIPELINE VALIDATION ONLY
```

### Gate 2 — Evidence E2E
Complete only when real admitted evidence exists:

- 40 real redacted traces
- required slice minimums met
- provenance/privacy metadata present
- all four backends run against all admitted traces
- blind/labeled judging complete
- judged ledger exists
- `/results` regenerated from judged ledger
- thresholds applied
- `docs/results/30_DAY_DECISION.md` produced

If missing, produce blockers and keep `evidence_e2e_complete=false`.

## Required PR lanes

| PR | Lane | Core deliverable | Exit gate |
|---|---|---|---|
| PR1 | Results contract docs | `docs/results/*.md` | contracts match V5; thresholds/slices fixed |
| PR2 | `packages/results-schema` | ledger/rubric/threshold/summary/uncertainty package | schema + derived utility tests pass |
| PR3 | Trace manifest + smoke traces | manifest, 4-8 smoke traces, coverage docs | smoke passes; production fails closed under 40 real traces |
| PR4 | `packages/eval-harness` | run all four backends uniformly | draft ledger produced, eval is read-only/fixture-safe |
| PR5 | Blind judging flow | packets + import | labels hidden, mapping private, missing production judgments fail |
| PR6 | `packages/results-site` | generated `/results` and summary | generated from ledger, low-N/negative/smoke warnings visible |
| PR7 | E2E smoke + production gates | `pnpm ocb:e2e:smoke`, blockers, run state | Engineering E2E true; Evidence E2E false unless real evidence exists |

## Canonical command surface

- `pnpm ocb:results:schema-test`
- `pnpm ocb:traces:validate`
- `pnpm ocb:eval:run`
- `pnpm ocb:eval:make-blind-packets`
- `pnpm ocb:judgments:import`
- `pnpm ocb:ledger:validate`
- `pnpm ocb:results:generate`
- `pnpm ocb:decision:generate`
- `pnpm ocb:e2e:smoke`
- `pnpm ocb:traces:from-session-logs`
- `pnpm ocb:judgments:judge-production`
- `pnpm ocb:e2e:production`

Equivalent command names are allowed only if mapped in `docs/results/COMMANDS.md`.

## Line-of-sight artifacts

- `docs/results/BLOCKERS.md`
- `docs/results/NEXT_DATA_NEEDED.md`
- `docs/results/PARTIAL_COMPLETION.md`
- `eval/results/<run-id>/RUN_STATE.json`
- `docs/results/index.md`
- `docs/results/summary.json`
- `docs/results/30_DAY_DECISION.blocked.md` for smoke/incomplete evidence
- `docs/results/30_DAY_DECISION.md` for production Evidence E2E

## Stop conditions

No new runtime architecture until all are working:

- ledger schema
- trace validation
- ablation harness
- blind judge packets
- judged ledger import
- results generation
- threshold decision logic
- smoke E2E command

## Swarm operating model

Implementation lanes work in isolated git worktrees and produce commits on PR branches. Audit lanes write artifacts only. GUCLAW principal merges/synthesizes in dependency order and verifies with the smallest meaningful gate at each step.
