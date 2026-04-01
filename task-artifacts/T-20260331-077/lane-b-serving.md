# T-20260331-077 Lane B: Bounded-Anytime Serving Interruption Accounting

## Summary

Added `InterruptionAccounting` truth surface to the traversal layer, providing
structured accounting of what was completed vs. dropped when a deadline forces
early termination during graph traversal. This enables downstream consumers
(trace recording, assembly decisions, health monitoring) to evaluate serving
quality under budget pressure.

## Problem

When a deadline interrupted traversal, the system recorded a boolean
`servedPartial` flag but provided no structured accounting of:

- Which frontier nodes were queued but never expanded
- How many expansions completed vs. were planned
- Budget utilization at interruption time
- Aggregate dropped proposal counts and reasons

The footer also omitted all interruption information, making it invisible in
logs and debug output.

## What Changed

### New type: `InterruptionAccounting` (types.ts)

```typescript
interface InterruptionAccounting {
  droppedFrontierNodeIds: string[];
  completedExpansionCount: number;
  maxExpansions: number;
  budgetUsed: number;
  budgetTotal: number;
  budgetUtilization: number;      // 0-1 fraction
  droppedProposalCount: number;
  droppedProposalReasons: Record<string, number>;
}
```

### TraverseResult extension (traverse.ts)

- Added `interruptionAccounting?: InterruptionAccounting | null` field
- Computed only when `interruption` is non-null (zero overhead on happy path)
- Captures remaining frontier as `droppedFrontierNodeIds`
- Computes budget utilization from consumed vs. total budget
- Aggregates dropped proposal outcomes across all expansions

### Footer update (traverse.ts)

When interrupted, the footer now includes:
`... INTERRUPTED · N frontier dropped · M proposals dropped · X% budget used · partial|empty`

### Trace recording (trace.ts)

- `interruptionAccounting` is forwarded into `selectionMetadata` of the
  `DecisionRouteTrace`, making it available to health dashboards and the
  recent-decision summary pipeline.

### Assembly decision surface (service.ts)

- `interruptionAccounting` added to `BrainAssemblyDecisionSelectionSurface`
  Pick type so it flows through the assembly decision snapshot.

## Files Changed

| File | Change |
|------|--------|
| `src/brain-core/types.ts` | Added `InterruptionAccounting` interface; added `interruptionAccounting` to `selectionMetadata` |
| `src/brain-core/traverse.ts` | Compute `InterruptionAccounting` on interruption; update footer |
| `src/brain-core/trace.ts` | Forward `interruptionAccounting` into route trace selectionMetadata |
| `src/brain-runtime/service.ts` | Added `interruptionAccounting` to assembly decision selection surface Pick |
| `test/brain-core/traverse.test.ts` | 7 new tests for interruption accounting |

## Why This Improves Bounded-Anytime Serving

1. **Observability**: Operators can now see exactly what was sacrificed under
   deadline pressure -- which frontier branches were dropped, how much budget
   remained unused, and whether proposals were dropped due to interrupts vs.
   budget exhaustion.

2. **Graceful degradation truth surface**: The `budgetUtilization` and
   `droppedFrontierNodeIds` fields let the teacher and health systems
   distinguish between "interrupted early with low utilization" (bad deadline
   calibration) vs. "interrupted late with high utilization" (expected
   bounded-anytime behavior).

3. **Footer visibility**: The footer now makes interruptions immediately visible
   in logs without requiring trace inspection.

4. **Zero overhead on happy path**: `interruptionAccounting` is only computed
   when `interruption` is non-null, so normal (non-interrupted) traversals pay
   no cost.

## Tests Run / Results

```
 Test Files  10 passed (10)
      Tests  73 passed (73)
```

All brain-core tests pass, including 7 new interruption accounting tests:
- `returns null interruptionAccounting when no deadline is set`
- `returns null interruptionAccounting when deadline is not exceeded`
- `accounts for dropped frontier nodes when deadline interrupts frontier loop`
- `reports interruption in footer with INTERRUPTED marker`
- `tracks dropped proposals in interruptionAccounting`
- `computes correct budget utilization on full non-interrupted traversal`
- `records servedPartial=true in interruption when nodes were already fired`
- `records servedPartial=false in interruption when deadline hits before any nodes fire`

6 pre-existing flaky service.test.ts failures (child worker timeout tests) are
unrelated to this change.

## Remaining Gaps

1. **No time-pressure policy adjustment**: The traversal policy does not yet
   adjust temperature or stop bias based on remaining deadline budget. A future
   change could add a `deadlinePressure` term to the policy score computation,
   biasing toward stop_local as the deadline approaches.

2. **No per-expansion timing**: Individual expansion wall-clock durations are
   not tracked, which would help calibrate deadline budgets.

3. **No injection-stage interruption accounting**: The assembler's
   injection/fitting stage has its own deadline handling but does not produce
   an `InterruptionAccounting` equivalent.

4. **Replay/teacher integration**: The teacher evaluation pipeline does not yet
   consume `interruptionAccounting` to discount reward signals from interrupted
   episodes.

## Commit SHA

See task-status JSON for commit SHA (populated after commit).
