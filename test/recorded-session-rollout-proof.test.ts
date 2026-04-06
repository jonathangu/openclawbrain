import assert from "node:assert/strict";
import path from "node:path";
import { describe, it } from "vitest";
import {
  buildRecordedSessionReplayRolloutVerdict,
  discoverRecordedSessionReplayTracePaths,
  evaluateRecordedSessionReplayRollout,
  type RecordedSessionReplayTraceRolloutEvaluation,
} from "../src/recorded-session-rollout-proof.ts";

function relationFromMargin(margin: number): "win" | "tie" | "loss" {
  if (margin > 0) {
    return "win";
  }
  if (margin < 0) {
    return "loss";
  }
  return "tie";
}

function makeTraceEvaluation(
  traceId: string,
  input: {
    classification?: "test_fixture" | "non_test_recorded_session";
    winnerMode?: "learned_route" | "graph_prior_only" | "vector_only" | "no_brain" | null;
    vectorMargin?: number;
    graphMargin?: number;
    noBrainMargin?: number;
  } = {},
): RecordedSessionReplayTraceRolloutEvaluation {
  const vectorMargin = input.vectorMargin ?? 0;
  const graphMargin = input.graphMargin ?? 0;
  const noBrainMargin = input.noBrainMargin ?? 100;
  return {
    traceId,
    tracePath: `/tmp/${traceId}.json`,
    bundleRoot: `/tmp/${traceId}`,
    source: "sanitized_recorded_session",
    privacyNotes: [],
    validationOk: true,
    winnerMode: input.winnerMode ?? "learned_route",
    learnedRouteQualityScore: 100,
    learnedRouteEvalTurnCount: 2,
    learnedRouteUsedLearnedRouteTurnCount: 2,
    evidence: {
      classification: input.classification ?? "non_test_recorded_session",
      reasons: [],
    },
    comparisons: [
      {
        baselineMode: "no_brain",
        learnedRouteQualityScore: 100,
        baselineQualityScore: 100 - noBrainMargin,
        margin: noBrainMargin,
        relation: relationFromMargin(noBrainMargin),
      },
      {
        baselineMode: "vector_only",
        learnedRouteQualityScore: 100,
        baselineQualityScore: 100 - vectorMargin,
        margin: vectorMargin,
        relation: relationFromMargin(vectorMargin),
      },
      {
        baselineMode: "graph_prior_only",
        learnedRouteQualityScore: 100,
        baselineQualityScore: 100 - graphMargin,
        margin: graphMargin,
        relation: relationFromMargin(graphMargin),
      },
    ],
    cleanWinAgainstRetrievalBaselines: vectorMargin > 0 && graphMargin > 0,
  };
}

describe("recorded-session-rollout-proof", () => {
  it("buildRecordedSessionReplayRolloutVerdict passes when learned route wins cleanly on enough non-test traces", () => {
    const verdict = buildRecordedSessionReplayRolloutVerdict([
      makeTraceEvaluation("trace-a", { vectorMargin: 8, graphMargin: 7 }),
      makeTraceEvaluation("trace-b", { vectorMargin: 5, graphMargin: 6 }),
      makeTraceEvaluation("trace-c", { vectorMargin: 9, graphMargin: 5 }),
    ]);

    assert.equal(verdict.ok, true);
    assert.equal(verdict.eligibleTraceCount, 3);
    assert.deepEqual(verdict.failureReasons, []);
    assert.equal(verdict.eligibleTraceSummary.cleanWinTraceCount, 3);
  });

  it("the shipped suite fails because the available replay traces are test fixtures and learned route never wins cleanly", () => {
    const tracePaths = discoverRecordedSessionReplayTracePaths(path.resolve("docs/evidence"));
    const verdict = evaluateRecordedSessionReplayRollout(tracePaths);

    assert.deepEqual(
      tracePaths.map((tracePath) => path.basename(path.dirname(tracePath))).sort(),
      ["trace-comparative-replay", "trace-train-freeze-eval"],
    );
    assert.equal(verdict.ok, false);
    assert.equal(verdict.totalTraceCount, 2);
    assert.equal(verdict.eligibleTraceCount, 0);
    assert.ok(verdict.failureReasons.includes("insufficient_eligible_trace_count"));
    assert.ok(verdict.failureReasons.includes("average_margin_vs_vector_only_below_bar"));
    assert.ok(verdict.failureReasons.includes("average_margin_vs_graph_prior_only_below_bar"));
    assert.ok(verdict.traces.every((trace) => trace.evidence.classification === "test_fixture"));
    assert.ok(verdict.traces.every((trace) => trace.validationOk === true));

    const vectorOnly = verdict.allTraceSummary.baselineAggregates.find(
      (aggregate) => aggregate.baselineMode === "vector_only",
    );
    const graphPriorOnly = verdict.allTraceSummary.baselineAggregates.find(
      (aggregate) => aggregate.baselineMode === "graph_prior_only",
    );

    assert.deepEqual(vectorOnly, {
      baselineMode: "vector_only",
      traceCount: 2,
      strictWinCount: 0,
      tieCount: 2,
      lossCount: 0,
      averageMargin: 0,
      averageLearnedRouteQualityScore: 100,
      averageBaselineQualityScore: 100,
    });
    assert.deepEqual(graphPriorOnly, {
      baselineMode: "graph_prior_only",
      traceCount: 2,
      strictWinCount: 0,
      tieCount: 2,
      lossCount: 0,
      averageMargin: 0,
      averageLearnedRouteQualityScore: 100,
      averageBaselineQualityScore: 100,
    });
    assert.equal(verdict.allTraceSummary.cleanWinTraceCount, 0);
    assert.equal(verdict.allTraceSummary.learnedWinnerTraceCount, 0);
  });
});
