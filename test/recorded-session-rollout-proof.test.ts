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

  it("the shipped suite still fails the rollout bar even with real recorded sessions because learned route rarely wins cleanly", () => {
    const tracePaths = discoverRecordedSessionReplayTracePaths(path.resolve("docs/evidence"));
    const verdict = evaluateRecordedSessionReplayRollout(tracePaths);

    const traceIds = tracePaths.map((tracePath) => path.basename(path.dirname(tracePath)));
    const uniqueTraceIds = [...new Set(traceIds)].sort();
    const classifications = [...new Set(verdict.traces.map((trace) => trace.evidence.classification))].sort();

    assert.ok(uniqueTraceIds.includes("trace-comparative-replay"));
    assert.ok(uniqueTraceIds.includes("trace-train-freeze-eval"));
    assert.ok(tracePaths.length >= 2);
    assert.equal(verdict.ok, false);
    assert.equal(verdict.totalTraceCount, tracePaths.length);
    assert.ok(verdict.eligibleTraceCount > 0);
    assert.ok(verdict.eligibleTraceCount < tracePaths.length);
    assert.ok(verdict.failureReasons.includes("clean_win_rate_below_bar"));
    assert.ok(verdict.failureReasons.includes("average_margin_vs_vector_only_below_bar"));
    assert.ok(verdict.failureReasons.includes("average_margin_vs_graph_prior_only_below_bar"));
    assert.ok(classifications.includes("non_test_recorded_session"));
    assert.ok(classifications.includes("test_fixture"));
    assert.ok(verdict.traces.every((trace) => trace.validationOk === true));

    const noBrain = verdict.allTraceSummary.baselineAggregates.find(
      (aggregate) => aggregate.baselineMode === "no_brain",
    );
    const vectorOnly = verdict.eligibleTraceSummary.baselineAggregates.find(
      (aggregate) => aggregate.baselineMode === "vector_only",
    );
    const graphPriorOnly = verdict.eligibleTraceSummary.baselineAggregates.find(
      (aggregate) => aggregate.baselineMode === "graph_prior_only",
    );

    assert.ok(noBrain);
    assert.equal(noBrain?.traceCount, tracePaths.length);
    assert.equal(noBrain?.lossCount, 0);

    assert.ok(vectorOnly);
    assert.equal(vectorOnly?.traceCount, verdict.eligibleTraceCount);
    assert.ok((vectorOnly?.lossCount ?? 0) > 0 || (vectorOnly?.averageMargin ?? 0) <= 0);

    assert.ok(graphPriorOnly);
    assert.equal(graphPriorOnly?.traceCount, verdict.eligibleTraceCount);
    assert.ok((graphPriorOnly?.strictWinCount ?? 0) < verdict.eligibleTraceSummary.requiredCleanWinCount);

    assert.ok(verdict.eligibleTraceSummary.cleanWinTraceCount < verdict.eligibleTraceSummary.requiredCleanWinCount);
    assert.ok(verdict.allTraceSummary.learnedWinnerTraceCount >= verdict.allTraceSummary.cleanWinTraceCount);
  });
});
