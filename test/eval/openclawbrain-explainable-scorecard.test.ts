import { describe, expect, it } from "vitest";
import {
  OPENCLAWBRAIN_EXPLAINABLE_EVAL_SCORECARD_CONTRACT,
  buildOpenClawBrainExplainableEvalScorecard,
  isOpenClawBrainExplainableEvalScorecard,
} from "../../src/eval/openclawbrain-explainable-scorecard.js";

describe("openclawbrain explainable eval scorecard", () => {
  it("separates public/operator metrics from internal diagnostics with explicit formula surfaces", () => {
    const scorecard = buildOpenClawBrainExplainableEvalScorecard({
      generatedAt: "2026-04-12T12:00:00.000Z",
      manifestId: "explainable-fixture",
      manifestContract: "frozen_recorded_session_eval_manifest.v1",
      modeOrder: ["no_brain", "vector_only", "graph_prior_only", "learned_route"],
      requestedTraceCount: 2,
      successfulTraceCount: 2,
      failedTraceCount: 0,
      modes: [
        {
          mode: "no_brain",
          traceCount: 2,
          rankedWinnerCount: 0,
          sharedTopScoreTraceCount: 0,
          meanQualityScore: 65,
          totalCompileOkCount: 3,
          totalTurnCount: 4,
          compileOkRate: 0.75,
          totalPhraseHitCount: 3,
          totalPhraseCount: 4,
          phraseHitRate: 0.75,
          totalPromotionCount: 0,
          totalUsedLearnedRouteTurnCount: 0,
          totalWarningCount: 0,
          totalSelectedContextBlockCount: 2,
          totalSelectedContextChars: 24,
          estimatedPromptTokens: 6,
          estimatedPromptCostUsd: 0.00003,
        },
        {
          mode: "vector_only",
          traceCount: 2,
          rankedWinnerCount: 0,
          sharedTopScoreTraceCount: 0,
          meanQualityScore: 72,
          totalCompileOkCount: 4,
          totalTurnCount: 4,
          compileOkRate: 1,
          totalPhraseHitCount: 3,
          totalPhraseCount: 4,
          phraseHitRate: 0.75,
          totalPromotionCount: 0,
          totalUsedLearnedRouteTurnCount: 0,
          totalWarningCount: 0,
          totalSelectedContextBlockCount: 4,
          totalSelectedContextChars: 40,
          estimatedPromptTokens: 10,
          estimatedPromptCostUsd: 0.00005,
        },
        {
          mode: "graph_prior_only",
          traceCount: 2,
          rankedWinnerCount: 1,
          sharedTopScoreTraceCount: 1,
          meanQualityScore: 84,
          totalCompileOkCount: 4,
          totalTurnCount: 4,
          compileOkRate: 1,
          totalPhraseHitCount: 4,
          totalPhraseCount: 4,
          phraseHitRate: 1,
          totalPromotionCount: 0,
          totalUsedLearnedRouteTurnCount: 0,
          totalWarningCount: 0,
          totalSelectedContextBlockCount: 6,
          totalSelectedContextChars: 48,
          estimatedPromptTokens: 12,
          estimatedPromptCostUsd: 0.00006,
        },
        {
          mode: "learned_route",
          traceCount: 2,
          rankedWinnerCount: 1,
          sharedTopScoreTraceCount: 2,
          meanQualityScore: 86,
          totalCompileOkCount: 4,
          totalTurnCount: 4,
          compileOkRate: 1,
          totalPhraseHitCount: 4,
          totalPhraseCount: 4,
          phraseHitRate: 1,
          totalPromotionCount: 1,
          totalUsedLearnedRouteTurnCount: 4,
          totalWarningCount: 0,
          totalSelectedContextBlockCount: 8,
          totalSelectedContextChars: 64,
          estimatedPromptTokens: 16,
          estimatedPromptCostUsd: 0.00008,
        },
      ],
      pairwise: [
        {
          leftMode: "no_brain",
          rightMode: "vector_only",
          comparableTraceCount: 2,
          comparableTurnCount: 4,
          traceWins: { left: 0, right: 2, ties: 0, leftRate: 0, rightRate: 1, tieRate: 0 },
          traceTieOrBetter: { left: 0, right: 2, leftRate: 0, rightRate: 1 },
          turnWins: { left: 0, right: 4, ties: 0, leftRate: 0, rightRate: 1, tieRate: 0 },
          turnTieOrBetter: { left: 0, right: 4, leftRate: 0, rightRate: 1 },
          aggregateDeltas: {
            qualityScoreDeltaLeftMinusRightSum: -14,
            qualityScoreDeltaLeftMinusRightMean: -7,
            compileOkDeltaLeftMinusRightSum: -1,
            phraseHitDeltaLeftMinusRightSum: 0,
            promotionDeltaLeftMinusRightSum: 0,
            tiePromotionDeltaLeftMinusRightSum: 0,
          },
        },
        {
          leftMode: "no_brain",
          rightMode: "graph_prior_only",
          comparableTraceCount: 2,
          comparableTurnCount: 4,
          traceWins: { left: 0, right: 2, ties: 0, leftRate: 0, rightRate: 1, tieRate: 0 },
          traceTieOrBetter: { left: 0, right: 2, leftRate: 0, rightRate: 1 },
          turnWins: { left: 0, right: 4, ties: 0, leftRate: 0, rightRate: 1, tieRate: 0 },
          turnTieOrBetter: { left: 0, right: 4, leftRate: 0, rightRate: 1 },
          aggregateDeltas: {
            qualityScoreDeltaLeftMinusRightSum: -38,
            qualityScoreDeltaLeftMinusRightMean: -19,
            compileOkDeltaLeftMinusRightSum: -1,
            phraseHitDeltaLeftMinusRightSum: -1,
            promotionDeltaLeftMinusRightSum: 0,
            tiePromotionDeltaLeftMinusRightSum: 0,
          },
        },
        {
          leftMode: "no_brain",
          rightMode: "learned_route",
          comparableTraceCount: 2,
          comparableTurnCount: 4,
          traceWins: { left: 0, right: 2, ties: 0, leftRate: 0, rightRate: 1, tieRate: 0 },
          traceTieOrBetter: { left: 0, right: 2, leftRate: 0, rightRate: 1 },
          turnWins: { left: 0, right: 4, ties: 0, leftRate: 0, rightRate: 1, tieRate: 0 },
          turnTieOrBetter: { left: 0, right: 4, leftRate: 0, rightRate: 1 },
          aggregateDeltas: {
            qualityScoreDeltaLeftMinusRightSum: -42,
            qualityScoreDeltaLeftMinusRightMean: -21,
            compileOkDeltaLeftMinusRightSum: -1,
            phraseHitDeltaLeftMinusRightSum: -1,
            promotionDeltaLeftMinusRightSum: -1,
            tiePromotionDeltaLeftMinusRightSum: 0,
          },
        },
        {
          leftMode: "vector_only",
          rightMode: "graph_prior_only",
          comparableTraceCount: 2,
          comparableTurnCount: 4,
          traceWins: { left: 0, right: 2, ties: 0, leftRate: 0, rightRate: 1, tieRate: 0 },
          traceTieOrBetter: { left: 0, right: 2, leftRate: 0, rightRate: 1 },
          turnWins: { left: 0, right: 4, ties: 0, leftRate: 0, rightRate: 1, tieRate: 0 },
          turnTieOrBetter: { left: 0, right: 4, leftRate: 0, rightRate: 1 },
          aggregateDeltas: {
            qualityScoreDeltaLeftMinusRightSum: -24,
            qualityScoreDeltaLeftMinusRightMean: -12,
            compileOkDeltaLeftMinusRightSum: 0,
            phraseHitDeltaLeftMinusRightSum: -1,
            promotionDeltaLeftMinusRightSum: 0,
            tiePromotionDeltaLeftMinusRightSum: 0,
          },
        },
        {
          leftMode: "vector_only",
          rightMode: "learned_route",
          comparableTraceCount: 2,
          comparableTurnCount: 4,
          traceWins: { left: 0, right: 2, ties: 0, leftRate: 0, rightRate: 1, tieRate: 0 },
          traceTieOrBetter: { left: 0, right: 2, leftRate: 0, rightRate: 1 },
          turnWins: { left: 0, right: 4, ties: 0, leftRate: 0, rightRate: 1, tieRate: 0 },
          turnTieOrBetter: { left: 0, right: 4, leftRate: 0, rightRate: 1 },
          aggregateDeltas: {
            qualityScoreDeltaLeftMinusRightSum: -28,
            qualityScoreDeltaLeftMinusRightMean: -14,
            compileOkDeltaLeftMinusRightSum: 0,
            phraseHitDeltaLeftMinusRightSum: -1,
            promotionDeltaLeftMinusRightSum: -1,
            tiePromotionDeltaLeftMinusRightSum: 0,
          },
        },
        {
          leftMode: "graph_prior_only",
          rightMode: "learned_route",
          comparableTraceCount: 2,
          comparableTurnCount: 4,
          traceWins: { left: 0, right: 1, ties: 1, leftRate: 0, rightRate: 0.5, tieRate: 0.5 },
          traceTieOrBetter: { left: 1, right: 2, leftRate: 0.5, rightRate: 1 },
          turnWins: { left: 0, right: 2, ties: 2, leftRate: 0, rightRate: 0.5, tieRate: 0.5 },
          turnTieOrBetter: { left: 2, right: 4, leftRate: 0.5, rightRate: 1 },
          aggregateDeltas: {
            qualityScoreDeltaLeftMinusRightSum: -2,
            qualityScoreDeltaLeftMinusRightMean: -1,
            compileOkDeltaLeftMinusRightSum: 0,
            phraseHitDeltaLeftMinusRightSum: 0,
            promotionDeltaLeftMinusRightSum: -1,
            tiePromotionDeltaLeftMinusRightSum: -1,
          },
        },
      ],
      traces: [
        {
          traceId: "trace-a",
          status: "ok",
          validationOk: true,
          winnerMode: "learned_route",
          topScoreModes: ["learned_route"],
          scoreSpread: 12,
          modes: [
            {
              mode: "no_brain",
              qualityScore: 60,
              compileOkCount: 1,
              turnCount: 2,
              phraseHitCount: 1,
              phraseCount: 2,
              selectedContextBlockCount: 1,
              selectedContextChars: 10,
              estimatedPromptTokens: 3,
              estimatedPromptCostUsd: 0.000015,
            },
            {
              mode: "vector_only",
              qualityScore: 74,
              compileOkCount: 2,
              turnCount: 2,
              phraseHitCount: 1,
              phraseCount: 2,
              selectedContextBlockCount: 2,
              selectedContextChars: 18,
              estimatedPromptTokens: 5,
              estimatedPromptCostUsd: 0.000025,
            },
            {
              mode: "graph_prior_only",
              qualityScore: 84,
              compileOkCount: 2,
              turnCount: 2,
              phraseHitCount: 2,
              phraseCount: 2,
              selectedContextBlockCount: 3,
              selectedContextChars: 22,
              estimatedPromptTokens: 6,
              estimatedPromptCostUsd: 0.00003,
            },
            {
              mode: "learned_route",
              qualityScore: 90,
              compileOkCount: 2,
              turnCount: 2,
              phraseHitCount: 2,
              phraseCount: 2,
              selectedContextBlockCount: 4,
              selectedContextChars: 30,
              estimatedPromptTokens: 8,
              estimatedPromptCostUsd: 0.00004,
            },
          ],
        },
        {
          traceId: "trace-b",
          status: "ok",
          validationOk: true,
          winnerMode: "graph_prior_only",
          topScoreModes: ["graph_prior_only", "learned_route"],
          scoreSpread: 8,
          modes: [
            {
              mode: "no_brain",
              qualityScore: 70,
              compileOkCount: 2,
              turnCount: 2,
              phraseHitCount: 2,
              phraseCount: 2,
              selectedContextBlockCount: 1,
              selectedContextChars: 14,
              estimatedPromptTokens: 3,
              estimatedPromptCostUsd: 0.000015,
            },
            {
              mode: "vector_only",
              qualityScore: 70,
              compileOkCount: 2,
              turnCount: 2,
              phraseHitCount: 2,
              phraseCount: 2,
              selectedContextBlockCount: 2,
              selectedContextChars: 22,
              estimatedPromptTokens: 5,
              estimatedPromptCostUsd: 0.000025,
            },
            {
              mode: "graph_prior_only",
              qualityScore: 84,
              compileOkCount: 2,
              turnCount: 2,
              phraseHitCount: 2,
              phraseCount: 2,
              selectedContextBlockCount: 3,
              selectedContextChars: 26,
              estimatedPromptTokens: 6,
              estimatedPromptCostUsd: 0.00003,
            },
            {
              mode: "learned_route",
              qualityScore: 84,
              compileOkCount: 2,
              turnCount: 2,
              phraseHitCount: 2,
              phraseCount: 2,
              selectedContextBlockCount: 4,
              selectedContextChars: 34,
              estimatedPromptTokens: 8,
              estimatedPromptCostUsd: 0.00004,
            },
          ],
        },
      ],
      notes: ["qualityScore is a deterministic replay proxy"],
    });

    expect(isOpenClawBrainExplainableEvalScorecard(scorecard)).toBe(true);
    expect(scorecard.contract).toBe(OPENCLAWBRAIN_EXPLAINABLE_EVAL_SCORECARD_CONTRACT);
    expect(scorecard.publicOperatorMetrics.length).toBeGreaterThan(0);
    expect(scorecard.internalMetrics.length).toBeGreaterThan(0);
    expect(scorecard.traceSuccessProxy.id).toBe("validated_replay_trace_success_proxy");
    expect(scorecard.diagnosticLanguage).toContain("qualityScore and winnerMode");

    const regressionMetric = scorecard.publicOperatorMetrics.find((metric) => metric.id === "brain_on_regression_rate_vs_prior");
    expect(regressionMetric?.audience).toBe("public_operator");
    expect(regressionMetric?.value).toBe(0);
    expect(regressionMetric?.formula.components[0]?.value).toBe(0);
    expect(regressionMetric?.formula.components[1]?.value).toBe(2);

    const tieMetric = scorecard.publicOperatorMetrics.find((metric) => metric.id === "tie_or_better_rate_vs_prior");
    expect(tieMetric?.value).toBe(1);
    expect(tieMetric?.formula.expression).toContain("betterThanPriorCount + tiedWithPriorCount");

    const promptCostDeltaMetric = scorecard.publicOperatorMetrics.find(
      (metric) => metric.id === "estimated_prompt_cost_per_successful_trace_delta_vs_prior",
    );
    expect(promptCostDeltaMetric?.value).toBeCloseTo(0.00001, 8);
    expect(promptCostDeltaMetric?.formula.components).toHaveLength(4);

    const safeFallbackMetric = scorecard.publicOperatorMetrics.find((metric) => metric.id === "safe_fallback_rate");
    expect(safeFallbackMetric?.availability).toBe("not_available");
    expect(safeFallbackMetric?.value).toBeNull();
    expect(safeFallbackMetric?.formula.components[0]?.value).toBeNull();
    expect(scorecard.failOpenLanguage).toContain("does not prove live safe-fallback");

    const brainDisabledMetric = scorecard.publicOperatorMetrics.find(
      (metric) => metric.id === "brain_disabled_comparable_success_rate",
    );
    expect(brainDisabledMetric?.availability).toBe("proxy");
    expect(brainDisabledMetric?.value).toBe(0.5);

    const qualityMetric = scorecard.internalMetrics.find((metric) => metric.id === "diagnostic_quality_score_mean_by_mode");
    expect(qualityMetric?.audience).toBe("internal");
    expect(qualityMetric?.kind).toBe("scalar");

    const winnerMetric = scorecard.internalMetrics.find((metric) => metric.id === "diagnostic_ranked_winner_count_by_mode");
    expect(winnerMetric?.audience).toBe("internal");
    expect(winnerMetric?.language).toContain("winnerMode");
  });
});
