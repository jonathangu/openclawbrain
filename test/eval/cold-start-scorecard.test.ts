import { describe, expect, it } from "vitest";

import {
  buildColdStartScorecardV1,
  COLD_START_SCORECARD_CONTRACT_V1,
} from "../../src/eval/cold-start-scorecard.js";

function descriptor(modes: Array<{
  mode: "no_brain" | "graph_prior_only" | "learned_route";
  meanQualityScore: number;
  totalPhraseHitCount: number;
  totalPhraseCount: number;
  totalSelectedContextBlockCount: number;
  totalSelectedContextChars: number;
  estimatedPromptTokens: number;
  totalWarningCount?: number;
}>) {
  return {
    outputDir: "/tmp/eval",
    report: { successfulTraceCount: 20 },
    scorecard: {
      modes: modes.map((mode) => ({
        traceCount: 20,
        rankedWinnerCount: 0,
        sharedTopScoreTraceCount: 0,
        totalCompileOkCount: 45,
        totalTurnCount: 45,
        compileOkRate: 1,
        totalPromotionCount: 0,
        ...mode,
        phraseHitRate: mode.totalPhraseCount === 0 ? null : mode.totalPhraseHitCount / mode.totalPhraseCount,
        totalWarningCount: mode.totalWarningCount ?? 0,
        estimatedPromptCostUsd: 0,
      })),
    },
  } as never;
}

describe("cold-start scorecard", () => {
  it("maps baseline and cold-start-prior evals into a frozen four-mode scorecard", () => {
    const baseline = descriptor([
      { mode: "no_brain", meanQualityScore: 0, totalPhraseHitCount: 0, totalPhraseCount: 74, totalSelectedContextBlockCount: 0, totalSelectedContextChars: 0, estimatedPromptTokens: 0 },
      { mode: "graph_prior_only", meanQualityScore: 92.8, totalPhraseHitCount: 65, totalPhraseCount: 74, totalSelectedContextBlockCount: 135, totalSelectedContextChars: 26125, estimatedPromptTokens: 6539 },
      { mode: "learned_route", meanQualityScore: 97.3, totalPhraseHitCount: 71, totalPhraseCount: 74, totalSelectedContextBlockCount: 135, totalSelectedContextChars: 27592, estimatedPromptTokens: 6906 },
    ]);
    const single = descriptor([
      { mode: "no_brain", meanQualityScore: 0, totalPhraseHitCount: 0, totalPhraseCount: 74, totalSelectedContextBlockCount: 0, totalSelectedContextChars: 0, estimatedPromptTokens: 0 },
      { mode: "graph_prior_only", meanQualityScore: 92.8, totalPhraseHitCount: 65, totalPhraseCount: 74, totalSelectedContextBlockCount: 135, totalSelectedContextChars: 26125, estimatedPromptTokens: 6539 },
      { mode: "learned_route", meanQualityScore: 92.05, totalPhraseHitCount: 64, totalPhraseCount: 74, totalSelectedContextBlockCount: 45, totalSelectedContextChars: 12501, estimatedPromptTokens: 3129, totalWarningCount: 20 },
    ]);
    const cold = descriptor([
      { mode: "no_brain", meanQualityScore: 0, totalPhraseHitCount: 0, totalPhraseCount: 74, totalSelectedContextBlockCount: 0, totalSelectedContextChars: 0, estimatedPromptTokens: 0 },
      { mode: "graph_prior_only", meanQualityScore: 92.8, totalPhraseHitCount: 65, totalPhraseCount: 74, totalSelectedContextBlockCount: 135, totalSelectedContextChars: 26125, estimatedPromptTokens: 6539 },
      { mode: "learned_route", meanQualityScore: 92.8, totalPhraseHitCount: 65, totalPhraseCount: 74, totalSelectedContextBlockCount: 48, totalSelectedContextChars: 13101, estimatedPromptTokens: 3279, totalWarningCount: 20 },
    ]);

    const scorecard = buildColdStartScorecardV1({
      generatedAt: "2026-04-25T14:30:00.000Z",
      manifestPath: "evals/recorded-session-replay/canonical-frozen-20/manifest.json",
      candidateArtifactDir: "artifacts/activation-first-gating-retune/T-20260419-269/candidate-artifact",
      baseline,
      coldStartPriorSingle: single,
      coldStartPrior: cold,
    });

    expect(scorecard.contract).toBe(COLD_START_SCORECARD_CONTRACT_V1);
    expect(scorecard.modes.map((mode) => mode.mode)).toEqual([
      "no_brain",
      "graph_prior_only",
      "cold_start_prior_single",
      "cold_start_prior",
      "learned_route",
    ]);
    expect(scorecard.deltas.find((item) => item.leftMode === "cold_start_prior_single" && item.rightMode === "cold_start_prior")).toMatchObject({
      phraseHitDeltaRightMinusLeft: 1,
      selectedContextBlockDeltaRightMinusLeft: 3,
    });
    expect(scorecard.deltas.find((item) => item.leftMode === "graph_prior_only" && item.rightMode === "cold_start_prior")).toMatchObject({
      phraseHitDeltaRightMinusLeft: 0,
      selectedContextBlockDeltaRightMinusLeft: -87,
    });
    expect(scorecard.verdict).toMatchObject({
      usefulContextWin: true,
      noOverfireVsGraphPrior: true,
    });
  });
});
