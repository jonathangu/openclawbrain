import { describe, expect, it } from "vitest";

import {
  ACTIVATION_FIRST_GATING_ONLY_INTERVENTION_HEAD_V1,
  classifyBroadLiveProofReadV1,
  coerceRestraintLanePolicyRowV1,
  deriveFinalCandidateStatusV1,
  derivePseudoRouteBucketPlanV1,
  summarizeFeltOptimizeScorecardV1,
} from "../scripts/run-activation-first-gating-retune.ts";
import type { PolicySupervisionRowV1 } from "../src/brain-core/policy-supervision-rows.ts";

function basePolicyRow(): PolicySupervisionRowV1 {
  return {
    schema_version: 1,
    contract: "policy_supervision_row.v1",
    row_id: "ps_row",
    trace_id: "trace-1",
    episode_id: null,
    decision_point_id: null,
    row_type: "abstain",
    focus_lane: "must-not-fire-100",
    trace_slice: {
      route_row_id: "route-row-1",
      route_fn_version: "activation-first-gating-retune.v1",
      chosen_action_kind: "stop_local",
      stop_label: "STOP_LOCAL",
      query_text_hash: "sha256:abc",
    },
    row_weight: 1.75,
    confidence_target: 0.9,
    hard_negative_class: "graph_prior_preferred",
    net_utility_delta: null,
    net_utility_delta_source: null,
    projection_status: "trace_projected",
    oracle_best_mode: "graph_prior_only",
    notes: ["base"],
  };
}

describe("activation-first gating retune runner helpers", () => {
  it("locks the candidate retune to the explicit gating-only intervention head", () => {
    expect(ACTIVATION_FIRST_GATING_ONLY_INTERVENTION_HEAD_V1).toEqual({
      decisionPolicyMode: "gating_only_v1",
      freezeCandidateSelection: true,
      freezeStopLocal: true,
      featureProfile: "resume_gate_v1",
    });
  });

  it("uses explicit activation and restraint bucket plans", () => {
    expect(derivePseudoRouteBucketPlanV1({ oracleBestMode: "learned_route" })).toEqual({
      candidateCount: 5,
      evidenceSpanCount: 3,
      hardNegativeCount: 1,
      outcomeGain: 1,
      stopLabel: "CONTINUE",
      chosenActionKind: "traverse",
    });

    expect(derivePseudoRouteBucketPlanV1({ oracleBestMode: "tie" })).toEqual({
      candidateCount: 1,
      evidenceSpanCount: 1,
      hardNegativeCount: 0,
      outcomeGain: 0.1,
      stopLabel: "STOP_LOCAL",
      chosenActionKind: "stop_local",
    });
  });

  it("coerces must-not-fire rows to stop_local without changing other lanes", () => {
    const restrained = coerceRestraintLanePolicyRowV1(basePolicyRow(), "mustNotFire");
    expect(restrained.row_type).toBe("stop_local");
    expect(restrained.notes.at(-1)).toContain("Must-not-fire restraint lane coerced to stop_local");

    const unchanged = coerceRestraintLanePolicyRowV1(basePolicyRow(), "feltResume");
    expect(unchanged.row_type).toBe("abstain");
    expect(unchanged.notes).toEqual(["base"]);
  });

  it("downgrades non-authoritative broad-live proof reads to architecture_verdict", () => {
    const proofRead = classifyBroadLiveProofReadV1({
      scorecard: {
        contract: "comparative_eval_scorecard.v1",
        requestedTraceCount: 3,
        successfulTraceCount: 3,
        failedTraceCount: 0,
        policy: { status: "pass" },
        explainableScorecard: {
          regressionVsBaseline: { count: 0, totalCount: 3 },
          traceOutcomeVsBaseline: {
            betterCount: 1,
            tiedCount: 2,
            worseCount: 0,
            totalCount: 3,
          },
        },
        modes: [{ mode: "learned_route", totalUsedLearnedRouteTurnCount: 0 }],
        traces: [
          { traceId: "felt-1", candidateRelationVsBaseline: "tied" },
          { traceId: "felt-2", candidateRelationVsBaseline: "better" },
          { traceId: "other-1", candidateRelationVsBaseline: "worse" },
        ],
      },
      report: {
        contract: "comparative_eval_runner_report.v1",
        status: "ok",
        gateStatus: "pass",
        gateDecisive: true,
      },
      summaryTables: {
        contract: "recorded_session_replay_proof_lane_summary_tables.v1",
        turns: [
          {
            traceId: "felt-1",
            modes: [{
              mode: "learned_route",
              usedLearnedRouteFn: false,
              routerIdentity: null,
              activationTaken: false,
            }],
          },
        ],
      },
      feltTraceIds: ["felt-1", "felt-2", "felt-3"],
    });

    expect(proofRead.authoritative).toBe(false);
    expect(proofRead.vetoResult).toBe("proof_read_only");
    expect(proofRead.feltOverlap).toEqual({
      availableTraceCount: 2,
      totalTraceCount: 3,
      betterCount: 1,
      tiedCount: 1,
      worseCount: 0,
    });

    expect(deriveFinalCandidateStatusV1({
      feltPassed: true,
      restraintPassed: true,
      broadLiveAuthoritative: proofRead.authoritative,
      broadLiveVetoResult: proofRead.vetoResult,
    })).toBe("architecture_verdict");

    expect(deriveFinalCandidateStatusV1({
      feltPassed: false,
      restraintPassed: true,
      broadLiveAuthoritative: true,
      broadLiveVetoResult: "pass",
    })).toBe("reject");

    expect(deriveFinalCandidateStatusV1({
      feltPassed: true,
      restraintPassed: true,
      broadLiveAuthoritative: true,
      broadLiveVetoResult: "pass",
    })).toBe("pass");
  });

  it("summarizes full felt optimize outcomes from comparative eval scorecards", () => {
    const feltSummary = summarizeFeltOptimizeScorecardV1({
      scorecard: {
        contract: "comparative_eval_scorecard.v1",
        manifestId: "felt-resume-25-eval",
        manifestContract: "frozen_recorded_session_eval_manifest.v1",
        modeOrder: ["graph_prior_only", "learned_route"],
        requestedTraceCount: 25,
        successfulTraceCount: 25,
        failedTraceCount: 0,
        scorecardHash: "sha256:test",
        pricingTable: {
          version: null,
          path: "pricing.json",
          charsPerToken: 4,
          promptPriceUsdPer1mTokens: 0,
        },
        scoringProxyNotes: [],
        modes: [],
        pairwise: [],
        policy: {
          status: "fail",
          decisive: true,
          thresholds: {
            candidateMode: "learned_route",
            baselineMode: "graph_prior_only",
            floorMode: "graph_prior_only",
            minTieOrBetterRateVsBaseline: null,
            maxRegressionRateVsBaseline: null,
            maxRegressionRateVsFloor: null,
            minMeanQualityDeltaVsBaseline: null,
          },
          observed: {
            comparableTraceCount: 25,
            tieOrBetterRateVsBaseline: null,
            regressionRateVsBaseline: null,
            regressionRateVsFloor: null,
            meanQualityDeltaVsBaseline: null,
          },
          reasons: [],
          checks: [],
        },
        explainableScorecard: {
          candidateMode: "learned_route",
          baselineMode: "graph_prior_only",
          floorMode: "graph_prior_only",
          comparableTraceCount: 25,
          comparableTurnCount: 25,
          traceOutcomeVsBaseline: {
            betterCount: 0,
            tiedCount: 24,
            worseCount: 1,
            betterRate: 0,
            tieRate: 0.96,
            worseRate: 0.04,
            totalCount: 25,
          },
          turnOutcomeVsBaseline: {
            betterCount: 0,
            tiedCount: 24,
            worseCount: 1,
            betterRate: 0,
            tieRate: 0.96,
            worseRate: 0.04,
            totalCount: 25,
          },
          traceTieOrBetterVsBaseline: { count: 24, rate: 0.96, totalCount: 25 },
          turnTieOrBetterVsBaseline: { count: 24, rate: 0.96, totalCount: 25 },
          regressionVsBaseline: { count: 1, rate: 0.04, totalCount: 25 },
          regressionVsFloor: { count: 1, rate: 0.04, totalCount: 25 },
          criticalRegressionCount: 1,
          requiredContextRecall: {
            available: true,
            candidateMode: "learned_route",
            baselineMode: "graph_prior_only",
            candidatePhraseHitCount: 3,
            candidatePhraseCount: 63,
            candidateRate: 0.047619,
            baselinePhraseHitCount: 4,
            baselinePhraseCount: 63,
            baselineRate: 0.063492,
            delta: -0.015873,
            summary: "learned_route recalled 3/63 required-context phrases vs graph_prior_only 4/63",
          },
          correctionAbsorption: {
            available: false,
            observedFeedbackTurnCount: 0,
            observedNonApprovalFeedbackTurnCount: 0,
            summary: "none",
          },
          successAdjustedEconomics: {
            available: false,
            successUnit: null,
            candidateMode: "learned_route",
            baselineMode: "graph_prior_only",
            successCount: 25,
            candidateEstimatedPromptTokensPerSuccess: null,
            baselineEstimatedPromptTokensPerSuccess: null,
            candidateEstimatedPromptCostUsdPerSuccess: null,
            baselineEstimatedPromptCostUsdPerSuccess: null,
            promptTokenDeltaCandidateMinusBaseline: null,
            promptCostUsdDeltaCandidateMinusBaseline: null,
            summary: "n/a",
            limitations: [],
          },
          failOpen: {
            available: false,
            failOpenRate: null,
            summary: "n/a",
          },
        },
        traces: [],
      } as any,
      outputDir: "artifacts/activation-first-gating-retune/T-20260415-257/felt-optimize-comparative-eval",
      notes: ["candidate-specific felt optimize eval"],
    });

    expect(feltSummary).toEqual({
      available: true,
      comparableTraceCount: 25,
      betterCount: 0,
      tiedCount: 24,
      worseCount: 1,
      regressions: 1,
      requiredContextRecallSummary: "learned_route recalled 3/63 required-context phrases vs graph_prior_only 4/63",
      outputDir: "artifacts/activation-first-gating-retune/T-20260415-257/felt-optimize-comparative-eval",
      notes: ["candidate-specific felt optimize eval"],
    });
  });

  it("records a fresh candidate-specific veto result without overstating authority", () => {
    const freshPass = classifyBroadLiveProofReadV1({
      scorecard: {
        contract: "comparative_eval_scorecard.v1",
        requestedTraceCount: 2,
        successfulTraceCount: 2,
        failedTraceCount: 0,
        policy: { status: "pass" },
        explainableScorecard: {
          regressionVsBaseline: { count: 0, totalCount: 2 },
          traceOutcomeVsBaseline: {
            betterCount: 1,
            tiedCount: 1,
            worseCount: 0,
            totalCount: 2,
          },
        },
        modes: [{ mode: "learned_route", totalUsedLearnedRouteTurnCount: 0 }],
        traces: [
          { traceId: "felt-1", candidateRelationVsBaseline: "better" },
          { traceId: "felt-2", candidateRelationVsBaseline: "tied" },
        ],
      },
      report: {
        contract: "comparative_eval_runner_report.v1",
        status: "ok",
        gateStatus: "pass",
        gateDecisive: true,
      },
      summaryTables: {
        contract: "recorded_session_replay_proof_lane_summary_tables.v1",
        turns: [
          {
            traceId: "felt-1",
            modes: [{
              mode: "learned_route",
              usedLearnedRouteFn: false,
              routerIdentity: null,
              activationTaken: false,
            }],
          },
        ],
      },
      feltTraceIds: ["felt-1", "felt-2"],
      freshRunExecuted: true,
    });

    expect(freshPass.authoritative).toBe(false);
    expect(freshPass.vetoResult).toBe("pass");
    expect(freshPass.notes).toContain(
      "Candidate-specific broad-live comparative eval executed for the just-trained gating-only candidate.",
    );
    expect(deriveFinalCandidateStatusV1({
      feltPassed: true,
      restraintPassed: true,
      broadLiveAuthoritative: freshPass.authoritative,
      broadLiveVetoResult: freshPass.vetoResult,
    })).toBe("architecture_verdict");

    const freshReject = classifyBroadLiveProofReadV1({
      scorecard: {
        contract: "comparative_eval_scorecard.v1",
        requestedTraceCount: 2,
        successfulTraceCount: 2,
        failedTraceCount: 0,
        policy: { status: "fail" },
        explainableScorecard: {
          regressionVsBaseline: { count: 1, totalCount: 2 },
          traceOutcomeVsBaseline: {
            betterCount: 0,
            tiedCount: 1,
            worseCount: 1,
            totalCount: 2,
          },
        },
        modes: [{ mode: "learned_route", totalUsedLearnedRouteTurnCount: 0 }],
        traces: [
          { traceId: "felt-1", candidateRelationVsBaseline: "worse" },
          { traceId: "felt-2", candidateRelationVsBaseline: "tied" },
        ],
      },
      report: {
        contract: "comparative_eval_runner_report.v1",
        status: "partial",
        gateStatus: "fail",
        gateDecisive: true,
      },
      summaryTables: {
        contract: "recorded_session_replay_proof_lane_summary_tables.v1",
        turns: [
          {
            traceId: "felt-1",
            modes: [{
              mode: "learned_route",
              usedLearnedRouteFn: false,
              routerIdentity: null,
              activationTaken: false,
            }],
          },
        ],
      },
      feltTraceIds: ["felt-1", "felt-2"],
      freshRunExecuted: true,
    });

    expect(freshReject.authoritative).toBe(false);
    expect(freshReject.vetoResult).toBe("reject");
  });
});
