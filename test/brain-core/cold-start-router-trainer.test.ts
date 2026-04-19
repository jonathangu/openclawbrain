import { execFileSync } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it } from "vitest";

import {
  summarizeRouterArtifactManifestV1,
  validateRouterArtifactManifestV1,
} from "../../src/brain-core/cold-start-router-contracts.js";
import type {
  DataRegistryEntryV1,
  RouteDecisionRowV1,
} from "../../src/brain-core/cold-start-router-contracts.js";
import {
  loadAndFilterColdStartRouterApprovedExportV1,
} from "../../src/brain-core/cold-start-router-approved-export-loader.js";
import {
  loadColdStartRouterArtifactBundleV1,
  materializeColdStartRouterLivePolicyFromArtifactBundleV1,
  scoreColdStartRouteRowFromArtifactBundleV1,
  selectColdStartRouteCandidateIdsFromArtifactBundleV1,
} from "../../src/brain-core/cold-start-router-runtime.js";
import { scoreAction } from "../../src/brain-core/policy.js";
import { applyWeightUpdates, computeReinforceUpdates } from "../../src/brain-core/update.js";
import {
  predictColdStartStopLabelV1,
  rankColdStartRouteCandidatesV1,
  scoreColdStartRouteRowV1,
  trainColdStartRouterArtifactV1,
} from "../../src/brain-core/cold-start-router-trainer.js";
import type { ColdStartRouterModelV1, ColdStartRouterTrainingInputV1 } from "../../src/brain-core/cold-start-router-trainer.js";
import { materializeColdStartRouterLivePolicyGraphV1 } from "../../src/brain-core/graph.js";
import {
  POLICY_SUPERVISION_ROW_CONTRACT_V1,
  POLICY_SUPERVISION_ROW_VERSION_V1,
} from "../../src/brain-core/policy-supervision-rows.js";
import type { PolicySupervisionRowV1 } from "../../src/brain-core/policy-supervision-rows.js";
import type { Episode, TraversalAction, TraversalState, TrajectoryExpansion, TrajectoryStep } from "../../src/brain-core/types.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..", "..");
const approvedExportPath = fileURLToPath(
  new URL("../../artifacts/cold-start-router-approved-export/approved-router-export.fixture.v1.json", import.meta.url),
);
const approvedExportV2Path = fileURLToPath(
  new URL("../../artifacts/cold-start-router-approved-export/real-approved-router-export.hotpotqa-musique.v2.json", import.meta.url),
);
const approvedTrainV2Dir = path.join(
  repoRoot,
  "artifacts",
  "cold-start-router-approved-export",
  "real-approved-router-train.hotpotqa-musique.v2",
);

const tempRoots: string[] = [];

afterEach(() => {
  while (tempRoots.length > 0) {
    rmSync(tempRoots.pop()!, { recursive: true, force: true });
  }
});

function createTempRoot(label: string): string {
  const root = mkdtempSync(path.join(os.tmpdir(), `${label}-`));
  tempRoots.push(root);
  return root;
}

function makeActivationFirstRegistryEntry(datasetId: string): DataRegistryEntryV1 {
  return {
    dataset_id: datasetId,
    source_family: "qa",
    upstream_url: "https://example.org/activation-first",
    original_creator: "OpenClaw",
    license: "CC BY 4.0",
    commercial_use_status: "allowed",
    redistribution_status: "allowed",
    pii_risk: "none",
    benchmark_split_status: "train",
    approval_status: "approved_train",
    reviewer: "operator",
    immutable_snapshot_ref: `snapshot:${datasetId}@sha256:activation-first`,
    exact_files: ["data/train.json"],
    file_hashes: {
      "data/train.json": "sha256:activation-first-train",
    },
    allowed_uses: ["route supervision", "ranking baselines"],
    disallowed_uses: ["private redistribution"],
    notes: ["activation-first fixture"],
    created_at: "2026-04-10T08:00:00Z",
    updated_at: "2026-04-10T08:00:00Z",
  };
}

function makeActivationFirstTrainingRows(datasetId: string): RouteDecisionRowV1[] {
  return [
    {
      row_id: "row_activation_first_continue",
      dataset_id: datasetId,
      query: "Find the current shipping correction memory before drafting the reply",
      cursor_path: ["source:billing-thread"],
      candidate_set: [
        { candidate_id: "mem:shipping_correction", candidate_type: "memory_node", authority: "human", freshness: "current", score_hint: 0.88 },
        { candidate_id: "doc:generic_shipping_policy", candidate_type: "doc_chunk", authority: "operator_policy", freshness: "stale", token_cost: 18, score_hint: 0.22 },
      ],
      teacher_action: { kind: "traverse", target_ids: ["mem:shipping_correction"] },
      stop_label: "CONTINUE",
      evidence_spans: [
        { source_ref: "message_201", start: 0, end: 32 },
        { source_ref: "memory_201", start: 0, end: 28 },
      ],
      hard_negatives: ["doc:generic_shipping_policy"],
      outcome_gain: 0.8,
      provenance: {
        dataset: datasetId,
        source_license: "CC BY 4.0",
        source_family: "qa",
        source_snapshot_ref: `snapshot:${datasetId}@sha256:activation-first`,
        recorded_by: "operator",
        recorded_at: "2026-04-10T08:01:00Z",
        review_status: "approved_train",
      },
      split_tag: "train",
      created_at: "2026-04-10T08:02:00Z",
    },
    {
      row_id: "row_activation_first_stop_local",
      dataset_id: datasetId,
      query: "Use the invoice tool and stop locally",
      cursor_path: ["source:billing-thread"],
      candidate_set: [
        { candidate_id: "tool:invoice_draft", candidate_type: "tool", authority: "runtime", freshness: "current", token_cost: 2, score_hint: 0.87 },
        { candidate_id: "mem:shipping_correction", candidate_type: "memory_node", authority: "human", freshness: "current", score_hint: 0.15 },
      ],
      teacher_action: { kind: "tool", tool_name: "tool:invoice_draft" },
      stop_label: "STOP_LOCAL",
      evidence_spans: [
        { source_ref: "message_202", start: 0, end: 28 },
      ],
      hard_negatives: [],
      outcome_gain: 0.8,
      provenance: {
        dataset: datasetId,
        source_license: "CC BY 4.0",
        source_family: "qa",
        source_snapshot_ref: `snapshot:${datasetId}@sha256:activation-first`,
        recorded_by: "operator",
        recorded_at: "2026-04-10T08:03:00Z",
        review_status: "approved_train",
      },
      split_tag: "train",
      created_at: "2026-04-10T08:04:00Z",
    },
  ];
}

function makeGreedyThresholdRouteRow(datasetId: string): RouteDecisionRowV1 {
  return {
    row_id: "row_activation_first_greedy_lane",
    dataset_id: datasetId,
    query: "Need the exact shipping correction memory, not the generic policy page",
    cursor_path: ["source:billing-thread"],
    candidate_set: [
      { candidate_id: "mem:shipping_correction", candidate_type: "memory_node", authority: "human", freshness: "current", score_hint: 0.28 },
      { candidate_id: "doc:generic_shipping_policy", candidate_type: "doc_chunk", authority: "operator_policy", freshness: "current", score_hint: 0.18 },
    ],
    teacher_action: { kind: "traverse", target_ids: ["mem:shipping_correction"] },
    stop_label: "CONTINUE",
    evidence_spans: [
      { source_ref: "message_203", start: 0, end: 30 },
      { source_ref: "memory_203", start: 0, end: 24 },
    ],
    hard_negatives: ["doc:generic_shipping_policy"],
    outcome_gain: 0.72,
    provenance: {
      dataset: datasetId,
      source_license: "CC BY 4.0",
      source_family: "qa",
      source_snapshot_ref: `snapshot:${datasetId}@sha256:activation-first`,
      recorded_by: "operator",
      recorded_at: "2026-04-10T08:05:00Z",
      review_status: "approved_train",
    },
    split_tag: "train",
    created_at: "2026-04-10T08:06:00Z",
  };
}

function makeReplayScopedFeltRouteRow(datasetId: string): RouteDecisionRowV1 {
  return {
    row_id: "row_replay_scoped_felt_training",
    dataset_id: datasetId,
    query: "Prefer the replay feedback block over the replay interaction block",
    cursor_path: ["felt_resume_25"],
    candidate_set: [
      {
        candidate_id: "pack-runtime:event:alpha:feedback",
        candidate_type: "graph_node",
        semantic_class: "feedback_context",
        authority: "recorded_session_replay",
        freshness: "replay_eval",
        token_cost: 64,
        score_hint: 0.3,
      },
      {
        candidate_id: "pack-runtime:event:alpha:interaction",
        candidate_type: "graph_node",
        semantic_class: "interaction_context",
        authority: "recorded_session_replay",
        freshness: "replay_eval",
        token_cost: 56,
        score_hint: 0.45,
      },
    ],
    teacher_action: { kind: "traverse", target_ids: ["pack-runtime:event:alpha:feedback"] },
    stop_label: "CONTINUE",
    evidence_spans: [
      { source_ref: "replay:evidence:0", start: 0, end: 39, excerpt: "Need the replay feedback block, not the cue." },
    ],
    hard_negatives: ["pack-runtime:event:alpha:interaction"],
    outcome_gain: 1,
    provenance: {
      dataset: datasetId,
      source_license: "internal_local_only",
      source_family: "agent_traces",
      source_snapshot_ref: `snapshot:${datasetId}@sha256:replay-scoped-felt-training`,
      recorded_by: "test",
      recorded_at: "2026-04-18T16:20:00Z",
      review_status: "approved_train",
    },
    split_tag: "train",
    created_at: "2026-04-18T16:20:00Z",
  };
}

function makePolicySupervisionRowFixture(params: {
  rowId: string;
  traceId: string;
  routeRowId: string;
  rowType: PolicySupervisionRowV1["row_type"];
  focusLane: string | null;
  rowWeight: number;
  hardNegativeClass?: PolicySupervisionRowV1["hard_negative_class"];
  oracleBestMode?: PolicySupervisionRowV1["oracle_best_mode"];
}): PolicySupervisionRowV1 {
  return {
    schema_version: POLICY_SUPERVISION_ROW_VERSION_V1,
    contract: POLICY_SUPERVISION_ROW_CONTRACT_V1,
    row_id: params.rowId,
    trace_id: params.traceId,
    episode_id: null,
    decision_point_id: null,
    row_type: params.rowType,
    focus_lane: params.focusLane,
    trace_slice: {
      route_row_id: params.routeRowId,
      route_fn_version: null,
      chosen_action_kind: null,
      stop_label: null,
      query_text_hash: null,
    },
    row_weight: params.rowWeight,
    confidence_target: null,
    hard_negative_class: params.hardNegativeClass ?? null,
    net_utility_delta: null,
    net_utility_delta_source: null,
    projection_status: "owner_labeled",
    oracle_best_mode: params.oracleBestMode ?? null,
    notes: ["trainer fixture"],
  };
}

function trainActivationFirstFixture(
  outputDir: string,
  datasetId: string,
  overrides: Partial<Pick<ColdStartRouterTrainingInputV1, "policySupervisionRows" | "focusLaneWeights" | "rowTypeWeights" | "interventionHead" | "calibrationOverrides">> = {},
) {
  return trainColdStartRouterArtifactV1({
    artifactId: `router-artifact-${datasetId}`,
    artifactVersion: "0.0.1",
    packType: "base",
    compatibleRuntimeVersion: "openclawbrain-runtime@0.4.43",
    registryEntries: [makeActivationFirstRegistryEntry(datasetId)],
    routeRows: makeActivationFirstTrainingRows(datasetId),
    outputDir,
    routerIdentity: `router:${datasetId}`,
    createdAt: "2026-04-10T08:10:00Z",
    trainingDataRefs: [datasetId],
    replayGateRefs: [`replay:${datasetId}`],
    ...overrides,
  });
}

describe("cold-start router trainer", () => {
  it("trains a manifest-compatible router artifact and ranks candidates from the approved export loader", () => {
    const outputDir = createTempRoot("cold-start-router-trainer");
    const loadedExport = loadAndFilterColdStartRouterApprovedExportV1(approvedExportPath);

    const result = trainColdStartRouterArtifactV1({
      artifactId: "router-artifact-approved-export-smoke",
      artifactVersion: "0.0.1",
      packType: "base",
      compatibleRuntimeVersion: "openclawbrain-runtime@0.4.43",
      registryEntries: loadedExport.registryEntries,
      routeRows: loadedExport.routeRows,
      outputDir,
      routerIdentity: "router:approved-export:base",
      createdAt: "2026-04-05T16:20:00Z",
      trainingDataRefs: loadedExport.summary.approvedDatasetIds,
      replayGateRefs: ["replay:approved-export:fixture-v1"],
    });

    expect(result.manifestPath).toBe(path.join(outputDir, "manifest.json"));
    expect(existsSync(result.manifestPath)).toBe(true);
    expect(existsSync(result.baseModelPath)).toBe(true);
    expect(existsSync(result.weightsPath)).toBe(true);
    expect(existsSync(result.calibrationPath)).toBe(true);
    expect(existsSync(result.featureNormalizersPath)).toBe(true);
    expect(existsSync(result.sourcePriorsPath)).toBe(true);
    expect(existsSync(result.safetyRulesPath)).toBe(true);

    const manifest = JSON.parse(readFileSync(result.manifestPath, "utf8")) as Record<string, unknown>;
    const validation = validateRouterArtifactManifestV1(manifest);
    expect(validation.valid).toBe(true);
    expect(summarizeRouterArtifactManifestV1(manifest as never)).toMatchObject({
      artifactId: "router-artifact-approved-export-smoke",
      packType: "base",
      trainingDataRefCount: 1,
      replayGateRefCount: 1,
      runtimeVersion: "openclawbrain-runtime@0.4.43",
    });

    expect(result.model.training).toMatchObject({
      totalRows: 2,
      eligibleRows: 2,
      usedRows: 2,
      skippedRows: 0,
      usedDatasetIds: ["router_fixture_train_v1"],
    });
    expect(result.model.sourcePriors.datasets["router_fixture_train_v1"]).toMatchObject({
      datasetId: "router_fixture_train_v1",
      rowCount: 2,
      usedRowCount: 2,
      skippedRowCount: 0,
    });
    expect(result.model.livePolicyInitializer).toMatchObject({
      contract: "cold_start_router_live_policy_initializer.v1",
      usedRowCount: 2,
      traverseRowCount: 1,
      toolRowCount: 1,
    });
    expect(result.model.training).toMatchObject({
      toolActionPriorCount: 2,
      toolActionSetCount: 2,
    });
    expect(result.model.calibration).toMatchObject({
      activationThreshold: 0.45,
      abstentionThreshold: 0.55,
      expectedUtilityThreshold: 0,
      stopLocalThreshold: 0.5,
      interventionHead: {
        decisionPolicyMode: "router_blended",
        freezeCandidateSelection: false,
        freezeStopLocal: false,
        featureProfile: "default_router",
      },
    });
    expect(result.model.toolActionPriors.length).toBeGreaterThan(0);
    expect(result.model.toolActionSets.length).toBeGreaterThan(0);
    expect(result.model.toolActionSets.some((entry) => entry.candidates.some((candidate) => candidate.candidate_type === "tool"))).toBe(true);

    const rowScore = scoreColdStartRouteRowV1({ model: result.model, row: loadedExport.routeRows[0] });
    expect(rowScore.rankedCandidates[0]?.candidate.candidate_id).toBe("mem:shipping_history");
    expect(rowScore.rankedCandidates[0]?.score).toBeGreaterThan(rowScore.rankedCandidates[1]?.score ?? -Infinity);
    expect(rowScore.stopPrediction.contributingBuckets).toHaveLength(4);
    expect(rowScore.stopPrediction.scores).toHaveProperty("CONTINUE");

    const ranking = rankColdStartRouteCandidatesV1({ model: result.model, candidates: loadedExport.routeRows[0].candidate_set });
    expect(ranking[0]?.candidate.candidate_id).toBe("mem:shipping_history");
    expect(rowScore.policyDistribution.actions).toHaveLength(loadedExport.routeRows[0].candidate_set.length + 1);
    expect(rowScore.policyDistribution.stopAction.action.type).toBe("stop_local");
    expect(rowScore.decisionSummary).toMatchObject({
      activated: true,
      selectedContextCount: 1,
      selectedTokenBudget: null,
      activationThreshold: 0.45,
      abstentionThreshold: 0.55,
      expectedUtilityThreshold: 0,
      stopLocalThreshold: 0.5,
      decisionPolicyMode: "router_blended",
      freezeCandidateSelection: false,
      freezeStopLocal: false,
      stopReason: null,
    });
    expect(rowScore.decisionSummary.activationProbability).toBeGreaterThan(0);
    expect(rowScore.decisionSummary.predictedUtility).toBeGreaterThan(0);
    expect(
      rowScore.policyDistribution.actions.reduce((sum, action) => sum + action.probability, 0),
    ).toBeCloseTo(1.0, 5);
    expect(
      rowScore.policyDistribution.actions.find((action) => action.action.type === "stop_local")
        ?.probability,
    ).toBeGreaterThan(0);
    expect(predictColdStartStopLabelV1({
      model: result.model,
      candidateCount: loadedExport.routeRows[0].candidate_set.length,
      evidenceSpanCount: loadedExport.routeRows[0].evidence_spans.length,
      hardNegativeCount: loadedExport.routeRows[0].hard_negatives.length,
      outcomeGain: loadedExport.routeRows[0].outcome_gain,
    })).toMatchObject({
      label: expect.any(String),
    });

    const runtimeBundle = loadColdStartRouterArtifactBundleV1(outputDir);
    expect(runtimeBundle.manifest.artifact_checksum).toBe(result.manifest.artifact_checksum);
    expect(runtimeBundle.model.training).toMatchObject(result.model.training);

    const runtimeRowScore = scoreColdStartRouteRowFromArtifactBundleV1({ artifactBundle: runtimeBundle, row: loadedExport.routeRows[0] });
    expect(runtimeRowScore.rankedCandidates[0]?.candidate.candidate_id).toBe("mem:shipping_history");
    expect(runtimeRowScore.policyDistribution.actions).toHaveLength(loadedExport.routeRows[0].candidate_set.length + 1);
    expect(runtimeRowScore.policyDistribution.stopAction.action.type).toBe("stop_local");
    expect(runtimeRowScore.decisionSummary.activated).toBe(true);
    expect(runtimeRowScore.decisionSummary.stopReason).toBeNull();

    const runtimeSelection = selectColdStartRouteCandidateIdsFromArtifactBundleV1({ artifactBundle: runtimeBundle, row: loadedExport.routeRows[0] });
    expect(runtimeSelection.stopped).toBe(false);
    expect(runtimeSelection.selectedCandidateIds).toEqual(["mem:shipping_history"]);
    expect(runtimeSelection.decisionSummary.activated).toBe(true);
  });

  it("overweights beneficial CONTINUE rows so greedy retrains lean activation-first", () => {
    const outputDir = createTempRoot("cold-start-router-activation-first-bias");
    const result = trainActivationFirstFixture(outputDir, "dataset_activation_first_bias");

    expect(result.model.calibration.activationThreshold).toBe(0.45);
    expect(result.model.stopLabelCounts.CONTINUE).toBeGreaterThan(result.model.stopLabelCounts.STOP_LOCAL);
    expect(result.model.stopLabelCounts.CONTINUE).toBeGreaterThan(0.8);
    expect(result.model.stopLabelCounts.STOP_LOCAL).toBeCloseTo(0.8, 6);
    expect(result.model.livePolicyInitializer.policyParams.stopBias).toBeLessThan(0);
  });

  it("persists an explicit gating-only intervention head and ignores stop-local gating when enabled", () => {
    const outputDir = createTempRoot("cold-start-router-gating-only-head");
    const result = trainActivationFirstFixture(outputDir, "dataset_activation_first_gating_only", {
      interventionHead: {
        decisionPolicyMode: "gating_only_v1",
        freezeCandidateSelection: true,
        freezeStopLocal: true,
        featureProfile: "resume_gate_v1",
      },
    });

    expect(result.model.calibration.interventionHead).toEqual({
      decisionPolicyMode: "gating_only_v1",
      freezeCandidateSelection: true,
      freezeStopLocal: true,
      featureProfile: "resume_gate_v1",
    });

    const runtimeBundle = loadColdStartRouterArtifactBundleV1(outputDir);
    expect(runtimeBundle.model.calibration.interventionHead).toEqual({
      decisionPolicyMode: "gating_only_v1",
      freezeCandidateSelection: true,
      freezeStopLocal: true,
      featureProfile: "resume_gate_v1",
    });

    const row = makeGreedyThresholdRouteRow("dataset_activation_first_greedy_lane");
    const blendedScoring = scoreColdStartRouteRowV1({
      model: {
        ...result.model,
        calibration: {
          ...result.model.calibration,
          interventionHead: {
            decisionPolicyMode: "router_blended",
            freezeCandidateSelection: false,
            freezeStopLocal: false,
            featureProfile: "default_router",
          },
          stopLocalThreshold: 0.01,
        },
      },
      row,
    });
    const gatingOnlyScoring = scoreColdStartRouteRowV1({
      model: {
        ...result.model,
        calibration: {
          ...result.model.calibration,
          stopLocalThreshold: 0.01,
        },
      },
      row,
    });

    expect(blendedScoring.decisionSummary.activated).toBe(false);
    expect(blendedScoring.decisionSummary.stopReason).toBe("stop_local");
    expect(gatingOnlyScoring.decisionSummary.activated).toBe(true);
    expect(gatingOnlyScoring.decisionSummary.stopReason).toBeNull();
    expect(gatingOnlyScoring.decisionSummary).toMatchObject({
      decisionPolicyMode: "gating_only_v1",
      freezeCandidateSelection: true,
      freezeStopLocal: true,
    });
    expect(gatingOnlyScoring.rankedCandidates[0]?.candidate.candidate_id).toBe(
      blendedScoring.rankedCandidates[0]?.candidate.candidate_id,
    );
  });

  it("backfills missing calibration thresholds when warm-starting a gating-only retrain", () => {
    const outputDir = createTempRoot("cold-start-router-gating-only-warm-start-thresholds");
    const result = trainColdStartRouterArtifactV1({
      artifactId: "router-artifact-gating-only-warm-start-thresholds",
      artifactVersion: "0.0.1",
      packType: "base",
      compatibleRuntimeVersion: "openclawbrain-runtime@0.4.43",
      registryEntries: [makeActivationFirstRegistryEntry("dataset_activation_first_warm_start")],
      routeRows: makeActivationFirstTrainingRows("dataset_activation_first_warm_start"),
      outputDir,
      routerIdentity: "router:dataset_activation_first_warm_start",
      createdAt: "2026-04-17T22:50:00Z",
      trainingDataRefs: ["dataset_activation_first_warm_start"],
      replayGateRefs: ["replay:dataset_activation_first_warm_start"],
      warmStartArtifactDir: approvedTrainV2Dir,
      interventionHead: {
        decisionPolicyMode: "gating_only_v1",
        freezeCandidateSelection: true,
        freezeStopLocal: true,
        featureProfile: "resume_gate_v1",
      },
    });

    expect(result.model.calibration).toMatchObject({
      activationThreshold: 0.45,
      abstentionThreshold: 0.55,
      expectedUtilityThreshold: 0,
      stopLocalThreshold: 0.5,
      interventionHead: {
        decisionPolicyMode: "gating_only_v1",
        freezeCandidateSelection: true,
        freezeStopLocal: true,
        featureProfile: "resume_gate_v1",
      },
    });

    const runtimeBundle = loadColdStartRouterArtifactBundleV1(outputDir);
    const scoring = scoreColdStartRouteRowFromArtifactBundleV1({
      artifactBundle: runtimeBundle,
      row: makeGreedyThresholdRouteRow("dataset_activation_first_greedy_lane"),
    });

    expect(scoring.decisionSummary).toMatchObject({
      activationThreshold: 0.45,
      abstentionThreshold: 0.55,
      expectedUtilityThreshold: 0,
      stopLocalThreshold: 0.5,
      decisionPolicyMode: "gating_only_v1",
      freezeCandidateSelection: true,
      freezeStopLocal: true,
    });
  });

  it("applies calibration overrides for gating-only resume-gate experiments", () => {
    const outputDir = createTempRoot("cold-start-router-gating-only-calibration-override");
    const result = trainColdStartRouterArtifactV1({
      artifactId: "router-artifact-gating-only-calibration-override",
      artifactVersion: "0.0.1",
      packType: "base",
      compatibleRuntimeVersion: "openclawbrain-runtime@0.4.43",
      registryEntries: [makeActivationFirstRegistryEntry("dataset_activation_first_override")],
      routeRows: makeActivationFirstTrainingRows("dataset_activation_first_override"),
      outputDir,
      routerIdentity: "router:dataset_activation_first_override",
      createdAt: "2026-04-18T10:45:00Z",
      trainingDataRefs: ["dataset_activation_first_override"],
      replayGateRefs: ["replay:dataset_activation_first_override"],
      warmStartArtifactDir: approvedTrainV2Dir,
      interventionHead: {
        decisionPolicyMode: "gating_only_v1",
        freezeCandidateSelection: true,
        freezeStopLocal: true,
        featureProfile: "resume_gate_v1",
      },
      calibrationOverrides: {
        activationThreshold: 0.38,
      },
    });

    expect(result.model.calibration).toMatchObject({
      activationThreshold: 0.38,
      abstentionThreshold: 0.55,
      expectedUtilityThreshold: 0,
      stopLocalThreshold: 0.5,
      interventionHead: {
        decisionPolicyMode: "gating_only_v1",
        freezeCandidateSelection: true,
        freezeStopLocal: true,
        featureProfile: "resume_gate_v1",
      },
    });

    const runtimeBundle = loadColdStartRouterArtifactBundleV1(outputDir);
    const scoring = scoreColdStartRouteRowFromArtifactBundleV1({
      artifactBundle: runtimeBundle,
      row: makeGreedyThresholdRouteRow("dataset_activation_first_greedy_lane"),
    });

    expect(scoring.decisionSummary).toMatchObject({
      activationThreshold: 0.38,
      abstentionThreshold: 0.55,
      expectedUtilityThreshold: 0,
      stopLocalThreshold: 0.5,
      decisionPolicyMode: "gating_only_v1",
      freezeCandidateSelection: true,
      freezeStopLocal: true,
    });
  });

  it("weights felt-lane activation supervision into gating without changing the top-ranked candidate", () => {
    const datasetId = "dataset_activation_first_policy_activate";
    const baseline = trainActivationFirstFixture(
      createTempRoot("cold-start-router-policy-activate-baseline"),
      datasetId,
    );
    const weighted = trainActivationFirstFixture(
      createTempRoot("cold-start-router-policy-activate-weighted"),
      datasetId,
      {
        policySupervisionRows: [
          makePolicySupervisionRowFixture({
            rowId: "ps_activation_first_felt_activate",
            traceId: "trace_activation_first_felt_activate",
            routeRowId: "row_activation_first_continue",
            rowType: "activate",
            focusLane: "felt_resume_25",
            rowWeight: 1.8,
            oracleBestMode: "learned_route",
          }),
        ],
        focusLaneWeights: { felt_resume_25: 4 },
        rowTypeWeights: { activate: 2 },
      },
    );

    const row = makeGreedyThresholdRouteRow(datasetId);
    const baselineScoring = scoreColdStartRouteRowV1({ model: baseline.model, row });
    const weightedScoring = scoreColdStartRouteRowV1({ model: weighted.model, row });

    expect(weighted.model.stopLabelCounts.CONTINUE).toBeGreaterThan(baseline.model.stopLabelCounts.CONTINUE);
    expect(weighted.model.livePolicyInitializer.policyParams.stopBias).toBeLessThan(
      baseline.model.livePolicyInitializer.policyParams.stopBias,
    );
    expect(weightedScoring.rankedCandidates[0]?.candidate.candidate_id).toBe(
      baselineScoring.rankedCandidates[0]?.candidate.candidate_id,
    );
    expect(weightedScoring.policyDistribution.stopAction.probability).toBeLessThan(
      baselineScoring.policyDistribution.stopAction.probability,
    );
    expect(weightedScoring.decisionSummary.activationProbability).toBeGreaterThan(
      baselineScoring.decisionSummary.activationProbability,
    );
  });

  it("lets must-not-fire abstain supervision push the same source lane into abstention", () => {
    const datasetId = "dataset_activation_first_policy_abstain";
    const baseline = trainActivationFirstFixture(
      createTempRoot("cold-start-router-policy-abstain-baseline"),
      datasetId,
    );
    const restrained = trainActivationFirstFixture(
      createTempRoot("cold-start-router-policy-abstain-restrained"),
      datasetId,
      {
        policySupervisionRows: [
          makePolicySupervisionRowFixture({
            rowId: "ps_activation_first_must_not_fire",
            traceId: "trace_activation_first_must_not_fire",
            routeRowId: "row_activation_first_continue",
            rowType: "abstain",
            focusLane: "must_not_fire_100",
            rowWeight: 2,
            hardNegativeClass: "unnecessary_activation",
            oracleBestMode: "graph_prior_only",
          }),
        ],
        focusLaneWeights: { must_not_fire_100: 8 },
        rowTypeWeights: { abstain: 2 },
      },
    );

    const row = makeGreedyThresholdRouteRow(datasetId);
    const baselineScoring = scoreColdStartRouteRowV1({ model: baseline.model, row });
    const restrainedScoring = scoreColdStartRouteRowV1({ model: restrained.model, row });

    expect(baselineScoring.decisionSummary.activated).toBe(true);
    expect(restrained.model.stopLabelCounts.STOP).toBeGreaterThan(baseline.model.stopLabelCounts.STOP);
    expect(restrained.model.livePolicyInitializer.policyParams.stopBias).toBeGreaterThan(
      baseline.model.livePolicyInitializer.policyParams.stopBias,
    );
    expect(restrainedScoring.policyDistribution.stopAction.probability).toBeGreaterThan(
      baselineScoring.policyDistribution.stopAction.probability,
    );
    expect(restrainedScoring.decisionSummary.activated).toBe(false);
    expect(restrainedScoring.decisionSummary.stopReason).toBe("abstention_threshold_met");
  });

  it("trains scoped felt replay source ids into the live policy initializer for resume-gate candidates", () => {
    const datasetId = "dataset_activation_first_scoped_felt_training";
    const replayRow = makeReplayScopedFeltRouteRow(datasetId);
    const trained = trainColdStartRouterArtifactV1({
      artifactId: `router-artifact-${datasetId}`,
      artifactVersion: "0.0.1",
      packType: "base",
      compatibleRuntimeVersion: "openclawbrain-runtime@0.4.43",
      registryEntries: [makeActivationFirstRegistryEntry(datasetId)],
      routeRows: [replayRow],
      outputDir: createTempRoot("cold-start-router-scoped-felt-training"),
      routerIdentity: `router:${datasetId}`,
      createdAt: "2026-04-18T16:21:00Z",
      trainingDataRefs: [datasetId],
      replayGateRefs: [`replay:${datasetId}`],
      interventionHead: {
        decisionPolicyMode: "gating_only_v1",
        freezeCandidateSelection: true,
        freezeStopLocal: true,
        featureProfile: "resume_gate_v1",
      },
    });

    expect(trained.model.livePolicyInitializer.stopLocalWeights).toContainEqual(
      expect.objectContaining({ sourceNodeId: "felt_resume_25:source:pack-runtime:event:alpha:feedback" }),
    );
    expect(trained.model.livePolicyInitializer.stopLocalWeights.some((entry) => entry.sourceNodeId === "felt_resume_25")).toBe(false);
    expect(trained.model.livePolicyInitializer.edgeWeights).toContainEqual(
      expect.objectContaining({
        sourceNodeId: "felt_resume_25:source:pack-runtime:event:alpha:feedback",
        targetNodeId: "pack-runtime:event:alpha:feedback",
      }),
    );

    const scoring = scoreColdStartRouteRowV1({ model: trained.model, row: replayRow });
    expect(scoring.rankedCandidates[0]?.candidate.candidate_id).toBe("pack-runtime:event:alpha:feedback");
  });

  it("lands felt policy supervision on the scoped replay source instead of the shared felt bucket", () => {
    const datasetId = "dataset_activation_first_scoped_felt_policy";
    const replayRow = makeReplayScopedFeltRouteRow(datasetId);
    const baseline = trainColdStartRouterArtifactV1({
      artifactId: `router-artifact-${datasetId}-baseline`,
      artifactVersion: "0.0.1",
      packType: "base",
      compatibleRuntimeVersion: "openclawbrain-runtime@0.4.43",
      registryEntries: [makeActivationFirstRegistryEntry(datasetId)],
      routeRows: [replayRow],
      outputDir: createTempRoot("cold-start-router-scoped-felt-policy-baseline"),
      routerIdentity: `router:${datasetId}:baseline`,
      createdAt: "2026-04-18T16:22:00Z",
      trainingDataRefs: [datasetId],
      replayGateRefs: [`replay:${datasetId}`],
      interventionHead: {
        decisionPolicyMode: "gating_only_v1",
        freezeCandidateSelection: true,
        freezeStopLocal: true,
        featureProfile: "resume_gate_v1",
      },
    });
    const weighted = trainColdStartRouterArtifactV1({
      artifactId: `router-artifact-${datasetId}-weighted`,
      artifactVersion: "0.0.1",
      packType: "base",
      compatibleRuntimeVersion: "openclawbrain-runtime@0.4.43",
      registryEntries: [makeActivationFirstRegistryEntry(datasetId)],
      routeRows: [replayRow],
      policySupervisionRows: [
        makePolicySupervisionRowFixture({
          rowId: "ps_scoped_felt_activate",
          traceId: "trace_scoped_felt_activate",
          routeRowId: replayRow.row_id,
          rowType: "activate",
          focusLane: "felt_resume_25",
          rowWeight: 2,
          oracleBestMode: "learned_route",
        }),
      ],
      focusLaneWeights: { felt_resume_25: 4 },
      rowTypeWeights: { activate: 2 },
      outputDir: createTempRoot("cold-start-router-scoped-felt-policy-weighted"),
      routerIdentity: `router:${datasetId}:weighted`,
      createdAt: "2026-04-18T16:23:00Z",
      trainingDataRefs: [datasetId],
      replayGateRefs: [`replay:${datasetId}`],
      interventionHead: {
        decisionPolicyMode: "gating_only_v1",
        freezeCandidateSelection: true,
        freezeStopLocal: true,
        featureProfile: "resume_gate_v1",
      },
    });

    const baselineScoped = baseline.model.livePolicyInitializer.stopLocalWeights.find(
      (entry) => entry.sourceNodeId === "felt_resume_25:source:pack-runtime:event:alpha:feedback",
    );
    const weightedScoped = weighted.model.livePolicyInitializer.stopLocalWeights.find(
      (entry) => entry.sourceNodeId === "felt_resume_25:source:pack-runtime:event:alpha:feedback",
    );

    expect(baselineScoped).toBeDefined();
    expect(weightedScoped).toBeDefined();
    expect(weightedScoped!.negative).toBeGreaterThan(baselineScoped!.negative);
    expect(weightedScoped!.support).toBeGreaterThan(baselineScoped!.support);
    expect(weighted.model.livePolicyInitializer.stopLocalWeights.some((entry) => entry.sourceNodeId === "felt_resume_25")).toBe(false);
  });

  it("activates a near-threshold must-fire row once it enters the greedy activation lane", () => {
    const outputDir = createTempRoot("cold-start-router-greedy-lane");
    const trained = trainActivationFirstFixture(outputDir, "dataset_activation_first_greedy_lane");
    const model: ColdStartRouterModelV1 = {
      ...trained.model,
      livePolicyInitializer: {
        ...trained.model.livePolicyInitializer,
        policyParams: {
          ...trained.model.livePolicyInitializer.policyParams,
          stopBias: 0.35,
        },
        seedWeights: [],
        stopLocalWeights: [
          {
            sourceNodeId: "source:billing-thread",
            positive: 1,
            negative: 4,
            support: 5,
            weight: 0,
          },
        ],
        edgeWeights: [
          {
            sourceNodeId: "source:billing-thread",
            targetNodeId: "mem:shipping_correction",
            positive: 5,
            negative: 1,
            support: 6,
            prior: 1,
            weight: 0.3,
          },
          {
            sourceNodeId: "source:billing-thread",
            targetNodeId: "doc:generic_shipping_policy",
            positive: 2,
            negative: 3,
            support: 5,
            prior: 1,
            weight: 0.05,
          },
        ],
        toolActionPriors: [],
        toolActionSets: [],
      },
    };
    const row = makeGreedyThresholdRouteRow("dataset_activation_first_greedy_lane");
    const scoring = scoreColdStartRouteRowV1({ model, row });
    const topCandidateProbability = scoring.policyDistribution.actions.find((action) => (
      action.action.type === "traverse" && action.action.candidate?.candidate_id === "mem:shipping_correction"
    ))?.probability ?? 0;

    expect(scoring.stopPrediction.label).toBe("CONTINUE");
    expect(scoring.decisionSummary.activationThreshold).toBe(0.45);
    expect(scoring.policyDistribution.stopAction.probability).toBeLessThan(0.55);
    expect(topCandidateProbability).toBeGreaterThan(0.45);
    expect(topCandidateProbability).toBeLessThan(0.5);
    expect(scoring.decisionSummary.activated).toBe(true);
    expect(scoring.decisionSummary.stopReason).toBeNull();
    expect(scoring.rankedCandidates[0]?.candidate.candidate_id).toBe("mem:shipping_correction");
  });

  it("transfers live-policy prior binding across replay-like source ids via semantic class fallback", () => {
    const outputDir = createTempRoot("cold-start-router-semantic-class-live-priors");
    const trained = trainActivationFirstFixture(outputDir, "dataset_activation_first_semantic_class_live_priors");
    const model: ColdStartRouterModelV1 = {
      ...trained.model,
      livePolicyInitializer: {
        ...trained.model.livePolicyInitializer,
        seedWeights: [],
        semanticClassSeedWeights: [
          {
            semanticClass: "event_context",
            positive: 5,
            negative: 0,
            support: 5,
            weight: 0.35,
          },
          {
            semanticClass: "init_context",
            positive: 0,
            negative: 5,
            support: 5,
            weight: -0.35,
          },
        ],
        stopLocalWeights: [
          {
            sourceNodeId: "recorded_session_replay",
            positive: 1,
            negative: 4,
            support: 5,
            weight: 0,
          },
        ],
        edgeWeights: [],
        semanticClassEdgeWeights: [
          {
            sourceBindingKey: "resume_replay_context",
            targetSemanticClass: "event_context",
            positive: 5,
            negative: 0,
            support: 5,
            prior: 1,
            weight: 0.3,
          },
          {
            sourceBindingKey: "resume_replay_context",
            targetSemanticClass: "init_context",
            positive: 0,
            negative: 5,
            support: 5,
            prior: 1,
            weight: -0.05,
          },
        ],
        toolActionPriors: [],
        toolActionSets: [],
      },
    };
    const row: RouteDecisionRowV1 = {
      row_id: "row_replay_semantic_class_transfer",
      dataset_id: "dataset_activation_first_semantic_class_live_priors",
      query: "Recover the replay event instead of defaulting to pointer-aware init",
      cursor_path: ["recorded_session_replay"],
      candidate_set: [
        {
          candidate_id: "pack-runtime:event:alpha",
          candidate_type: "graph_node",
          semantic_class: "event_context",
          authority: "recorded_session_replay",
          freshness: "replay_eval",
          token_cost: 72,
          score_hint: 0.9,
        },
        {
          candidate_id: "pack-runtime:pointer-aware-init",
          candidate_type: "graph_node",
          semantic_class: "init_context",
          authority: "recorded_session_replay",
          freshness: "replay_eval",
          token_cost: 96,
          score_hint: 0.2,
        },
      ],
      teacher_action: { kind: "traverse", target_ids: ["pack-runtime:event:alpha"] },
      stop_label: "CONTINUE",
      evidence_spans: [
        { source_ref: "replay:evidence:0", start: 0, end: 24, excerpt: "Need the replay event." },
      ],
      hard_negatives: ["pack-runtime:pointer-aware-init"],
      outcome_gain: 1,
      provenance: {
        dataset: "dataset_activation_first_semantic_class_live_priors",
        source_license: "internal_local_only",
        source_family: "agent_traces",
        source_snapshot_ref: "snapshot:replay-semantic-class-transfer",
        recorded_by: "test",
        recorded_at: "2026-04-18T05:10:00Z",
        review_status: "approved_train",
      },
      split_tag: "train",
      created_at: "2026-04-18T05:10:00Z",
    };

    const scoring = scoreColdStartRouteRowV1({ model, row });

    expect(scoring.decisionSummary.activated).toBe(true);
    expect(scoring.rankedCandidates[0]?.candidate.candidate_id).toBe("pack-runtime:event:alpha");
    expect(scoring.rankedCandidates[0]?.score).toBeGreaterThan(scoring.rankedCandidates[1]?.score ?? Number.NEGATIVE_INFINITY);
  });

  it("applies the resume-gate fallback floor only to felt/replay event_context semantic edges", () => {
    const outputDir = createTempRoot("cold-start-router-replay-fallback-floor-materialization");
    const trained = trainActivationFirstFixture(outputDir, "dataset_activation_first_replay_fallback_floor_materialization");
    const initializer = {
      ...trained.model.livePolicyInitializer,
      seedWeights: [],
      semanticClassSeedWeights: [],
      edgeWeights: [],
      semanticClassEdgeWeights: [
        {
          sourceBindingKey: "resume_replay_context",
          targetSemanticClass: "event_context",
          positive: 0,
          negative: 0,
          support: 0,
          prior: 1,
          weight: 0,
        },
        {
          sourceBindingKey: "resume_replay_context",
          targetSemanticClass: "init_context",
          positive: 3,
          negative: 1,
          support: 4,
          prior: 1,
          weight: 0.18,
        },
      ],
      toolActionPriors: [],
      toolActionSets: [],
    };
    const feltReplayRow: RouteDecisionRowV1 = {
      row_id: "row_felt_replay_fallback_floor",
      dataset_id: "dataset_activation_first_replay_fallback_floor_materialization",
      query: "Recover the felt replay event instead of pointer-aware init",
      cursor_path: ["felt_resume_25"],
      candidate_set: [
        {
          candidate_id: "pack-runtime:event:alpha",
          candidate_type: "graph_node",
          semantic_class: "event_context",
          authority: "recorded_session_replay",
          freshness: "replay_eval",
          token_cost: 48,
          score_hint: 0.28,
        },
        {
          candidate_id: "pack-runtime:pointer-aware-init",
          candidate_type: "graph_node",
          semantic_class: "init_context",
          authority: "recorded_session_replay",
          freshness: "replay_eval",
          token_cost: 48,
          score_hint: 0.45,
        },
      ],
      teacher_action: { kind: "traverse", target_ids: ["pack-runtime:event:alpha"] },
      stop_label: "CONTINUE",
      evidence_spans: [
        { source_ref: "replay:evidence:0", start: 0, end: 30, excerpt: "Need the replay event." },
      ],
      hard_negatives: ["pack-runtime:pointer-aware-init"],
      outcome_gain: 1,
      provenance: {
        dataset: "dataset_activation_first_replay_fallback_floor_materialization",
        source_license: "internal_local_only",
        source_family: "agent_traces",
        source_snapshot_ref: "snapshot:replay-fallback-floor-materialization",
        recorded_by: "test",
        recorded_at: "2026-04-18T05:20:00Z",
        review_status: "approved_train",
      },
      split_tag: "train",
      created_at: "2026-04-18T05:20:00Z",
    };
    const nonReplayRow: RouteDecisionRowV1 = {
      ...feltReplayRow,
      row_id: "row_non_replay_fallback_floor",
      candidate_set: feltReplayRow.candidate_set.map((candidate) => ({
        ...candidate,
        authority: "snapshot_context",
        freshness: "eval_only",
      })),
    };

    const baseline = materializeColdStartRouterLivePolicyGraphV1({
      initializer,
      row: feltReplayRow,
      applyResumeGateReplaySemanticFallbackBoost: false,
    });
    const boosted = materializeColdStartRouterLivePolicyGraphV1({
      initializer,
      row: feltReplayRow,
      applyResumeGateReplaySemanticFallbackBoost: true,
    });
    const contained = materializeColdStartRouterLivePolicyGraphV1({
      initializer,
      row: nonReplayRow,
      applyResumeGateReplaySemanticFallbackBoost: true,
    });

    expect(baseline.graph.getEdge("felt_resume_25", "pack-runtime:event:alpha")?.weight).toBe(0);
    expect(boosted.graph.getEdge("felt_resume_25", "pack-runtime:event:alpha")?.weight).toBe(0.4);
    expect(boosted.graph.getEdge("felt_resume_25", "pack-runtime:event:alpha")?.metadata).toMatchObject({
      fallbackExperiment: "resume_gate_replay_event_context_fallback_edge_floor.v1",
      fallbackBaseWeight: 0,
      fallbackAdjustedWeight: 0.4,
      fallbackAppliedBoost: 0.4,
    });
    expect(boosted.graph.getEdge("felt_resume_25", "pack-runtime:pointer-aware-init")?.weight).toBe(0.18);
    expect(contained.graph.getEdge("felt_resume_25", "pack-runtime:event:alpha")?.weight).toBe(0);
  });

  it("applies the replay fallback floor to feedback_context without boosting interaction_context", () => {
    const outputDir = createTempRoot("cold-start-router-replay-feedback-floor-materialization");
    const trained = trainActivationFirstFixture(outputDir, "dataset_activation_first_replay_feedback_floor_materialization");
    const initializer = {
      ...trained.model.livePolicyInitializer,
      seedWeights: [],
      semanticClassSeedWeights: [],
      edgeWeights: [],
      semanticClassEdgeWeights: [
        {
          sourceBindingKey: "resume_replay_context",
          targetSemanticClass: "feedback_context",
          positive: 0,
          negative: 0,
          support: 0,
          prior: 1,
          weight: 0,
        },
        {
          sourceBindingKey: "resume_replay_context",
          targetSemanticClass: "interaction_context",
          positive: 2,
          negative: 1,
          support: 3,
          prior: 1,
          weight: 0.22,
        },
      ],
      toolActionPriors: [],
      toolActionSets: [],
    };
    const replayRow: RouteDecisionRowV1 = {
      row_id: "row_replay_feedback_floor",
      dataset_id: "dataset_activation_first_replay_feedback_floor_materialization",
      query: "Prefer the replay feedback block over the replay interaction block",
      cursor_path: ["felt_resume_25"],
      candidate_set: [
        {
          candidate_id: "pack-runtime:event:alpha:feedback",
          candidate_type: "graph_node",
          semantic_class: "feedback_context",
          authority: "recorded_session_replay",
          freshness: "replay_eval",
          token_cost: 64,
          score_hint: 0.3,
        },
        {
          candidate_id: "pack-runtime:event:alpha:interaction",
          candidate_type: "graph_node",
          semantic_class: "interaction_context",
          authority: "recorded_session_replay",
          freshness: "replay_eval",
          token_cost: 56,
          score_hint: 0.45,
        },
      ],
      teacher_action: { kind: "traverse", target_ids: ["pack-runtime:event:alpha:feedback"] },
      stop_label: "CONTINUE",
      evidence_spans: [
        { source_ref: "replay:evidence:0", start: 0, end: 39, excerpt: "Need the replay feedback block, not the cue." },
      ],
      hard_negatives: ["pack-runtime:event:alpha:interaction"],
      outcome_gain: 1,
      provenance: {
        dataset: "dataset_activation_first_replay_feedback_floor_materialization",
        source_license: "internal_local_only",
        source_family: "agent_traces",
        source_snapshot_ref: "snapshot:replay-feedback-floor-materialization",
        recorded_by: "test",
        recorded_at: "2026-04-18T12:05:00Z",
        review_status: "approved_train",
      },
      split_tag: "train",
      created_at: "2026-04-18T12:05:00Z",
    };

    const boosted = materializeColdStartRouterLivePolicyGraphV1({
      initializer,
      row: replayRow,
      applyResumeGateReplaySemanticFallbackBoost: true,
    });

    expect(boosted.graph.getEdge("felt_resume_25", "pack-runtime:event:alpha:feedback")?.weight).toBe(0.4);
    expect(boosted.graph.getEdge("felt_resume_25", "pack-runtime:event:alpha:feedback")?.metadata).toMatchObject({
      fallbackExperiment: "resume_gate_replay_event_context_fallback_edge_floor.v1",
      fallbackBaseWeight: 0,
      fallbackAdjustedWeight: 0.4,
      fallbackAppliedBoost: 0.4,
    });
    expect(boosted.graph.getEdge("felt_resume_25", "pack-runtime:event:alpha:interaction")?.weight).toBe(0.22);
    expect(boosted.graph.getEdge("felt_resume_25", "pack-runtime:event:alpha:interaction")?.metadata).not.toHaveProperty("fallbackExperiment");
  });

  it("reuses exact scoped felt source ids at scoring time when replay rows have one anchored feedback target", () => {
    const outputDir = createTempRoot("cold-start-router-scoped-felt-source-materialization");
    const trained = trainActivationFirstFixture(outputDir, "dataset_activation_first_scoped_felt_source_materialization");
    const initializer = {
      ...trained.model.livePolicyInitializer,
      seedWeights: [],
      semanticClassSeedWeights: [],
      stopLocalWeights: [
        {
          sourceNodeId: "felt_resume_25:source:pack-runtime:event:alpha:feedback",
          positive: 0,
          negative: 4,
          support: 4,
          weight: -0.4,
        },
      ],
      edgeWeights: [
        {
          sourceNodeId: "felt_resume_25:source:pack-runtime:event:alpha:feedback",
          targetNodeId: "pack-runtime:event:alpha:feedback",
          positive: 4,
          negative: 0,
          support: 4,
          prior: 1,
          weight: 0.7,
        },
      ],
      semanticClassEdgeWeights: [
        {
          sourceBindingKey: "resume_replay_context",
          targetSemanticClass: "feedback_context",
          positive: 0,
          negative: 0,
          support: 0,
          prior: 1,
          weight: 0,
        },
        {
          sourceBindingKey: "resume_replay_context",
          targetSemanticClass: "interaction_context",
          positive: 4,
          negative: 0,
          support: 4,
          prior: 1,
          weight: 0.25,
        },
      ],
      toolActionPriors: [],
      toolActionSets: [],
    };
    const row: RouteDecisionRowV1 = {
      row_id: "row_scoped_felt_source_materialization",
      dataset_id: "dataset_activation_first_scoped_felt_source_materialization",
      query: "Use the replay feedback target, not the generic interaction chunk.",
      cursor_path: ["felt_resume_25"],
      candidate_set: [
        {
          candidate_id: "pack-runtime:event:alpha:feedback",
          candidate_type: "graph_node",
          semantic_class: "feedback_context",
          authority: "recorded_session_replay",
          freshness: "replay_eval",
          token_cost: 64,
          score_hint: 0.31,
        },
        {
          candidate_id: "pack-runtime:event:alpha:interaction",
          candidate_type: "graph_node",
          semantic_class: "interaction_context",
          authority: "recorded_session_replay",
          freshness: "replay_eval",
          token_cost: 48,
          score_hint: 0.45,
        },
      ],
      teacher_action: { kind: "traverse", target_ids: ["pack-runtime:event:alpha:feedback"] },
      stop_label: "CONTINUE",
      evidence_spans: [
        { source_ref: "replay:evidence:0", start: 0, end: 38, excerpt: "Need the replay feedback target." },
      ],
      hard_negatives: ["pack-runtime:event:alpha:interaction"],
      outcome_gain: 1,
      provenance: {
        dataset: "dataset_activation_first_scoped_felt_source_materialization",
        source_license: "internal_local_only",
        source_family: "agent_traces",
        source_snapshot_ref: "snapshot:scoped-felt-source-materialization",
        recorded_by: "test",
        recorded_at: "2026-04-18T15:55:00Z",
        review_status: "approved_train",
      },
      split_tag: "train",
      created_at: "2026-04-18T15:55:00Z",
    };

    const materialized = materializeColdStartRouterLivePolicyGraphV1({
      initializer,
      row,
      applyResumeGateReplaySemanticFallbackBoost: true,
    });

    expect(materialized.sourceNodeId).toBe("felt_resume_25:source:pack-runtime:event:alpha:feedback");
    expect(materialized.graph.getEdge(
      "felt_resume_25:source:pack-runtime:event:alpha:feedback",
      "pack-runtime:event:alpha:feedback",
    )?.weight).toBe(0.7);

    const model: ColdStartRouterModelV1 = {
      ...trained.model,
      calibration: {
        ...trained.model.calibration,
        interventionHead: {
          decisionPolicyMode: "gating_only_v1",
          freezeCandidateSelection: true,
          freezeStopLocal: true,
          featureProfile: "resume_gate_v1",
        },
      },
      livePolicyInitializer: initializer,
    };
    const scoring = scoreColdStartRouteRowV1({ model, row });

    expect(scoring.rankedCandidates[0]?.candidate.candidate_id).toBe("pack-runtime:event:alpha:feedback");
  });

  it("keeps shared felt source binding when replay rows do not have exactly one scoped anchor", () => {
    const outputDir = createTempRoot("cold-start-router-scoped-felt-source-ambiguous");
    const trained = trainActivationFirstFixture(outputDir, "dataset_activation_first_scoped_felt_source_ambiguous");
    const initializer = {
      ...trained.model.livePolicyInitializer,
      seedWeights: [],
      semanticClassSeedWeights: [],
      stopLocalWeights: [
        {
          sourceNodeId: "felt_resume_25:source:pack-runtime:event:alpha:feedback",
          positive: 0,
          negative: 4,
          support: 4,
          weight: -0.4,
        },
      ],
      edgeWeights: [
        {
          sourceNodeId: "felt_resume_25:source:pack-runtime:event:alpha:feedback",
          targetNodeId: "pack-runtime:event:alpha:feedback",
          positive: 4,
          negative: 0,
          support: 4,
          prior: 1,
          weight: 0.7,
        },
      ],
      semanticClassEdgeWeights: [],
      toolActionPriors: [],
      toolActionSets: [],
    };
    const row: RouteDecisionRowV1 = {
      row_id: "row_scoped_felt_source_ambiguous",
      dataset_id: "dataset_activation_first_scoped_felt_source_ambiguous",
      query: "Two replay anchors are present, so keep the shared binding.",
      cursor_path: ["felt_resume_25"],
      candidate_set: [
        {
          candidate_id: "pack-runtime:event:alpha:feedback",
          candidate_type: "graph_node",
          semantic_class: "feedback_context",
          authority: "recorded_session_replay",
          freshness: "replay_eval",
          token_cost: 64,
          score_hint: 0.31,
        },
        {
          candidate_id: "pack-runtime:event:beta:feedback",
          candidate_type: "graph_node",
          semantic_class: "feedback_context",
          authority: "recorded_session_replay",
          freshness: "replay_eval",
          token_cost: 60,
          score_hint: 0.29,
        },
      ],
      teacher_action: { kind: "traverse", target_ids: ["pack-runtime:event:alpha:feedback"] },
      stop_label: "CONTINUE",
      evidence_spans: [
        { source_ref: "replay:evidence:0", start: 0, end: 30, excerpt: "Ambiguous replay anchors." },
      ],
      hard_negatives: ["pack-runtime:event:beta:feedback"],
      outcome_gain: 1,
      provenance: {
        dataset: "dataset_activation_first_scoped_felt_source_ambiguous",
        source_license: "internal_local_only",
        source_family: "agent_traces",
        source_snapshot_ref: "snapshot:scoped-felt-source-ambiguous",
        recorded_by: "test",
        recorded_at: "2026-04-18T15:56:00Z",
        review_status: "approved_train",
      },
      split_tag: "train",
      created_at: "2026-04-18T15:56:00Z",
    };

    const materialized = materializeColdStartRouterLivePolicyGraphV1({
      initializer,
      row,
      applyResumeGateReplaySemanticFallbackBoost: true,
    });

    expect(materialized.sourceNodeId).toBe("felt_resume_25");
    expect(materialized.graph.getEdge("felt_resume_25", "pack-runtime:event:alpha:feedback")).toBeUndefined();
  });

  it("uses the resume-gate fallback floor to flip replay ranking without touching non-replay rows", () => {
    const outputDir = createTempRoot("cold-start-router-replay-fallback-floor-scoring");
    const trained = trainActivationFirstFixture(outputDir, "dataset_activation_first_replay_fallback_floor_scoring");
    const baseModel: ColdStartRouterModelV1 = {
      ...trained.model,
      calibration: {
        ...trained.model.calibration,
        interventionHead: {
          decisionPolicyMode: "gating_only_v1",
          freezeCandidateSelection: true,
          freezeStopLocal: true,
          featureProfile: "default_router",
        },
      },
      livePolicyInitializer: {
        ...trained.model.livePolicyInitializer,
        seedWeights: [],
        semanticClassSeedWeights: [],
        stopLocalWeights: [
          {
            sourceNodeId: "recorded_session_replay",
            positive: 1,
            negative: 4,
            support: 5,
            weight: 0,
          },
        ],
        edgeWeights: [],
        semanticClassEdgeWeights: [
          {
            sourceBindingKey: "resume_replay_context",
            targetSemanticClass: "event_context",
            positive: 0,
            negative: 0,
            support: 0,
            prior: 1,
            weight: 0,
          },
          {
            sourceBindingKey: "resume_replay_context",
            targetSemanticClass: "init_context",
            positive: 3,
            negative: 1,
            support: 4,
            prior: 1,
            weight: 0.18,
          },
        ],
        toolActionPriors: [],
        toolActionSets: [],
      },
    };
    const replayRow: RouteDecisionRowV1 = {
      row_id: "row_replay_fallback_floor_scoring",
      dataset_id: "dataset_activation_first_replay_fallback_floor_scoring",
      query: "Use the replay event, not the init fallback.",
      cursor_path: ["recorded_session_replay"],
      candidate_set: [
        {
          candidate_id: "pack-runtime:event:alpha",
          candidate_type: "graph_node",
          semantic_class: "event_context",
          authority: "recorded_session_replay",
          freshness: "replay_eval",
          token_cost: 48,
          score_hint: 0.28,
        },
        {
          candidate_id: "pack-runtime:pointer-aware-init",
          candidate_type: "graph_node",
          semantic_class: "init_context",
          authority: "recorded_session_replay",
          freshness: "replay_eval",
          token_cost: 48,
          score_hint: 0.45,
        },
      ],
      teacher_action: { kind: "traverse", target_ids: ["pack-runtime:event:alpha"] },
      stop_label: "CONTINUE",
      evidence_spans: [
        { source_ref: "replay:evidence:0", start: 0, end: 30, excerpt: "Need the replay event." },
      ],
      hard_negatives: ["pack-runtime:pointer-aware-init"],
      outcome_gain: 1,
      provenance: {
        dataset: "dataset_activation_first_replay_fallback_floor_scoring",
        source_license: "internal_local_only",
        source_family: "agent_traces",
        source_snapshot_ref: "snapshot:replay-fallback-floor-scoring",
        recorded_by: "test",
        recorded_at: "2026-04-18T05:25:00Z",
        review_status: "approved_train",
      },
      split_tag: "train",
      created_at: "2026-04-18T05:25:00Z",
    };
    const baselineScoring = scoreColdStartRouteRowV1({ model: baseModel, row: replayRow });
    const boostedScoring = scoreColdStartRouteRowV1({
      model: {
        ...baseModel,
        calibration: {
          ...baseModel.calibration,
          interventionHead: {
            ...baseModel.calibration.interventionHead,
            featureProfile: "resume_gate_v1",
          },
        },
      },
      row: replayRow,
    });
    const nonReplayScoring = scoreColdStartRouteRowV1({
      model: {
        ...baseModel,
        calibration: {
          ...baseModel.calibration,
          interventionHead: {
            ...baseModel.calibration.interventionHead,
            featureProfile: "resume_gate_v1",
          },
        },
      },
      row: {
        ...replayRow,
        row_id: "row_non_replay_fallback_floor_scoring",
        candidate_set: replayRow.candidate_set.map((candidate) => ({
          ...candidate,
          authority: "snapshot_context",
          freshness: "eval_only",
        })),
      },
    });

    expect(baselineScoring.rankedCandidates[0]?.candidate.candidate_id).toBe("pack-runtime:pointer-aware-init");
    expect(boostedScoring.rankedCandidates[0]?.candidate.candidate_id).toBe("pack-runtime:event:alpha");
    expect(nonReplayScoring.rankedCandidates[0]?.candidate.candidate_id).toBe("pack-runtime:pointer-aware-init");
  });

  it("predicts STOP_LOCAL for the two MuSiQue replay rows that only have a single evidence span", () => {
    const loadedExport = loadAndFilterColdStartRouterApprovedExportV1(approvedExportV2Path);
    const runtimeBundle = loadColdStartRouterArtifactBundleV1(approvedTrainV2Dir);

    for (const rowId of ["musique-dev-export-candidate-1", "musique-dev-export-candidate-11"]) {
      const row = loadedExport.routeRows.find((entry) => entry.row_id === rowId);
      expect(row).toBeDefined();
      const teacherAction = row!.teacher_action;
      if (teacherAction.kind !== "traverse") {
        throw new Error(`expected traverse action for ${rowId}`);
      }
      const scoring = scoreColdStartRouteRowFromArtifactBundleV1({ artifactBundle: runtimeBundle, row: row! });
      expect(scoring.stopPrediction.label).toBe("STOP_LOCAL");
      expect(scoring.policyDistribution.stopAction.probability).toBeGreaterThan(0);
      expect(scoring.decisionSummary.stopLocalProbability).toBeGreaterThan(0);
      expect(scoring.rankedCandidates[0]?.candidate.candidate_id).toBe(teacherAction.target_ids[0]);
    }
  });

  it("continues from a warm-start bundle and matches a full retrain on the accumulated rows", () => {
    const loadedExport = loadAndFilterColdStartRouterApprovedExportV1(approvedExportPath);
    const priorOutputDir = createTempRoot("cold-start-router-warm-start-prior");
    const fullOutputDir = createTempRoot("cold-start-router-warm-start-full");
    const warmOutputDir = createTempRoot("cold-start-router-warm-start-continuation");

    trainColdStartRouterArtifactV1({
      artifactId: "router-artifact-warm-start-prior",
      artifactVersion: "0.0.1",
      packType: "base",
      compatibleRuntimeVersion: "openclawbrain-runtime@0.4.43",
      registryEntries: loadedExport.registryEntries,
      routeRows: loadedExport.routeRows.slice(0, 1),
      outputDir: priorOutputDir,
      routerIdentity: "router:warm-start:prior",
      createdAt: "2026-04-05T16:22:00Z",
      trainingDataRefs: loadedExport.summary.approvedDatasetIds,
      replayGateRefs: ["replay:approved-export:fixture-v1"],
    });

    const priorBundle = loadColdStartRouterArtifactBundleV1(priorOutputDir);
    const full = trainColdStartRouterArtifactV1({
      artifactId: "router-artifact-warm-start-target",
      artifactVersion: "0.0.2",
      packType: "base",
      compatibleRuntimeVersion: "openclawbrain-runtime@0.4.43",
      registryEntries: loadedExport.registryEntries,
      routeRows: loadedExport.routeRows,
      outputDir: fullOutputDir,
      routerIdentity: "router:warm-start:target",
      createdAt: "2026-04-05T16:25:00Z",
      trainingDataRefs: loadedExport.summary.approvedDatasetIds,
      replayGateRefs: ["replay:approved-export:fixture-v1"],
    });
    const warm = trainColdStartRouterArtifactV1({
      artifactId: "router-artifact-warm-start-target",
      artifactVersion: "0.0.2",
      packType: "base",
      compatibleRuntimeVersion: "openclawbrain-runtime@0.4.43",
      registryEntries: loadedExport.registryEntries,
      routeRows: loadedExport.routeRows.slice(1),
      outputDir: warmOutputDir,
      routerIdentity: "router:warm-start:target",
      createdAt: "2026-04-05T16:25:00Z",
      trainingDataRefs: loadedExport.summary.approvedDatasetIds,
      replayGateRefs: ["replay:approved-export:fixture-v1"],
      warmStartArtifactBundle: {
        artifactDir: priorBundle.artifactDir,
        manifest: priorBundle.manifest,
        model: priorBundle.model,
      },
      warmStartMode: "strict",
    });

    expect(warm.model).toEqual(full.model);
    expect(warm.manifest.warm_start_applied).toBe(true);
    expect(warm.manifest.warm_start_from_artifact_id).toBe(priorBundle.manifest.artifact_id);
    expect(warm.manifest.warm_start_from_artifact_checksum).toBe(priorBundle.manifest.artifact_checksum);
    expect(warm.manifest.prior_base_artifact_id).toBe(priorBundle.manifest.artifact_id);
    expect(warm.manifest.prior_base_artifact_checksum).toBe(priorBundle.manifest.artifact_checksum);
    expect(warm.manifest.warm_start_summary).toContain("warm-start continuation");
    expect(loadColdStartRouterArtifactBundleV1(warmOutputDir).manifest.warm_start_applied).toBe(true);
  });

  it("materializes the live runtime policy families that hot-path PG updates", () => {
    const outputDir = createTempRoot("cold-start-router-live-family");
    const loadedExport = loadAndFilterColdStartRouterApprovedExportV1(approvedExportPath);

    trainColdStartRouterArtifactV1({
      artifactId: "router-artifact-live-family",
      artifactVersion: "0.0.1",
      packType: "base",
      compatibleRuntimeVersion: "openclawbrain-runtime@0.4.43",
      registryEntries: loadedExport.registryEntries,
      routeRows: loadedExport.routeRows,
      outputDir,
      routerIdentity: "router:approved-export:live-family",
      createdAt: "2026-04-05T16:30:00Z",
      trainingDataRefs: loadedExport.summary.approvedDatasetIds,
      replayGateRefs: ["replay:approved-export:fixture-v1"],
    });

    const runtimeBundle = loadColdStartRouterArtifactBundleV1(outputDir);
    const row = loadedExport.routeRows[0];
    const materialized = materializeColdStartRouterLivePolicyFromArtifactBundleV1({
      artifactBundle: runtimeBundle,
      row,
    });

    expect(materialized.sourceNodeId).toBe("mem:user_profile");
    expect(materialized.graph.getToolActionPrior(materialized.sourceNodeId, "tool:gmail_search")).not.toBe(0);
    expect(materialized.graph.getActionSet(materialized.sourceNodeId, new Set<string>()).some((action) => action.type === "traverse" && action.targetNodeId === "tool:gmail_search")).toBe(true);

    const traverseAction: TraversalAction = { type: "traverse", targetNodeId: "mem:shipping_history" };
    const stopAction: TraversalAction = { type: "stop_local" };
    const seedState: TraversalState = {
      sourceNodeId: null,
      queryEmbedding: new Float32Array(0),
      frontier: [],
      visited: new Set(),
      fired: [],
      budgetRemaining: 1000,
      initialBudget: 1000,
      reservedTokenCost: 0,
      expansionCount: 0,
      maxHops: 8,
    };
    const localState: TraversalState = {
      sourceNodeId: materialized.sourceNodeId,
      queryEmbedding: new Float32Array(0),
      frontier: [],
      visited: new Set([materialized.sourceNodeId ?? ""]),
      fired: [],
      budgetRemaining: 1000,
      initialBudget: 1000,
      reservedTokenCost: 0,
      expansionCount: 1,
      maxHops: 8,
    };

    const beforeSeedWeight = materialized.graph.getSeedWeight("mem:shipping_history");
    const beforeEdgeWeight = materialized.graph.getEdge("mem:user_profile", "mem:shipping_history")?.weight ?? 0;
    const beforeStopWeight = materialized.graph.getStopLocalWeight(materialized.sourceNodeId);

    const seedEpisode: Episode = {
      id: "live-family-seed",
      conversationId: null,
      queryText: row.query,
      queryEmbedding: new Float32Array(0),
      trajectory: [{
        sourceNodeId: null,
        expansionIndex: 0,
        frontierBefore: [],
        frontierAfter: ["mem:shipping_history"],
        budgetBefore: 1000,
        budgetAfter: 900,
        substeps: [{
          stateSnapshot: {
            sourceNodeId: null,
            expansionIndex: 0,
            selectionIndex: 0,
            budgetRemaining: 1000,
            initialBudget: 1000,
            reservedTokenCost: 0,
            maxHops: 8,
            frontierSize: 0,
            frontierNodeIds: [],
            visitedCount: 0,
            firedCount: 0,
          },
          candidates: [
            { action: traverseAction, score: scoreAction(traverseAction, seedState, materialized.graph, materialized.policyParams), probability: 0.8 },
            { action: stopAction, score: scoreAction(stopAction, seedState, materialized.graph, materialized.policyParams), probability: 0.2 },
          ],
          chosenAction: traverseAction,
          chosenActionProbability: 0.8,
          stopProbability: 0.2,
        }],
        selectedTargets: ["mem:shipping_history"],
        acceptedTargets: ["mem:shipping_history"],
        vetoedTargets: [],
      }],
      firedNodes: ["mem:shipping_history"],
      vetoedNodes: [],
      contextChars: 0,
      reward: 1,
      rewardSource: "human",
      packVersion: 1,
      createdAt: Date.now(),
    };
    const seedUpdates = computeReinforceUpdates(seedEpisode, 0.1, 0.0);
    applyWeightUpdates(materialized.graph, seedUpdates);
    expect(materialized.graph.getSeedWeight("mem:shipping_history")).toBeGreaterThan(beforeSeedWeight);

    const edgeEpisode: Episode = {
      id: "live-family-edge",
      conversationId: null,
      queryText: row.query,
      queryEmbedding: new Float32Array(0),
      trajectory: [{
        sourceNodeId: materialized.sourceNodeId,
        expansionIndex: 0,
        frontierBefore: ["mem:shipping_history"],
        frontierAfter: ["mem:shipping_history"],
        budgetBefore: 1000,
        budgetAfter: 900,
        substeps: [{
          stateSnapshot: {
            sourceNodeId: materialized.sourceNodeId,
            expansionIndex: 0,
            selectionIndex: 0,
            budgetRemaining: 1000,
            initialBudget: 1000,
            reservedTokenCost: 0,
            maxHops: 8,
            frontierSize: 0,
            frontierNodeIds: [],
            visitedCount: 1,
            firedCount: 0,
          },
          candidates: [
            { action: traverseAction, score: scoreAction(traverseAction, localState, materialized.graph, materialized.policyParams), probability: 0.7 },
            { action: stopAction, score: scoreAction(stopAction, localState, materialized.graph, materialized.policyParams), probability: 0.3 },
          ],
          chosenAction: traverseAction,
          chosenActionProbability: 0.7,
          stopProbability: 0.3,
        }],
        selectedTargets: ["mem:shipping_history"],
        acceptedTargets: ["mem:shipping_history"],
        vetoedTargets: [],
      }],
      firedNodes: ["mem:shipping_history"],
      vetoedNodes: [],
      contextChars: 0,
      reward: 1,
      rewardSource: "human",
      packVersion: 1,
      createdAt: Date.now(),
    };
    const edgeUpdates = computeReinforceUpdates(edgeEpisode, 0.1, 0.0);
    applyWeightUpdates(materialized.graph, edgeUpdates);
    expect(materialized.graph.getEdge("mem:user_profile", "mem:shipping_history")?.weight ?? 0).toBeGreaterThan(beforeEdgeWeight);

    const stopEpisode: Episode = {
      id: "live-family-stop",
      conversationId: null,
      queryText: row.query,
      queryEmbedding: new Float32Array(0),
      trajectory: [{
        sourceNodeId: materialized.sourceNodeId,
        expansionIndex: 0,
        frontierBefore: ["mem:shipping_history"],
        frontierAfter: [],
        budgetBefore: 1000,
        budgetAfter: 1000,
        substeps: [{
          stateSnapshot: {
            sourceNodeId: materialized.sourceNodeId,
            expansionIndex: 0,
            selectionIndex: 0,
            budgetRemaining: 1000,
            initialBudget: 1000,
            reservedTokenCost: 0,
            maxHops: 8,
            frontierSize: 0,
            frontierNodeIds: [],
            visitedCount: 1,
            firedCount: 0,
          },
          candidates: [
            { action: traverseAction, score: scoreAction(traverseAction, localState, materialized.graph, materialized.policyParams), probability: 0.4 },
            { action: stopAction, score: scoreAction(stopAction, localState, materialized.graph, materialized.policyParams), probability: 0.6 },
          ],
          chosenAction: stopAction,
          chosenActionProbability: 0.6,
          stopProbability: 0.6,
          stopTruth: "chosen",
          stopReason: "policy_stop",
        }],
        selectedTargets: [],
        acceptedTargets: [],
        vetoedTargets: [],
        terminationReason: "policy_stop",
      }],
      firedNodes: [],
      vetoedNodes: [],
      contextChars: 0,
      reward: 1,
      rewardSource: "human",
      packVersion: 1,
      createdAt: Date.now(),
    };
    const stopUpdates = computeReinforceUpdates(stopEpisode, 0.1, 0.0, materialized.graph);
    applyWeightUpdates(materialized.graph, stopUpdates);
    expect(materialized.graph.getStopLocalWeight(materialized.sourceNodeId)).toBeGreaterThan(beforeStopWeight);

    const beforeToolWeight = materialized.graph.getToolActionPrior(materialized.sourceNodeId, "tool:gmail_search");
    const toolEpisode: Episode = {
      id: "live-family-tool",
      conversationId: null,
      queryText: row.query,
      queryEmbedding: new Float32Array(0),
      trajectory: [{
        sourceNodeId: materialized.sourceNodeId,
        expansionIndex: 1,
        frontierBefore: ["mem:shipping_history"],
        frontierAfter: ["tool:gmail_search"],
        budgetBefore: 1000,
        budgetAfter: 900,
        substeps: [{
          stateSnapshot: {
            sourceNodeId: materialized.sourceNodeId,
            expansionIndex: 1,
            selectionIndex: 0,
            budgetRemaining: 1000,
            initialBudget: 1000,
            reservedTokenCost: 0,
            maxHops: 8,
            frontierSize: 0,
            frontierNodeIds: [],
            visitedCount: 1,
            firedCount: 0,
          },
          candidates: [
            { action: { type: "traverse", targetNodeId: "tool:gmail_search" }, score: 0.9, probability: 0.7 },
            { action: stopAction, score: 0.1, probability: 0.3 },
          ],
          chosenAction: { type: "traverse", targetNodeId: "tool:gmail_search" },
          chosenActionProbability: 0.7,
          stopProbability: 0.3,
        }],
        selectedTargets: ["tool:gmail_search"],
        acceptedTargets: ["tool:gmail_search"],
        vetoedTargets: [],
      }],
      firedNodes: ["tool:gmail_search"],
      vetoedNodes: [],
      contextChars: 0,
      reward: 1,
      rewardSource: "human",
      packVersion: 1,
      createdAt: Date.now(),
    };
    const toolUpdates = computeReinforceUpdates(toolEpisode, 0.1, 0.0, materialized.graph);
    applyWeightUpdates(materialized.graph, toolUpdates);
    expect(materialized.graph.getToolActionPrior(materialized.sourceNodeId, "tool:gmail_search")).toBeGreaterThan(beforeToolWeight);
  });

  it("exposes a runnable smoke script backed by the approved export fixture", () => {
    const stdout = execFileSync(
      "node",
      [
        "--experimental-transform-types",
        "scripts/train-cold-start-router-smoke.ts",
      ],
      {
        cwd: repoRoot,
        encoding: "utf8",
      },
    );

    expect(stdout).toContain("Cold-start router smoke: ok");
    expect(stdout).toContain("approvedExportSummary:");
    expect(stdout).toContain("manifestChecksum:");
    expect(stdout).toContain("topCandidate: mem:shipping_history");
    expect(stdout).toContain("stopPrediction:");
  });
});
