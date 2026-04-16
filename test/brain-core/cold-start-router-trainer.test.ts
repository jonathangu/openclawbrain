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
      activationThreshold: 0.5,
      abstentionThreshold: 0.55,
      expectedUtilityThreshold: 0,
      stopLocalThreshold: 0.5,
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
      activationThreshold: 0.5,
      abstentionThreshold: 0.55,
      expectedUtilityThreshold: 0,
      stopLocalThreshold: 0.5,
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
