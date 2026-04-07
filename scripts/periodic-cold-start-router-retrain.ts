#!/usr/bin/env node

import path from "node:path";
import { fileURLToPath } from "node:url";

import { runColdStartRouterPeriodicRetrainV1 } from "../src/brain-core/cold-start-router-periodic-retrain.ts";
import { readContinuousLearningControl } from "../src/brain-runtime/continuous-learning-status.ts";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..");
const workspaceRoot = path.resolve(repoRoot, "..");

const trainExportPath = process.env.COLD_START_TRAIN_EXPORT_PATH ?? path.join(
  repoRoot,
  "artifacts",
  "cold-start-router-approved-export",
  "real-approved-router-export.hotpotqa-musique.v3.json",
);
const evalExportPath = process.env.COLD_START_EVAL_EXPORT_PATH ?? path.join(
  repoRoot,
  "artifacts",
  "cold-start-router-approved-export",
  "real-disjoint-eval-only-router-export.hotpotqa-musique.v1.json",
);
const priorBaseArtifactDir = process.env.COLD_START_PRIOR_BASE_ARTIFACT_DIR ?? path.join(
  repoRoot,
  "artifacts",
  "cold-start-router-approved-export",
  "real-approved-router-train.hotpotqa-musique.v3",
);
const candidateArtifactDir = process.env.COLD_START_CANDIDATE_ARTIFACT_DIR ?? path.join(
  repoRoot,
  "scratch",
  "cold-start-router-periodic-retrain",
  "candidate.v1",
);
const reportDir = process.env.COLD_START_REPORT_DIR ?? path.join(
  repoRoot,
  "scratch",
  "cold-start-router-periodic-retrain",
  "report.v1",
);

async function main(): Promise<void> {
  const retrainControl = readContinuousLearningControl(workspaceRoot, "retrain");
  if (retrainControl?.paused) {
    console.log(JSON.stringify({
      generatedAt: new Date().toISOString(),
      status: "paused",
      control: retrainControl,
      workspaceRoot,
      reportDir,
      candidateArtifactDir,
      trainExportPath,
      evalExportPath,
      priorBaseArtifactDir,
    }, null, 2));
    return;
  }

  const result = runColdStartRouterPeriodicRetrainV1({
    trainExportPath,
    evalExportPath,
    candidateArtifactDir,
    reportDir,
    candidateArtifactId: process.env.COLD_START_CANDIDATE_ARTIFACT_ID ?? "router-artifact-periodic-retrain-v1",
    candidateArtifactVersion: process.env.COLD_START_CANDIDATE_ARTIFACT_VERSION ?? "0.0.1",
    candidateRouterIdentity: process.env.COLD_START_CANDIDATE_ROUTER_IDENTITY ?? "router:periodic-retrain:v1",
    compatibleRuntimeVersion: process.env.COLD_START_COMPATIBLE_RUNTIME_VERSION ?? "openclawbrain-runtime@0.3.8",
    packType: (process.env.COLD_START_PACK_TYPE ?? "base") as "base" | "user_delta" | "mixed",
    registryId: process.env.COLD_START_REGISTRY_ID ?? "cold-start-router-periodic-retrain-v1",
    previousBaseArtifactDir: priorBaseArtifactDir,
    previousBaseArtifactId: process.env.COLD_START_PREVIOUS_BASE_ARTIFACT_ID ?? "router-artifact-real-approved-export-hotpotqa-musique-stoplocal-v3",
    trainingDataRefs: process.env.COLD_START_TRAINING_DATA_REFS ? process.env.COLD_START_TRAINING_DATA_REFS.split(",") : undefined,
    replayGateRefs: process.env.COLD_START_REPLAY_GATE_REFS ? process.env.COLD_START_REPLAY_GATE_REFS.split(",") : undefined,
    createdAt: process.env.COLD_START_CREATED_AT,
  });

  console.log(JSON.stringify({
    generatedAt: result.generatedAt,
    splitRegistryPath: result.paths.splitRegistryPath,
    replayReportPath: result.paths.replayReportPath,
    promotionPackagePath: result.paths.promotionPackagePath,
    candidateArtifactDir: result.candidate.outputDir,
    candidateManifestChecksum: result.candidate.manifest.artifact_checksum,
    trainReplay: result.trainReplay,
    evalReplay: result.evalReplay,
    gatePassed: result.report.gatePassed,
    promotionDecision: result.promotionPackage.decision,
    summary: result.report.summary,
  }, null, 2));
}

main().catch((error) => {
  console.error(error instanceof Error ? error.stack ?? error.message : String(error));
  process.exitCode = 1;
});
