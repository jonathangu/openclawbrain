#!/usr/bin/env node

import { mkdirSync, rmSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import {
  buildColdStartQaSnapshotExportCandidateV1,
} from "../src/brain-core/cold-start-qa-export-candidate.ts";
import {
  loadAndFilterColdStartRouterApprovedExportV1,
} from "../src/brain-core/cold-start-router-approved-export-loader.ts";
import {
  replayColdStartRouterArtifactV1,
} from "../src/brain-core/cold-start-router-replay-gate.ts";
import {
  summarizeRouterArtifactManifestV1,
  validateRouterArtifactManifestV1,
} from "../src/brain-core/cold-start-router-contracts.ts";
import {
  scoreColdStartRouteRowFromArtifactBundleV1,
  loadColdStartRouterArtifactBundleV1,
} from "../src/brain-core/cold-start-router-runtime.ts";
import {
  trainColdStartRouterArtifactV1,
} from "../src/brain-core/cold-start-router-trainer.ts";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..");
const workspaceRoot = path.resolve(repoRoot, "..");
const registryPath = path.join(repoRoot, "data", "cold-start", "registry.bootstrap.json");
const previousExportPath = path.join(
  repoRoot,
  "artifacts",
  "cold-start-router-approved-export",
  "real-approved-router-export.hotpotqa-musique.v2.json",
);
const exportPath = path.join(
  repoRoot,
  "artifacts",
  "cold-start-router-approved-export",
  "real-approved-router-export.hotpotqa-musique.v3.json",
);
const trainDir = path.join(
  repoRoot,
  "artifacts",
  "cold-start-router-approved-export",
  "real-approved-router-train.hotpotqa-musique.v3",
);

function ensureCleanDir(dirPath: string): void {
  rmSync(dirPath, { recursive: true, force: true });
  mkdirSync(dirPath, { recursive: true });
}

function countStopLocal(routeRows: Array<{ stop_label: string }>): number {
  return routeRows.filter((row) => row.stop_label === "STOP_LOCAL").length;
}

async function main(): Promise<void> {
  const previousExport = loadAndFilterColdStartRouterApprovedExportV1(previousExportPath);

  const candidate = buildColdStartQaSnapshotExportCandidateV1({
    registryPath,
    workspaceRoot,
    generatedAt: "2026-04-06T00:40:00Z",
    hotpotSampleCount: 3,
    musiqueSampleCount: 100,
  });

  const approvedExport = {
    contract: "cold_start_router_approved_export.v1",
    export_id: "cold-start-router-approved-export-real-hotpotqa-musique-stoplocal-v3",
    generated_at: candidate.generated_at,
    registry_entries: candidate.registry_entries,
    route_rows: candidate.route_rows.map((row) => row.teacher_action.kind === "traverse"
      ? { ...row, teacher_action: { kind: "traverse", target_ids: [...row.teacher_action.target_ids].sort() } }
      : row),
    notes: [
      "Real governed approved export compiled from the explicitly approved HotpotQA and MuSiQue snapshot-backed QA rows.",
      "This expansion widens STOP_LOCAL coverage by including the first 100 supporting MuSiQue examples while preserving the approved HotpotQA rows.",
      "Registry eligibility is preserved exactly as emitted by the governed source-intake registry and snapshot compiler.",
    ],
  };

  mkdirSync(path.dirname(exportPath), { recursive: true });
  writeFileSync(exportPath, `${JSON.stringify(approvedExport, null, 2)}\n`, "utf8");

  const loadedExport = loadAndFilterColdStartRouterApprovedExportV1(exportPath);
  ensureCleanDir(trainDir);
  const result = trainColdStartRouterArtifactV1({
    artifactId: "router-artifact-real-approved-export-hotpotqa-musique-stoplocal-v3",
    artifactVersion: "0.0.3",
    packType: "base",
    compatibleRuntimeVersion: "openclawbrain-runtime@0.4.43",
    registryEntries: loadedExport.registryEntries,
    routeRows: loadedExport.routeRows,
    outputDir: trainDir,
    routerIdentity: "router:real-approved-export:hotpotqa-musique-stoplocal-v3",
    createdAt: candidate.generated_at,
    trainingDataRefs: loadedExport.summary.approvedDatasetIds,
    replayGateRefs: ["replay:real-approved-export:hotpotqa-musique-stoplocal-v3"],
  });

  const manifestValidation = validateRouterArtifactManifestV1(result.manifest);
  if (!manifestValidation.valid) {
    throw new Error(`manifest validation failed: ${manifestValidation.issues.join("; ")}`);
  }

  const runtimeBundle = loadColdStartRouterArtifactBundleV1(trainDir);
  const replayVerdict = replayColdStartRouterArtifactV1({
    artifactDir: trainDir,
    routeRows: loadedExport.routeRows,
  });

  const firstRow = loadedExport.routeRows[0];
  const firstRowScore = firstRow
    ? scoreColdStartRouteRowFromArtifactBundleV1({ artifactBundle: runtimeBundle, row: firstRow })
    : null;

  const summary = {
    before: {
      exportPath: previousExportPath,
      approvedRowCount: previousExport.summary.approvedRowCount,
      stopLocalRowCount: countStopLocal(previousExport.routeRows),
      approvedDatasetIds: previousExport.summary.approvedDatasetIds,
    },
    after: {
      exportPath,
      approvedRowCount: loadedExport.summary.approvedRowCount,
      stopLocalRowCount: countStopLocal(loadedExport.routeRows),
      approvedDatasetIds: loadedExport.summary.approvedDatasetIds,
    },
    trainDir,
    manifestSummary: summarizeRouterArtifactManifestV1(result.manifest),
    training: result.model.training,
    usedDatasetIds: result.model.training.usedDatasetIds,
    replay: replayVerdict,
    firstRowScore: firstRowScore
      ? {
          rowId: firstRow.row_id,
          topCandidateId: firstRowScore.rankedCandidates[0]?.candidate.candidate_id ?? null,
          topCandidateProbability: firstRowScore.policyDistribution.actions[0]?.probability ?? 0,
          stopProbability: firstRowScore.policyDistribution.stopAction.probability,
        }
      : null,
    loadMatch: runtimeBundle.manifest.artifact_checksum === result.manifest.artifact_checksum,
    addedStopLocalRowIds: loadedExport.routeRows
      .filter((row) => row.stop_label === "STOP_LOCAL")
      .map((row) => row.row_id)
      .filter((rowId) => !previousExport.routeRows.some((row) => row.row_id === rowId)),
  };

  console.log(JSON.stringify(summary, null, 2));
}

main().catch((error) => {
  console.error(error instanceof Error ? error.stack ?? error.message : String(error));
  process.exitCode = 1;
});
