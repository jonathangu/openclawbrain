#!/usr/bin/env node

import { mkdtempSync, rmSync, mkdirSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { validateRouterArtifactManifestV1 } from "../src/brain-core/cold-start-router-contracts.ts";
import { loadAndFilterColdStartRouterApprovedExportV1 } from "../src/brain-core/cold-start-router-approved-export-loader.ts";
import {
  loadColdStartRouterArtifactBundleV1,
  scoreColdStartRouteRowFromArtifactBundleV1,
  selectColdStartRouteCandidateIdsFromArtifactBundleV1,
} from "../src/brain-core/cold-start-router-runtime.ts";
import {
  predictColdStartStopLabelV1,
  rankColdStartRouteCandidatesV1,
  scoreColdStartRouteRowV1,
  trainColdStartRouterArtifactV1,
} from "../src/brain-core/cold-start-router-trainer.ts";

const tempRoots: string[] = [];
const approvedExportPath = fileURLToPath(
  new URL("../artifacts/cold-start-router-approved-export/approved-router-export.fixture.v1.json", import.meta.url),
);

function createTempRoot(label: string): string {
  const root = mkdtempSync(path.join(os.tmpdir(), `${label}-`));
  tempRoots.push(root);
  return root;
}

function cleanup(): void {
  while (tempRoots.length > 0) {
    rmSync(tempRoots.pop()!, { recursive: true, force: true });
  }
}

async function main(): Promise<void> {
  const outputDir = createTempRoot("cold-start-router-train-smoke");
  mkdirSync(outputDir, { recursive: true });

  const loadedExport = loadAndFilterColdStartRouterApprovedExportV1(approvedExportPath);
  const result = trainColdStartRouterArtifactV1({
    artifactId: "router-artifact-approved-export-smoke",
    artifactVersion: "0.0.1",
    packType: "base",
    compatibleRuntimeVersion: "openclawbrain-runtime@0.3.8",
    registryEntries: loadedExport.registryEntries,
    routeRows: loadedExport.routeRows,
    outputDir,
    routerIdentity: "router:approved-export:base",
    createdAt: "2026-04-05T16:20:00Z",
    trainingDataRefs: loadedExport.summary.approvedDatasetIds,
    replayGateRefs: ["replay:approved-export:fixture-v1"],
  });

  const manifestValidation = validateRouterArtifactManifestV1(result.manifest);
  if (!manifestValidation.valid) {
    throw new Error(`manifest validation failed: ${manifestValidation.issues.join("; ")}`);
  }

  const runtimeBundle = loadColdStartRouterArtifactBundleV1(result.outputDir);
  const scoring = scoreColdStartRouteRowV1({ model: result.model, row: loadedExport.routeRows[0] });
  const runtimeScoring = scoreColdStartRouteRowFromArtifactBundleV1({ artifactBundle: runtimeBundle, row: loadedExport.routeRows[0] });
  const ranking = rankColdStartRouteCandidatesV1({ model: result.model, candidates: loadedExport.routeRows[0].candidate_set });
  const runtimeSelection = selectColdStartRouteCandidateIdsFromArtifactBundleV1({ artifactBundle: runtimeBundle, row: loadedExport.routeRows[0] });
  const stopPrediction = predictColdStartStopLabelV1({
    model: result.model,
    candidateCount: loadedExport.routeRows[0].candidate_set.length,
    evidenceSpanCount: loadedExport.routeRows[0].evidence_spans.length,
    hardNegativeCount: loadedExport.routeRows[0].hard_negatives.length,
    outcomeGain: loadedExport.routeRows[0].outcome_gain,
  });

  console.log("Cold-start router smoke: ok");
  console.log(`approvedExport: ${approvedExportPath}`);
  console.log(`approvedExportSummary: ${JSON.stringify(loadedExport.summary)}`);
  console.log(`outputDir: ${result.outputDir}`);
  console.log(`manifestChecksum: ${result.manifest.artifact_checksum}`);
  console.log(`topCandidate: ${ranking[0]?.candidate.candidate_id ?? "none"}`);
  console.log(`stopPrediction: ${stopPrediction.label}`);
  console.log(`row0TopScore: ${scoring.rankedCandidates[0]?.score.toFixed(3) ?? "n/a"}`);
  console.log(`runtimeLoad: ${runtimeBundle.manifest.artifact_checksum === result.manifest.artifact_checksum ? "ok" : "mismatch"}`);
  console.log(`runtimeTopCandidate: ${runtimeSelection.selectedCandidateIds[0] ?? "none"}`);
  console.log(`runtimeStopped: ${runtimeSelection.stopped}`);
  console.log(`runtimeTopScore: ${runtimeScoring.rankedCandidates[0]?.score.toFixed(3) ?? "n/a"}`);
  console.log(`manifestSummary: ${JSON.stringify({ ...result.manifest, artifact_checksum: result.manifest.artifact_checksum })}`);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.stack ?? error.message : String(error));
  process.exitCode = 1;
}).finally(() => {
  cleanup();
});
