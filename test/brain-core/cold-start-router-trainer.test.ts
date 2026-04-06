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
  scoreColdStartRouteRowFromArtifactBundleV1,
  selectColdStartRouteCandidateIdsFromArtifactBundleV1,
} from "../../src/brain-core/cold-start-router-runtime.js";
import {
  predictColdStartStopLabelV1,
  rankColdStartRouteCandidatesV1,
  scoreColdStartRouteRowV1,
  trainColdStartRouterArtifactV1,
} from "../../src/brain-core/cold-start-router-trainer.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..", "..");
const approvedExportPath = fileURLToPath(
  new URL("../../artifacts/cold-start-router-approved-export/approved-router-export.fixture.v1.json", import.meta.url),
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
      compatibleRuntimeVersion: "openclawbrain-runtime@0.3.8",
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
      runtimeVersion: "openclawbrain-runtime@0.3.8",
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

    const rowScore = scoreColdStartRouteRowV1({ model: result.model, row: loadedExport.routeRows[0] });
    expect(rowScore.rankedCandidates[0]?.candidate.candidate_id).toBe("mem:shipping_history");
    expect(rowScore.rankedCandidates[0]?.score).toBeGreaterThan(rowScore.rankedCandidates[1]?.score ?? -Infinity);
    expect(rowScore.stopPrediction.contributingBuckets).toHaveLength(4);
    expect(rowScore.stopPrediction.scores).toHaveProperty("CONTINUE");

    const ranking = rankColdStartRouteCandidatesV1({ model: result.model, candidates: loadedExport.routeRows[0].candidate_set });
    expect(ranking[0]?.candidate.candidate_id).toBe("mem:shipping_history");
    expect(rowScore.policyDistribution.actions).toHaveLength(loadedExport.routeRows[0].candidate_set.length + 1);
    expect(rowScore.policyDistribution.stopAction.action.type).toBe("stop_local");
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

    const runtimeSelection = selectColdStartRouteCandidateIdsFromArtifactBundleV1({ artifactBundle: runtimeBundle, row: loadedExport.routeRows[0] });
    expect(runtimeSelection.stopped).toBe(false);
    expect(runtimeSelection.selectedCandidateIds).toEqual(["mem:shipping_history"]);
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
