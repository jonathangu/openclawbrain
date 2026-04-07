import { execFileSync } from "node:child_process";
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it } from "vitest";

import { runColdStartRouterPeriodicRetrainV1 } from "../../src/brain-core/cold-start-router-periodic-retrain.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "../..");

const trainExportPath = path.join(
  repoRoot,
  "artifacts",
  "cold-start-router-approved-export",
  "real-approved-router-export.hotpotqa-musique.v3.json",
);
const evalExportPath = path.join(
  repoRoot,
  "artifacts",
  "cold-start-router-approved-export",
  "real-disjoint-eval-only-router-export.hotpotqa-musique.v1.json",
);
const priorBaseArtifactDir = path.join(
  repoRoot,
  "artifacts",
  "cold-start-router-approved-export",
  "real-approved-router-train.hotpotqa-musique.v3",
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

describe("cold-start router periodic retrain", () => {
  it("builds a bounded split registry and replay-gated promotion package from accumulated route rows", () => {
    const candidateArtifactDir = createTempRoot("cold-start-router-periodic-retrain-candidate");
    const reportDir = createTempRoot("cold-start-router-periodic-retrain-report");

    const result = runColdStartRouterPeriodicRetrainV1({
      trainExportPath,
      evalExportPath,
      candidateArtifactDir,
      reportDir,
      candidateArtifactId: "router-artifact-periodic-retrain-smoke",
      candidateArtifactVersion: "0.0.1",
      candidateRouterIdentity: "router:periodic-retrain:smoke",
      compatibleRuntimeVersion: "openclawbrain-runtime@0.3.8",
      packType: "base",
      registryId: "cold-start-router-periodic-retrain-smoke",
      previousBaseArtifactDir: priorBaseArtifactDir,
      previousBaseArtifactId: "router-artifact-real-approved-export-hotpotqa-musique-stoplocal-v3",
      trainingDataRefs: ["train:hotpotqa-musique:v3"],
      replayGateRefs: ["replay:hotpotqa-musique:eval-only-v1"],
      createdAt: "2026-04-07T12:00:00Z",
    });

    expect(result.splitRegistry).toMatchObject({
      contract: "cold_start_router_route_split_registry.v1",
      registryId: "cold-start-router-periodic-retrain-smoke",
      trainRows: expect.any(Array),
      evalRows: expect.any(Array),
      quarantinedRows: expect.any(Array),
      overlapRowIds: [],
    });
    expect(result.splitRegistry.trainRows).toHaveLength(44);
    expect(result.splitRegistry.evalRows).toHaveLength(2);
    expect(result.splitRegistry.quarantinedRows).toHaveLength(0);
    expect(result.splitRegistry.trainSource.datasetIds).toEqual(["hotpotqa_v1", "musique_v1"]);
    expect(result.splitRegistry.evalSource.datasetIds).toEqual(["hotpotqa_v1", "musique_v1"]);
    expect(result.splitRegistry.summary).toContain("44 train rows and 2 eval-only rows");

    expect(result.candidate.model.training.usedRows).toBe(44);
    expect(result.candidate.manifest.training_data_refs).toContain("train:hotpotqa-musique:v3");
    expect(result.candidate.manifest.replay_gate_refs).toContain("replay:hotpotqa-musique:eval-only-v1");
    expect(result.trainReplay).toMatchObject({ passed: true, verdict: "pass" });
    expect(result.evalReplay).toMatchObject({ passed: true, verdict: "pass" });
    expect(result.report.gatePassed).toBe(true);
    expect(result.report.summary).toContain("both replay-gated the next same-family base prior");
    expect(result.promotionPackage.decision).toBe("promote");
    expect(result.promotionPackage.rollbackKey).toBe("rollback:router-artifact-real-approved-export-hotpotqa-musique-stoplocal-v3:0.0.3");
    expect(result.promotionPackage.gatePassed).toBe(true);
    expect(result.promotionPackage.blockers).toEqual([]);

    expect(existsSync(result.paths.splitRegistryPath)).toBe(true);
    expect(existsSync(result.paths.replayReportPath)).toBe(true);
    expect(existsSync(result.paths.promotionPackagePath)).toBe(true);
  });

  it("runs the periodic retrain entrypoint end to end", () => {
    const candidateArtifactDir = createTempRoot("cold-start-router-periodic-retrain-script-candidate");
    const reportDir = createTempRoot("cold-start-router-periodic-retrain-script-report");

    const stdout = execFileSync(
      "node",
      ["--experimental-transform-types", "scripts/periodic-cold-start-router-retrain.ts"],
      {
        cwd: repoRoot,
        env: {
          ...process.env,
          COLD_START_CANDIDATE_ARTIFACT_DIR: candidateArtifactDir,
          COLD_START_REPORT_DIR: reportDir,
          COLD_START_CANDIDATE_ARTIFACT_ID: "router-artifact-periodic-retrain-script-smoke",
          COLD_START_CANDIDATE_ARTIFACT_VERSION: "0.0.1",
          COLD_START_CANDIDATE_ROUTER_IDENTITY: "router:periodic-retrain:script",
          COLD_START_REGISTRY_ID: "cold-start-router-periodic-retrain-script-smoke",
          COLD_START_PREVIOUS_BASE_ARTIFACT_ID: "router-artifact-real-approved-export-hotpotqa-musique-stoplocal-v3",
        },
        encoding: "utf8",
      },
    );

    expect(stdout).toContain("\"gatePassed\": true");
    expect(stdout).toContain("\"promotionDecision\": \"promote\"");
    expect(stdout).toContain("\"candidateManifestChecksum\":");
  });
});
