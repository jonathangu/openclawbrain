import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it } from "vitest";

import {
  buildContinuousLearningOperatorStatus,
  continuousLearningControlPath,
  readContinuousLearningControl,
  writeContinuousLearningControl,
} from "../../src/brain-runtime/continuous-learning-status.js";

const tempDirs: string[] = [];
const approvedTrainV2Dir = fileURLToPath(
  new URL("../../artifacts/cold-start-router-approved-export/real-approved-router-train.hotpotqa-musique.v2", import.meta.url),
);
const approvedTrainV3Dir = fileURLToPath(
  new URL("../../artifacts/cold-start-router-approved-export/real-approved-router-train.hotpotqa-musique.v3", import.meta.url),
);

afterEach(() => {
  while (tempDirs.length > 0) {
    rmSync(tempDirs.pop()!, { recursive: true, force: true });
  }
});

function makeTempDir(prefix: string): string {
  const dir = mkdtempSync(join(os.tmpdir(), prefix));
  tempDirs.push(dir);
  return dir;
}

function writeJson(path: string, value: unknown): void {
  mkdirSync(join(path, ".."), { recursive: true });
  writeFileSync(path, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

describe("continuous learning operator status", () => {
  it("surfaces graphify cadence runs, retrain/promotion truth, queue visibility, and pause controls", () => {
    const workspaceRoot = makeTempDir("openclawbrain-continuous-learning-status-");
    const graphifyRegistryPath = join(workspaceRoot, "artifacts", "graphify-scheduler", "registry.json");
    const graphifyDeltaRunRoot = join(workspaceRoot, "artifacts", "graphify-scheduler", "delta", "delta-20260407-0001");
    const graphifyReorgRunRoot = join(workspaceRoot, "artifacts", "graphify-scheduler", "reorg", "reorg-20260407-0001");
    const retrainReportDir = join(workspaceRoot, "scratch", "cold-start-router-periodic-retrain", "report.v1");

    writeJson(graphifyRegistryPath, {
      contract: "graphify_scheduler_registry.v1",
      schedulerVersion: "graphify-scheduler@1",
      runCount: 2,
      latestByCadence: {
        delta: {
          runId: "delta-20260407-0001",
          runRoot: join("artifacts", "graphify-scheduler", "delta", "delta-20260407-0001"),
          generatedAt: "2026-04-07T12:00:00.000Z",
          status: "completed",
        },
        reorg: {
          runId: "reorg-20260407-0001",
          runRoot: join("artifacts", "graphify-scheduler", "reorg", "reorg-20260407-0001"),
          generatedAt: "2026-04-07T12:15:00.000Z",
          status: "completed",
        },
      },
      runs: [],
    });

    writeJson(join(graphifyDeltaRunRoot, "status.json"), {
      contract: "graphify_scheduler_run_status.v1",
      schedulerVersion: "graphify-scheduler@1",
      cadence: "delta",
      runId: "delta-20260407-0001",
      generatedAt: "2026-04-07T12:00:00.000Z",
      status: "completed",
      offPath: true,
      inspectable: true,
      replayable: true,
      registryRunCount: 2,
      latestByCadence: {},
      graphifyRun: {
        runId: "delta-run",
        runDir: join("artifacts", "graphify-scheduler", "delta", "delta-20260407-0001", "run"),
        graph: { hash: "sha256:delta-graph" },
        graphifyMode: "delta-cadence",
        graphifyVersion: "graphify-scheduler@1",
        graphifyConfigHash: "sha256:delta-config",
        outputs: {},
      },
      importSlice: {
        outputDir: join("artifacts", "graphify-scheduler", "delta", "delta-20260407-0001", "import"),
        digest: { bundleHash: "sha256:delta-import" },
        counts: {
          hubPriors: 3,
          neighborhoodPriors: 5,
          evidencePointers: 8,
          rationalePointers: 2,
          sourceArtifacts: 11,
        },
      },
      maintenanceDiff: {
        verdict: { verdict: "clean" },
      },
    });

    writeJson(join(graphifyReorgRunRoot, "status.json"), {
      contract: "graphify_scheduler_run_status.v1",
      schedulerVersion: "graphify-scheduler@1",
      cadence: "reorg",
      runId: "reorg-20260407-0001",
      generatedAt: "2026-04-07T12:15:00.000Z",
      status: "completed",
      offPath: true,
      inspectable: true,
      replayable: true,
      registryRunCount: 2,
      latestByCadence: {},
      graphifyRun: {
        runId: "reorg-run",
        runDir: join("artifacts", "graphify-scheduler", "reorg", "reorg-20260407-0001", "run"),
        graph: { hash: "sha256:reorg-graph" },
        graphifyMode: "reorg-cadence",
        graphifyVersion: "graphify-scheduler@1",
        graphifyConfigHash: "sha256:reorg-config",
        outputs: {},
      },
      importSlice: {
        outputDir: join("artifacts", "graphify-scheduler", "reorg", "reorg-20260407-0001", "import"),
        digest: { bundleHash: "sha256:reorg-import" },
        counts: {
          hubPriors: 1,
          neighborhoodPriors: 2,
          evidencePointers: 3,
          rationalePointers: 1,
          sourceArtifacts: 5,
        },
      },
      maintenanceDiff: {
        verdict: { verdict: "review" },
      },
    });

    writeJson(join(retrainReportDir, "route-split-registry.v1.json"), {
      contract: "cold_start_router_route_split_registry.v1",
      registryId: "cold-start-router-periodic-retrain-v1",
      generatedAt: "2026-04-07T12:30:00.000Z",
      trainRows: [{ rowId: "row-1" }, { rowId: "row-2" }],
      evalRows: [{ rowId: "row-3" }],
      quarantinedRows: [],
      overlapRowIds: [],
      summary: "2 train rows and 1 eval-only row are cleanly partitioned",
    });
    writeJson(join(retrainReportDir, "replay-eval-report.v1.json"), {
      contract: "cold_start_router_replay_eval_report.v1",
      generatedAt: "2026-04-07T12:30:00.000Z",
      registryId: "cold-start-router-periodic-retrain-v1",
      gatePassed: true,
      summary: "2 train rows and 1 eval-only row both replay-gated the next same-family base prior.",
      trainReplay: { passed: true, verdict: "pass", summary: "pass" },
      evalReplay: { passed: true, verdict: "pass", summary: "pass" },
    });
    writeJson(join(retrainReportDir, "promotion-package.v1.json"), {
      contract: "cold_start_router_promotion_package.v1",
      packageId: "cold-start-router-periodic-retrain-v1:router-artifact-periodic-retrain-v1",
      generatedAt: "2026-04-07T12:30:00.000Z",
      gatePassed: true,
      registryId: "cold-start-router-periodic-retrain-v1",
      candidateArtifactId: "router-artifact-periodic-retrain-v1",
      candidateArtifactVersion: "v1",
      candidateArtifactDir: join(retrainReportDir, "candidate"),
      candidateArtifactChecksum: "sha256:candidate-router-checksum",
      priorBaseArtifactId: "router-base-prior-v0",
      priorBaseArtifactVersion: "v0",
      priorBaseArtifactDir: join(retrainReportDir, "prior-base"),
      priorBaseArtifactChecksum: "sha256:prior-base-router-checksum",
      rollbackKey: "rollback-key-123",
      decision: "promote",
      summary: "bounded periodic retrain package is promotable: route rows are split into 2 train rows and 1 replay-gated eval row.",
      blockers: [],
      splitRegistryPath: join(retrainReportDir, "route-split-registry.v1.json"),
      replayReportPath: join(retrainReportDir, "replay-eval-report.v1.json"),
      trainingDataRefs: ["training-data-ref"],
      replayGateRefs: ["replay-gate-ref"],
    });

    writeContinuousLearningControl(workspaceRoot, "graphify-import", true, "operator pause", "test");
    writeContinuousLearningControl(workspaceRoot, "retrain", false, null, "test");

    const status = buildContinuousLearningOperatorStatus({
      workspaceRoot,
      controlRoot: join(workspaceRoot, "artifacts", "continuous-learning-controls"),
      now: Date.UTC(2026, 3, 7, 12, 45, 0),
      store: {
        getTrainingState: (key: string) => {
          if (key === "last_promotion_reason") {
            return "promote";
          }
          if (key === "continuous_learning_rows_added_since_last_retrain") {
            return "7";
          }
          return null;
        },
        getTrainingStateJson: <T>(key: string): T | null => {
          if (key === "last_promotion_verdict_json") {
            return {
              summary: "bounded periodic retrain package is promotable",
              decision: "promote",
              gatePassed: true,
            } as T;
          }
          return null;
        },
        countMutationsByStatus: () => ({ pending: 1, promoted: 2 }),
        getTeacherQueueSummary: () => ({ pendingCount: 1, readyCount: 0, delayedCount: 0, budgetDeferredCount: 0, sparseReadyCount: 0 }),
      },
    });

    expect(status.controls.graphifyImportPaused).toBe(true);
    expect(status.controls.retrainPaused).toBe(false);
    expect(status.graphify.delta?.cadence).toBe("delta");
    expect(status.graphify.delta?.status).toBe("completed");
    expect(status.graphify.reorg?.cadence).toBe("reorg");
    expect(status.graphify.reorg?.status).toBe("completed");
    expect(status.retrain.lastRetrain?.gatePassed).toBe(true);
    expect(status.retrain.lastRetrain?.promotionDecision).toBe("promote");
    expect(status.retrain.lastPromotionReason).toBe("promote");
    expect(status.retrain.lastPromotionVerdict).toMatchObject({ decision: "promote", gatePassed: true });
    expect(status.retrain.rowsAddedSinceLastRetrain).toBe(7);
    expect(status.retrain.lineage).toMatchObject({
      priorBaseArtifactId: "router-base-prior-v0",
      priorBaseArtifactChecksum: "sha256:prior-base-router-checksum",
      candidateArtifactId: "router-artifact-periodic-retrain-v1",
      candidateArtifactChecksum: "sha256:candidate-router-checksum",
      priorRooted: true,
      promotionValid: true,
      residualUpdateCount: 7,
    });
    expect(status.retrain.lineage?.summary).toContain("prior-rooted=yes");
    expect(status.retrain.lineage?.summary).toContain("promotion-valid=yes");
    expect(status.retrain.lineage?.summary).toContain("residual updates=7");
    expect(status.queueVisibility.teacherQueue).toMatchObject({ pendingCount: 1 });
    expect(status.queueVisibility.mutationBacklog).toEqual({ pending: 1, promoted: 2 });
    expect(status.operatorSummary.improved).toEqual(expect.arrayContaining([
      "last Graphify delta run is surfaced",
      "last Graphify reorg run is surfaced",
      "last retrain and promotion result are surfaced",
      "cold-start prior lineage and residual update truth are surfaced",
      "approved base prior vs live delta truth are surfaced",
      "graphify-import pause control is surfaced",
      "retraining pause control is surfaced",
      "teacher and mutation queue visibility is surfaced",
    ]));
    expect(status.operatorSummary.diagnosticOnly).toEqual([
      "live-delta weight magnitude remains unavailable until current graph and promoted pack snapshot are both readable",
    ]);

    expect(readContinuousLearningControl(workspaceRoot, "graphify-import")).toMatchObject({
      paused: true,
      reason: "operator pause",
    });
    expect(continuousLearningControlPath(workspaceRoot, "graphify-import")).toContain("graphify-import.json");
  });

  it("surfaces approved base prior versus live delta runtime truth", () => {
    const workspaceRoot = makeTempDir("openclawbrain-continuous-learning-runtime-truth-");
    const retrainReportDir = join(workspaceRoot, "scratch", "cold-start-router-periodic-retrain", "report.v1");
    const currentPackVersion = 10;
    const candidatePackVersion = 12;

    writeJson(join(retrainReportDir, "route-split-registry.v1.json"), {
      contract: "cold_start_router_route_split_registry.v1",
      registryId: "cold-start-router-periodic-retrain-runtime-truth-v1",
      generatedAt: "2026-04-07T13:00:00.000Z",
      trainRows: [{ rowId: "row-1" }],
      evalRows: [{ rowId: "row-2" }],
      quarantinedRows: [],
      overlapRowIds: [],
      summary: "runtime truth retrain split stayed bounded",
    });
    writeJson(join(retrainReportDir, "replay-eval-report.v1.json"), {
      contract: "cold_start_router_replay_eval_report.v1",
      generatedAt: "2026-04-07T13:00:00.000Z",
      registryId: "cold-start-router-periodic-retrain-runtime-truth-v1",
      gatePassed: true,
      summary: "candidate base prior stayed promotable",
      candidateManifestSummary: {
        artifactId: "router-artifact-real-approved-export-hotpotqa-musique-stoplocal-v3",
        artifactVersion: "0.0.3",
        checksum: "sha256:4ef43329a36bea9e9c9d2fe18aecff29037b636dd6912ee3f60b24532ccee834",
        packType: "base",
      },
      priorBaseManifestSummary: {
        artifactId: "router-artifact-real-approved-export-hotpotqa-musique-stoplocal-v2",
        artifactVersion: "0.0.2",
        checksum: "sha256:2cd6babae7a19722006a5d6a1415a9e769af8ba00087e8327bdfb788295f14ed",
        packType: "base",
      },
    });
    writeJson(join(retrainReportDir, "promotion-package.v1.json"), {
      contract: "cold_start_router_promotion_package.v1",
      packageId: "cold-start-router-periodic-retrain-runtime-truth-v1:router-artifact-real-approved-export-hotpotqa-musique-stoplocal-v3",
      generatedAt: "2026-04-07T13:00:00.000Z",
      gatePassed: true,
      registryId: "cold-start-router-periodic-retrain-runtime-truth-v1",
      candidateArtifactId: "router-artifact-real-approved-export-hotpotqa-musique-stoplocal-v3",
      candidateArtifactVersion: "0.0.3",
      candidateArtifactDir: approvedTrainV3Dir,
      candidateArtifactChecksum: "sha256:4ef43329a36bea9e9c9d2fe18aecff29037b636dd6912ee3f60b24532ccee834",
      priorBaseArtifactId: "router-artifact-real-approved-export-hotpotqa-musique-stoplocal-v2",
      priorBaseArtifactVersion: "0.0.2",
      priorBaseArtifactDir: approvedTrainV2Dir,
      priorBaseArtifactChecksum: "sha256:2cd6babae7a19722006a5d6a1415a9e769af8ba00087e8327bdfb788295f14ed",
      rollbackKey: "rollback-key-runtime-truth",
      decision: "promote",
      summary: "runtime truth package promoted the approved base prior",
      blockers: [],
      splitRegistryPath: join(retrainReportDir, "route-split-registry.v1.json"),
      replayReportPath: join(retrainReportDir, "replay-eval-report.v1.json"),
      trainingDataRefs: ["training-data-ref"],
      replayGateRefs: ["replay-gate-ref"],
    });

    const baseSnapshot = {
      nodes: [],
      edges: [
        {
          source: "node-a",
          target: "node-b",
          kind: "learned" as const,
          weight: 0.6,
          prior: 0.5,
          metadata: {},
          decayedAt: 0,
          createdAt: 0,
        },
      ],
      seedWeights: [{ nodeId: "seed-a", weight: 0.2, updatedAt: 0 }],
      stopLocalWeights: [{ sourceNodeId: "node-a", weight: 0.3, updatedAt: 0 }],
      toolActionPriors: [{ sourceNodeId: "node-a", toolNodeId: "tool-x", weight: 0.1, updatedAt: 0 }],
    };

    const status = buildContinuousLearningOperatorStatus({
      workspaceRoot,
      controlRoot: join(workspaceRoot, "artifacts", "continuous-learning-controls"),
      now: Date.UTC(2026, 3, 7, 13, 15, 0),
      store: {
        getTrainingState: (key: string) => {
          if (key === "last_promotion_reason") {
            return "promote";
          }
          if (key === "continuous_learning_rows_added_since_last_retrain") {
            return "5";
          }
          if (key === "last_pg_candidate_pack_version") {
            return String(candidatePackVersion);
          }
          return null;
        },
        getTrainingStateJson: <T>(key: string): T | null => {
          if (key === "last_promotion_verdict_json") {
            return {
              summary: "runtime truth package promoted the approved base prior",
              decision: "promote",
              gatePassed: true,
            } as T;
          }
          if (key === "last_pg_candidate_update_json") {
            return { updateCount: 9 } as T;
          }
          return null;
        },
        getCurrentPackVersion: () => currentPackVersion,
        readPackSnapshot: (version: number) => {
          if (version === currentPackVersion) {
            return {
              ...baseSnapshot,
              metadata: { reason: "promotion_base" },
            };
          }
          if (version === currentPackVersion + 1 || version === candidatePackVersion) {
            return {
              ...baseSnapshot,
              metadata: { reason: "pg_update_candidate" },
            };
          }
          return null;
        },
      },
      graph: {
        getAllSeedWeights: () => [{ nodeId: "seed-a", weight: 0.45, updatedAt: 1 }],
        getAllStopLocalWeights: () => [{ sourceNodeId: "node-a", weight: 0.45, updatedAt: 1 }],
        getAllToolActionPriors: () => [{ sourceNodeId: "node-a", toolNodeId: "tool-x", weight: 0.25, updatedAt: 1 }],
        getAllEdges: () => [{
          source: "node-a",
          target: "node-b",
          kind: "learned" as const,
          weight: 0.85,
          prior: 0.5,
          metadata: {},
          decayedAt: 0,
          createdAt: 0,
        }],
      },
    });

    expect(status.runtimeTruth).toMatchObject({
      baseArtifactId: "router-artifact-real-approved-export-hotpotqa-musique-stoplocal-v3",
      baseArtifactVersion: "0.0.3",
      baseArtifactChecksum: "sha256:5ebedc06407909e4f1801ac819e97f9a8bc2f2d02f84cd3286233b73b0a5887e",
      baseArtifactSource: "candidate",
      basePackType: "base",
      mixedPackFromBaseArtifactId: null,
      liveDeltaUpdateCount: 2,
      liveDeltaWeightCount: 4,
    });
    expect(status.runtimeTruth?.liveDeltaMagnitudeSummary).toMatchObject({
      changedWeightCount: 4,
      changedSeedWeightCount: 1,
      changedStopLocalWeightCount: 1,
      changedToolActionWeightCount: 1,
      changedEdgeWeightCount: 1,
      totalAbsoluteDelta: 0.8,
      maxAbsoluteDelta: 0.25,
      meanAbsoluteDelta: 0.2,
      summary: "4 changed weight(s); total|delta|=0.800; max|delta|=0.250; mean|delta|=0.200; seed=1; stop_local=1; tool_action=1; edge=1",
    });
    expect(status.runtimeTruth?.summary).toContain("approved base prior=router-artifact-real-approved-export-hotpotqa-musique-stoplocal-v3@0.0.3");
    expect(status.runtimeTruth?.summary).toContain("live delta updates=2");
    expect(status.operatorSummary.improved).toContain("approved base prior vs live delta truth are surfaced");
    expect(status.operatorSummary.diagnosticOnly).not.toContain(
      "approved base prior vs live delta truth is unavailable for the current retrain report",
    );
    expect(status.operatorSummary.diagnosticOnly).not.toContain(
      "live-delta weight magnitude remains unavailable until current graph and promoted pack snapshot are both readable",
    );
  });
});
