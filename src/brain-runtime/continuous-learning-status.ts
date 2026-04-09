import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";

type JsonRecord = Record<string, unknown>;

export type ContinuousLearningControlScope = "graphify-import" | "retrain";

export interface ContinuousLearningControlStateV1 {
  contract: "continuous_learning_control.v1";
  scope: ContinuousLearningControlScope;
  paused: boolean;
  reason: string | null;
  updatedAt: string;
  source: string;
}

export interface ContinuousLearningRetrainLineageV1 {
  priorBaseArtifactId: string | null;
  priorBaseArtifactVersion: string | null;
  priorBaseArtifactChecksum: string | null;
  candidateArtifactId: string | null;
  candidateArtifactVersion: string | null;
  candidateArtifactChecksum: string | null;
  priorRooted: boolean;
  promotionValid: boolean;
  residualUpdateCount: number | null;
  summary: string;
}

export interface ContinuousLearningOperatorStatusParams {
  store?: {
    getTrainingState(key: string): string | null;
    getTrainingStateJson<T>(key: string): T | null;
    countMutationsByStatus?(): Record<string, number>;
    getTeacherQueueSummary?(readyBefore: number, limit: number): unknown;
  } | null;
  workspaceRoot?: string | null;
  controlRoot?: string | null;
  brainRoot?: string | null;
  now?: number;
}

export interface ContinuousLearningOperatorStatusV1 {
  contract: "continuous_learning_operator_status.v1";
  observedAt: string;
  workspaceRoot: string | null;
  controlRoot: string | null;
  controls: {
    graphifyImport: ContinuousLearningControlStateV1 | null;
    retrain: ContinuousLearningControlStateV1 | null;
    graphifyImportPaused: boolean | null;
    retrainPaused: boolean | null;
  };
  graphify: {
    registryPath: string | null;
    delta: JsonRecord | null;
    reorg: JsonRecord | null;
    runCount: number | null;
  };
  retrain: {
    reportDir: string | null;
    lastRetrain: JsonRecord | null;
    lastPromotionReason: string | null;
    lastPromotionVerdict: JsonRecord | null;
    rowsAddedSinceLastRetrain: number | null;
    lineage: ContinuousLearningRetrainLineageV1 | null;
  };
  queueVisibility: JsonRecord;
  operatorSummary: {
    improved: string[];
    diagnosticOnly: string[];
    summary: string;
  };
}

function normalizeString(value: unknown): string | null {
  if (typeof value !== "string") {
    return null;
  }
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
}

function normalizeNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string" && value.trim().length > 0) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function isRecord(value: unknown): value is JsonRecord {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function candidateWorkspaceRoots(workspaceRoot: string | null | undefined): string[] {
  const root = normalizeString(workspaceRoot);
  if (!root) {
    return [];
  }
  return [
    root,
    resolve(root, ".."),
    join(root, "openclawbrain"),
  ];
}

function readJsonIfExists<T>(filePath: string): T | null {
  if (!existsSync(filePath)) {
    return null;
  }
  try {
    return JSON.parse(readFileSync(filePath, "utf8")) as T;
  } catch {
    return null;
  }
}

function ensureDir(dirPath: string): void {
  mkdirSync(dirPath, { recursive: true });
}

export function continuousLearningControlDir(workspaceRoot: string): string {
  return join(workspaceRoot, "artifacts", "continuous-learning-controls");
}

export function continuousLearningControlPath(workspaceRoot: string, scope: ContinuousLearningControlScope): string {
  return join(continuousLearningControlDir(workspaceRoot), `${scope}.json`);
}

export function readContinuousLearningControl(
  workspaceRoot: string | null | undefined,
  scope: ContinuousLearningControlScope,
): ContinuousLearningControlStateV1 | null {
  for (const root of candidateWorkspaceRoots(workspaceRoot)) {
    const control = readJsonIfExists<ContinuousLearningControlStateV1>(continuousLearningControlPath(root, scope));
    if (control) {
      return control;
    }
  }
  return null;
}

export function writeContinuousLearningControl(
  workspaceRoot: string,
  scope: ContinuousLearningControlScope,
  paused: boolean,
  reason: string | null | undefined,
  source = "openclawbrain-cli",
): ContinuousLearningControlStateV1 {
  const root = resolve(workspaceRoot);
  const controlRoot = continuousLearningControlDir(root);
  ensureDir(controlRoot);
  const state: ContinuousLearningControlStateV1 = {
    contract: "continuous_learning_control.v1",
    scope,
    paused,
    reason: normalizeString(reason),
    updatedAt: new Date().toISOString(),
    source,
  };
  writeFileSync(continuousLearningControlPath(root, scope), `${JSON.stringify(state, null, 2)}\n`, "utf8");
  return state;
}

export function clearContinuousLearningControl(
  workspaceRoot: string,
  scope: ContinuousLearningControlScope,
): void {
  const controlPath = continuousLearningControlPath(resolve(workspaceRoot), scope);
  if (existsSync(controlPath)) {
    rmSync(controlPath);
  }
}

function readGraphifySchedulerStatus(
  workspaceRoot: string | null,
  cadence: "delta" | "reorg",
): JsonRecord | null {
  for (const root of candidateWorkspaceRoots(workspaceRoot)) {
    const registryPath = join(root, "artifacts", "graphify-scheduler", "registry.json");
    const registry = readJsonIfExists<JsonRecord>(registryPath);
    if (!registry) {
      continue;
    }
    const latestByCadence = isRecord(registry.latestByCadence) ? registry.latestByCadence : null;
    const latest = latestByCadence && isRecord(latestByCadence[cadence]) ? (latestByCadence[cadence] as JsonRecord) : null;
    if (!latest) {
      return {
        contract: registry.contract ?? "graphify_scheduler.v1",
        schedulerVersion: registry.schedulerVersion ?? null,
        cadence,
        status: "missing",
        registryPath,
        registryRunCount: typeof registry.runCount === "number" ? registry.runCount : null,
        latestByCadence: registry.latestByCadence ?? null,
      };
    }

    const runRoot = normalizeString(latest.runRoot) ? resolve(root, String(latest.runRoot)) : null;
    const statusPath = runRoot ? join(runRoot, "status.json") : null;
    const runStatus = statusPath ? readJsonIfExists<JsonRecord>(statusPath) : null;
    if (runStatus) {
      return {
        ...runStatus,
        registryPath,
        registryRunCount: typeof registry.runCount === "number" ? registry.runCount : null,
        latestByCadence: registry.latestByCadence ?? null,
        registryEntry: latest,
        statusPath,
      };
    }

    return {
      contract: registry.contract ?? "graphify_scheduler.v1",
      schedulerVersion: registry.schedulerVersion ?? null,
      cadence,
      runId: latest.runId ?? null,
      generatedAt: latest.generatedAt ?? null,
      status: latest.status ?? "unknown",
      runRoot: latest.runRoot ?? null,
      registryPath,
      registryRunCount: typeof registry.runCount === "number" ? registry.runCount : null,
      latestByCadence: registry.latestByCadence ?? null,
      registryEntry: latest,
      statusPath,
    };
  }
  return null;
}

function readPeriodicRetrainStatus(workspaceRoot: string | null): JsonRecord | null {
  for (const root of candidateWorkspaceRoots(workspaceRoot)) {
    const reportDirCandidates = [
      join(root, "scratch", "cold-start-router-periodic-retrain", "report.v1"),
      join(root, "openclawbrain", "scratch", "cold-start-router-periodic-retrain", "report.v1"),
    ];
    const reportDir = reportDirCandidates.find((candidate) => existsSync(join(candidate, "route-split-registry.v1.json")) || existsSync(join(candidate, "replay-eval-report.v1.json")) || existsSync(join(candidate, "promotion-package.v1.json")) || existsSync(join(candidate, "status.json")))
      ?? reportDirCandidates[0];
    const statusPath = join(reportDir, "status.json");
    const status = readJsonIfExists<JsonRecord>(statusPath);
    if (status) {
      return {
        ...status,
        reportDir,
        statusPath,
      };
    }

    const splitRegistryPath = join(reportDir, "route-split-registry.v1.json");
    const replayReportPath = join(reportDir, "replay-eval-report.v1.json");
    const promotionPackagePath = join(reportDir, "promotion-package.v1.json");
    const splitRegistry = readJsonIfExists<JsonRecord>(splitRegistryPath);
    const replayReport = readJsonIfExists<JsonRecord>(replayReportPath);
    const promotionPackage = readJsonIfExists<JsonRecord>(promotionPackagePath);

    if (!splitRegistry && !replayReport && !promotionPackage) {
      continue;
    }

    return {
      contract: replayReport?.contract ?? promotionPackage?.contract ?? "cold_start_router_periodic_retrain_status.v1",
      reportDir,
      status: replayReport?.gatePassed === true ? "promotable" : replayReport?.gatePassed === false ? "held" : "unknown",
      generatedAt: replayReport?.generatedAt ?? promotionPackage?.generatedAt ?? null,
      gatePassed: typeof replayReport?.gatePassed === "boolean"
        ? replayReport.gatePassed
        : typeof promotionPackage?.gatePassed === "boolean"
          ? promotionPackage.gatePassed
          : null,
      promotionDecision: promotionPackage?.decision ?? null,
      summary: replayReport?.summary ?? promotionPackage?.summary ?? null,
      trainRowCount: Array.isArray(splitRegistry?.trainRows) ? splitRegistry.trainRows.length : null,
      evalRowCount: Array.isArray(splitRegistry?.evalRows) ? splitRegistry.evalRows.length : null,
      quarantinedRowCount: Array.isArray(splitRegistry?.quarantinedRows) ? splitRegistry.quarantinedRows.length : null,
      overlapRowCount: Array.isArray(splitRegistry?.overlapRowIds) ? splitRegistry.overlapRowIds.length : null,
      splitRegistryPath,
      replayReportPath,
      promotionPackagePath,
      splitRegistry,
      replayReport,
      promotionPackage,
      rowsAddedSinceLastRetrain: null,
      statusPath,
    };
  }
  return null;
}

function readContinuousLearningRowsAddedSinceLastRetrain(
  store: ContinuousLearningOperatorStatusParams["store"],
  retrainStatus: JsonRecord | null,
): number | null {
  const stateValue = store?.getTrainingState("continuous_learning_rows_added_since_last_retrain");
  const parsedStateValue = normalizeNumber(stateValue);
  if (parsedStateValue !== null) {
    return parsedStateValue;
  }
  const retrainRows = normalizeNumber(retrainStatus?.rowsAddedSinceLastRetrain);
  return retrainRows;
}

function summarizeRetrainLineage(params: {
  retrainStatus: JsonRecord | null;
  rowsAddedSinceLastRetrain: number | null;
}): ContinuousLearningRetrainLineageV1 | null {
  const promotionPackage = isRecord(params.retrainStatus?.promotionPackage)
    ? params.retrainStatus.promotionPackage
    : null;
  if (!promotionPackage) {
    return null;
  }

  const priorBaseArtifactId = normalizeString(promotionPackage.priorBaseArtifactId);
  const priorBaseArtifactVersion = normalizeString(promotionPackage.priorBaseArtifactVersion);
  const priorBaseArtifactChecksum = normalizeString(promotionPackage.priorBaseArtifactChecksum);
  const candidateArtifactId = normalizeString(promotionPackage.candidateArtifactId);
  const candidateArtifactVersion = normalizeString(promotionPackage.candidateArtifactVersion);
  const candidateArtifactChecksum = normalizeString(promotionPackage.candidateArtifactChecksum);
  const priorRooted = priorBaseArtifactId !== null && priorBaseArtifactChecksum !== null;
  const promotionValid = params.retrainStatus?.gatePassed === true && normalizeString(promotionPackage.decision) === "promote";
  const residualUpdateCount = params.rowsAddedSinceLastRetrain;

  return {
    priorBaseArtifactId,
    priorBaseArtifactVersion,
    priorBaseArtifactChecksum,
    candidateArtifactId,
    candidateArtifactVersion,
    candidateArtifactChecksum,
    priorRooted,
    promotionValid,
    residualUpdateCount,
    summary: [
      `seeded by ${priorBaseArtifactId ?? "unknown prior"}@${priorBaseArtifactVersion ?? "unknown version"}`,
      `seed checksum=${priorBaseArtifactChecksum ?? "unknown checksum"}`,
      `current router=${candidateArtifactId ?? "unknown candidate"}@${candidateArtifactVersion ?? "unknown version"}`,
      `router checksum=${candidateArtifactChecksum ?? "unknown checksum"}`,
      `prior-rooted=${priorRooted ? "yes" : "no"}`,
      `promotion-valid=${promotionValid ? "yes" : "no"}`,
      `residual updates=${residualUpdateCount ?? "unknown"}`,
    ].join("; "),
  };
}

function summarizeQueueVisibility(params: {
  store?: ContinuousLearningOperatorStatusParams["store"] | null;
  now: number;
  graphifyDelta: JsonRecord | null;
  graphifyReorg: JsonRecord | null;
  retrain: JsonRecord | null;
}): JsonRecord {
  const teacherReadyBefore = params.now - 30_000;
  const teacherQueue = params.store?.getTeacherQueueSummary?.(teacherReadyBefore, 20) ?? null;
  const mutationBacklog = params.store?.countMutationsByStatus?.() ?? null;
  const graphifyRegistryRunCount = normalizeNumber(params.graphifyDelta?.registryRunCount) ?? normalizeNumber(params.graphifyReorg?.registryRunCount) ?? null;
  const retrainPromotionPackage = isRecord(params.retrain?.promotionPackage) ? params.retrain.promotionPackage : null;
  return {
    teacherQueue,
    mutationBacklog,
    graphifyRegistryRunCount,
    graphifyDeltaRunId: normalizeString(params.graphifyDelta?.runId),
    graphifyReorgRunId: normalizeString(params.graphifyReorg?.runId),
    retrainStatus: params.retrain?.status ?? null,
    retrainGatePassed: typeof params.retrain?.gatePassed === "boolean" ? params.retrain.gatePassed : null,
    retrainDecision: normalizeString(params.retrain?.promotionDecision ?? retrainPromotionPackage?.decision),
  };
}

export function buildContinuousLearningOperatorStatus(
  params: ContinuousLearningOperatorStatusParams,
): ContinuousLearningOperatorStatusV1 {
  const workspaceRoot = normalizeString(params.workspaceRoot ?? process.env.OPENCLAWBRAIN_WORKSPACE_ROOT ?? null);
  const controlRoot = normalizeString(params.controlRoot ?? (workspaceRoot ? continuousLearningControlDir(workspaceRoot) : null));
  const now = params.now ?? Date.now();
  const graphifyDelta = readGraphifySchedulerStatus(workspaceRoot, "delta");
  const graphifyReorg = readGraphifySchedulerStatus(workspaceRoot, "reorg");
  const retrainStatus = readPeriodicRetrainStatus(workspaceRoot);
  const graphifyImportControl = readContinuousLearningControl(workspaceRoot, "graphify-import");
  const retrainControl = readContinuousLearningControl(workspaceRoot, "retrain");
  const rowsAddedSinceLastRetrain = readContinuousLearningRowsAddedSinceLastRetrain(params.store, retrainStatus);
  const retrainLineage = summarizeRetrainLineage({ retrainStatus, rowsAddedSinceLastRetrain });
  const retrainPromotionPackage = isRecord(retrainStatus?.promotionPackage) ? retrainStatus.promotionPackage : null;
  const storedPromotionVerdict = params.store?.getTrainingStateJson<JsonRecord>("last_promotion_verdict_json") ?? null;
  const lastPromotionReason = normalizeString(params.store?.getTrainingState("last_promotion_reason") ?? null)
    ?? normalizeString(retrainPromotionPackage?.summary)
    ?? null;
  const lastPromotionVerdict = storedPromotionVerdict
    ?? retrainPromotionPackage
    ?? null;

  const improved = [
    graphifyDelta ? "last Graphify delta run is surfaced" : null,
    graphifyReorg ? "last Graphify reorg run is surfaced" : null,
    retrainStatus ? "last retrain and promotion result are surfaced" : null,
    retrainLineage ? "cold-start prior lineage and residual update truth are surfaced" : null,
    graphifyImportControl ? "graphify-import pause control is surfaced" : null,
    retrainControl ? "retraining pause control is surfaced" : null,
    params.store ? "teacher and mutation queue visibility is surfaced" : null,
  ].filter((entry): entry is string => entry !== null);

  const diagnosticOnly = [
    workspaceRoot ? null : "workspace-root-backed learning status is unavailable in this process",
    rowsAddedSinceLastRetrain === null ? "rows-added-since-last-retrain remains diagnostic-only until a durable counter is recorded" : null,
    graphifyDelta === null ? "Graphify delta registry is missing in the current workspace" : null,
    graphifyReorg === null ? "Graphify reorg registry is missing in the current workspace" : null,
    retrainStatus === null ? "last retrain report is missing in the current workspace" : null,
  ].filter((entry): entry is string => entry !== null);

  return {
    contract: "continuous_learning_operator_status.v1",
    observedAt: new Date(now).toISOString(),
    workspaceRoot,
    controlRoot,
    controls: {
      graphifyImport: graphifyImportControl,
      retrain: retrainControl,
      graphifyImportPaused: graphifyImportControl?.paused ?? null,
      retrainPaused: retrainControl?.paused ?? null,
    },
    graphify: {
      registryPath: workspaceRoot ? join(workspaceRoot, "artifacts", "graphify-scheduler", "registry.json") : null,
      delta: graphifyDelta,
      reorg: graphifyReorg,
      runCount: normalizeNumber(graphifyDelta?.registryRunCount) ?? normalizeNumber(graphifyReorg?.registryRunCount) ?? null,
    },
    retrain: {
      reportDir: workspaceRoot ? join(workspaceRoot, "scratch", "cold-start-router-periodic-retrain", "report.v1") : null,
      lastRetrain: retrainStatus,
      lastPromotionReason,
      lastPromotionVerdict,
      rowsAddedSinceLastRetrain,
      lineage: retrainLineage,
    },
    queueVisibility: summarizeQueueVisibility({
      store: params.store,
      now,
      graphifyDelta,
      graphifyReorg,
      retrain: retrainStatus,
    }),
    operatorSummary: {
      improved,
      diagnosticOnly,
      summary: improved.length > 0
        ? `surfaced ${improved.length} operator control/status improvement${improved.length === 1 ? "" : "s"}`
        : "continuous-learning operator surface is still diagnostic-only",
    },
  };
}
