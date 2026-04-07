import { createHash } from "node:crypto";
import { mkdirSync, rmSync, writeFileSync } from "node:fs";
import path from "node:path";

import type {
  ColdStartApprovalStatusV1,
  ColdStartStopLabelV1,
  ColdStartPackTypeV1,
  DataRegistryEntryV1,
  RouteDecisionRowV1,
} from "./cold-start-router-contracts.ts";
import type { ColdStartRouterApprovedExportV1 } from "./cold-start-router-approved-export-loader.ts";
import {
  loadColdStartRouterApprovedExportV1,
} from "./cold-start-router-approved-export-loader.ts";
import {
  summarizeRouterArtifactManifestV1,
} from "./cold-start-router-contracts.ts";
import {
  replayColdStartRouterArtifactV1,
} from "./cold-start-router-replay-gate.ts";
import {
  loadColdStartRouterArtifactBundleV1 as loadRuntimeArtifactBundleV1,
} from "./cold-start-router-runtime.ts";
import {
  trainColdStartRouterArtifactV1,
  type ColdStartRouterTrainingResultV1,
} from "./cold-start-router-trainer.ts";

export const COLD_START_ROUTER_PERIODIC_RETRAIN_CONTRACT_V1 = "cold_start_router_periodic_retrain.v1" as const;
export const COLD_START_ROUTER_ROUTE_SPLIT_REGISTRY_CONTRACT_V1 = "cold_start_router_route_split_registry.v1" as const;
export const COLD_START_ROUTER_REPLAY_EVAL_REPORT_CONTRACT_V1 = "cold_start_router_replay_eval_report.v1" as const;
export const COLD_START_ROUTER_PROMOTION_PACKAGE_CONTRACT_V1 = "cold_start_router_promotion_package.v1" as const;

export type ColdStartRouterRouteSliceKindV1 = "train" | "eval_only" | "quarantine";

export interface ColdStartRouterPeriodicRetrainSliceRowV1 {
  rowId: string;
  datasetId: string;
  splitTag: string;
  reviewStatus: ColdStartApprovalStatusV1;
  slice: ColdStartRouterRouteSliceKindV1;
  teacherActionKind: RouteDecisionRowV1["teacher_action"]["kind"];
  stopLabel: ColdStartStopLabelV1;
  candidateCount: number;
  evidenceSpanCount: number;
  reason: string;
}

export interface ColdStartRouterPeriodicRetrainSplitSourceV1 {
  exportId: string;
  generatedAt: string;
  datasetIds: string[];
  registryEntryCount: number;
  rowCount: number;
  approvalStatuses: ColdStartApprovalStatusV1[];
}

export interface ColdStartRouterPeriodicRetrainSplitRegistryV1 {
  contract: typeof COLD_START_ROUTER_ROUTE_SPLIT_REGISTRY_CONTRACT_V1;
  registryId: string;
  generatedAt: string;
  trainSource: ColdStartRouterPeriodicRetrainSplitSourceV1;
  evalSource: ColdStartRouterPeriodicRetrainSplitSourceV1;
  trainRows: ColdStartRouterPeriodicRetrainSliceRowV1[];
  evalRows: ColdStartRouterPeriodicRetrainSliceRowV1[];
  quarantinedRows: ColdStartRouterPeriodicRetrainSliceRowV1[];
  overlapRowIds: string[];
  summary: string;
  notes: string[];
}

export interface ColdStartRouterPeriodicRetrainReplaySummaryV1 {
  verdict: "pass" | "warn" | "fail";
  passed: boolean;
  summary: string;
  evaluatedRowCount: number;
  passedRowCount: number;
  failedRowCount: number;
  skippedRowCount: number;
  rowResults: ReturnType<typeof replayColdStartRouterArtifactV1>["rowResults"];
}

export interface ColdStartRouterPeriodicRetrainEvalReportV1 {
  contract: typeof COLD_START_ROUTER_REPLAY_EVAL_REPORT_CONTRACT_V1;
  generatedAt: string;
  registryId: string;
  candidateArtifactDir: string;
  priorBaseArtifactDir: string;
  candidateManifestSummary: ReturnType<typeof summarizeRouterArtifactManifestV1>;
  priorBaseManifestSummary: ReturnType<typeof summarizeRouterArtifactManifestV1>;
  trainReplay: ColdStartRouterPeriodicRetrainReplaySummaryV1;
  evalReplay: ColdStartRouterPeriodicRetrainReplaySummaryV1;
  gatePassed: boolean;
  summary: string;
}

export interface ColdStartRouterPeriodicRetrainPromotionPackageV1 {
  contract: typeof COLD_START_ROUTER_PROMOTION_PACKAGE_CONTRACT_V1;
  packageId: string;
  generatedAt: string;
  registryId: string;
  candidateArtifactId: string;
  candidateArtifactVersion: string;
  candidateArtifactDir: string;
  candidateArtifactChecksum: string;
  priorBaseArtifactId: string;
  priorBaseArtifactVersion: string;
  priorBaseArtifactDir: string;
  priorBaseArtifactChecksum: string;
  rollbackKey: string;
  decision: "promote" | "hold" | "rollback";
  gatePassed: boolean;
  blockers: string[];
  splitRegistryPath: string;
  replayReportPath: string;
  trainingDataRefs: string[];
  replayGateRefs: string[];
  summary: string;
}

export interface ColdStartRouterPeriodicRetrainRunInputV1 {
  trainExportPath: string;
  evalExportPath: string;
  candidateArtifactDir: string;
  reportDir: string;
  candidateArtifactId: string;
  candidateArtifactVersion: string;
  candidateRouterIdentity: string;
  compatibleRuntimeVersion: string;
  packType: ColdStartPackTypeV1;
  registryId: string;
  previousBaseArtifactDir: string;
  previousBaseArtifactId: string;
  trainingDataRefs?: string[];
  replayGateRefs?: string[];
  createdAt?: string;
}

export interface ColdStartRouterPeriodicRetrainRunPathsV1 {
  splitRegistryPath: string;
  replayReportPath: string;
  promotionPackagePath: string;
}

export interface ColdStartRouterPeriodicRetrainRunResultV1 {
  generatedAt: string;
  splitRegistry: ColdStartRouterPeriodicRetrainSplitRegistryV1;
  candidate: ColdStartRouterTrainingResultV1;
  trainExport: ColdStartRouterApprovedExportV1;
  evalExport: ColdStartRouterApprovedExportV1;
  priorBaseArtifactDir: string;
  priorBaseManifestSummary: ReturnType<typeof summarizeRouterArtifactManifestV1>;
  trainReplay: ColdStartRouterPeriodicRetrainReplaySummaryV1;
  evalReplay: ColdStartRouterPeriodicRetrainReplaySummaryV1;
  report: ColdStartRouterPeriodicRetrainEvalReportV1;
  promotionPackage: ColdStartRouterPeriodicRetrainPromotionPackageV1;
  paths: ColdStartRouterPeriodicRetrainRunPathsV1;
}

function sha256Text(value: string): string {
  return `sha256:${createHash("sha256").update(value, "utf8").digest("hex")}`;
}

function writeJsonArtifact(filePath: string, value: unknown): { path: string; digest: string } {
  const text = `${JSON.stringify(value, null, 2)}\n`;
  mkdirSync(path.dirname(filePath), { recursive: true });
  writeFileSync(filePath, text, "utf8");
  return { path: filePath, digest: sha256Text(text) };
}

function uniqueStrings(values: readonly string[]): string[] {
  return [...new Set(values.map((value) => value.trim()).filter((value) => value.length > 0))].sort();
}

function normalizeText(value: unknown): string {
  return typeof value === "string" ? value.trim() : "";
}

function classifyRows(params: {
  exportBundle: ColdStartRouterApprovedExportV1;
  expectedReviewStatus: ColdStartApprovalStatusV1;
  slice: ColdStartRouterRouteSliceKindV1;
}): {
  acceptedRows: ColdStartRouterPeriodicRetrainSliceRowV1[];
  quarantinedRows: ColdStartRouterPeriodicRetrainSliceRowV1[];
  source: ColdStartRouterPeriodicRetrainSplitSourceV1;
} {
  const registryDatasetIds = new Set(params.exportBundle.registry_entries.map((entry) => entry.dataset_id));
  const datasetIds = uniqueStrings(params.exportBundle.registry_entries.map((entry) => entry.dataset_id));
  const approvalStatuses = uniqueStrings(params.exportBundle.registry_entries.map((entry) => entry.approval_status)) as ColdStartApprovalStatusV1[];

  const acceptedRows: ColdStartRouterPeriodicRetrainSliceRowV1[] = [];
  const quarantinedRows: ColdStartRouterPeriodicRetrainSliceRowV1[] = [];

  for (const row of params.exportBundle.route_rows) {
    const reviewStatus = row.provenance.review_status;
    const provenanceDataset = normalizeText(row.provenance.dataset);
    const datasetKnown = registryDatasetIds.has(row.dataset_id);
    const reviewMatches = reviewStatus === params.expectedReviewStatus;
    const provenanceMatches = provenanceDataset.length > 0 && provenanceDataset === row.dataset_id;
    const teacherActionKind = row.teacher_action.kind;
    const rowRecord: ColdStartRouterPeriodicRetrainSliceRowV1 = {
      rowId: row.row_id,
      datasetId: row.dataset_id,
      splitTag: row.split_tag,
      reviewStatus,
      slice: datasetKnown && reviewMatches && provenanceMatches ? params.slice : "quarantine",
      teacherActionKind,
      stopLabel: row.stop_label,
      candidateCount: row.candidate_set.length,
      evidenceSpanCount: row.evidence_spans.length,
      reason: datasetKnown && reviewMatches && provenanceMatches
        ? `${params.slice} slice accepted from ${params.expectedReviewStatus} export`
        : [
          datasetKnown ? null : `dataset ${row.dataset_id} is not present in the registry`,
          reviewMatches ? null : `review_status ${reviewStatus} does not match ${params.expectedReviewStatus}`,
          provenanceMatches ? null : `provenance dataset ${row.provenance.dataset} does not match row dataset ${row.dataset_id}`,
        ].filter(Boolean).join("; "),
    };

    if (rowRecord.slice === params.slice) {
      acceptedRows.push(rowRecord);
    } else {
      quarantinedRows.push(rowRecord);
    }
  }

  return {
    acceptedRows,
    quarantinedRows,
    source: {
      exportId: params.exportBundle.export_id,
      generatedAt: params.exportBundle.generated_at,
      datasetIds,
      registryEntryCount: params.exportBundle.registry_entries.length,
      rowCount: params.exportBundle.route_rows.length,
      approvalStatuses,
    },
  };
}

function summarizeReplayVerdict(verdict: ReturnType<typeof replayColdStartRouterArtifactV1>): ColdStartRouterPeriodicRetrainReplaySummaryV1 {
  return {
    verdict: verdict.verdict,
    passed: verdict.passed,
    summary: verdict.summary,
    evaluatedRowCount: verdict.evaluatedRowCount,
    passedRowCount: verdict.passedRowCount,
    failedRowCount: verdict.failedRowCount,
    skippedRowCount: verdict.skippedRowCount,
    rowResults: verdict.rowResults,
  };
}

function buildPromotionDecision(params: {
  gatePassed: boolean;
  blockers: string[];
}): "promote" | "hold" | "rollback" {
  if (params.blockers.length > 0) {
    return "hold";
  }
  return params.gatePassed ? "promote" : "hold";
}

function loadPriorBaseArtifactSummary(priorBaseArtifactDir: string) {
  const priorBaseArtifactBundle = loadRuntimeArtifactBundleV1(priorBaseArtifactDir);
  return {
    bundle: priorBaseArtifactBundle,
    summary: summarizeRouterArtifactManifestV1(priorBaseArtifactBundle.manifest),
  };
}

export function buildColdStartRouterPeriodicRetrainSplitRegistryV1(params: {
  registryId: string;
  generatedAt?: string;
  trainExport: ColdStartRouterApprovedExportV1;
  evalExport: ColdStartRouterApprovedExportV1;
}): ColdStartRouterPeriodicRetrainSplitRegistryV1 {
  const generatedAt = params.generatedAt ?? new Date().toISOString();
  const trainClassified = classifyRows({
    exportBundle: params.trainExport,
    expectedReviewStatus: "approved_train",
    slice: "train",
  });
  const evalClassified = classifyRows({
    exportBundle: params.evalExport,
    expectedReviewStatus: "approved_eval_only",
    slice: "eval_only",
  });

  const trainRowIds = new Set(trainClassified.acceptedRows.map((row) => row.rowId));
  const evalRowIds = new Set(evalClassified.acceptedRows.map((row) => row.rowId));
  const overlapRowIds = uniqueStrings([...trainRowIds].filter((rowId) => evalRowIds.has(rowId)));

  return {
    contract: COLD_START_ROUTER_ROUTE_SPLIT_REGISTRY_CONTRACT_V1,
    registryId: params.registryId,
    generatedAt,
    trainSource: trainClassified.source,
    evalSource: evalClassified.source,
    trainRows: trainClassified.acceptedRows,
    evalRows: evalClassified.acceptedRows,
    quarantinedRows: [...trainClassified.quarantinedRows, ...evalClassified.quarantinedRows],
    overlapRowIds,
    summary: overlapRowIds.length === 0
      ? `${trainClassified.acceptedRows.length} train rows and ${evalClassified.acceptedRows.length} eval-only rows are cleanly partitioned`
      : `split registry found ${overlapRowIds.length} overlapping row id${overlapRowIds.length === 1 ? "" : "s"}; quarantine before promotion`,
    notes: [
      `train export ${trainClassified.source.exportId} supplies approved_train route rows for the next same-family base prior.`,
      `eval export ${evalClassified.source.exportId} supplies approved_eval_only route rows for the replay gate.`,
      "The registry stays bounded: rows are either train, eval_only, or quarantined. No broad speculative slice taxonomy is introduced.",
    ],
  };
}

export function runColdStartRouterPeriodicRetrainV1(params: ColdStartRouterPeriodicRetrainRunInputV1): ColdStartRouterPeriodicRetrainRunResultV1 {
  const generatedAt = params.createdAt ?? new Date().toISOString();
  const trainExport = loadColdStartRouterApprovedExportV1(params.trainExportPath);
  const evalExport = loadColdStartRouterApprovedExportV1(params.evalExportPath);
  const splitRegistry = buildColdStartRouterPeriodicRetrainSplitRegistryV1({
    registryId: params.registryId,
    generatedAt,
    trainExport,
    evalExport,
  });

  rmSync(params.candidateArtifactDir, { recursive: true, force: true });
  mkdirSync(params.candidateArtifactDir, { recursive: true });
  rmSync(params.reportDir, { recursive: true, force: true });
  mkdirSync(params.reportDir, { recursive: true });

  const priorBase = loadPriorBaseArtifactSummary(params.previousBaseArtifactDir);
  const trainRows = splitRegistry.trainRows.map((row) => trainExport.route_rows.find((candidate) => candidate.row_id === row.rowId)).filter((row): row is RouteDecisionRowV1 => row !== undefined);
  const evalRows = splitRegistry.evalRows.map((row) => evalExport.route_rows.find((candidate) => candidate.row_id === row.rowId)).filter((row): row is RouteDecisionRowV1 => row !== undefined);

  const trainingDataRefs = uniqueStrings([
    params.registryId,
    trainExport.export_id,
    ...trainExport.registry_entries.map((entry) => entry.dataset_id),
    ...(params.trainingDataRefs ?? []),
  ]);
  const replayGateRefs = uniqueStrings([
    params.registryId,
    evalExport.export_id,
    ...(params.replayGateRefs ?? []),
  ]);

  const candidate = trainColdStartRouterArtifactV1({
    artifactId: params.candidateArtifactId,
    artifactVersion: params.candidateArtifactVersion,
    packType: params.packType,
    compatibleRuntimeVersion: params.compatibleRuntimeVersion,
    registryEntries: trainExport.registry_entries,
    routeRows: trainRows,
    outputDir: params.candidateArtifactDir,
    routerIdentity: params.candidateRouterIdentity,
    createdAt: generatedAt,
    trainingDataRefs,
    replayGateRefs,
  });

  const candidateBundle = loadRuntimeArtifactBundleV1(params.candidateArtifactDir);
  if (candidateBundle.manifest.artifact_checksum !== candidate.manifest.artifact_checksum) {
    throw new Error(
      `candidate bundle checksum mismatch: manifest=${candidate.manifest.artifact_checksum} runtime=${candidateBundle.manifest.artifact_checksum}`,
    );
  }
  const trainReplay = summarizeReplayVerdict(replayColdStartRouterArtifactV1({ artifactDir: params.candidateArtifactDir, routeRows: trainRows }));
  const evalReplay = summarizeReplayVerdict(replayColdStartRouterArtifactV1({ artifactDir: params.candidateArtifactDir, routeRows: evalRows }));

  const blockers = [
    ...(splitRegistry.overlapRowIds.length > 0 ? [`split registry overlaps on ${splitRegistry.overlapRowIds.join(", ")}`] : []),
    ...(trainReplay.passed ? [] : [`train replay gate failed: ${trainReplay.summary}`]),
    ...(evalReplay.passed ? [] : [`eval replay gate failed: ${evalReplay.summary}`]),
  ];

  const gatePassed = blockers.length === 0;
  const decision = buildPromotionDecision({ gatePassed, blockers });
  const splitRegistryPath = path.join(params.reportDir, "route-split-registry.v1.json");
  const replayReportPath = path.join(params.reportDir, "replay-eval-report.v1.json");
  const promotionPackagePath = path.join(params.reportDir, "promotion-package.v1.json");

  const report: ColdStartRouterPeriodicRetrainEvalReportV1 = {
    contract: COLD_START_ROUTER_REPLAY_EVAL_REPORT_CONTRACT_V1,
    generatedAt,
    registryId: params.registryId,
    candidateArtifactDir: params.candidateArtifactDir,
    priorBaseArtifactDir: params.previousBaseArtifactDir,
    candidateManifestSummary: summarizeRouterArtifactManifestV1(candidate.manifest),
    priorBaseManifestSummary: priorBase.summary,
    trainReplay,
    evalReplay,
    gatePassed,
    summary: gatePassed
      ? `${splitRegistry.trainRows.length} train rows and ${splitRegistry.evalRows.length} eval-only rows both replay-gated the next same-family base prior.`
      : `periodic retrain remains held: ${blockers.join("; ")}`,
  };

  const promotionPackage: ColdStartRouterPeriodicRetrainPromotionPackageV1 = {
    contract: COLD_START_ROUTER_PROMOTION_PACKAGE_CONTRACT_V1,
    packageId: `${params.registryId}:${params.candidateArtifactId}`,
    generatedAt,
    registryId: params.registryId,
    candidateArtifactId: params.candidateArtifactId,
    candidateArtifactVersion: params.candidateArtifactVersion,
    candidateArtifactDir: params.candidateArtifactDir,
    candidateArtifactChecksum: candidate.manifest.artifact_checksum,
    priorBaseArtifactId: params.previousBaseArtifactId,
    priorBaseArtifactVersion: priorBase.bundle.manifest.artifact_version,
    priorBaseArtifactDir: params.previousBaseArtifactDir,
    priorBaseArtifactChecksum: priorBase.bundle.manifest.artifact_checksum,
    rollbackKey: `rollback:${params.previousBaseArtifactId}:${priorBase.bundle.manifest.artifact_version}`,
    decision,
    gatePassed,
    blockers,
    splitRegistryPath,
    replayReportPath,
    trainingDataRefs,
    replayGateRefs,
    summary: gatePassed
      ? `bounded periodic retrain package is promotable: route rows are split into ${splitRegistry.trainRows.length} train rows and ${splitRegistry.evalRows.length} replay-gated eval rows.`
      : `bounded periodic retrain package is held back: ${blockers.join("; ")}`,
  };

  writeJsonArtifact(splitRegistryPath, splitRegistry);
  writeJsonArtifact(replayReportPath, report);
  writeJsonArtifact(promotionPackagePath, promotionPackage);

  return {
    generatedAt,
    splitRegistry,
    candidate,
    trainExport,
    evalExport,
    priorBaseArtifactDir: params.previousBaseArtifactDir,
    priorBaseManifestSummary: priorBase.summary,
    trainReplay,
    evalReplay,
    report,
    promotionPackage,
    paths: {
      splitRegistryPath,
      replayReportPath,
      promotionPackagePath,
    },
  };
}
