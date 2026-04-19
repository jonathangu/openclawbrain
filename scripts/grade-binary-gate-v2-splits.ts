#!/usr/bin/env tsx

import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

import type { RouteDecisionRowV1 } from "../src/brain-core/cold-start-router-contracts.ts";
import {
  replayColdStartRouterArtifactV1,
  type ColdStartRouterReplayGateVerdictV1,
} from "../src/brain-core/cold-start-router-replay-gate.ts";
import type { PolicySupervisionRowV1 } from "../src/brain-core/policy-supervision-rows.ts";
import {
  BINARY_GATE_V2_MERGED_ABSTENTION_TRANCHE_ID,
  BINARY_GATE_V2_MERGED_POSITIVE_TRANCHE_ID,
} from "./build-binary-gate-v2-tranches.ts";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..");
const workspaceRoot = path.resolve(repoRoot, "..");

interface ParsedArgs {
  taskId: string;
  splitManifestPath: string;
  sourceRunDir: string;
  candidateArtifactDir: string;
  outputDir: string;
  generatedAt: string;
}

interface TrancheAnchor {
  traceId: string;
  sourcePath: string;
  bucket?: string;
  whyIncluded?: string;
}

export interface TrancheManifest {
  contract: string;
  trancheId: string;
  anchors: TrancheAnchor[];
}

export interface BinaryGateV2SplitDedupeSummaryV1 {
  exactDuplicateRouteRowsCollapsed: number;
  exactDuplicatePolicyRowsCollapsed: number;
  conflictingDuplicateRouteRowIds: string[];
  conflictingDuplicatePolicyRowIds: string[];
}

export interface PreparedBinaryGateV2SplitReplayInputsV1 {
  manifest: TrancheManifest;
  laneKey: "mustFire" | "mustNotFire";
  sourceLaneTrancheId: string;
  routeRows: RouteDecisionRowV1[];
  policyRows: PolicySupervisionRowV1[];
  traceIds: string[];
  dedupe: BinaryGateV2SplitDedupeSummaryV1;
}

export interface BinaryGateV2SplitSummaryV1 {
  taskId: string;
  generatedAt: string;
  laneKey: "mustFire" | "mustNotFire";
  trancheId: string;
  traceCount: number;
  routeRowCount: number;
  policyRowCount: number;
  splitManifestPath: string;
  sourceRunDir: string;
  candidateArtifactDir: string;
  dedupe: BinaryGateV2SplitDedupeSummaryV1;
  replay: {
    verdict: ColdStartRouterReplayGateVerdictV1["verdict"];
    passed: boolean;
    summary: string;
    evaluatedRowCount: number;
    passedRowCount: number;
    failedRowCount: number;
    skippedRowCount: number;
    policyExpectations: ReturnType<typeof summarizePolicyExpectationsV1>;
  };
  activation: {
    activationMatchCount: number;
    activationExpectationCount: number;
    failedPolicyExpectations: number;
    totalPolicyExpectations: number;
  } | null;
  restraint: {
    unnecessaryActivations: number;
    totalEvaluatedRows: number;
    failedPolicyExpectations: number;
    totalPolicyExpectations: number;
    abstainMatchCount: number;
    abstainExpectationCount: number;
    stopLocalMatchCount: number;
    stopLocalExpectationCount: number;
  } | null;
}

function usage(): void {
  process.stderr.write(
    [
      "Usage: tsx scripts/grade-binary-gate-v2-splits.ts [options]",
      "",
      "Options:",
      "  --task-id <id>                  Required.",
      "  --split-manifest <path>         Required.",
      "  --source-run-dir <path>         Required.",
      "  --candidate-artifact-dir <path> Required.",
      "  --output-dir <path>             Required.",
      "  --generated-at <iso>            Override generated timestamp.",
      "  --help                          Show this help.",
    ].join("\n") + "\n",
  );
}

function normalizeCliString(value: string | undefined): string | null {
  if (typeof value !== "string") {
    return null;
  }
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
}

function parseArgs(argv: string[]): ParsedArgs {
  const parsed: Partial<ParsedArgs> = {
    generatedAt: new Date().toISOString(),
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    switch (arg) {
      case "--task-id":
        parsed.taskId = normalizeCliString(argv[index + 1]) ?? undefined;
        index += 1;
        break;
      case "--split-manifest":
        parsed.splitManifestPath = path.resolve(argv[index + 1] ?? "");
        index += 1;
        break;
      case "--source-run-dir":
        parsed.sourceRunDir = path.resolve(argv[index + 1] ?? "");
        index += 1;
        break;
      case "--candidate-artifact-dir":
        parsed.candidateArtifactDir = path.resolve(argv[index + 1] ?? "");
        index += 1;
        break;
      case "--output-dir":
        parsed.outputDir = path.resolve(argv[index + 1] ?? "");
        index += 1;
        break;
      case "--generated-at":
        parsed.generatedAt = normalizeCliString(argv[index + 1]) ?? parsed.generatedAt;
        index += 1;
        break;
      case "--help":
      case "-h":
        usage();
        process.exit(0);
        break;
      default:
        throw new Error(`Unknown argument: ${arg}`);
    }
  }

  if (!parsed.taskId || !parsed.splitManifestPath || !parsed.sourceRunDir || !parsed.candidateArtifactDir || !parsed.outputDir) {
    usage();
    throw new Error("Missing required arguments");
  }

  return parsed as ParsedArgs;
}

function readJson<T>(filePath: string): T {
  return JSON.parse(readFileSync(filePath, "utf8")) as T;
}

function writeJson(filePath: string, value: unknown): void {
  mkdirSync(path.dirname(filePath), { recursive: true });
  writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

function portableRelativePath(fromDir: string, toPath: string): string {
  return path.relative(fromDir, toPath).split(path.sep).join("/");
}

function shouldCollapseExactDuplicatesForTrancheV1(trancheId: string): boolean {
  return trancheId !== BINARY_GATE_V2_MERGED_POSITIVE_TRANCHE_ID
    && trancheId !== BINARY_GATE_V2_MERGED_ABSTENTION_TRANCHE_ID;
}

function dedupeExactRowsByIdV1<T extends { row_id: string }>(rows: readonly T[]): {
  rows: T[];
  collapsedExactDuplicateCount: number;
  conflictingDuplicateIds: string[];
} {
  const result: T[] = [];
  const serializedByRowId = new Map<string, string>();
  const conflictingDuplicateIds = new Set<string>();
  let collapsedExactDuplicateCount = 0;

  for (const row of rows) {
    const serialized = JSON.stringify(row);
    const priorSerialized = serializedByRowId.get(row.row_id);
    if (priorSerialized === undefined) {
      serializedByRowId.set(row.row_id, serialized);
      result.push(row);
      continue;
    }
    if (priorSerialized === serialized) {
      collapsedExactDuplicateCount += 1;
      continue;
    }
    conflictingDuplicateIds.add(row.row_id);
    result.push(row);
  }

  return {
    rows: result,
    collapsedExactDuplicateCount,
    conflictingDuplicateIds: [...conflictingDuplicateIds].sort(),
  };
}

export function inferBinaryGateV2SplitLaneKeyV1(trancheId: string): "mustFire" | "mustNotFire" {
  return trancheId.startsWith("must_fire_") ? "mustFire" : "mustNotFire";
}

export function sourceLaneTrancheIdForBinaryGateV2SplitLaneV1(laneKey: "mustFire" | "mustNotFire"): string {
  return laneKey === "mustFire" ? BINARY_GATE_V2_MERGED_POSITIVE_TRANCHE_ID : BINARY_GATE_V2_MERGED_ABSTENTION_TRANCHE_ID;
}

function routeRowMatchesTraceId(row: RouteDecisionRowV1, traceId: string, sourceLaneTrancheId: string): boolean {
  return row.row_id === `activation-first:${sourceLaneTrancheId}:${traceId}`;
}

export function summarizePolicyExpectationsV1(
  rowResults: ReturnType<typeof replayColdStartRouterArtifactV1>["rowResults"],
) {
  const policyExpectationResults = rowResults.flatMap((rowResult) => rowResult.policyExpectationResults);
  return {
    total: policyExpectationResults.length,
    passed: policyExpectationResults.filter((result) => result.passed).length,
    failed: policyExpectationResults.filter((result) => !result.passed).length,
    activationExpectationCount: policyExpectationResults.filter((result) => result.expectedActivated === true).length,
    activationMatchCount: policyExpectationResults.filter((result) => result.expectedActivated === true && result.actualActivated === true).length,
    abstainExpectationCount: policyExpectationResults.filter((result) => result.expectedAbstained === true).length,
    abstainMatchCount: policyExpectationResults.filter((result) => result.expectedAbstained === true && result.actualAbstained === true).length,
    stopLocalExpectationCount: policyExpectationResults.filter((result) => result.expectedStopLocal === true).length,
    stopLocalMatchCount: policyExpectationResults.filter((result) => result.expectedStopLocal === true && result.actualStopLocal === true).length,
  };
}

export function prepareBinaryGateV2SplitReplayInputsV1(params: {
  manifest: TrancheManifest;
  routeRows: RouteDecisionRowV1[];
  policyRows: PolicySupervisionRowV1[];
}): PreparedBinaryGateV2SplitReplayInputsV1 {
  const traceIds = params.manifest.anchors.map((anchor) => anchor.traceId);
  const traceIdSet = new Set(traceIds);
  const laneKey = inferBinaryGateV2SplitLaneKeyV1(params.manifest.trancheId);
  const sourceLaneTrancheId = sourceLaneTrancheIdForBinaryGateV2SplitLaneV1(laneKey);

  const filteredRouteRows = params.routeRows.filter((row) => (
    traceIds.some((traceId) => routeRowMatchesTraceId(row, traceId, sourceLaneTrancheId))
  ));
  const filteredPolicyRows = params.policyRows.filter((row) => (
    traceIdSet.has(row.trace_id)
    && row.trace_slice.route_row_id.startsWith(`activation-first:${sourceLaneTrancheId}:`)
  ));

  if (!shouldCollapseExactDuplicatesForTrancheV1(params.manifest.trancheId)) {
    return {
      manifest: params.manifest,
      laneKey,
      sourceLaneTrancheId,
      routeRows: filteredRouteRows,
      policyRows: filteredPolicyRows,
      traceIds,
      dedupe: {
        exactDuplicateRouteRowsCollapsed: 0,
        exactDuplicatePolicyRowsCollapsed: 0,
        conflictingDuplicateRouteRowIds: [],
        conflictingDuplicatePolicyRowIds: [],
      },
    };
  }

  const routeRowDedupe = dedupeExactRowsByIdV1(filteredRouteRows);
  const policyRowDedupe = dedupeExactRowsByIdV1(filteredPolicyRows);
  return {
    manifest: params.manifest,
    laneKey,
    sourceLaneTrancheId,
    routeRows: routeRowDedupe.rows,
    policyRows: policyRowDedupe.rows,
    traceIds,
    dedupe: {
      exactDuplicateRouteRowsCollapsed: routeRowDedupe.collapsedExactDuplicateCount,
      exactDuplicatePolicyRowsCollapsed: policyRowDedupe.collapsedExactDuplicateCount,
      conflictingDuplicateRouteRowIds: routeRowDedupe.conflictingDuplicateIds,
      conflictingDuplicatePolicyRowIds: policyRowDedupe.conflictingDuplicateIds,
    },
  };
}

export function buildBinaryGateV2SplitSummaryV1(params: {
  taskId: string;
  generatedAt: string;
  prepared: PreparedBinaryGateV2SplitReplayInputsV1;
  replay: ColdStartRouterReplayGateVerdictV1;
  splitManifestPath: string;
  sourceRunDir: string;
  candidateArtifactDir: string;
}): BinaryGateV2SplitSummaryV1 {
  const policySummary = summarizePolicyExpectationsV1(params.replay.rowResults);
  const evaluatedRows = params.replay.rowResults.filter((rowResult) => rowResult.gateEvaluated);
  const unnecessaryActivations = evaluatedRows.filter((rowResult) => rowResult.actualActivated === true).length;

  return {
    taskId: params.taskId,
    generatedAt: params.generatedAt,
    laneKey: params.prepared.laneKey,
    trancheId: params.prepared.manifest.trancheId,
    traceCount: params.prepared.traceIds.length,
    routeRowCount: params.prepared.routeRows.length,
    policyRowCount: params.prepared.policyRows.length,
    splitManifestPath: portableRelativePath(workspaceRoot, params.splitManifestPath),
    sourceRunDir: portableRelativePath(repoRoot, params.sourceRunDir),
    candidateArtifactDir: portableRelativePath(repoRoot, params.candidateArtifactDir),
    dedupe: params.prepared.dedupe,
    replay: {
      verdict: params.replay.verdict,
      passed: params.replay.passed,
      summary: params.replay.summary,
      evaluatedRowCount: params.replay.evaluatedRowCount,
      passedRowCount: params.replay.passedRowCount,
      failedRowCount: params.replay.failedRowCount,
      skippedRowCount: params.replay.skippedRowCount,
      policyExpectations: policySummary,
    },
    activation: params.prepared.laneKey === "mustFire"
      ? {
        activationMatchCount: policySummary.activationMatchCount,
        activationExpectationCount: policySummary.activationExpectationCount,
        failedPolicyExpectations: policySummary.failed,
        totalPolicyExpectations: policySummary.total,
      }
      : null,
    restraint: params.prepared.laneKey === "mustNotFire"
      ? {
        unnecessaryActivations,
        totalEvaluatedRows: evaluatedRows.length,
        failedPolicyExpectations: policySummary.failed,
        totalPolicyExpectations: policySummary.total,
        abstainMatchCount: policySummary.abstainMatchCount,
        abstainExpectationCount: policySummary.abstainExpectationCount,
        stopLocalMatchCount: policySummary.stopLocalMatchCount,
        stopLocalExpectationCount: policySummary.stopLocalExpectationCount,
      }
      : null,
  };
}

export function gradeBinaryGateV2SplitV1(params: {
  taskId: string;
  splitManifestPath: string;
  sourceRunDir: string;
  candidateArtifactDir: string;
  generatedAt: string;
}): {
  prepared: PreparedBinaryGateV2SplitReplayInputsV1;
  replay: ColdStartRouterReplayGateVerdictV1;
  summary: BinaryGateV2SplitSummaryV1;
} {
  const manifest = readJson<TrancheManifest>(params.splitManifestPath);
  const routeRows = readJson<RouteDecisionRowV1[]>(path.join(params.sourceRunDir, "route-rows.json"));
  const policyRows = readJson<PolicySupervisionRowV1[]>(path.join(params.sourceRunDir, "policy-supervision-rows.json"));
  const prepared = prepareBinaryGateV2SplitReplayInputsV1({
    manifest,
    routeRows,
    policyRows,
  });
  const replay = replayColdStartRouterArtifactV1({
    artifactDir: params.candidateArtifactDir,
    routeRows: prepared.routeRows,
    policySupervisionRows: prepared.policyRows,
  });
  const summary = buildBinaryGateV2SplitSummaryV1({
    taskId: params.taskId,
    generatedAt: params.generatedAt,
    prepared,
    replay,
    splitManifestPath: params.splitManifestPath,
    sourceRunDir: params.sourceRunDir,
    candidateArtifactDir: params.candidateArtifactDir,
  });
  return { prepared, replay, summary };
}

function main(): void {
  const args = parseArgs(process.argv.slice(2));
  mkdirSync(args.outputDir, { recursive: true });

  const graded = gradeBinaryGateV2SplitV1({
    taskId: args.taskId,
    splitManifestPath: args.splitManifestPath,
    sourceRunDir: args.sourceRunDir,
    candidateArtifactDir: args.candidateArtifactDir,
    generatedAt: args.generatedAt,
  });

  writeJson(path.join(args.outputDir, "route-rows.json"), graded.prepared.routeRows);
  writeJson(path.join(args.outputDir, "policy-supervision-rows.json"), graded.prepared.policyRows);
  writeJson(path.join(args.outputDir, "replay-verdict.json"), graded.replay);
  writeJson(path.join(args.outputDir, "summary.json"), graded.summary);
  writeFileSync(
    path.join(args.outputDir, "README.md"),
    [
      `# ${graded.summary.trancheId}`,
      "",
      `- laneKey: ${graded.summary.laneKey}`,
      `- traceCount: ${graded.summary.traceCount}`,
      `- routeRowCount: ${graded.summary.routeRowCount}`,
      `- policyRowCount: ${graded.summary.policyRowCount}`,
      `- replay verdict: ${graded.replay.verdict}`,
      `- replay passed: ${graded.replay.passed}`,
      `- summary: ${graded.replay.summary}`,
      ...(graded.summary.dedupe.exactDuplicateRouteRowsCollapsed > 0
        ? [`- collapsed exact duplicate route rows: ${graded.summary.dedupe.exactDuplicateRouteRowsCollapsed}`]
        : []),
      ...(graded.summary.dedupe.exactDuplicatePolicyRowsCollapsed > 0
        ? [`- collapsed exact duplicate policy rows: ${graded.summary.dedupe.exactDuplicatePolicyRowsCollapsed}`]
        : []),
      ...(graded.summary.dedupe.conflictingDuplicateRouteRowIds.length > 0
        ? [`- conflicting duplicate route row ids left intact: ${graded.summary.dedupe.conflictingDuplicateRouteRowIds.join(", ")}`]
        : []),
      ...(graded.summary.dedupe.conflictingDuplicatePolicyRowIds.length > 0
        ? [`- conflicting duplicate policy row ids left intact: ${graded.summary.dedupe.conflictingDuplicatePolicyRowIds.join(", ")}`]
        : []),
      ...(graded.summary.laneKey === "mustFire"
        ? [
          `- activation expectation passes: ${graded.summary.activation?.activationMatchCount ?? 0}/${graded.summary.activation?.activationExpectationCount ?? 0}`,
          `- failed policy expectations: ${graded.summary.activation?.failedPolicyExpectations ?? 0}/${graded.summary.activation?.totalPolicyExpectations ?? 0}`,
        ]
        : [
          `- unnecessary activations: ${graded.summary.restraint?.unnecessaryActivations ?? 0}/${graded.summary.restraint?.totalEvaluatedRows ?? 0}`,
          `- abstain matches: ${graded.summary.restraint?.abstainMatchCount ?? 0}/${graded.summary.restraint?.abstainExpectationCount ?? 0}`,
          `- stop_local matches: ${graded.summary.restraint?.stopLocalMatchCount ?? 0}/${graded.summary.restraint?.stopLocalExpectationCount ?? 0}`,
          `- failed policy expectations: ${graded.summary.restraint?.failedPolicyExpectations ?? 0}/${graded.summary.restraint?.totalPolicyExpectations ?? 0}`,
        ]),
      "",
      "Artifacts:",
      `- summary.json`,
      `- replay-verdict.json`,
      `- route-rows.json`,
      `- policy-supervision-rows.json`,
    ].join("\n") + "\n",
    "utf8",
  );

  process.stdout.write(`${JSON.stringify(graded.summary, null, 2)}\n`);
}

const isMain = process.argv[1] ? path.resolve(process.argv[1]) === __filename : false;

if (isMain) {
  try {
    main();
  } catch (error) {
    process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
    process.exit(1);
  }
}
