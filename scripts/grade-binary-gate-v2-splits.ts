#!/usr/bin/env tsx

import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

import type { RouteDecisionRowV1 } from "../src/brain-core/cold-start-router-contracts.ts";
import { replayColdStartRouterArtifactV1 } from "../src/brain-core/cold-start-router-replay-gate.ts";
import type { PolicySupervisionRowV1 } from "../src/brain-core/policy-supervision-rows.ts";

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

interface TrancheManifest {
  contract: string;
  trancheId: string;
  anchors: TrancheAnchor[];
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

function inferLaneKey(trancheId: string): "mustFire" | "mustNotFire" {
  return trancheId.startsWith("must_fire_") ? "mustFire" : "mustNotFire";
}

function sourceLaneTrancheIdForSplitLane(laneKey: "mustFire" | "mustNotFire"): string {
  return laneKey === "mustFire" ? "must_fire_binary_gate_v2" : "must_not_fire_binary_gate_v2";
}

function routeRowMatchesTraceId(row: RouteDecisionRowV1, traceId: string, sourceLaneTrancheId: string): boolean {
  return row.row_id === `activation-first:${sourceLaneTrancheId}:${traceId}`;
}

function summarizePolicyExpectations(
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

function main(): void {
  const args = parseArgs(process.argv.slice(2));
  mkdirSync(args.outputDir, { recursive: true });

  const manifest = readJson<TrancheManifest>(args.splitManifestPath);
  const traceIds = manifest.anchors.map((anchor) => anchor.traceId);
  const traceIdSet = new Set(traceIds);
  const laneKey = inferLaneKey(manifest.trancheId);
  const sourceLaneTrancheId = sourceLaneTrancheIdForSplitLane(laneKey);
  const routeRows = readJson<RouteDecisionRowV1[]>(path.join(args.sourceRunDir, "route-rows.json"));
  const policyRows = readJson<PolicySupervisionRowV1[]>(path.join(args.sourceRunDir, "policy-supervision-rows.json"));

  const filteredRouteRows = routeRows.filter((row) => traceIds.some((traceId) => routeRowMatchesTraceId(row, traceId, sourceLaneTrancheId)));
  const filteredPolicyRows = policyRows.filter((row) => (
    traceIdSet.has(row.trace_id)
    && row.trace_slice.route_row_id.startsWith(`activation-first:${sourceLaneTrancheId}:`)
  ));

  const replay = replayColdStartRouterArtifactV1({
    artifactDir: args.candidateArtifactDir,
    routeRows: filteredRouteRows,
    policySupervisionRows: filteredPolicyRows,
  });

  const policySummary = summarizePolicyExpectations(replay.rowResults);
  const evaluatedRows = replay.rowResults.filter((rowResult) => rowResult.gateEvaluated);
  const unnecessaryActivations = evaluatedRows.filter((rowResult) => rowResult.actualActivated === true).length;

  const summary = {
    taskId: args.taskId,
    generatedAt: args.generatedAt,
    laneKey,
    trancheId: manifest.trancheId,
    traceCount: traceIds.length,
    routeRowCount: filteredRouteRows.length,
    policyRowCount: filteredPolicyRows.length,
    splitManifestPath: portableRelativePath(workspaceRoot, args.splitManifestPath),
    sourceRunDir: portableRelativePath(repoRoot, args.sourceRunDir),
    candidateArtifactDir: portableRelativePath(repoRoot, args.candidateArtifactDir),
    replay: {
      verdict: replay.verdict,
      passed: replay.passed,
      summary: replay.summary,
      evaluatedRowCount: replay.evaluatedRowCount,
      passedRowCount: replay.passedRowCount,
      failedRowCount: replay.failedRowCount,
      skippedRowCount: replay.skippedRowCount,
      policyExpectations: policySummary,
    },
    activation: laneKey === "mustFire"
      ? {
        activationMatchCount: policySummary.activationMatchCount,
        activationExpectationCount: policySummary.activationExpectationCount,
        failedPolicyExpectations: policySummary.failed,
        totalPolicyExpectations: policySummary.total,
      }
      : null,
    restraint: laneKey === "mustNotFire"
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

  writeJson(path.join(args.outputDir, "route-rows.json"), filteredRouteRows);
  writeJson(path.join(args.outputDir, "policy-supervision-rows.json"), filteredPolicyRows);
  writeJson(path.join(args.outputDir, "replay-verdict.json"), replay);
  writeJson(path.join(args.outputDir, "summary.json"), summary);
  writeFileSync(
    path.join(args.outputDir, "README.md"),
    [
      `# ${manifest.trancheId}`,
      "",
      `- laneKey: ${laneKey}`,
      `- traceCount: ${traceIds.length}`,
      `- routeRowCount: ${filteredRouteRows.length}`,
      `- policyRowCount: ${filteredPolicyRows.length}`,
      `- replay verdict: ${replay.verdict}`,
      `- replay passed: ${replay.passed}`,
      `- summary: ${replay.summary}`,
      ...(laneKey === "mustFire"
        ? [
          `- activation expectation passes: ${policySummary.activationMatchCount}/${policySummary.activationExpectationCount}`,
          `- failed policy expectations: ${policySummary.failed}/${policySummary.total}`,
        ]
        : [
          `- unnecessary activations: ${unnecessaryActivations}/${evaluatedRows.length}`,
          `- abstain matches: ${policySummary.abstainMatchCount}/${policySummary.abstainExpectationCount}`,
          `- stop_local matches: ${policySummary.stopLocalMatchCount}/${policySummary.stopLocalExpectationCount}`,
          `- failed policy expectations: ${policySummary.failed}/${policySummary.total}`,
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

  process.stdout.write(`${JSON.stringify(summary, null, 2)}\n`);
}

main();
