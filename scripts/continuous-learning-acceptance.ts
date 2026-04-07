#!/usr/bin/env tsx

import { mkdirSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { writeGraphifyFinalReplayProof } from "./graphify-final-replay-proof.mjs";
import { writeGraphifySchedulerRun } from "./graphify-scheduler.mjs";
import { verifyProofSmoke } from "./verify-proof-smoke.mjs";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const defaultRepoRoot = path.resolve(__dirname, "..");
const defaultWorkspaceRoot = path.resolve(defaultRepoRoot, "..");
const defaultOutputRoot = path.join(defaultWorkspaceRoot, "artifacts", "continuous-learning-acceptance");
const defaultOpenClawHome = path.join(os.homedir(), ".openclaw");

export const CONTINUOUS_LEARNING_ACCEPTANCE_CONTRACT = "continuous_learning_acceptance_lane.v1" as const;
export const CONTINUOUS_LEARNING_ACCEPTANCE_LAYOUT = {
  summary: "summary.md",
  status: "status.json",
  graphifyScheduler: "graphify-scheduler",
  finalReplayProof: "final-replay-proof",
} as const;

export interface ContinuousLearningAcceptanceSchedulerCheckV1 {
  cadence: "delta" | "reorg";
  ok: boolean;
  summary: string;
  outputRoot: string;
  runId: string;
  summaryPath: string;
  statusPath: string;
  registryPath: string;
}

export interface ContinuousLearningAcceptanceFinalReplayCheckV1 {
  ok: boolean;
  summary: string;
  proofRoot: string;
  reportPath: string;
  statusPath: string;
}

export interface ContinuousLearningAcceptanceProofSmokeCheckV1 {
  ok: boolean;
  enforced: boolean;
  summary: string;
  bundlesChecked: number;
}

export interface ContinuousLearningAcceptanceLaneOptionsV1 {
  repoRoot?: string;
  workspaceRoot?: string;
  outputRoot?: string;
  openclawHome?: string;
  activationRoot?: string | null;
  generatedAt?: string;
  runId?: string;
  clean?: boolean;
  proofSmokeMaxAgeDays?: number;
}

export interface ContinuousLearningAcceptanceLaneResultV1 {
  contract: typeof CONTINUOUS_LEARNING_ACCEPTANCE_CONTRACT;
  ok: boolean;
  generatedAt: string;
  repoRoot: string;
  workspaceRoot: string;
  outputRoot: string;
  runId: string;
  summaryPath: string;
  statusPath: string;
  blockers: string[];
  checks: {
    proofSmoke: ContinuousLearningAcceptanceProofSmokeCheckV1;
    graphifyDelta: ContinuousLearningAcceptanceSchedulerCheckV1;
    graphifyReorg: ContinuousLearningAcceptanceSchedulerCheckV1;
    finalReplayProof: ContinuousLearningAcceptanceFinalReplayCheckV1;
  };
  outputs: {
    graphifySchedulerRoot: string;
    finalReplayProofRoot: string;
  };
}

function canonicalizeJsonValue(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map((entry) => canonicalizeJsonValue(entry));
  }
  if (value === null || typeof value !== "object") {
    return value;
  }
  const result: Record<string, unknown> = {};
  for (const key of Object.keys(value as Record<string, unknown>).sort((left, right) => left.localeCompare(right))) {
    const nextValue = canonicalizeJsonValue((value as Record<string, unknown>)[key]);
    if (nextValue !== undefined) {
      result[key] = nextValue;
    }
  }
  return result;
}

function stableJsonStringify(value: unknown): string {
  return `${JSON.stringify(canonicalizeJsonValue(value), null, 2)}\n`;
}

function ensureDir(dirPath: string): void {
  mkdirSync(dirPath, { recursive: true });
}

function writeJson(filePath: string, value: unknown): string {
  ensureDir(path.dirname(filePath));
  writeFileSync(filePath, stableJsonStringify(value), "utf8");
  return filePath;
}

function writeText(filePath: string, value: string): string {
  ensureDir(path.dirname(filePath));
  writeFileSync(filePath, `${value}\n`, "utf8");
  return filePath;
}

function timestampToken(value: string): string {
  return value.replace(/[:]/g, "-");
}

function renderCheckLine(label: string, ok: boolean, summary: string): string {
  return `- ${label}: ${ok ? "ok" : "blocker"} — ${summary}`;
}

function summarizeSchedulerResult(result: ReturnType<typeof writeGraphifySchedulerRun>): ContinuousLearningAcceptanceSchedulerCheckV1 {
  return {
    cadence: result.cadence,
    ok: result.status === "completed" && result.offPath === true && result.inspectable === true && result.replayable === true,
    summary: `${result.status}; downstream=${result.downstreamArtifacts.map((artifact: { kind: string }) => artifact.kind).join(", ")}`,
    outputRoot: result.outputRoot,
    runId: result.runId,
    summaryPath: result.summaryPath,
    statusPath: result.statusPath,
    registryPath: result.registryPath,
  };
}

function summarizeFinalReplayResult(result: ReturnType<typeof writeGraphifyFinalReplayProof>): ContinuousLearningAcceptanceFinalReplayCheckV1 {
  return {
    ok: result.verdict.status === "pass" && result.report.modes.length > 0 && result.report.modeRanking.length > 0,
    summary: `${result.verdict.status}; coldStartDelta=${result.verdict.coldStartDelta}`,
    proofRoot: result.proofRoot,
    reportPath: result.reportPath,
    statusPath: result.statusPath,
  };
}

function buildSummaryMarkdown(result: ContinuousLearningAcceptanceLaneResultV1): string {
  const lines = [
    "# Continuous learning acceptance lane",
    "",
    `- contract: \`${result.contract}\``,
    `- run id: \`${result.runId}\``,
    `- generated at: ${result.generatedAt}`,
    `- overall: ${result.ok ? "pass" : "blocked"}`,
    "",
    "## Checks",
    renderCheckLine("proof smoke", result.checks.proofSmoke.ok, result.checks.proofSmoke.summary),
    renderCheckLine("graphify delta", result.checks.graphifyDelta.ok, result.checks.graphifyDelta.summary),
    renderCheckLine("graphify reorg", result.checks.graphifyReorg.ok, result.checks.graphifyReorg.summary),
    renderCheckLine("final replay proof", result.checks.finalReplayProof.ok, result.checks.finalReplayProof.summary),
    "",
    "## Artifact roots",
    `- graphify scheduler: \`${result.outputs.graphifySchedulerRoot}\``,
    `- final replay proof: \`${result.outputs.finalReplayProofRoot}\``,
    `- summary: \`${result.summaryPath}\``,
    `- status: \`${result.statusPath}\``,
  ];
  if (result.blockers.length > 0) {
    lines.push("", "## Blockers");
    for (const blocker of result.blockers) {
      lines.push(`- ${blocker}`);
    }
  }
  return lines.join("\n");
}

export function writeContinuousLearningAcceptanceLane(options: ContinuousLearningAcceptanceLaneOptionsV1 = {}): ContinuousLearningAcceptanceLaneResultV1 {
  const repoRoot = path.resolve(options.repoRoot ?? defaultRepoRoot);
  const workspaceRoot = path.resolve(options.workspaceRoot ?? defaultWorkspaceRoot);
  const outputRoot = path.resolve(options.outputRoot ?? defaultOutputRoot);
  const openclawHome = path.resolve(options.openclawHome ?? defaultOpenClawHome);
  const activationRoot = options.activationRoot === undefined || options.activationRoot === null
    ? null
    : path.resolve(options.activationRoot);
  const generatedAt = options.generatedAt ?? new Date().toISOString();
  const runId = options.runId ?? `continuous-learning-${timestampToken(generatedAt)}`;
  const laneRoot = path.join(outputRoot, runId);
  const summaryPath = path.join(laneRoot, CONTINUOUS_LEARNING_ACCEPTANCE_LAYOUT.summary);
  const statusPath = path.join(laneRoot, CONTINUOUS_LEARNING_ACCEPTANCE_LAYOUT.status);
  const graphifySchedulerRoot = path.join(laneRoot, CONTINUOUS_LEARNING_ACCEPTANCE_LAYOUT.graphifyScheduler);
  const finalReplayArtifactRoot = path.join(laneRoot, CONTINUOUS_LEARNING_ACCEPTANCE_LAYOUT.finalReplayProof);
  const finalReplayProofRoot = path.join(finalReplayArtifactRoot, "proof");
  const finalReplayReportPath = path.join(finalReplayArtifactRoot, "report.md");
  const finalReplayStatusPath = path.join(finalReplayArtifactRoot, "status.json");

  if (options.clean !== false) {
    rmSync(laneRoot, { recursive: true, force: true });
  }
  ensureDir(laneRoot);

  const schedulerBaseOptions = {
    repoRoot,
    workspaceRoot,
    openclawHome,
    ...(activationRoot === null ? {} : { activationRoot }),
    outputRoot: graphifySchedulerRoot,
    generatedAt,
    clean: true,
  };

  const delta = summarizeSchedulerResult(writeGraphifySchedulerRun({
    ...schedulerBaseOptions,
    cadence: "delta",
    runId: `${runId}-delta`,
  }));
  const reorg = summarizeSchedulerResult(writeGraphifySchedulerRun({
    ...schedulerBaseOptions,
    cadence: "reorg",
    runId: `${runId}-reorg`,
  }));

  const finalReplay = summarizeFinalReplayResult(writeGraphifyFinalReplayProof({
    repoRoot,
    workspaceRoot,
    artifactRoot: finalReplayArtifactRoot,
    proofRoot: finalReplayProofRoot,
    reportPath: finalReplayReportPath,
    statusPath: finalReplayStatusPath,
    generatedAt,
  }));

  const proofSmokeResult = verifyProofSmoke({
    repoRoot,
    maxAgeDays: options.proofSmokeMaxAgeDays ?? 21,
    now: new Date(generatedAt),
  }) as {
    ok: boolean;
    enforced: boolean;
    message: string;
    bundlesChecked: number;
  };

  const proofSmoke: ContinuousLearningAcceptanceProofSmokeCheckV1 = {
    ok: proofSmokeResult.ok,
    enforced: proofSmokeResult.enforced,
    summary: proofSmokeResult.message,
    bundlesChecked: proofSmokeResult.bundlesChecked,
  };

  const blockers = [
    ...(proofSmoke.ok ? [] : [`proof smoke failed: ${proofSmoke.summary}`]),
    ...(delta.ok ? [] : [`delta scheduler run failed: ${delta.summary}`]),
    ...(reorg.ok ? [] : [`reorg scheduler run failed: ${reorg.summary}`]),
    ...(finalReplay.ok ? [] : [`final replay proof failed: ${finalReplay.summary}`]),
  ];

  const result: ContinuousLearningAcceptanceLaneResultV1 = {
    contract: CONTINUOUS_LEARNING_ACCEPTANCE_CONTRACT,
    ok: blockers.length === 0,
    generatedAt,
    repoRoot,
    workspaceRoot,
    outputRoot,
    runId,
    summaryPath,
    statusPath,
    blockers,
    checks: {
      proofSmoke,
      graphifyDelta: delta,
      graphifyReorg: reorg,
      finalReplayProof: finalReplay,
    },
    outputs: {
      graphifySchedulerRoot,
      finalReplayProofRoot,
    },
  };

  writeText(summaryPath, buildSummaryMarkdown(result));
  writeJson(statusPath, result);

  return result;
}

export function runContinuousLearningAcceptanceLaneCli(argv = process.argv.slice(2)): void {
  const args = [...argv];
  const options: ContinuousLearningAcceptanceLaneOptionsV1 = {};

  while (args.length > 0) {
    const arg = args.shift();
    switch (arg) {
      case "--repo-root":
        options.repoRoot = path.resolve(args.shift() ?? "");
        break;
      case "--workspace-root":
        options.workspaceRoot = path.resolve(args.shift() ?? "");
        break;
      case "--output-root":
        options.outputRoot = path.resolve(args.shift() ?? "");
        break;
      case "--openclaw-home":
        options.openclawHome = path.resolve(args.shift() ?? "");
        break;
      case "--activation-root":
        options.activationRoot = path.resolve(args.shift() ?? "");
        break;
      case "--generated-at":
        options.generatedAt = args.shift() ?? undefined;
        break;
      case "--run-id":
        options.runId = args.shift() ?? undefined;
        break;
      case "--keep":
        options.clean = false;
        break;
      case "--proof-smoke-max-age-days":
        options.proofSmokeMaxAgeDays = Number(args.shift() ?? "21");
        break;
      case "--help":
      case "-h":
        process.stdout.write([
          "Usage: node scripts/continuous-learning-acceptance.ts [options]",
          "",
          "Options:",
          `  --repo-root <path>            Repo root (default: ${defaultRepoRoot})`,
          `  --workspace-root <path>       Workspace root (default: ${defaultWorkspaceRoot})`,
          `  --output-root <path>          Output root (default: ${defaultOutputRoot})`,
          `  --openclaw-home <path>        OpenClaw home used for the graphify scheduler lane (default: ${defaultOpenClawHome})`,
          "  --activation-root <path>      Optional activation root for scheduler source bundle exports",
          "  --generated-at <iso>          Override generated timestamp",
          "  --run-id <id>                 Override the lane run id",
          "  --proof-smoke-max-age-days <n>  Max bundle age for the proof smoke gate (default: 21)",
          "  --keep                        Do not clean the lane output root before writing",
          "  --help                        Show this help",
        ].join("\n") + "\n");
        return;
      default:
        throw new Error(`Unknown argument: ${arg}`);
    }
  }

  const result = writeContinuousLearningAcceptanceLane(options);
  process.stdout.write(`${JSON.stringify({
    contract: result.contract,
    ok: result.ok,
    runId: result.runId,
    summaryPath: result.summaryPath,
    statusPath: result.statusPath,
    blockers: result.blockers,
  }, null, 2)}\n`);
  if (!result.ok) {
    process.exitCode = 1;
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === __filename) {
  try {
    runContinuousLearningAcceptanceLaneCli();
  } catch (error) {
    process.stderr.write(`${error instanceof Error ? error.stack ?? error.message : String(error)}\n`);
    process.exit(1);
  }
}
