#!/usr/bin/env node

import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

import type {
  ColdStartStopLabelV1,
  DataRegistryEntryV1,
  RouteCandidateV1,
  RouteDecisionRowV1 as ColdStartRouteDecisionRowV1,
} from "../src/brain-core/cold-start-router-contracts.ts";
import { loadColdStartRouterArtifactBundleV1 } from "../src/brain-core/cold-start-router-runtime.ts";
import {
  replayColdStartRouterArtifactV1,
  type ColdStartRouterReplayGateLaneSummaryV1,
  type ColdStartRouterReplayGateVerdictV1,
} from "../src/brain-core/cold-start-router-replay-gate.ts";
import {
  trainColdStartRouterArtifactV1,
  type ColdStartRouterInterventionHeadConfigV1,
} from "../src/brain-core/cold-start-router-trainer.ts";
import type { GraphifyRouteObjectiveTraceLabelLikeV1 } from "../src/brain-core/graphify-training-bridge.ts";
import {
  buildPolicySupervisionRowV1,
  type PolicySupervisionRowV1,
} from "../src/brain-core/policy-supervision-rows.ts";
import type { RouteDecisionRowV1 as PolicyProjectionRouteRowV1 } from "../src/brain-core/route-rows.ts";
import {
  type ComparativeEvalScorecardV1,
  runComparativeEval,
} from "../src/eval/comparative-eval-runner.ts";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..");
const workspaceRoot = path.resolve(repoRoot, "..");

const DEFAULT_TASK_ID = "T-20260415-257";
const DEFAULT_HARNESS_DIR = path.join(
  workspaceRoot,
  "task-artifacts",
  DEFAULT_TASK_ID,
  "activation-first-retune-harness",
);
const DEFAULT_OUTPUT_DIR = path.join(
  repoRoot,
  "artifacts",
  "activation-first-gating-retune",
  DEFAULT_TASK_ID,
);
const DEFAULT_WARM_START_ARTIFACT_DIR = path.join(
  repoRoot,
  "artifacts",
  "cold-start-router-approved-export",
  "real-approved-router-train.hotpotqa-musique.v3",
);
const CANDIDATE_SPECIFIC_BROAD_LIVE_RUN_NOTE =
  "Candidate-specific broad-live comparative eval executed for the just-trained gating-only candidate.";
export const ACTIVATION_FIRST_GATING_ONLY_INTERVENTION_HEAD_V1: ColdStartRouterInterventionHeadConfigV1 = {
  decisionPolicyMode: "gating_only_v1",
  freezeCandidateSelection: true,
  freezeStopLocal: true,
  featureProfile: "resume_gate_v1",
};
export const ACTIVATION_FIRST_GATING_ONLY_CALIBRATION_OVERRIDES_V1 = {
  activationThreshold: 0.38,
} as const;
export const FELT_SAFE_POSITIVE_PRESSURE_ROW_WEIGHT_OVERRIDES_V1: Record<string, number> = {
  "activation-first:felt_resume_25:live-main-6688d40b-5220-45ca-83f4-835184de4116-window-002": 3,
  "activation-first:felt_resume_25:live-main-4c69091d-1290-4bcd-a74c-7166c46e5670-window-002": 3,
  "activation-first:felt_resume_25:live-main-971973d8-2a63-4883-a18f-bfa883f844ea-window-003": 3,
  "activation-first:felt_resume_25:live-main-b8b03b3e-6e68-4062-8dd5-0439897868c4-window-003": 3,
  "activation-first:felt_resume_25:live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-002": 3,
  "activation-first:felt_resume_25:live-main-7498149c-ca61-4cda-b16f-880f2c1cf323-window-003": 3,
  "activation-first:felt_resume_25:live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-016": 3,
};
export const FELT_SAFE_POSITIVE_PRESSURE_SUPPLEMENTAL_ROW_WEIGHT_V1 = 1.5;

function parseNonNegativeIntegerEnvV1(name: string, fallback: number): number {
  const raw = process.env[name];
  if (!raw) {
    return fallback;
  }
  const value = Number.parseInt(raw, 10);
  if (!Number.isFinite(value) || value < 0) {
    return fallback;
  }
  return value;
}

function parseFiniteNumberEnvV1(name: string, fallback: number): number {
  const raw = process.env[name];
  if (!raw) {
    return fallback;
  }
  const value = Number(raw);
  if (!Number.isFinite(value)) {
    return fallback;
  }
  return value;
}

export const SAFE_FEEDBACK_ROUTE_DUPLICATE_COUNT_V1 = parseNonNegativeIntegerEnvV1(
  "OCB_SAFE_FEEDBACK_ROUTE_DUPLICATE_COUNT",
  0,
);
export const SAFE_FEEDBACK_ROUTE_OUTCOME_GAIN_BONUS_V1 = parseFiniteNumberEnvV1(
  "OCB_SAFE_FEEDBACK_ROUTE_OUTCOME_GAIN_BONUS",
  0,
);

export type ActivationFirstLaneKeyV1 = "feltResume" | "mustFire" | "mustNotFire";
export type ActivationFirstCandidateStatusV1 = "pass" | "reject" | "architecture_verdict";
export type BroadLiveVetoResultV1 = "pass" | "reject" | "proof_read_only";

interface ParsedArgs {
  taskId: string;
  harnessDir: string;
  outputDir: string;
  warmStartArtifactDir: string;
  includeMustFire: boolean;
  generatedAt: string;
}

interface ActivationFirstHarnessLaneV1 {
  trancheId: string;
  purpose?: string | null;
  anchorTraceCount?: number | null;
  targetTraceCount?: number | null;
  manifestPath: string;
  sourceManifestPath?: string;
}

interface ActivationFirstRetuneHarnessV1 {
  contract: string;
  taskId: string;
  generatedAt: string;
  missionTestCommand?: string;
  labelsTemplatePath: string;
  routeObjectiveInputPath: string;
  suggestedTrainingConfig: {
    focusLaneWeights: Record<string, number>;
    rowTypeWeights: Record<string, number>;
  };
  primaryOptimizeLane?: string;
  feltResume: ActivationFirstHarnessLaneV1;
  mustFire: ActivationFirstHarnessLaneV1;
  mustNotFire: ActivationFirstHarnessLaneV1;
  broadLiveGuardrail?: {
    manifestPath?: string | null;
    manifestContract?: string | null;
    manifestId?: string | null;
    traceCount?: number | null;
  };
  hardNegativeMining?: {
    weightFloorsByClass?: Record<string, number>;
  };
}

interface ActivationFirstRouteObjectiveLaneV1 {
  trancheId: string;
  traceCount: number;
  manifestPath: string;
  sourceManifestPath: string;
  traceIds: string[];
  labelsByTraceId: Record<string, GraphifyRouteObjectiveTraceLabelLikeV1>;
}

interface ActivationFirstRouteObjectiveInputV1 {
  contract: string;
  taskId: string;
  generatedAt: string;
  labelsTemplatePath: string;
  lanes: {
    feltResume: ActivationFirstRouteObjectiveLaneV1;
    mustFire: ActivationFirstRouteObjectiveLaneV1;
    mustNotFire: ActivationFirstRouteObjectiveLaneV1;
  };
  suggestedTrainingConfig: {
    focusLaneWeights: Record<string, number>;
    rowTypeWeights: Record<string, number>;
  };
}

interface FrozenRecordedSessionEvalManifestV1 {
  contract: string;
  manifestId: string;
  expectedTraceCount?: number;
  traces: Array<{
    traceId?: string;
    tracePath: string;
    notes?: string[];
  }>;
}

interface RecordedSessionTraceTurnV1 {
  turnId?: string;
  userMessage?: string;
  runtimeHints?: unknown;
  expectedContextPhrases?: unknown;
}

interface RecordedSessionTraceSeedCueV1 {
  cueId?: string;
  content?: string;
  text?: string;
  summary?: string;
  kind?: string;
}

interface RecordedSessionTraceV1 {
  contract: string;
  traceId: string;
  seedCues?: RecordedSessionTraceSeedCueV1[];
  turns: RecordedSessionTraceTurnV1[];
}

interface LearnedRouteBundleTurnV1 {
  selectedContextIds?: unknown;
  selectedContextTexts?: unknown;
  expectedContextPhrases?: unknown;
  usedLearnedRouteFn?: unknown;
  activationTaken?: unknown;
  routerIdentity?: unknown;
}

interface LearnedRouteBundleV1 {
  mode: string;
  turns: LearnedRouteBundleTurnV1[];
}

interface ReplayBundleSurfaceV1 {
  traceId: string;
  bundleDir: string;
  selectedContextIds: string[];
  selectedContextTexts: string[];
  expectedContextPhrases: string[];
  modeSelectedContextIds: Record<string, string[]>;
  modeSelectedContextTexts: Record<string, string[]>;
  usedLearnedRouteTurnCount: number;
  activatedTurnCount: number;
  boundTurnCount: number;
}

interface LoadedFrozenTraceV1 {
  traceId: string;
  tracePath: string;
  trace: RecordedSessionTraceV1;
  traceLabel: GraphifyRouteObjectiveTraceLabelLikeV1;
}

export interface PseudoRouteBucketPlanV1 {
  candidateCount: number;
  evidenceSpanCount: number;
  hardNegativeCount: number;
  outcomeGain: number;
  stopLabel: ColdStartStopLabelV1;
  chosenActionKind: PolicyProjectionRouteRowV1["chosen_action_kind"];
}

interface MaterializedLaneRowsV1 {
  laneKey: ActivationFirstLaneKeyV1;
  laneName: string;
  datasetId: string;
  registryEntry: DataRegistryEntryV1;
  routeRows: ColdStartRouteDecisionRowV1[];
  policyRows: PolicySupervisionRowV1[];
}

type SyntheticCandidateMetadataProfileV1 = "synthetic_eval" | "runtime_like_replay";

interface FeltOverlapSummaryV1 {
  availableTraceCount: number;
  totalTraceCount: number;
  betterCount: number;
  tiedCount: number;
  worseCount: number;
}

export interface FeltOptimizeComparativeSummaryV1 {
  available: boolean;
  comparableTraceCount: number;
  betterCount: number;
  tiedCount: number;
  worseCount: number;
  regressions: number;
  requiredContextRecallSummary: string | null;
  outputDir: string | null;
  notes: string[];
}

interface BroadLiveTraceScorecardRowV1 {
  traceId: string;
  candidateRelationVsBaseline: "better" | "tied" | "worse" | null;
}

interface BroadLiveScorecardModeRowV1 {
  mode: string;
  totalUsedLearnedRouteTurnCount?: number | null;
}

interface BroadLiveScorecardV1 {
  contract: string;
  requestedTraceCount: number;
  successfulTraceCount: number;
  failedTraceCount: number;
  policy: {
    status: string;
  };
  explainableScorecard: {
    regressionVsBaseline: {
      count: number;
      totalCount: number;
    };
    traceOutcomeVsBaseline: {
      betterCount: number;
      tiedCount: number;
      worseCount: number;
      totalCount: number;
    };
  };
  modes: BroadLiveScorecardModeRowV1[];
  traces: BroadLiveTraceScorecardRowV1[];
}

interface BroadLiveReportV1 {
  contract: string;
  status: string;
  gateStatus: string;
  gateDecisive: boolean;
}

interface BroadLiveSummaryTablesModeRowV1 {
  mode: string;
  usedLearnedRouteFn?: boolean | null;
  routerIdentity?: string | null;
  activationTaken?: boolean | null;
}

interface BroadLiveSummaryTablesTurnV1 {
  traceId: string;
  modes: BroadLiveSummaryTablesModeRowV1[];
}

interface BroadLiveSummaryTablesV1 {
  contract: string;
  turns: BroadLiveSummaryTablesTurnV1[];
}

export interface BroadLiveProofReadSummaryV1 {
  available: boolean;
  authoritative: boolean;
  vetoResult: BroadLiveVetoResultV1;
  regressions: number;
  comparableTraceCount: number;
  policyStatus: string | null;
  gateStatus: string | null;
  usedLearnedRouteTurnCount: number;
  boundLearnedRouteTurnCount: number;
  activatedLearnedRouteTurnCount: number;
  feltOverlap: FeltOverlapSummaryV1;
  notes: string[];
}

export interface CandidateStatusInputV1 {
  feltPassed: boolean;
  restraintPassed: boolean;
  broadLiveAuthoritative: boolean;
  broadLiveVetoResult: BroadLiveVetoResultV1;
}

interface RunnerArtifactsV1 {
  routeRowsPath: string;
  policyRowsPath: string;
  registryEntriesPath: string;
  materializationPath: string;
  candidateArtifactDir: string;
  feltOptimizeEvalDir: string;
  broadLiveEvalDir: string;
  scorecardPath: string;
  runPath: string;
}

interface RunnerScorecardV1 {
  candidateArtifactDir: string;
  feltUniqueWinsTiesRegressions: FeltOverlapSummaryV1;
  feltComparativeEval: FeltOptimizeComparativeSummaryV1;
  feltBroadLiveOverlap: FeltOverlapSummaryV1;
  feltReplayGateExpectationPassCount: {
    passed: number;
    total: number;
  };
  unnecessaryActivations: {
    count: number;
    total: number;
  };
  mustNotFireFailures: {
    count: number;
    total: number;
  };
  broadLive: BroadLiveProofReadSummaryV1;
  finalCandidateStatus: ActivationFirstCandidateStatusV1;
}

interface CandidateArtifactRunSummaryV1 {
  artifactId: string;
  artifactVersion: string;
  artifactChecksum: string;
  outputDir: string;
  routerIdentity: string | null;
}

function usage(): void {
  process.stderr.write(
    [
      "Usage: node --experimental-transform-types scripts/run-activation-first-gating-retune.ts [options]",
      "",
      "Options:",
      `  --task-id <id>                 Defaults to ${DEFAULT_TASK_ID}`,
      `  --harness-dir <path>          Defaults to ${DEFAULT_HARNESS_DIR}`,
      `  --output-dir <path>           Defaults to ${DEFAULT_OUTPUT_DIR}`,
      `  --warm-start-artifact-dir <path> Defaults to ${DEFAULT_WARM_START_ARTIFACT_DIR}`,
      "  --include-must-fire           Include must-fire-30 as secondary training input (default)",
      "  --no-must-fire                Exclude must-fire-30 from training input",
      "  --generated-at <iso>          Override generated timestamp",
      "  --help                        Show this help",
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

function toErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function parseArgs(argv: string[]): ParsedArgs {
  const parsed: ParsedArgs = {
    taskId: DEFAULT_TASK_ID,
    harnessDir: DEFAULT_HARNESS_DIR,
    outputDir: DEFAULT_OUTPUT_DIR,
    warmStartArtifactDir: DEFAULT_WARM_START_ARTIFACT_DIR,
    includeMustFire: true,
    generatedAt: new Date().toISOString(),
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    switch (arg) {
      case "--task-id":
        parsed.taskId = normalizeCliString(argv[index + 1]) ?? parsed.taskId;
        index += 1;
        break;
      case "--harness-dir":
        parsed.harnessDir = path.resolve(argv[index + 1] ?? "");
        index += 1;
        break;
      case "--output-dir":
        parsed.outputDir = path.resolve(argv[index + 1] ?? "");
        index += 1;
        break;
      case "--warm-start-artifact-dir":
        parsed.warmStartArtifactDir = path.resolve(argv[index + 1] ?? "");
        index += 1;
        break;
      case "--include-must-fire":
        parsed.includeMustFire = true;
        break;
      case "--no-must-fire":
        parsed.includeMustFire = false;
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

  return parsed;
}

function portableRelativePath(fromPath: string, toPath: string): string {
  return path.relative(fromPath, toPath).split(path.sep).join("/");
}

function readJson<T>(filePath: string): T {
  return JSON.parse(readFileSync(filePath, "utf8")) as T;
}

function writeJson(filePath: string, value: unknown): void {
  mkdirSync(path.dirname(filePath), { recursive: true });
  writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

function sha256Text(text: string): string {
  return createHash("sha256").update(text, "utf8").digest("hex");
}

function sha256File(filePath: string): string {
  return sha256Text(readFileSync(filePath, "utf8"));
}

function resolveHarnessPath(harnessDir: string, relativeOrAbsolutePath: string): string {
  return path.isAbsolute(relativeOrAbsolutePath)
    ? relativeOrAbsolutePath
    : path.resolve(harnessDir, relativeOrAbsolutePath);
}

function stringifyUnknown(value: unknown): string[] {
  if (typeof value === "string") {
    const trimmed = value.trim();
    return trimmed.length > 0 ? [trimmed] : [];
  }
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .flatMap((entry) => stringifyUnknown(entry))
    .filter((entry, index, entries) => entries.indexOf(entry) === index);
}

function uniqueStrings(values: Array<string | null | undefined>): string[] {
  const normalized = values
    .map((value) => typeof value === "string" ? value.trim() : "")
    .filter((value) => value.length > 0);
  return [...new Set(normalized)];
}

function normalizeTraceExcerpt(value: string, maxLength = 160): string {
  const compact = value.replace(/\s+/g, " ").trim();
  return compact.length <= maxLength ? compact : `${compact.slice(0, maxLength - 1)}…`;
}

export function derivePseudoRouteBucketPlanV1(
  traceLabel: GraphifyRouteObjectiveTraceLabelLikeV1,
): PseudoRouteBucketPlanV1 {
  const oracleBestMode = traceLabel.oracleBestMode ?? null;
  if (oracleBestMode === "learned_route") {
    return {
      candidateCount: 5,
      evidenceSpanCount: 3,
      hardNegativeCount: 1,
      outcomeGain: 1,
      stopLabel: "CONTINUE",
      chosenActionKind: "traverse",
    };
  }
  return {
    candidateCount: 1,
    evidenceSpanCount: 1,
    hardNegativeCount: 0,
    outcomeGain: oracleBestMode === "tie" ? 0.1 : 0,
    stopLabel: "STOP_LOCAL",
    chosenActionKind: "stop_local",
  };
}

function hashId(prefix: string, ...parts: Array<string | number | null | undefined>): string {
  return `${prefix}-${sha256Text(parts.map((part) => String(part ?? "")).join("\u001f")).slice(0, 16)}`;
}

const REPLAY_TRAINING_SURFACE_MODE_PRIORITY = [
  "graph_prior_only",
  "vector_only",
  "learned_route",
  "no_brain",
] as const;

export function collectBundleReplaySurfaces(bundleRoot: string): ReplayBundleSurfaceV1[] {
  if (!existsSync(bundleRoot)) {
    return [];
  }
  const surfaces: ReplayBundleSurfaceV1[] = [];
  for (const entry of readdirSync(bundleRoot, { withFileTypes: true })) {
    if (!entry.isDirectory() || entry.name.startsWith("_")) {
      continue;
    }
    const tracePath = path.join(bundleRoot, entry.name, "trace.json");
    const learnedRoutePath = path.join(bundleRoot, entry.name, "modes", "learned_route.json");
    if (!existsSync(tracePath) || !existsSync(learnedRoutePath)) {
      continue;
    }
    const trace = readJson<RecordedSessionTraceV1>(tracePath);
    const learnedRoute = readJson<LearnedRouteBundleV1>(learnedRoutePath);
    const modeDir = path.join(bundleRoot, entry.name, "modes");
    const prioritizedModeNames = [
      ...REPLAY_TRAINING_SURFACE_MODE_PRIORITY,
      ...readdirSync(modeDir, { withFileTypes: true })
        .filter((modeEntry) => modeEntry.isFile() && modeEntry.name.endsWith(".json"))
        .map((modeEntry) => modeEntry.name.replace(/\.json$/, ""))
        .filter((modeName) => !REPLAY_TRAINING_SURFACE_MODE_PRIORITY.includes(modeName as typeof REPLAY_TRAINING_SURFACE_MODE_PRIORITY[number])),
    ];
    const modeBundles = prioritizedModeNames
      .map((modeName) => ({ modeName, modePath: path.join(modeDir, `${modeName}.json`) }))
      .filter(({ modePath }) => existsSync(modePath))
      .map(({ modeName, modePath }) => ({ modeName, bundle: readJson<LearnedRouteBundleV1>(modePath) }));
    const modeSelectedContextIds = Object.fromEntries(
      modeBundles.map(({ modeName, bundle }) => [
        modeName,
        uniqueStrings(bundle.turns.flatMap((turn) => stringifyUnknown(turn.selectedContextIds))),
      ]),
    );
    const modeSelectedContextTexts = Object.fromEntries(
      modeBundles.map(({ modeName, bundle }) => [
        modeName,
        uniqueStrings(bundle.turns.flatMap((turn) => stringifyUnknown(turn.selectedContextTexts))),
      ]),
    );
    const selectedContextIds = uniqueStrings(
      Object.values(modeSelectedContextIds).flat(),
    );
    const selectedContextTexts = uniqueStrings(
      Object.values(modeSelectedContextTexts).flat(),
    );
    const expectedContextPhrases = uniqueStrings(
      modeBundles.flatMap(({ bundle }) => bundle.turns.flatMap((turn) => stringifyUnknown(turn.expectedContextPhrases))),
    );
    const usedLearnedRouteTurnCount = learnedRoute.turns.filter((turn) => turn.usedLearnedRouteFn === true).length;
    const activatedTurnCount = learnedRoute.turns.filter((turn) => turn.activationTaken === true).length;
    const boundTurnCount = learnedRoute.turns.filter((turn) => (
      typeof turn.routerIdentity === "string" && turn.routerIdentity.trim().length > 0
    )).length;
    surfaces.push({
      traceId: trace.traceId,
      bundleDir: path.join(bundleRoot, entry.name),
      selectedContextIds,
      selectedContextTexts,
      expectedContextPhrases,
      modeSelectedContextIds,
      modeSelectedContextTexts,
      usedLearnedRouteTurnCount,
      activatedTurnCount,
      boundTurnCount,
    });
  }
  return surfaces;
}

function buildReplaySurfaceIndex(harnessDir: string): Map<string, ReplayBundleSurfaceV1[]> {
  const bundleRoots = readdirSync(harnessDir, { withFileTypes: true })
    .filter((entry) => entry.isDirectory() && entry.name.endsWith("-replay-lane"))
    .map((entry) => path.join(harnessDir, entry.name));
  const broadLiveTraceRoot = path.join(harnessDir, "semantic-rich-broad-live-eval", "traces");
  if (existsSync(broadLiveTraceRoot)) {
    bundleRoots.push(broadLiveTraceRoot);
  }
  const surfacesByTraceId = new Map<string, ReplayBundleSurfaceV1[]>();
  for (const bundleRoot of bundleRoots) {
    for (const surface of collectBundleReplaySurfaces(bundleRoot)) {
      const existing = surfacesByTraceId.get(surface.traceId) ?? [];
      existing.push(surface);
      surfacesByTraceId.set(surface.traceId, existing);
    }
  }
  return surfacesByTraceId;
}

function packNativeWinnerCandidateIds(bundleSurfaces: ReplayBundleSurfaceV1[]): string[] {
  const graphPriorWinnerIds = uniqueStrings(
    bundleSurfaces.flatMap((surface) => surface.modeSelectedContextIds.graph_prior_only ?? []),
  );
  if (graphPriorWinnerIds.length > 0) {
    return graphPriorWinnerIds;
  }
  return uniqueStrings(
    bundleSurfaces.flatMap((surface) => surface.modeSelectedContextIds.vector_only ?? []),
  );
}

function buildCandidateIds(params: {
  trace: RecordedSessionTraceV1;
  traceLabel: GraphifyRouteObjectiveTraceLabelLikeV1;
  bundleSurfaces: ReplayBundleSurfaceV1[];
  targetCount: number;
}): string[] {
  const selectedContextIds = uniqueStrings(
    params.bundleSurfaces.flatMap((surface) => surface.selectedContextIds),
  );
  const winnerCandidateIds = packNativeWinnerCandidateIds(params.bundleSurfaces);
  const learnedRouteCandidateIds = uniqueStrings(
    params.bundleSurfaces.flatMap((surface) => surface.modeSelectedContextIds.learned_route ?? []),
  );
  const expectedContextPhrases = uniqueStrings([
    ...stringifyUnknown(params.trace.turns[0]?.expectedContextPhrases),
    ...params.bundleSurfaces.flatMap((surface) => surface.expectedContextPhrases),
  ]);
  const seedCueTexts = uniqueStrings(
    (params.trace.seedCues ?? []).flatMap((cue) => [
      typeof cue.content === "string" ? cue.content : null,
      typeof cue.text === "string" ? cue.text : null,
      typeof cue.summary === "string" ? cue.summary : null,
    ]),
  );

  const candidateIds: string[] = [];
  const pushCandidateId = (candidateId: string): void => {
    const normalized = candidateId.trim();
    if (normalized.length === 0 || candidateIds.includes(normalized) || candidateIds.length >= params.targetCount) {
      return;
    }
    candidateIds.push(normalized);
  };

  for (const candidateId of winnerCandidateIds) {
    pushCandidateId(candidateId);
  }
  for (const candidateId of learnedRouteCandidateIds) {
    pushCandidateId(candidateId);
  }
  for (const candidateId of selectedContextIds) {
    pushCandidateId(candidateId);
  }
  for (const phrase of expectedContextPhrases) {
    pushCandidateId(hashId(`phrase-context:${params.trace.traceId}`, phrase));
  }
  for (const cueText of seedCueTexts) {
    pushCandidateId(hashId(`cue-context:${params.trace.traceId}`, cueText));
  }
  for (let index = candidateIds.length; index < params.targetCount; index += 1) {
    pushCandidateId(`synthetic-context:${params.trace.traceId}:${index}`);
  }

  return candidateIds.slice(0, params.targetCount);
}

function candidateOrigin(candidateId: string): "feedback" | "interaction" | "event" | "init" | "phrase" | "cue" | "synthetic" | "other" {
  if (candidateId.includes(":feedback")) {
    return "feedback";
  }
  if (candidateId.includes(":interaction")) {
    return "interaction";
  }
  if (candidateId.includes(":event:")) {
    return "event";
  }
  if (
    candidateId.includes(":pointer-aware-init")
    || candidateId.includes(":pointer-passive-expansion")
    || candidateId.includes(":workspace-init:")
  ) {
    return "init";
  }
  if (candidateId.startsWith("phrase-context:")) {
    return "phrase";
  }
  if (candidateId.startsWith("cue-context:")) {
    return "cue";
  }
  if (candidateId.startsWith("synthetic-context:")) {
    return "synthetic";
  }
  return "other";
}

function candidateSemanticClass(candidateId: string): string {
  const origin = candidateOrigin(candidateId);
  switch (origin) {
    case "feedback":
      return "feedback_context";
    case "interaction":
      return "interaction_context";
    case "event":
      return "event_context";
    case "init":
      return "init_context";
    case "phrase":
      return "phrase_context";
    case "cue":
      return "cue_context";
    case "synthetic":
      return "synthetic_context";
    default:
      return "other_context";
  }
}

export function buildSyntheticRouteCandidatesV1(
  candidateIds: string[],
  profile: SyntheticCandidateMetadataProfileV1 = "synthetic_eval",
): RouteCandidateV1[] {
  return candidateIds.map((candidateId, index) => {
    const origin = candidateOrigin(candidateId);
    const scoreHint = profile === "runtime_like_replay"
      ? origin === "feedback"
        ? 0.95
        : origin === "interaction"
          ? 0.55
          : origin === "event"
        ? 0.9
        : origin === "init"
          ? 0.2
          : origin === "phrase"
            ? 0.65
            : origin === "cue"
              ? 0.45
              : origin === "synthetic"
                ? 0.1
                : Math.max(0.15, 0.45 - (index * 0.05))
      : origin === "feedback"
        ? 0.98
        : origin === "interaction"
          ? 0.7
          : origin === "event"
        ? 0.95
        : origin === "init"
          ? 0.35
          : origin === "phrase"
            ? 0.75
            : origin === "cue"
              ? 0.55
              : origin === "synthetic"
                ? 0.05
                : Math.max(0.1, 0.5 - (index * 0.05));
    const authority = profile === "runtime_like_replay"
      ? "recorded_session_replay"
      : origin === "feedback" || origin === "event"
        ? "snapshot_supporting_fact"
        : "snapshot_context";
    const freshness = profile === "runtime_like_replay" ? "replay_eval" : "eval_only";
    const tokenCost = profile === "runtime_like_replay"
      ? origin === "feedback"
        ? 80
        : origin === "interaction"
          ? 56
          : origin === "event"
        ? 72
        : origin === "init"
          ? 96
          : origin === "phrase"
            ? 64
            : origin === "cue"
              ? 48
              : origin === "synthetic"
                ? 32
                : 48
      : origin === "feedback"
        ? 24
        : origin === "interaction"
          ? 20
          : origin === "event"
        ? 24
        : origin === "init"
          ? 48
          : 16;
    return {
      candidate_id: candidateId,
      candidate_type: "graph_node",
      semantic_class: candidateSemanticClass(candidateId),
      authority,
      freshness,
      score_hint: Number(scoreHint.toFixed(2)),
      token_cost: tokenCost,
    };
  });
}

export function chooseTraverseTargetCandidateIdV1(candidateIds: string[]): string | null {
  for (const candidateId of candidateIds) {
    const origin = candidateOrigin(candidateId);
    if (origin === "feedback" || origin === "event") {
      return candidateId;
    }
  }
  return candidateIds[0] ?? null;
}

function chooseTraverseTargetCandidateIdsV1(bundleSurfaces: ReplayBundleSurfaceV1[], candidateIds: string[]): string[] {
  const winnerCandidateIds = packNativeWinnerCandidateIds(bundleSurfaces).filter((candidateId) => candidateIds.includes(candidateId));
  const highAuthorityWinnerIds = winnerCandidateIds.filter((candidateId) => {
    const origin = candidateOrigin(candidateId);
    return origin === "feedback" || origin === "event";
  });
  const highestAuthorityWinner = chooseTraverseTargetCandidateIdV1(highAuthorityWinnerIds);
  if (highestAuthorityWinner !== null) {
    return [highestAuthorityWinner];
  }
  const fallback = chooseTraverseTargetCandidateIdV1(candidateIds);
  return fallback === null ? [] : [fallback];
}

export function deriveFeltRouteSourceNodeIdV1(params: {
  laneName: string;
  traceId: string;
  traverseTargetCandidateIds: string[];
}): string {
  const preferredTarget = params.traverseTargetCandidateIds[0] ?? null;
  if (preferredTarget) {
    return `${params.laneName}:source:${preferredTarget}`;
  }
  return `${params.laneName}:source:trace:${params.traceId}`;
}

function isSafeFeedbackCohortRowIdV1(rowId: string): boolean {
  return Object.prototype.hasOwnProperty.call(FELT_SAFE_POSITIVE_PRESSURE_ROW_WEIGHT_OVERRIDES_V1, rowId);
}

function shouldUseFeedbackScopedFeltSourceNodeV1(rowId: string, traverseTargetCandidateIds: string[]): boolean {
  return traverseTargetCandidateIds.length > 0 && isSafeFeedbackCohortRowIdV1(rowId);
}

export function appendSafeFeedbackRankingRouteRowsV1(
  routeRow: ColdStartRouteDecisionRowV1,
  laneKey: ActivationFirstLaneKeyV1,
): ColdStartRouteDecisionRowV1[] {
  if (laneKey !== "feltResume" || !isSafeFeedbackCohortRowIdV1(routeRow.row_id) || SAFE_FEEDBACK_ROUTE_DUPLICATE_COUNT_V1 <= 0) {
    return [routeRow];
  }
  const rows = [routeRow];
  for (let index = 0; index < SAFE_FEEDBACK_ROUTE_DUPLICATE_COUNT_V1; index += 1) {
    rows.push({
      ...routeRow,
      row_id: `${routeRow.row_id}:feedback_rank_bonus:${index + 1}`,
      outcome_gain: routeRow.outcome_gain + SAFE_FEEDBACK_ROUTE_OUTCOME_GAIN_BONUS_V1,
      cursor_path: [...routeRow.cursor_path],
      candidate_set: routeRow.candidate_set.map((candidate) => ({ ...candidate })),
      evidence_spans: routeRow.evidence_spans.map((span) => ({ ...span })),
      hard_negatives: [...routeRow.hard_negatives],
      teacher_action: routeRow.teacher_action.kind === "traverse"
        ? { kind: "traverse", target_ids: [...routeRow.teacher_action.target_ids] }
        : { ...routeRow.teacher_action },
      provenance: { ...routeRow.provenance },
    });
  }
  return rows;
}

function buildEvidenceSpans(params: {
  trace: RecordedSessionTraceV1;
  bundleSurfaces: ReplayBundleSurfaceV1[];
  targetCount: number;
}): ColdStartRouteDecisionRowV1["evidence_spans"] {
  const excerpts = uniqueStrings([
    ...stringifyUnknown(params.trace.turns[0]?.expectedContextPhrases),
    ...params.bundleSurfaces.flatMap((surface) => surface.expectedContextPhrases),
    ...(params.trace.seedCues ?? []).flatMap((cue) => [
      typeof cue.content === "string" ? cue.content : null,
      typeof cue.text === "string" ? cue.text : null,
      typeof cue.summary === "string" ? cue.summary : null,
    ]),
    ...params.bundleSurfaces.flatMap((surface) => surface.selectedContextTexts.map((text) => normalizeTraceExcerpt(text))),
    typeof params.trace.turns[0]?.userMessage === "string"
      ? normalizeTraceExcerpt(params.trace.turns[0].userMessage)
      : params.trace.traceId,
  ]).slice(0, params.targetCount);

  return excerpts.map((excerpt, index) => ({
    source_ref: `trace:${params.trace.traceId}:evidence:${index}`,
    start: 0,
    end: Math.max(1, excerpt.length),
    excerpt,
  }));
}

function buildHardNegatives(
  traceId: string,
  traceLabel: GraphifyRouteObjectiveTraceLabelLikeV1,
  bundleSurfaces: ReplayBundleSurfaceV1[],
  count: number,
  targetCandidateId: string | null,
): string[] {
  if (count <= 0) {
    return [];
  }

  const learnedRouteNegativeIds = uniqueStrings(
    bundleSurfaces.flatMap((surface) => surface.modeSelectedContextIds.learned_route ?? []),
  )
    .filter((candidateId) => candidateId !== targetCandidateId)
    .sort((left, right) => {
      const leftInit = candidateOrigin(left) === "init" ? 1 : 0;
      const rightInit = candidateOrigin(right) === "init" ? 1 : 0;
      if (rightInit !== leftInit) {
        return rightInit - leftInit;
      }
      return left.localeCompare(right);
    });

  if (learnedRouteNegativeIds.length > 0) {
    return learnedRouteNegativeIds.slice(0, count);
  }

  const suffix = traceLabel.hardNegativeClass ? `:${traceLabel.hardNegativeClass}` : "";
  return Array.from({ length: count }, (_, index) => `gating-negative:${traceId}:${index}${suffix}`);
}

function firstUserMessage(trace: RecordedSessionTraceV1): string {
  const userMessage = typeof trace.turns[0]?.userMessage === "string"
    ? trace.turns[0]?.userMessage?.trim()
    : "";
  return userMessage.length > 0 ? userMessage : `Activation-first gating trace ${trace.traceId}`;
}

export function buildColdStartRouteRow(params: {
  taskId: string;
  laneKey: ActivationFirstLaneKeyV1;
  laneName: string;
  datasetId: string;
  trace: RecordedSessionTraceV1;
  traceLabel: GraphifyRouteObjectiveTraceLabelLikeV1;
  bundleSurfaces: ReplayBundleSurfaceV1[];
  generatedAt: string;
}): ColdStartRouteDecisionRowV1 {
  const bucketPlan = derivePseudoRouteBucketPlanV1(params.traceLabel);
  const candidateIds = buildCandidateIds({
    trace: params.trace,
    traceLabel: params.traceLabel,
    bundleSurfaces: params.bundleSurfaces,
    targetCount: bucketPlan.candidateCount,
  });
  const candidates = buildSyntheticRouteCandidatesV1(
    candidateIds,
    params.laneKey === "feltResume" ? "runtime_like_replay" : "synthetic_eval",
  );
  const evidenceSpans = buildEvidenceSpans({
    trace: params.trace,
    bundleSurfaces: params.bundleSurfaces,
    targetCount: bucketPlan.evidenceSpanCount,
  });
  const traverseTargetCandidateIds = params.laneKey === "feltResume" && bucketPlan.chosenActionKind === "traverse"
    ? chooseTraverseTargetCandidateIdsV1(params.bundleSurfaces, candidateIds)
    : [];
  const hardNegatives = buildHardNegatives(
    params.trace.traceId,
    params.traceLabel,
    params.bundleSurfaces,
    bucketPlan.hardNegativeCount,
    traverseTargetCandidateIds[0] ?? null,
  );

  const rowId = `activation-first:${params.laneName}:${params.trace.traceId}`;
  const cursorPath = params.laneKey === "feltResume"
    && shouldUseFeedbackScopedFeltSourceNodeV1(rowId, traverseTargetCandidateIds)
    ? [
        params.laneName,
        deriveFeltRouteSourceNodeIdV1({
          laneName: params.laneName,
          traceId: params.trace.traceId,
          traverseTargetCandidateIds,
        }),
      ]
    : [params.laneName];

  return {
    row_id: rowId,
    dataset_id: params.datasetId,
    query: firstUserMessage(params.trace),
    cursor_path: cursorPath,
    candidate_set: candidates,
    teacher_action: traverseTargetCandidateIds.length === 0
      ? {
          kind: "tool",
          tool_name: `__gating_only__:${params.laneName}`,
        }
      : {
          kind: "traverse",
          target_ids: traverseTargetCandidateIds,
        },
    stop_label: bucketPlan.stopLabel,
    evidence_spans: evidenceSpans,
    hard_negatives: hardNegatives,
    outcome_gain: bucketPlan.outcomeGain,
    provenance: {
      dataset: params.datasetId,
      source_license: "internal_local_only",
      source_family: "agent_traces",
      source_snapshot_ref: `task-artifacts/${params.taskId}/activation-first-retune-harness`,
      recorded_by: "run-activation-first-gating-retune",
      recorded_at: params.generatedAt,
      review_status: "approved_train",
    },
    split_tag: "train",
    created_at: params.generatedAt,
  };
}

function buildPolicyProjectionRouteRow(params: {
  coldStartRouteRow: ColdStartRouteDecisionRowV1;
  traceId: string;
  generatedAt: string;
  traceLabel: GraphifyRouteObjectiveTraceLabelLikeV1;
}): PolicyProjectionRouteRowV1 {
  const bucketPlan = derivePseudoRouteBucketPlanV1(params.traceLabel);
  const primaryCandidateId = params.coldStartRouteRow.candidate_set[0]?.candidate_id ?? `candidate:${params.traceId}:0`;
  const continueAction = {
    action_id: `${params.coldStartRouteRow.row_id}:continue`,
    action_kind: "traverse" as const,
    node_id: primaryCandidateId,
    tool_name: null,
    tool_capability_id: null,
    tool_instance_id: null,
    tool_args_shape: null,
    prior_score: 0,
    probability: bucketPlan.chosenActionKind === "traverse" ? 0.9 : 0.1,
    retrieval_features: null,
  };
  const stopAction = {
    action_id: `${params.coldStartRouteRow.row_id}:stop_local`,
    action_kind: "stop_local" as const,
    node_id: null,
    tool_name: null,
    tool_capability_id: null,
    tool_instance_id: null,
    tool_args_shape: null,
    prior_score: 0,
    probability: bucketPlan.chosenActionKind === "stop_local" ? 0.9 : 0.1,
    retrieval_features: null,
  };
  const chosenAction = bucketPlan.chosenActionKind === "traverse" ? continueAction : stopAction;

  return {
    schema_version: 1,
    row_id: params.coldStartRouteRow.row_id,
    trace_id: params.traceId,
    episode_id: null,
    conversation_id: null,
    route_fn_version: `activation-first-gating-retune.${params.traceLabel.focusLane ?? "lane"}.v1`,
    decision_point_id: `decision:${params.coldStartRouteRow.row_id}`,
    decision_point_kind: "seed",
    source_node_id: null,
    expansion_index: 0,
    selection_index: 0,
    query_text: params.coldStartRouteRow.query,
    cursor_path: [...params.coldStartRouteRow.cursor_path],
    local_action_set: [continueAction, stopAction],
    chosen_action_id: chosenAction.action_id,
    chosen_action_kind: chosenAction.action_kind,
    chosen_node_id: chosenAction.node_id,
    chosen_tool_name: chosenAction.tool_name,
    chosen_tool_capability_id: chosenAction.tool_capability_id,
    chosen_tool_instance_id: chosenAction.tool_instance_id,
    chosen_action_probability: chosenAction.probability,
    stop_probability: bucketPlan.chosenActionKind === "stop_local" ? 0.9 : 0.1,
    stop_label: bucketPlan.stopLabel,
    budget_context: {
      budget_remaining: 1,
      initial_budget: 1,
      reserved_token_cost: 0,
      budget_used: 0,
      budget_used_fraction: 0,
      max_hops: 1,
      max_frontier_size: null,
      frontier_size: params.coldStartRouteRow.candidate_set.length,
      visited_count: 0,
      fired_count: 0,
      pending_selection_count: 0,
      pressure_level: null,
      frontier_pressure: null,
      budget_pressure: null,
      budget_fraction: null,
      query_budget_chars: null,
      max_context_chars: null,
      injected_chars: null,
      dropped_chars: null,
      context_clipped: null,
      route_selection_ms: null,
      total_query_ms: null,
      compile_deadline_ms: null,
      compile_deadline_hit: null,
    },
    route_context: {
      request_digest: null,
      active_pack_id: null,
      router_identity: null,
      candidate_node_ids: params.coldStartRouteRow.candidate_set.map((candidate) => candidate.candidate_id),
      selected_node_ids: bucketPlan.chosenActionKind === "traverse" ? [primaryCandidateId] : [],
      selected_traversal_node_ids: bucketPlan.chosenActionKind === "traverse" ? [primaryCandidateId] : [],
      selected_path_node_ids: bucketPlan.chosenActionKind === "traverse" ? [primaryCandidateId] : [],
      selected_seed_node_ids: [],
    },
    label_provenance: {
      provenance_id: `provenance:${params.coldStartRouteRow.row_id}`,
      state: "matched",
      basis: "manual",
      confidence: 0.9,
      detail: null,
      source: "teacher",
      kind: "teacher_review",
      binding_mode: "trace_id",
      attribution_quality: "fallback",
      feedback_richness: "sparse",
      trace_id: params.traceId,
      episode_id: null,
      decision_point_id: `decision:${params.coldStartRouteRow.row_id}`,
      observation_id: null,
      supervision_id: null,
      update_id: null,
      candidate_ids: params.coldStartRouteRow.candidate_set.map((candidate) => candidate.candidate_id),
      provenance_ref: params.coldStartRouteRow.dataset_id,
      content_hash: sha256Text(params.coldStartRouteRow.query),
      lineage_hash: sha256Text(params.coldStartRouteRow.row_id),
      created_at: Date.parse(params.generatedAt),
    },
    evidence_spans: params.coldStartRouteRow.evidence_spans.map((span) => ({
      source_ref: span.source_ref,
      start: span.start,
      end: span.end,
      excerpt: span.excerpt,
    })),
    hard_negatives: [...params.coldStartRouteRow.hard_negatives],
    created_at: params.generatedAt,
  };
}

function applyHardNegativeWeightFloor(
  policyRow: PolicySupervisionRowV1,
  weightFloorsByClass: Record<string, number> | undefined,
): PolicySupervisionRowV1 {
  const hardNegativeClass = policyRow.hard_negative_class ?? null;
  if (!hardNegativeClass) {
    return policyRow;
  }
  const configuredFloor = weightFloorsByClass?.[hardNegativeClass];
  if (!Number.isFinite(configuredFloor) || configuredFloor === undefined || configuredFloor <= policyRow.row_weight) {
    return policyRow;
  }
  return {
    ...policyRow,
    row_weight: Number(configuredFloor.toFixed(6)),
    notes: [
      ...policyRow.notes,
      `Row weight floored to ${configuredFloor} by the activation-first harness hard-negative spec.`,
    ],
  };
}

export function applyPositivePressureWeightOverrideV1(
  policyRow: PolicySupervisionRowV1,
  routeRowId: string,
  laneKey: ActivationFirstLaneKeyV1,
): PolicySupervisionRowV1 {
  if (laneKey !== "feltResume" || policyRow.row_type !== "activate") {
    return policyRow;
  }
  const configuredWeight = FELT_SAFE_POSITIVE_PRESSURE_ROW_WEIGHT_OVERRIDES_V1[routeRowId];
  if (!Number.isFinite(configuredWeight) || configuredWeight === undefined || configuredWeight <= policyRow.row_weight) {
    return policyRow;
  }
  return {
    ...policyRow,
    row_weight: Number(configuredWeight.toFixed(6)),
    notes: [
      ...policyRow.notes,
      `Row weight raised to ${configuredWeight} by the activation-first safe positive-pressure cohort.`,
    ],
  };
}

export function appendPositivePressureSupplementalRowsV1(
  policyRow: PolicySupervisionRowV1,
  routeRowId: string,
  laneKey: ActivationFirstLaneKeyV1,
): PolicySupervisionRowV1[] {
  if (laneKey !== "feltResume" || policyRow.row_type !== "activate") {
    return [policyRow];
  }
  if (!(routeRowId in FELT_SAFE_POSITIVE_PRESSURE_ROW_WEIGHT_OVERRIDES_V1)) {
    return [policyRow];
  }
  const supplementalWeight = FELT_SAFE_POSITIVE_PRESSURE_SUPPLEMENTAL_ROW_WEIGHT_V1;
  if (!Number.isFinite(supplementalWeight) || supplementalWeight <= 0) {
    return [policyRow];
  }
  return [
    policyRow,
    {
      ...policyRow,
      row_id: `${policyRow.row_id}:felt_feedback_bonus`,
      row_weight: Number(supplementalWeight.toFixed(6)),
      notes: [
        ...policyRow.notes,
        `Supplemental activation example added at weight ${supplementalWeight} for the safe feedback-target positive-pressure cohort.`,
      ],
    },
  ];
}

export function coerceRestraintLanePolicyRowV1(
  policyRow: PolicySupervisionRowV1,
  laneKey: ActivationFirstLaneKeyV1,
): PolicySupervisionRowV1 {
  if (laneKey !== "mustNotFire" || policyRow.row_type === "stop_local") {
    return policyRow;
  }
  return {
    ...policyRow,
    row_type: "stop_local",
    notes: [
      ...policyRow.notes,
      "Must-not-fire restraint lane coerced to stop_local so replay gating evaluates explicit deactivation instead of generic abstain.",
    ],
  };
}

function loadLaneTraces(
  harnessDir: string,
  lane: ActivationFirstRouteObjectiveLaneV1,
): LoadedFrozenTraceV1[] {
  const manifestPath = resolveHarnessPath(harnessDir, lane.manifestPath);
  const manifest = readJson<FrozenRecordedSessionEvalManifestV1>(manifestPath);
  const manifestDir = path.dirname(manifestPath);
  return manifest.traces.map((entry) => {
    const tracePath = path.resolve(manifestDir, entry.tracePath);
    const trace = readJson<RecordedSessionTraceV1>(tracePath);
    const traceLabel = lane.labelsByTraceId[trace.traceId];
    if (!traceLabel) {
      throw new Error(`Lane ${lane.trancheId} is missing a route-objective label for trace ${trace.traceId}`);
    }
    return {
      traceId: trace.traceId,
      tracePath,
      trace,
      traceLabel,
    };
  });
}

function buildRegistryEntry(params: {
  datasetId: string;
  laneName: string;
  generatedAt: string;
  exactFiles: string[];
}): DataRegistryEntryV1 {
  const exactFiles = params.exactFiles.map((filePath) => path.resolve(filePath));
  const fileHashes = Object.fromEntries(exactFiles.map((filePath) => [filePath, `sha256:${sha256File(filePath)}`]));
  return {
    dataset_id: params.datasetId,
    source_family: "agent_traces",
    upstream_url: `file:${portableRelativePath(workspaceRoot, exactFiles[0] ?? repoRoot)}`,
    original_creator: "openclawbrain-internal",
    license: "internal_local_only",
    commercial_use_status: "allowed",
    redistribution_status: "allowed",
    pii_risk: "none",
    benchmark_split_status: "train",
    approval_status: "approved_train",
    reviewer: "activation-first-gating-retune",
    immutable_snapshot_ref: `sha256:${sha256Text(JSON.stringify(fileHashes))}`,
    exact_files: exactFiles,
    file_hashes: fileHashes,
    allowed_uses: ["training", "replay_gate_eval"],
    disallowed_uses: ["public_export"],
    notes: [
      `Activation-first gating-only synthetic lane dataset for ${params.laneName}.`,
      "Lane-scoped pseudo rows intentionally preserve route-objective overlap without flattening conflicting labels across lanes.",
      "Candidate features stay neutral so the continuation only retunes stop/gating surfaces on top of the approved warm-start artifact.",
    ],
    created_at: params.generatedAt,
    updated_at: params.generatedAt,
  };
}

function materializeLaneRows(params: {
  taskId: string;
  harnessDir: string;
  generatedAt: string;
  laneKey: ActivationFirstLaneKeyV1;
  lane: ActivationFirstRouteObjectiveLaneV1;
  replaySurfaceIndex: Map<string, ReplayBundleSurfaceV1[]>;
  weightFloorsByClass: Record<string, number> | undefined;
  harnessFiles: string[];
}): MaterializedLaneRowsV1 {
  const datasetId = `activation-first-gating:${params.taskId}:${params.lane.trancheId}`;
  const loadedTraces = loadLaneTraces(params.harnessDir, params.lane);
  const routeRows: ColdStartRouteDecisionRowV1[] = [];
  const policyRows: PolicySupervisionRowV1[] = [];

  for (const loadedTrace of loadedTraces) {
    const bundleSurfaces = params.replaySurfaceIndex.get(loadedTrace.traceId) ?? [];
    const coldStartRouteRow = buildColdStartRouteRow({
      taskId: params.taskId,
      laneKey: params.laneKey,
      laneName: params.lane.trancheId,
      datasetId,
      trace: loadedTrace.trace,
      traceLabel: loadedTrace.traceLabel,
      bundleSurfaces,
      generatedAt: params.generatedAt,
    });
    const projectionRouteRow = buildPolicyProjectionRouteRow({
      coldStartRouteRow,
      traceId: loadedTrace.traceId,
      generatedAt: params.generatedAt,
      traceLabel: loadedTrace.traceLabel,
    });
    const policyRow = coerceRestraintLanePolicyRowV1(
      applyPositivePressureWeightOverrideV1(
        applyHardNegativeWeightFloor(
          buildPolicySupervisionRowV1({
            routeRow: projectionRouteRow,
            traceLabel: loadedTrace.traceLabel,
          }),
          params.weightFloorsByClass,
        ),
        coldStartRouteRow.row_id,
        params.laneKey,
      ),
      params.laneKey,
    );
    routeRows.push(...appendSafeFeedbackRankingRouteRowsV1(coldStartRouteRow, params.laneKey));
    policyRows.push(...appendPositivePressureSupplementalRowsV1(policyRow, coldStartRouteRow.row_id, params.laneKey));
  }

  return {
    laneKey: params.laneKey,
    laneName: params.lane.trancheId,
    datasetId,
    registryEntry: buildRegistryEntry({
      datasetId,
      laneName: params.lane.trancheId,
      generatedAt: params.generatedAt,
      exactFiles: params.harnessFiles,
    }),
    routeRows,
    policyRows,
  };
}

function ensureCandidateArtifactDir(candidateArtifactDir: string): void {
  rmSync(candidateArtifactDir, { recursive: true, force: true });
  mkdirSync(candidateArtifactDir, { recursive: true });
}

function findLaneSummary(
  verdict: ColdStartRouterReplayGateVerdictV1,
  laneName: string,
): ColdStartRouterReplayGateLaneSummaryV1 {
  return verdict.laneSummaries.find((laneSummary) => laneSummary.lane === laneName) ?? {
    lane: laneName,
    policyExpectationCount: 0,
    passedPolicyExpectationCount: 0,
    failedPolicyExpectationCount: 0,
    totalWeight: 0,
    activationExpectationCount: 0,
    activationMatchCount: 0,
    abstainExpectationCount: 0,
    abstainMatchCount: 0,
    stopLocalExpectationCount: 0,
    stopLocalMatchCount: 0,
  };
}

function summarizeUnnecessaryActivations(verdict: ColdStartRouterReplayGateVerdictV1): { count: number; total: number } {
  const relevantRows = verdict.rowResults.filter((rowResult) => rowResult.gateEvaluated);
  return {
    count: relevantRows.filter((rowResult) => rowResult.actualActivated === true).length,
    total: relevantRows.length,
  };
}

function summarizeMustNotFireFailures(laneSummary: ColdStartRouterReplayGateLaneSummaryV1): { count: number; total: number } {
  return {
    count: laneSummary.failedPolicyExpectationCount,
    total: laneSummary.policyExpectationCount,
  };
}

function buildFeltOverlapSummary(
  feltTraceIds: string[],
  scorecard: BroadLiveScorecardV1 | null,
): FeltOverlapSummaryV1 {
  if (!scorecard) {
    return {
      availableTraceCount: 0,
      totalTraceCount: feltTraceIds.length,
      betterCount: 0,
      tiedCount: 0,
      worseCount: 0,
    };
  }
  const traceRowById = new Map(scorecard.traces.map((traceRow) => [traceRow.traceId, traceRow]));
  let availableTraceCount = 0;
  let betterCount = 0;
  let tiedCount = 0;
  let worseCount = 0;
  for (const traceId of feltTraceIds) {
    const traceRow = traceRowById.get(traceId);
    if (!traceRow || traceRow.candidateRelationVsBaseline === null) {
      continue;
    }
    availableTraceCount += 1;
    if (traceRow.candidateRelationVsBaseline === "better") {
      betterCount += 1;
    } else if (traceRow.candidateRelationVsBaseline === "tied") {
      tiedCount += 1;
    } else {
      worseCount += 1;
    }
  }
  return {
    availableTraceCount,
    totalTraceCount: feltTraceIds.length,
    betterCount,
    tiedCount,
    worseCount,
  };
}

export function classifyBroadLiveProofReadV1(params: {
  scorecard: BroadLiveScorecardV1 | null;
  report: BroadLiveReportV1 | null;
  summaryTables: BroadLiveSummaryTablesV1 | null;
  feltTraceIds: string[];
  freshRunExecuted?: boolean;
}): BroadLiveProofReadSummaryV1 {
  const freshRunExecuted = params.freshRunExecuted === true;
  if (!params.scorecard || !params.report || !params.summaryTables) {
    return {
      available: false,
      authoritative: false,
      vetoResult: "proof_read_only",
      regressions: 0,
      comparableTraceCount: 0,
      policyStatus: null,
      gateStatus: null,
      usedLearnedRouteTurnCount: 0,
      boundLearnedRouteTurnCount: 0,
      activatedLearnedRouteTurnCount: 0,
      feltOverlap: {
        availableTraceCount: 0,
        totalTraceCount: params.feltTraceIds.length,
        betterCount: 0,
        tiedCount: 0,
        worseCount: 0,
      },
      notes: [
        freshRunExecuted
          ? "Candidate-specific broad-live comparative eval did not produce a complete scorecard/report/summary surface; no fresh veto can be made."
          : "Broad-live proof artifacts are unavailable; no candidate-specific veto can be made.",
      ],
    };
  }

  const learnedRouteMode = params.scorecard.modes.find((mode) => mode.mode === "learned_route") ?? null;
  const usedLearnedRouteTurnCount = Number(learnedRouteMode?.totalUsedLearnedRouteTurnCount ?? 0);
  let boundLearnedRouteTurnCount = 0;
  let activatedLearnedRouteTurnCount = 0;
  for (const turn of params.summaryTables.turns) {
    const learnedRouteTurn = turn.modes.find((mode) => mode.mode === "learned_route");
    if (!learnedRouteTurn) {
      continue;
    }
    if (learnedRouteTurn.usedLearnedRouteFn === true || (
      typeof learnedRouteTurn.routerIdentity === "string" && learnedRouteTurn.routerIdentity.trim().length > 0
    )) {
      boundLearnedRouteTurnCount += 1;
    }
    if (learnedRouteTurn.activationTaken === true) {
      activatedLearnedRouteTurnCount += 1;
    }
  }

  const authoritative = usedLearnedRouteTurnCount > 0 && boundLearnedRouteTurnCount > 0;
  const regressions = Number(params.scorecard.explainableScorecard.regressionVsBaseline.count ?? 0);
  const gatePassed = params.scorecard.policy.status === "pass" && params.report.gateStatus === "pass";
  const vetoResult: BroadLiveVetoResultV1 = freshRunExecuted
    ? regressions > 0 || !gatePassed
      ? "reject"
      : "pass"
    : !authoritative
      ? "proof_read_only"
      : regressions > 0 || !gatePassed
        ? "reject"
        : "pass";

  const notes: string[] = [];
  if (freshRunExecuted) {
    notes.push(CANDIDATE_SPECIFIC_BROAD_LIVE_RUN_NOTE);
  }
  if (!authoritative) {
    notes.push("learned_route never bound a live router function in the broad-live proof surface");
    if (freshRunExecuted) {
      notes.push(
        "Fresh broad-live veto outcome is recorded, but final status stays architecture_verdict until the replay surface binds a learned router identity.",
      );
    }
  }
  if (usedLearnedRouteTurnCount === 0) {
    notes.push("scorecard reports zero learned-route turns that used the learned router function");
  }
  if (boundLearnedRouteTurnCount === 0) {
    notes.push("summary tables report zero learned-route turns with a bound router identity");
  }

  return {
    available: true,
    authoritative,
    vetoResult,
    regressions,
    comparableTraceCount: Number(params.scorecard.explainableScorecard.regressionVsBaseline.totalCount ?? 0),
    policyStatus: params.scorecard.policy.status,
    gateStatus: params.report.gateStatus,
    usedLearnedRouteTurnCount,
    boundLearnedRouteTurnCount,
    activatedLearnedRouteTurnCount,
    feltOverlap: buildFeltOverlapSummary(params.feltTraceIds, params.scorecard),
    notes,
  };
}

export function deriveFinalCandidateStatusV1(
  input: CandidateStatusInputV1,
): ActivationFirstCandidateStatusV1 {
  if (!input.feltPassed || !input.restraintPassed) {
    return "reject";
  }
  if (!input.broadLiveAuthoritative || input.broadLiveVetoResult === "proof_read_only") {
    return "architecture_verdict";
  }
  return input.broadLiveVetoResult === "pass" ? "pass" : "reject";
}

export function summarizeFeltOptimizeScorecardV1(params: {
  scorecard: ComparativeEvalScorecardV1;
  outputDir?: string | null;
  notes?: string[];
}): FeltOptimizeComparativeSummaryV1 {
  const explainable = params.scorecard.explainableScorecard;
  const traceOutcome = explainable.traceOutcomeVsBaseline;
  return {
    available: true,
    comparableTraceCount: traceOutcome.totalCount,
    betterCount: traceOutcome.betterCount,
    tiedCount: traceOutcome.tiedCount,
    worseCount: traceOutcome.worseCount,
    regressions: explainable.regressionVsBaseline.count,
    requiredContextRecallSummary: explainable.requiredContextRecall.summary,
    outputDir: params.outputDir ?? null,
    notes: params.notes ?? [],
  };
}

function unavailableFeltOptimizeComparativeSummary(notes: string[]): FeltOptimizeComparativeSummaryV1 {
  return {
    available: false,
    comparableTraceCount: 0,
    betterCount: 0,
    tiedCount: 0,
    worseCount: 0,
    regressions: 0,
    requiredContextRecallSummary: null,
    outputDir: null,
    notes,
  };
}

function readBroadLiveArtifacts(broadLiveDir: string): {
  scorecard: BroadLiveScorecardV1 | null;
  report: BroadLiveReportV1 | null;
  summaryTables: BroadLiveSummaryTablesV1 | null;
} {
  const scorecardPath = path.join(broadLiveDir, "scorecard.json");
  const reportPath = path.join(broadLiveDir, "report.json");
  const summaryTablesPath = path.join(broadLiveDir, "traces", "_lane", "summary-tables.json");

  return {
    scorecard: existsSync(scorecardPath) ? readJson<BroadLiveScorecardV1>(scorecardPath) : null,
    report: existsSync(reportPath) ? readJson<BroadLiveReportV1>(reportPath) : null,
    summaryTables: existsSync(summaryTablesPath) ? readJson<BroadLiveSummaryTablesV1>(summaryTablesPath) : null,
  };
}

function unavailableBroadLiveProofRead(params: {
  feltTraceIds: string[];
  notes: string[];
}): BroadLiveProofReadSummaryV1 {
  return {
    available: false,
    authoritative: false,
    vetoResult: "proof_read_only",
    regressions: 0,
    comparableTraceCount: 0,
    policyStatus: null,
    gateStatus: null,
    usedLearnedRouteTurnCount: 0,
    boundLearnedRouteTurnCount: 0,
    activatedLearnedRouteTurnCount: 0,
    feltOverlap: {
      availableTraceCount: 0,
      totalTraceCount: params.feltTraceIds.length,
      betterCount: 0,
      tiedCount: 0,
      worseCount: 0,
    },
    notes: [...params.notes],
  };
}

function loadStaleBroadLiveProofRead(params: {
  harnessDir: string;
  feltTraceIds: string[];
}): BroadLiveProofReadSummaryV1 {
  const broadLiveDir = path.join(params.harnessDir, "semantic-rich-broad-live-eval");
  const { scorecard, report, summaryTables } = readBroadLiveArtifacts(broadLiveDir);
  return classifyBroadLiveProofReadV1({
    scorecard,
    report,
    summaryTables,
    feltTraceIds: params.feltTraceIds,
  });
}

function runCandidateSpecificFeltOptimizeEval(params: {
  harnessDir: string;
  feltOptimizeOutputDir: string;
  harness: ActivationFirstRetuneHarnessV1;
  candidateArtifact: CandidateArtifactRunSummaryV1;
}): FeltOptimizeComparativeSummaryV1 {
  const manifestPath = resolveHarnessPath(params.harnessDir, params.harness.feltResume.manifestPath);
  if (!existsSync(manifestPath)) {
    return unavailableFeltOptimizeComparativeSummary([
      `Felt optimize manifest is missing at ${manifestPath}; no candidate-specific felt comparative eval was run.`,
    ]);
  }

  rmSync(params.feltOptimizeOutputDir, { recursive: true, force: true });
  try {
    const descriptor = runComparativeEval({
      manifestPath,
      outputDir: params.feltOptimizeOutputDir,
      learnedRouteCandidateArtifactDir: params.candidateArtifact.outputDir,
      learnedRouteCandidateArtifactMode: "graph_walk_score_boost",
      learnedRouteCandidateReplayCursorPath: ["felt_resume_25"],
    });
    return summarizeFeltOptimizeScorecardV1({
      scorecard: descriptor.scorecard,
      outputDir: portableRelativePath(repoRoot, descriptor.outputDir),
      notes: [
        "Candidate-specific felt optimize comparative eval executed for the just-trained gating-only candidate.",
        `felt optimize manifest: ${portableRelativePath(repoRoot, manifestPath)}`,
        `candidate artifact: ${params.candidateArtifact.artifactId}@${params.candidateArtifact.artifactVersion}`,
        `candidate artifact checksum: ${params.candidateArtifact.artifactChecksum}`,
      ],
    });
  } catch (error) {
    return unavailableFeltOptimizeComparativeSummary([
      `Candidate-specific felt optimize comparative eval failed: ${toErrorMessage(error)}`,
      `candidate artifact: ${params.candidateArtifact.artifactId}@${params.candidateArtifact.artifactVersion}`,
      `candidate artifact checksum: ${params.candidateArtifact.artifactChecksum}`,
    ]);
  }
}

function runCandidateSpecificBroadLiveProofRead(params: {
  harnessDir: string;
  broadLiveOutputDir: string;
  harness: ActivationFirstRetuneHarnessV1;
  candidateArtifact: CandidateArtifactRunSummaryV1;
  feltTraceIds: string[];
}): BroadLiveProofReadSummaryV1 {
  const manifestRef = normalizeCliString(params.harness.broadLiveGuardrail?.manifestPath ?? undefined);
  if (manifestRef === null) {
    return unavailableBroadLiveProofRead({
      feltTraceIds: params.feltTraceIds,
      notes: ["Harness broad-live guardrail manifest is missing; no fresh candidate-specific veto can be run."],
    });
  }

  const manifestPath = resolveHarnessPath(params.harnessDir, manifestRef);
  if (!existsSync(manifestPath)) {
    return unavailableBroadLiveProofRead({
      feltTraceIds: params.feltTraceIds,
      notes: [`Broad-live guardrail manifest is missing at ${manifestPath}; no fresh candidate-specific veto can be run.`],
    });
  }

  rmSync(params.broadLiveOutputDir, { recursive: true, force: true });
  try {
    const descriptor = runComparativeEval({
      manifestPath,
      outputDir: params.broadLiveOutputDir,
      learnedRouteCandidateArtifactDir: params.candidateArtifact.outputDir,
      learnedRouteCandidateArtifactMode: "served_pack_adapter",
    });
    const laneSummaryTablesPath = descriptor.report.files.laneSummaryTables === null
      ? null
      : path.join(descriptor.outputDir, descriptor.report.files.laneSummaryTables);
    const summaryTables = laneSummaryTablesPath !== null && existsSync(laneSummaryTablesPath)
      ? readJson<BroadLiveSummaryTablesV1>(laneSummaryTablesPath)
      : null;
    const broadLive = classifyBroadLiveProofReadV1({
      scorecard: descriptor.scorecard,
      report: descriptor.report,
      summaryTables,
      feltTraceIds: params.feltTraceIds,
      freshRunExecuted: true,
    });
    broadLive.notes = uniqueStrings([
      ...broadLive.notes,
      `broad-live candidate run output: ${portableRelativePath(repoRoot, params.broadLiveOutputDir)}`,
      `broad-live guardrail manifest: ${portableRelativePath(repoRoot, manifestPath)}`,
      `candidate artifact: ${params.candidateArtifact.artifactId}@${params.candidateArtifact.artifactVersion}`,
      `candidate artifact checksum: ${params.candidateArtifact.artifactChecksum}`,
      `candidate router identity: ${params.candidateArtifact.routerIdentity ?? "null"}`,
    ]);
    return broadLive;
  } catch (error) {
    return unavailableBroadLiveProofRead({
      feltTraceIds: params.feltTraceIds,
      notes: [
        `Candidate-specific broad-live comparative eval failed: ${toErrorMessage(error)}`,
        `candidate artifact: ${params.candidateArtifact.artifactId}@${params.candidateArtifact.artifactVersion}`,
        `candidate artifact checksum: ${params.candidateArtifact.artifactChecksum}`,
      ],
    });
  }
}

function loadBroadLiveProofRead(params: {
  harnessDir: string;
  broadLiveOutputDir: string;
  harness: ActivationFirstRetuneHarnessV1;
  candidateArtifact: CandidateArtifactRunSummaryV1;
  feltTraceIds: string[];
}): BroadLiveProofReadSummaryV1 {
  const candidateRun = runCandidateSpecificBroadLiveProofRead(params);
  if (candidateRun.available) {
    return candidateRun;
  }

  const staleProofRead = loadStaleBroadLiveProofRead({
    harnessDir: params.harnessDir,
    feltTraceIds: params.feltTraceIds,
  });
  return {
    ...staleProofRead,
    notes: uniqueStrings([
      ...candidateRun.notes,
      ...staleProofRead.notes,
    ]),
  };
}

function broadLiveQualificationSuffix(broadLive: BroadLiveProofReadSummaryV1): string {
  if (broadLive.authoritative) {
    return "";
  }
  return broadLive.notes.includes(CANDIDATE_SPECIFIC_BROAD_LIVE_RUN_NOTE)
    ? " (non-authoritative candidate-specific run)"
    : " (non-authoritative proof read)";
}

function renderBrutalScorecard(scorecard: RunnerScorecardV1): string {
  const broadLive = scorecard.broadLive;
  const feltEval = scorecard.feltUniqueWinsTiesRegressions;
  const overlap = scorecard.feltBroadLiveOverlap;
  const feltSuffix = scorecard.feltComparativeEval.available
    ? ` (full felt eval ${feltEval.availableTraceCount}/${feltEval.totalTraceCount}; broad-live overlap ${overlap.availableTraceCount}/${overlap.totalTraceCount})`
    : feltEval.availableTraceCount > 0
      ? ` (broad-live overlap ${feltEval.availableTraceCount}/${feltEval.totalTraceCount})`
      : " (felt eval unavailable)";
  return [
    `felt unique wins / ties / regressions: ${feltEval.betterCount}/${feltEval.tiedCount}/${feltEval.worseCount}${feltSuffix}`,
    `felt replay-gate expectation pass count: ${scorecard.feltReplayGateExpectationPassCount.passed}/${scorecard.feltReplayGateExpectationPassCount.total}`,
    `unnecessary activations / must-not-fire failures: ${scorecard.unnecessaryActivations.count}/${scorecard.unnecessaryActivations.total} / ${scorecard.mustNotFireFailures.count}/${scorecard.mustNotFireFailures.total}`,
    `broad-live regressions or veto result: ${broadLive.regressions}/${broadLive.comparableTraceCount}; ${broadLive.vetoResult}${broadLiveQualificationSuffix(broadLive)}`,
    `final candidate status: ${scorecard.finalCandidateStatus}`,
  ].join("\n");
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  mkdirSync(args.outputDir, { recursive: true });

  const harnessPath = path.join(args.harnessDir, "activation-first-retune-harness.json");
  if (!existsSync(harnessPath)) {
    throw new Error(`Harness descriptor missing at ${harnessPath}`);
  }
  const harness = readJson<ActivationFirstRetuneHarnessV1>(harnessPath);
  const routeObjectiveInputPath = resolveHarnessPath(args.harnessDir, harness.routeObjectiveInputPath);
  const routeObjectiveInput = readJson<ActivationFirstRouteObjectiveInputV1>(routeObjectiveInputPath);
  const weightFloorsByClass = harness.hardNegativeMining?.weightFloorsByClass;
  const replaySurfaceIndex = buildReplaySurfaceIndex(args.harnessDir);

  const harnessFiles = uniqueStrings([
    harnessPath,
    routeObjectiveInputPath,
    resolveHarnessPath(args.harnessDir, harness.feltResume.manifestPath),
    resolveHarnessPath(args.harnessDir, harness.mustNotFire.manifestPath),
    args.includeMustFire ? resolveHarnessPath(args.harnessDir, harness.mustFire.manifestPath) : null,
  ]);

  const selectedLanes: Array<[ActivationFirstLaneKeyV1, ActivationFirstRouteObjectiveLaneV1]> = [
    ["feltResume", routeObjectiveInput.lanes.feltResume],
    ...(args.includeMustFire ? [["mustFire", routeObjectiveInput.lanes.mustFire] as const] : []),
    ["mustNotFire", routeObjectiveInput.lanes.mustNotFire],
  ];

  const materializedLanes = selectedLanes.map(([laneKey, lane]) => materializeLaneRows({
    taskId: args.taskId,
    harnessDir: args.harnessDir,
    generatedAt: args.generatedAt,
    laneKey,
    lane,
    replaySurfaceIndex,
    weightFloorsByClass,
    harnessFiles,
  }));

  const routeRows = materializedLanes.flatMap((lane) => lane.routeRows);
  const policyRows = materializedLanes.flatMap((lane) => lane.policyRows);
  const registryEntries = materializedLanes.map((lane) => lane.registryEntry);

  const artifacts: RunnerArtifactsV1 = {
    routeRowsPath: path.join(args.outputDir, "route-rows.json"),
    policyRowsPath: path.join(args.outputDir, "policy-supervision-rows.json"),
    registryEntriesPath: path.join(args.outputDir, "registry-entries.json"),
    materializationPath: path.join(args.outputDir, "materialization-summary.json"),
    candidateArtifactDir: path.join(args.outputDir, "candidate-artifact"),
    feltOptimizeEvalDir: path.join(args.outputDir, "felt-optimize-comparative-eval"),
    broadLiveEvalDir: path.join(args.outputDir, "broad-live-comparative-eval"),
    scorecardPath: path.join(args.outputDir, "scorecard.json"),
    runPath: path.join(args.outputDir, "run.json"),
  };

  writeJson(artifacts.routeRowsPath, routeRows);
  writeJson(artifacts.policyRowsPath, policyRows);
  writeJson(artifacts.registryEntriesPath, registryEntries);
  writeJson(artifacts.materializationPath, {
    taskId: args.taskId,
    generatedAt: args.generatedAt,
    includeMustFire: args.includeMustFire,
    laneCounts: Object.fromEntries(materializedLanes.map((lane) => [lane.laneName, lane.routeRows.length])),
    replaySurfaceCoverage: Object.fromEntries(
      materializedLanes.map((lane) => [
        lane.laneName,
        lane.routeRows.filter((row) => (replaySurfaceIndex.get(row.row_id.split(":").slice(-1)[0] ?? "") ?? []).length > 0).length,
      ]),
    ),
  });

  ensureCandidateArtifactDir(artifacts.candidateArtifactDir);

  const warmStartBundle = loadColdStartRouterArtifactBundleV1(args.warmStartArtifactDir);
  const candidateArtifactId = `router-artifact-${args.taskId.toLowerCase()}-activation-first-gating-only-v1`;
  const candidateArtifactVersion = "0.0.1";
  const candidateRouterIdentity = `router:${args.taskId}:activation-first-gating-only:v1`;
  const trainingDataRefs = uniqueStrings([
    `activation-first-harness:${portableRelativePath(repoRoot, harnessPath)}`,
    `activation-first-route-objective:${portableRelativePath(repoRoot, routeObjectiveInputPath)}`,
    ...materializedLanes.map((lane) => lane.datasetId),
  ]);

  const candidate = trainColdStartRouterArtifactV1({
    artifactId: candidateArtifactId,
    artifactVersion: candidateArtifactVersion,
    packType: "base",
    compatibleRuntimeVersion: warmStartBundle.manifest.compatible_runtime_version,
    registryEntries,
    routeRows,
    policySupervisionRows: policyRows,
    focusLaneWeights: harness.suggestedTrainingConfig.focusLaneWeights,
    rowTypeWeights: harness.suggestedTrainingConfig.rowTypeWeights,
    interventionHead: ACTIVATION_FIRST_GATING_ONLY_INTERVENTION_HEAD_V1,
    calibrationOverrides: ACTIVATION_FIRST_GATING_ONLY_CALIBRATION_OVERRIDES_V1,
    outputDir: artifacts.candidateArtifactDir,
    routerIdentity: candidateRouterIdentity,
    createdAt: args.generatedAt,
    trainingDataRefs,
    replayGateRefs: [
      `replay:${routeObjectiveInput.lanes.feltResume.trancheId}`,
      `replay:${routeObjectiveInput.lanes.mustNotFire.trancheId}`,
      ...(args.includeMustFire ? [`replay:${routeObjectiveInput.lanes.mustFire.trancheId}`] : []),
    ],
    warmStartArtifactDir: args.warmStartArtifactDir,
    warmStartMode: "strict",
    warmStartRef: args.warmStartArtifactDir,
  });

  const feltLane = materializedLanes.find((lane) => lane.laneKey === "feltResume");
  const mustNotFireLane = materializedLanes.find((lane) => lane.laneKey === "mustNotFire");
  if (!feltLane || !mustNotFireLane) {
    throw new Error("feltResume and mustNotFire lanes are required");
  }

  const feltReplay = replayColdStartRouterArtifactV1({
    artifactDir: artifacts.candidateArtifactDir,
    routeRows: feltLane.routeRows,
    policySupervisionRows: feltLane.policyRows,
  });
  const mustNotFireReplay = replayColdStartRouterArtifactV1({
    artifactDir: artifacts.candidateArtifactDir,
    routeRows: mustNotFireLane.routeRows,
    policySupervisionRows: mustNotFireLane.policyRows,
  });

  const feltLaneSummary = findLaneSummary(feltReplay, feltLane.laneName);
  const mustNotFireLaneSummary = findLaneSummary(mustNotFireReplay, mustNotFireLane.laneName);
  const unnecessaryActivations = summarizeUnnecessaryActivations(mustNotFireReplay);
  const mustNotFireFailures = summarizeMustNotFireFailures(mustNotFireLaneSummary);
  const candidateArtifact = {
    artifactId: candidate.manifest.artifact_id,
    artifactVersion: candidate.manifest.artifact_version,
    artifactChecksum: candidate.manifest.artifact_checksum,
    outputDir: artifacts.candidateArtifactDir,
    routerIdentity: candidateRouterIdentity,
  };
  const feltComparativeEval = runCandidateSpecificFeltOptimizeEval({
    harnessDir: args.harnessDir,
    feltOptimizeOutputDir: artifacts.feltOptimizeEvalDir,
    harness,
    candidateArtifact,
  });
  const broadLive = loadBroadLiveProofRead({
    harnessDir: args.harnessDir,
    broadLiveOutputDir: artifacts.broadLiveEvalDir,
    harness,
    candidateArtifact,
    feltTraceIds: routeObjectiveInput.lanes.feltResume.traceIds,
  });
  const feltUniqueWinsTiesRegressions: FeltOverlapSummaryV1 = feltComparativeEval.available
    ? {
      availableTraceCount: feltComparativeEval.comparableTraceCount,
      totalTraceCount: routeObjectiveInput.lanes.feltResume.traceIds.length,
      betterCount: feltComparativeEval.betterCount,
      tiedCount: feltComparativeEval.tiedCount,
      worseCount: feltComparativeEval.worseCount,
    }
    : broadLive.feltOverlap;

  const finalCandidateStatus = deriveFinalCandidateStatusV1({
    feltPassed: feltReplay.passed && feltLaneSummary.failedPolicyExpectationCount === 0,
    restraintPassed: mustNotFireReplay.passed && mustNotFireLaneSummary.failedPolicyExpectationCount === 0,
    broadLiveAuthoritative: broadLive.authoritative,
    broadLiveVetoResult: broadLive.vetoResult,
  });

  const scorecard: RunnerScorecardV1 = {
    candidateArtifactDir: portableRelativePath(repoRoot, artifacts.candidateArtifactDir),
    feltUniqueWinsTiesRegressions,
    feltComparativeEval,
    feltBroadLiveOverlap: broadLive.feltOverlap,
    feltReplayGateExpectationPassCount: {
      passed: feltLaneSummary.passedPolicyExpectationCount,
      total: feltLaneSummary.policyExpectationCount,
    },
    unnecessaryActivations,
    mustNotFireFailures,
    broadLive,
    finalCandidateStatus,
  };

  writeJson(artifacts.scorecardPath, scorecard);
  writeJson(artifacts.runPath, {
    taskId: args.taskId,
    generatedAt: args.generatedAt,
    includeMustFire: args.includeMustFire,
    harnessPath,
    routeObjectiveInputPath,
    warmStartArtifactDir: args.warmStartArtifactDir,
    candidateArtifact: {
      outputDir: artifacts.candidateArtifactDir,
      artifactId: candidate.manifest.artifact_id,
      artifactVersion: candidate.manifest.artifact_version,
      artifactChecksum: candidate.manifest.artifact_checksum,
      priorBaseArtifactId: candidate.manifest.prior_base_artifact_id ?? null,
      priorBaseArtifactChecksum: candidate.manifest.prior_base_artifact_checksum ?? null,
      routerIdentity: candidateRouterIdentity,
    },
    training: candidate.model.training,
    feltReplay,
    feltComparativeEvaluation: feltComparativeEval,
    mustNotFireReplay,
    broadLiveEvaluation: {
      outputDir: artifacts.broadLiveEvalDir,
      guardrailManifestPath: harness.broadLiveGuardrail?.manifestPath ?? null,
    },
    broadLive,
    scorecard,
  });

  process.stdout.write(`${renderBrutalScorecard(scorecard)}\n`);
}

const isMain = process.argv[1] ? path.resolve(process.argv[1]) === __filename : false;

if (isMain) {
  main().catch((error) => {
    process.stderr.write(`${error instanceof Error ? error.stack ?? error.message : String(error)}\n`);
    process.exitCode = 1;
  });
}
