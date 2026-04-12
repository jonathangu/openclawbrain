import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { canonicalJson, checksumJsonPayload } from "@openclawbrain/contracts";
import type {
  RecordedSessionReplayModeReportV1,
  RecordedSessionReplayProofBundleDescriptorV1,
  RecordedSessionTraceV1,
} from "../../packages/cli/dist/src/index.js";
import {
  buildOpenClawBrainExplainableEvalScorecard,
  type OpenClawBrainExplainableEvalScorecardV1,
} from "./openclawbrain-explainable-scorecard.ts";
import {
  RECORDED_SESSION_REPLAY_PROOF_LANE_MODE_ORDER,
  writeRecordedSessionReplayProofLane,
  type RecordedSessionReplayProofLaneDescriptorV1,
} from "../replay-proof-lane.ts";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..", "..");

export const CANONICAL_RECORDED_SESSION_TRACE_SET_MANIFEST_CONTRACT = "canonical_recorded_session_trace_set_manifest.v1";
export const FROZEN_RECORDED_SESSION_EVAL_MANIFEST_CONTRACT = "frozen_recorded_session_eval_manifest.v1";
export const COMPARATIVE_EVAL_RUNNER_REPORT_CONTRACT = "comparative_eval_runner_report.v1";
export const COMPARATIVE_EVAL_SCORECARD_CONTRACT = "comparative_eval_scorecard.v1";
export const DEFAULT_COMPARATIVE_EVAL_MANIFEST_PATH = path.resolve(
  repoRoot,
  "evals",
  "recorded-session-replay",
  "canonical-frozen-20",
  "manifest.json",
);

export const COMPARATIVE_EVAL_RUNNER_LAYOUT = {
  sourceManifest: "source-manifest.json",
  report: "report.json",
  scorecard: "scorecard.json",
  explainableScorecard: "explainable-scorecard.json",
  summary: "summary.md",
  traceDir: "traces",
} as const;

type ComparativeEvalMode = (typeof RECORDED_SESSION_REPLAY_PROOF_LANE_MODE_ORDER)[number];
type ComparativeEvalStatus = "ok" | "partial" | "blocked";
type ComparativeEvalGateStatus = "pass" | "fail" | "partial" | "blocked";

interface PricingTable {
  version: string | null;
  path: string;
  charsPerToken: number;
  promptPriceUsdPer1mTokens: number;
}

interface LoadedTraceInput {
  traceId: string;
  traceHash: string;
  tracePath: string;
  relativeTracePath: string;
  trace: RecordedSessionTraceV1;
}

interface LoadedManifestInputs {
  manifest: Record<string, unknown> | null;
  manifestContract: string | null;
  manifestId: string | null;
  expectedTraceCount: number | null;
  notes: string[];
  traces: LoadedTraceInput[];
  issues: string[];
}

export interface FrozenRecordedSessionEvalManifestTrace {
  tracePath: string;
  traceId?: string;
  traceHash?: string;
  notes?: string[];
}

export interface FrozenRecordedSessionEvalManifestV1 {
  contract: typeof FROZEN_RECORDED_SESSION_EVAL_MANIFEST_CONTRACT;
  manifestId: string;
  generatedAt?: string;
  expectedTraceCount?: number;
  notes?: string[];
  traces: FrozenRecordedSessionEvalManifestTrace[];
}

export interface CanonicalRecordedSessionTraceSetManifestV1 {
  contract: typeof CANONICAL_RECORDED_SESSION_TRACE_SET_MANIFEST_CONTRACT;
  setId: string;
  traceCount?: number;
  entries: Array<{
    slotId?: string;
    path?: string;
  }>;
  realTraceCoverage?: {
    summary?: string;
  };
  redactionPolicy?: {
    summary?: string;
  };
}

export interface ComparativeEvalTraceModeScorecardRowV1 {
  mode: ComparativeEvalMode;
  qualityScore: number;
  compileOkCount: number;
  turnCount: number;
  compileOkRate: number | null;
  phraseHitCount: number;
  phraseCount: number;
  phraseHitRate: number | null;
  promotionCount: number;
  usedLearnedRouteTurnCount: number;
  warningCount: number;
  selectedContextBlockCount: number;
  selectedContextChars: number;
  estimatedPromptTokens: number;
  estimatedPromptCostUsd: number | null;
}

export interface ComparativeEvalTraceScorecardRowV1 {
  traceId: string;
  tracePath: string | null;
  relativeTracePath: string | null;
  bundleDir: string | null;
  status: "ok" | "failed";
  validationOk: boolean | null;
  winnerMode: ComparativeEvalMode | null;
  topScoreModes: ComparativeEvalMode[];
  scoreSpread: number | null;
  error: string | null;
  modes: ComparativeEvalTraceModeScorecardRowV1[];
}

export interface ComparativeEvalModeScorecardRowV1 {
  mode: ComparativeEvalMode;
  traceCount: number;
  rankedWinnerCount: number;
  sharedTopScoreTraceCount: number;
  meanQualityScore: number | null;
  totalCompileOkCount: number;
  totalTurnCount: number;
  compileOkRate: number | null;
  totalPhraseHitCount: number;
  totalPhraseCount: number;
  phraseHitRate: number | null;
  totalPromotionCount: number;
  totalUsedLearnedRouteTurnCount: number;
  totalWarningCount: number;
  totalSelectedContextBlockCount: number;
  totalSelectedContextChars: number;
  estimatedPromptTokens: number;
  estimatedPromptCostUsd: number | null;
}

export interface ComparativeEvalWinRateRowV1 {
  left: number;
  right: number;
  ties: number;
  leftRate: number | null;
  rightRate: number | null;
  tieRate: number | null;
}

export interface ComparativeEvalTieOrBetterRowV1 {
  left: number;
  right: number;
  leftRate: number | null;
  rightRate: number | null;
}

export interface ComparativeEvalPairwiseScorecardRowV1 {
  leftMode: ComparativeEvalMode;
  rightMode: ComparativeEvalMode;
  comparableTraceCount: number;
  comparableTurnCount: number;
  traceWins: ComparativeEvalWinRateRowV1;
  traceTieOrBetter: ComparativeEvalTieOrBetterRowV1;
  turnWins: ComparativeEvalWinRateRowV1;
  turnTieOrBetter: ComparativeEvalTieOrBetterRowV1;
  aggregateDeltas: {
    qualityScoreDeltaLeftMinusRightSum: number;
    qualityScoreDeltaLeftMinusRightMean: number | null;
    compileOkDeltaLeftMinusRightSum: number;
    phraseHitDeltaLeftMinusRightSum: number;
    promotionDeltaLeftMinusRightSum: number;
    tiePromotionDeltaLeftMinusRightSum: number;
  };
}

export interface ComparativeEvalPolicyThresholdsV1 {
  candidateMode: ComparativeEvalMode;
  baselineMode: ComparativeEvalMode;
  floorMode: ComparativeEvalMode;
  maxFailedTraceCount: number;
  minCandidateTraceTieOrBetterRateVsBaseline: number;
  maxCandidateMeanQualityRegressionVsBaseline: number;
  minBaselineMeanQualityGainVsFloor: number;
  maxCandidateTiePromotionDeltaVsBaseline: number;
}

export interface ComparativeEvalPolicyCheckV1 {
  id: string;
  status: "pass" | "fail";
  summary: string;
  detail: string;
  observed: Record<string, number | string | boolean | null>;
  threshold: Record<string, number | string | boolean | null>;
}

export interface ComparativeEvalPolicyObservedV1 {
  requestedTraceCount: number;
  successfulTraceCount: number;
  failedTraceCount: number;
  comparableTraceCount: number;
  comparableTurnCount: number;
  baselineMeanQualityScore: number | null;
  candidateMeanQualityScore: number | null;
  floorMeanQualityScore: number | null;
  candidateTraceTieOrBetterCountVsBaseline: number | null;
  candidateTraceTieOrBetterRateVsBaseline: number | null;
  candidateTurnTieOrBetterCountVsBaseline: number | null;
  candidateTurnTieOrBetterRateVsBaseline: number | null;
  candidateTieTraceCountVsBaseline: number | null;
  candidateTiePromotionDeltaVsBaseline: number | null;
  candidateMeanQualityRegressionVsBaseline: number | null;
  baselineMeanQualityGainVsFloor: number | null;
}

export interface ComparativeEvalPolicyV1 {
  status: ComparativeEvalGateStatus;
  decisive: boolean;
  thresholds: ComparativeEvalPolicyThresholdsV1;
  observed: ComparativeEvalPolicyObservedV1;
  reasons: string[];
  checks: ComparativeEvalPolicyCheckV1[];
}

export interface ComparativeEvalScorecardV1 {
  contract: typeof COMPARATIVE_EVAL_SCORECARD_CONTRACT;
  manifestId: string | null;
  manifestContract: string | null;
  modeOrder: ComparativeEvalMode[];
  requestedTraceCount: number;
  successfulTraceCount: number;
  failedTraceCount: number;
  scorecardHash: string;
  pricingTable: PricingTable;
  scoringProxyNotes: string[];
  modes: ComparativeEvalModeScorecardRowV1[];
  pairwise: ComparativeEvalPairwiseScorecardRowV1[];
  policy: ComparativeEvalPolicyV1;
  traces: ComparativeEvalTraceScorecardRowV1[];
}

export interface ComparativeEvalRunnerReportV1 {
  contract: typeof COMPARATIVE_EVAL_RUNNER_REPORT_CONTRACT;
  status: ComparativeEvalStatus;
  generatedAt: string;
  repoRoot: string;
  gitSha: string;
  manifestPath: string;
  manifestContract: string | null;
  manifestId: string | null;
  outputDir: string;
  traceRoot: string;
  requestedTraceCount: number;
  expectedTraceCount: number | null;
  successfulTraceCount: number;
  failedTraceCount: number;
  notes: string[];
  assumptions: string[];
  issues: string[];
  pricingTable: PricingTable;
  scorecardHash: string;
  explainableScorecardHash: string;
  gateStatus: ComparativeEvalGateStatus;
  gateDecisive: boolean;
  gateFailedCheckIds: string[];
  files: {
    sourceManifest: string | null;
    report: string;
    scorecard: string;
    explainableScorecard: string;
    summary: string;
    traceDir: string;
    laneDir: string | null;
    laneIndex: string | null;
    laneSummaryTables: string | null;
    lanePairwiseDeltas: string | null;
    laneWinRateMatrix: string | null;
    laneWorkedTraces: string | null;
    laneGenerationReport: string | null;
  };
}

export interface ComparativeEvalRunnerDescriptor {
  outputDir: string;
  traceRoot: string;
  sourceManifestPath: string | null;
  reportPath: string;
  scorecardPath: string;
  explainableScorecardPath: string;
  summaryPath: string;
  report: ComparativeEvalRunnerReportV1;
  scorecard: ComparativeEvalScorecardV1;
  explainableScorecard: OpenClawBrainExplainableEvalScorecardV1;
}

export interface RunComparativeEvalInput {
  manifestPath?: string;
  outputDir?: string;
  scratchRootDir?: string;
  workedTraceLimit?: number | null;
  policy?: Partial<ComparativeEvalPolicyThresholdsV1>;
}

const DEFAULT_COMPARATIVE_EVAL_POLICY_THRESHOLDS: ComparativeEvalPolicyThresholdsV1 = {
  candidateMode: "learned_route",
  baselineMode: "graph_prior_only",
  floorMode: "no_brain",
  maxFailedTraceCount: 0,
  minCandidateTraceTieOrBetterRateVsBaseline: 1,
  maxCandidateMeanQualityRegressionVsBaseline: 5,
  minBaselineMeanQualityGainVsFloor: 5,
  maxCandidateTiePromotionDeltaVsBaseline: 0,
};

function normalizeCliString(value: string | undefined): string | null {
  if (typeof value !== "string") {
    return null;
  }
  const trimmed = value.trim();
  return trimmed.length === 0 ? null : trimmed;
}

function toObjectRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function round(value: number, places = 6): number {
  const factor = 10 ** places;
  return Math.round(value * factor) / factor;
}

function countTextChars(value: unknown): number {
  if (typeof value === "string") {
    return value.length;
  }
  if (!Array.isArray(value)) {
    return 0;
  }
  return value.reduce((total, item) => total + (typeof item === "string" ? item.length : 0), 0);
}

function estimateTokensFromChars(chars: number, charsPerToken: number): number {
  if (!Number.isFinite(chars) || !Number.isFinite(charsPerToken) || charsPerToken <= 0) {
    return 0;
  }
  return Math.ceil(chars / charsPerToken);
}

function estimateUsdFromTokens(tokens: number, pricePer1mTokens: number): number | null {
  if (!Number.isFinite(tokens) || !Number.isFinite(pricePer1mTokens)) {
    return null;
  }
  return round((tokens / 1_000_000) * pricePer1mTokens, 6);
}

function portableRelativePath(fromPath: string, toPath: string): string {
  return path.relative(fromPath, toPath).split(path.sep).join("/");
}

function toErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function gitShaOrUnknown(): string {
  try {
    return execFileSync("git", ["rev-parse", "HEAD"], {
      cwd: repoRoot,
      encoding: "utf8",
    }).trim();
  } catch {
    return "unknown-git-sha";
  }
}

function defaultOutputDir(manifestPath: string): string {
  const artifactDate = new Date().toISOString().slice(0, 10);
  const manifestLabel = path.basename(manifestPath, path.extname(manifestPath));
  return path.resolve(
    repoRoot,
    "docs",
    "evidence",
    artifactDate,
    gitShaOrUnknown(),
    "comparative-eval",
    manifestLabel,
  );
}

function loadPricingTable(): PricingTable {
  const pricingTablePath = path.resolve(repoRoot, "scripts", "pricing-table.v1.json");
  const pricingTable = JSON.parse(readFileSync(pricingTablePath, "utf8")) as {
    version?: string;
    charsPerToken?: number;
    promptPriceUsdPer1mTokens?: number;
  };
  const charsPerToken = Number(pricingTable.charsPerToken ?? 4);
  const promptPriceUsdPer1mTokens = Number(pricingTable.promptPriceUsdPer1mTokens ?? 0);
  if (!Number.isFinite(charsPerToken) || charsPerToken <= 0) {
    throw new Error(`pricing table charsPerToken is invalid at ${pricingTablePath}`);
  }
  if (!Number.isFinite(promptPriceUsdPer1mTokens)) {
    throw new Error(`pricing table promptPriceUsdPer1mTokens is invalid at ${pricingTablePath}`);
  }
  return {
    version: typeof pricingTable.version === "string" ? pricingTable.version : null,
    path: portableRelativePath(repoRoot, pricingTablePath),
    charsPerToken,
    promptPriceUsdPer1mTokens,
  };
}

function writeJson(filePath: string, payload: unknown): void {
  mkdirSync(path.dirname(filePath), { recursive: true });
  writeFileSync(filePath, `${canonicalJson(payload)}\n`, "utf8");
}

function writeText(filePath: string, value: string): void {
  mkdirSync(path.dirname(filePath), { recursive: true });
  writeFileSync(filePath, value.endsWith("\n") ? value : `${value}\n`, "utf8");
}

function isFrozenRecordedSessionEvalManifest(
  manifest: unknown,
): manifest is FrozenRecordedSessionEvalManifestV1 {
  return toObjectRecord(manifest)?.contract === FROZEN_RECORDED_SESSION_EVAL_MANIFEST_CONTRACT;
}

function isCanonicalRecordedSessionTraceSetManifest(
  manifest: unknown,
): manifest is CanonicalRecordedSessionTraceSetManifestV1 {
  return toObjectRecord(manifest)?.contract === CANONICAL_RECORDED_SESSION_TRACE_SET_MANIFEST_CONTRACT;
}

function loadTraceInput(params: {
  tracePath: string;
  relativeTracePath: string;
  expectedTraceId?: string | null;
  expectedTraceHash?: string | null;
  issues: string[];
  seenTraceIds: Set<string>;
}): LoadedTraceInput | null {
  if (!existsSync(params.tracePath)) {
    params.issues.push(`trace missing at ${params.tracePath}`);
    return null;
  }

  let trace: RecordedSessionTraceV1;
  try {
    trace = JSON.parse(readFileSync(params.tracePath, "utf8")) as RecordedSessionTraceV1;
  } catch (error) {
    params.issues.push(`trace at ${params.tracePath} is not valid JSON: ${toErrorMessage(error)}`);
    return null;
  }

  if ((trace as { contract?: string }).contract !== "recorded_session_trace.v1") {
    params.issues.push(`trace at ${params.tracePath} must use contract recorded_session_trace.v1`);
    return null;
  }

  const traceId = normalizeCliString(trace.traceId);
  if (traceId === null) {
    params.issues.push(`trace at ${params.tracePath} is missing traceId`);
    return null;
  }
  if (params.expectedTraceId !== null && params.expectedTraceId !== undefined && params.expectedTraceId !== traceId) {
    params.issues.push(`manifest traceId mismatch for ${params.tracePath}: expected ${params.expectedTraceId}, received ${traceId}`);
  }
  if (params.seenTraceIds.has(traceId)) {
    params.issues.push(`duplicate traceId in manifest: ${traceId}`);
    return null;
  }
  params.seenTraceIds.add(traceId);

  const traceHash = checksumJsonPayload(trace);
  if (params.expectedTraceHash !== null && params.expectedTraceHash !== undefined && params.expectedTraceHash !== traceHash) {
    params.issues.push(`manifest traceHash mismatch for ${traceId}: expected ${params.expectedTraceHash}, received ${traceHash}`);
  }

  return {
    traceId,
    traceHash,
    tracePath: params.tracePath,
    relativeTracePath: params.relativeTracePath,
    trace,
  };
}

function loadManifestInputs(manifestPath: string): LoadedManifestInputs {
  const issues: string[] = [];
  if (!existsSync(manifestPath)) {
    return {
      manifest: null,
      manifestContract: null,
      manifestId: null,
      expectedTraceCount: null,
      notes: [],
      traces: [],
      issues: [`manifest missing at ${manifestPath}`],
    };
  }

  let parsedManifest: Record<string, unknown> | null = null;
  try {
    parsedManifest = JSON.parse(readFileSync(manifestPath, "utf8")) as Record<string, unknown>;
  } catch (error) {
    return {
      manifest: null,
      manifestContract: null,
      manifestId: null,
      expectedTraceCount: null,
      notes: [],
      traces: [],
      issues: [`manifest is not valid JSON: ${toErrorMessage(error)}`],
    };
  }

  const manifestRecord = toObjectRecord(parsedManifest);
  if (!manifestRecord) {
    return {
      manifest: null,
      manifestContract: null,
      manifestId: null,
      expectedTraceCount: null,
      notes: [],
      traces: [],
      issues: ["manifest must be a JSON object"],
    };
  }

  const manifestDir = path.dirname(manifestPath);
  const manifestContract = normalizeCliString(typeof manifestRecord.contract === "string" ? manifestRecord.contract : undefined);
  const seenTraceIds = new Set<string>();
  const traces: LoadedTraceInput[] = [];
  const notes: string[] = [];
  let manifestId: string | null = null;
  let expectedTraceCount: number | null = null;

  if (isFrozenRecordedSessionEvalManifest(manifestRecord)) {
    manifestId = normalizeCliString(manifestRecord.manifestId) ?? null;
    if (manifestId === null) {
      issues.push("manifestId is required");
    }
    if (!Array.isArray(manifestRecord.traces)) {
      issues.push("manifest traces must be an array");
      return {
        manifest: manifestRecord,
        manifestContract,
        manifestId,
        expectedTraceCount,
        notes,
        traces,
        issues,
      };
    }
    expectedTraceCount = Number.isInteger(manifestRecord.expectedTraceCount) && Number(manifestRecord.expectedTraceCount) > 0
      ? Number(manifestRecord.expectedTraceCount)
      : null;
    if (manifestRecord.expectedTraceCount !== undefined && expectedTraceCount === null) {
      issues.push("expectedTraceCount must be a positive integer when provided");
    }
    if (expectedTraceCount !== null && manifestRecord.traces.length !== expectedTraceCount) {
      issues.push(`manifest expectedTraceCount=${expectedTraceCount} but found ${manifestRecord.traces.length} traces`);
    }
    if (Array.isArray(manifestRecord.notes)) {
      notes.push(...manifestRecord.notes.map((note) => String(note)));
    }
    for (const [index, entry] of manifestRecord.traces.entries()) {
      if (typeof entry?.tracePath !== "string" || entry.tracePath.trim().length === 0) {
        issues.push(`traces[${index}].tracePath is required`);
        continue;
      }
      const tracePath = path.resolve(manifestDir, entry.tracePath);
      const loadedTrace = loadTraceInput({
        tracePath,
        relativeTracePath: portableRelativePath(manifestDir, tracePath),
        expectedTraceId: normalizeCliString(entry.traceId),
        expectedTraceHash: normalizeCliString(entry.traceHash),
        issues,
        seenTraceIds,
      });
      if (loadedTrace) {
        traces.push(loadedTrace);
      }
    }
  } else if (isCanonicalRecordedSessionTraceSetManifest(manifestRecord)) {
    manifestId = normalizeCliString(manifestRecord.setId) ?? null;
    if (manifestId === null) {
      issues.push("setId is required");
    }
    if (!Array.isArray(manifestRecord.entries)) {
      issues.push("canonical manifest entries must be an array");
      return {
        manifest: manifestRecord,
        manifestContract,
        manifestId,
        expectedTraceCount,
        notes,
        traces,
        issues,
      };
    }
    expectedTraceCount = Number.isInteger(manifestRecord.traceCount) && Number(manifestRecord.traceCount) > 0
      ? Number(manifestRecord.traceCount)
      : null;
    if (manifestRecord.traceCount !== undefined && expectedTraceCount === null) {
      issues.push("traceCount must be a positive integer when provided");
    }
    if (expectedTraceCount !== null && manifestRecord.entries.length !== expectedTraceCount) {
      issues.push(`manifest traceCount=${expectedTraceCount} but found ${manifestRecord.entries.length} entries`);
    }
    const realTraceCoverageSummary = normalizeCliString(
      toObjectRecord(manifestRecord.realTraceCoverage)?.summary as string | undefined,
    );
    if (realTraceCoverageSummary !== null) {
      notes.push(`truth boundary: ${realTraceCoverageSummary}`);
    }
    const redactionPolicySummary = normalizeCliString(
      toObjectRecord(manifestRecord.redactionPolicy)?.summary as string | undefined,
    );
    if (redactionPolicySummary !== null) {
      notes.push(`redaction policy: ${redactionPolicySummary}`);
    }
    for (const [index, entry] of manifestRecord.entries.entries()) {
      const relativeTracePath = normalizeCliString(entry?.path);
      if (relativeTracePath === null) {
        issues.push(`entries[${index}].path is required`);
        continue;
      }
      const tracePath = path.resolve(manifestDir, relativeTracePath);
      const loadedTrace = loadTraceInput({
        tracePath,
        relativeTracePath,
        issues,
        seenTraceIds,
      });
      if (loadedTrace) {
        traces.push(loadedTrace);
      }
    }
  } else {
    issues.push(
      `manifest contract must be ${CANONICAL_RECORDED_SESSION_TRACE_SET_MANIFEST_CONTRACT} or ${FROZEN_RECORDED_SESSION_EVAL_MANIFEST_CONTRACT}`,
    );
  }

  return {
    manifest: manifestRecord,
    manifestContract,
    manifestId,
    expectedTraceCount,
    notes,
    traces,
    issues,
  };
}

function toRate(numerator: number, denominator: number): number | null {
  return denominator > 0 ? round(numerator / denominator, 6) : null;
}

function mergePolicyThresholds(
  overrides: Partial<ComparativeEvalPolicyThresholdsV1> | undefined,
): ComparativeEvalPolicyThresholdsV1 {
  const merged = {
    ...DEFAULT_COMPARATIVE_EVAL_POLICY_THRESHOLDS,
    ...(overrides ?? {}),
  };
  if (!Number.isInteger(merged.maxFailedTraceCount) || merged.maxFailedTraceCount < 0) {
    throw new Error("policy.maxFailedTraceCount must be a non-negative integer");
  }
  if (
    !Number.isFinite(merged.minCandidateTraceTieOrBetterRateVsBaseline)
    || merged.minCandidateTraceTieOrBetterRateVsBaseline < 0
    || merged.minCandidateTraceTieOrBetterRateVsBaseline > 1
  ) {
    throw new Error("policy.minCandidateTraceTieOrBetterRateVsBaseline must be between 0 and 1");
  }
  if (!Number.isFinite(merged.maxCandidateMeanQualityRegressionVsBaseline)) {
    throw new Error("policy.maxCandidateMeanQualityRegressionVsBaseline must be finite");
  }
  if (!Number.isFinite(merged.minBaselineMeanQualityGainVsFloor)) {
    throw new Error("policy.minBaselineMeanQualityGainVsFloor must be finite");
  }
  if (!Number.isFinite(merged.maxCandidateTiePromotionDeltaVsBaseline)) {
    throw new Error("policy.maxCandidateTiePromotionDeltaVsBaseline must be finite");
  }
  return merged;
}

function buildTraceModeRow(
  mode: RecordedSessionReplayModeReportV1,
  pricingTable: PricingTable,
): ComparativeEvalTraceModeScorecardRowV1 {
  const selectedContextBlockCount = mode.turns.reduce((total, turn) => total + (
    Array.isArray(turn.selectedContextIds)
      ? turn.selectedContextIds.length
      : Array.isArray(turn.selectedContextTexts)
        ? turn.selectedContextTexts.length
        : 0
  ), 0);
  const selectedContextChars = mode.turns.reduce(
    (total, turn) => total + countTextChars(turn.selectedContextTexts),
    0,
  );
  const estimatedPromptTokens = estimateTokensFromChars(selectedContextChars, pricingTable.charsPerToken);
  return {
    mode: mode.mode as ComparativeEvalMode,
    qualityScore: mode.summary.qualityScore,
    compileOkCount: mode.summary.compileOkCount,
    turnCount: mode.turns.length,
    compileOkRate: toRate(mode.summary.compileOkCount, mode.turns.length),
    phraseHitCount: mode.summary.phraseHitCount,
    phraseCount: mode.summary.phraseCount,
    phraseHitRate: toRate(mode.summary.phraseHitCount, mode.summary.phraseCount),
    promotionCount: mode.summary.promotionCount,
    usedLearnedRouteTurnCount: mode.summary.usedLearnedRouteTurnCount,
    warningCount: mode.summary.scannerEvidence.warnings.length,
    selectedContextBlockCount,
    selectedContextChars,
    estimatedPromptTokens,
    estimatedPromptCostUsd: estimateUsdFromTokens(estimatedPromptTokens, pricingTable.promptPriceUsdPer1mTokens),
  };
}

function topScoreModes(rows: ComparativeEvalTraceModeScorecardRowV1[]): ComparativeEvalMode[] {
  if (rows.length === 0) {
    return [];
  }
  const maxScore = rows.reduce((best, row) => Math.max(best, row.qualityScore), Number.NEGATIVE_INFINITY);
  return rows.filter((row) => row.qualityScore === maxScore).map((row) => row.mode);
}

function scoreSpread(rows: ComparativeEvalTraceModeScorecardRowV1[]): number | null {
  if (rows.length === 0) {
    return null;
  }
  const scores = rows.map((row) => row.qualityScore);
  return Math.max(...scores) - Math.min(...scores);
}

function buildTraceScorecardRows(params: {
  outputDir: string;
  manifestTraces: LoadedTraceInput[];
  laneDescriptor: RecordedSessionReplayProofLaneDescriptorV1 | null;
  pricingTable: PricingTable;
}): ComparativeEvalTraceScorecardRowV1[] {
  const descriptorByTraceId = new Map<string, RecordedSessionReplayProofBundleDescriptorV1>(
    (params.laneDescriptor?.successfulBundles ?? []).map((descriptor) => [descriptor.bundle.traceId, descriptor]),
  );
  const generationByTraceId = new Map(
    (params.laneDescriptor?.generationReport.entries ?? []).map((entry) => [entry.traceId, entry]),
  );

  return params.manifestTraces
    .map((traceInput) => {
      const descriptor = descriptorByTraceId.get(traceInput.traceId);
      const generationEntry = generationByTraceId.get(traceInput.traceId);
      if (!descriptor || !generationEntry) {
        return {
          traceId: traceInput.traceId,
          tracePath: traceInput.tracePath,
          relativeTracePath: traceInput.relativeTracePath,
          bundleDir: null,
          status: "failed" as const,
          validationOk: null,
          winnerMode: null,
          topScoreModes: [],
          scoreSpread: null,
          error: generationEntry?.error ?? "trace bundle was not generated",
          modes: [],
        };
      }

      const modeRows = RECORDED_SESSION_REPLAY_PROOF_LANE_MODE_ORDER.map((mode) => {
        const modeReport = descriptor.bundle.modes.find((candidate) => candidate.mode === mode);
        if (!modeReport) {
          throw new Error(`Missing mode ${mode} for trace ${descriptor.bundle.traceId}`);
        }
        return buildTraceModeRow(modeReport, params.pricingTable);
      });
      return {
        traceId: traceInput.traceId,
        tracePath: traceInput.tracePath,
        relativeTracePath: traceInput.relativeTracePath,
        bundleDir: portableRelativePath(params.outputDir, descriptor.rootDir),
        status: generationEntry.result === "passed" ? "ok" as const : "failed" as const,
        validationOk: generationEntry.validation?.ok ?? null,
        winnerMode: descriptor.bundle.summary.winnerMode as ComparativeEvalMode | null,
        topScoreModes: topScoreModes(modeRows),
        scoreSpread: scoreSpread(modeRows),
        error: generationEntry.error,
        modes: modeRows,
      };
    })
    .sort((left, right) => left.traceId.localeCompare(right.traceId));
}

function buildModeScorecardRows(traceRows: ComparativeEvalTraceScorecardRowV1[]): ComparativeEvalModeScorecardRowV1[] {
  const successfulTraceRows = traceRows.filter((traceRow) => traceRow.status === "ok" && traceRow.validationOk === true);
  return RECORDED_SESSION_REPLAY_PROOF_LANE_MODE_ORDER.map((mode) => {
    const modeRows = successfulTraceRows
      .map((traceRow) => traceRow.modes.find((candidate) => candidate.mode === mode))
      .filter((candidate): candidate is ComparativeEvalTraceModeScorecardRowV1 => candidate !== undefined);
    const traceCount = modeRows.length;
    const totalTurnCount = modeRows.reduce((sum, row) => sum + row.turnCount, 0);
    const totalCompileOkCount = modeRows.reduce((sum, row) => sum + row.compileOkCount, 0);
    const totalPhraseHitCount = modeRows.reduce((sum, row) => sum + row.phraseHitCount, 0);
    const totalPhraseCount = modeRows.reduce((sum, row) => sum + row.phraseCount, 0);
    const totalPromotionCount = modeRows.reduce((sum, row) => sum + row.promotionCount, 0);
    const totalUsedLearnedRouteTurnCount = modeRows.reduce((sum, row) => sum + row.usedLearnedRouteTurnCount, 0);
    const totalWarningCount = modeRows.reduce((sum, row) => sum + row.warningCount, 0);
    const totalSelectedContextBlockCount = modeRows.reduce((sum, row) => sum + row.selectedContextBlockCount, 0);
    const totalSelectedContextChars = modeRows.reduce((sum, row) => sum + row.selectedContextChars, 0);
    const estimatedPromptTokens = modeRows.reduce((sum, row) => sum + row.estimatedPromptTokens, 0);
    const estimatedPromptCosts = modeRows.map((row) => row.estimatedPromptCostUsd).filter((value): value is number => value !== null);
    return {
      mode,
      traceCount,
      rankedWinnerCount: successfulTraceRows.filter((traceRow) => traceRow.winnerMode === mode).length,
      sharedTopScoreTraceCount: successfulTraceRows.filter((traceRow) => traceRow.topScoreModes.includes(mode)).length,
      meanQualityScore: traceCount > 0
        ? round(modeRows.reduce((sum, row) => sum + row.qualityScore, 0) / traceCount, 6)
        : null,
      totalCompileOkCount,
      totalTurnCount,
      compileOkRate: toRate(totalCompileOkCount, totalTurnCount),
      totalPhraseHitCount,
      totalPhraseCount,
      phraseHitRate: toRate(totalPhraseHitCount, totalPhraseCount),
      totalPromotionCount,
      totalUsedLearnedRouteTurnCount,
      totalWarningCount,
      totalSelectedContextBlockCount,
      totalSelectedContextChars,
      estimatedPromptTokens,
      estimatedPromptCostUsd: estimatedPromptCosts.length === traceCount
        ? round(estimatedPromptCosts.reduce((sum, value) => sum + value, 0), 6)
        : null,
    };
  });
}

function buildWinRateRow(left: number, right: number, ties: number): ComparativeEvalWinRateRowV1 {
  const comparableCount = left + right + ties;
  return {
    left,
    right,
    ties,
    leftRate: toRate(left, comparableCount),
    rightRate: toRate(right, comparableCount),
    tieRate: toRate(ties, comparableCount),
  };
}

function buildTieOrBetterRow(
  wins: ComparativeEvalWinRateRowV1,
  comparableCount: number,
): ComparativeEvalTieOrBetterRowV1 {
  const left = wins.left + wins.ties;
  const right = wins.right + wins.ties;
  return {
    left,
    right,
    leftRate: toRate(left, comparableCount),
    rightRate: toRate(right, comparableCount),
  };
}

function buildPairwiseScorecardRows(params: {
  laneDescriptor: RecordedSessionReplayProofLaneDescriptorV1 | null;
  successfulTraceIds: Set<string>;
}): ComparativeEvalPairwiseScorecardRowV1[] {
  if (params.laneDescriptor === null || params.successfulTraceIds.size === 0) {
    return [];
  }

  const traceRows = params.laneDescriptor.summaryTables.traces.filter((trace) =>
    trace.validationOk === true && params.successfulTraceIds.has(trace.traceId)
  );
  const turnRows = params.laneDescriptor.summaryTables.turns.filter((turn) => params.successfulTraceIds.has(turn.traceId));
  const pairwiseRows: ComparativeEvalPairwiseScorecardRowV1[] = [];

  for (let leftIndex = 0; leftIndex < RECORDED_SESSION_REPLAY_PROOF_LANE_MODE_ORDER.length; leftIndex += 1) {
    for (let rightIndex = leftIndex + 1; rightIndex < RECORDED_SESSION_REPLAY_PROOF_LANE_MODE_ORDER.length; rightIndex += 1) {
      const leftMode = RECORDED_SESSION_REPLAY_PROOF_LANE_MODE_ORDER[leftIndex];
      const rightMode = RECORDED_SESSION_REPLAY_PROOF_LANE_MODE_ORDER[rightIndex];
      let traceLeftWins = 0;
      let traceRightWins = 0;
      let traceTies = 0;
      let turnLeftWins = 0;
      let turnRightWins = 0;
      let turnTies = 0;
      let qualityScoreDeltaLeftMinusRightSum = 0;
      let compileOkDeltaLeftMinusRightSum = 0;
      let phraseHitDeltaLeftMinusRightSum = 0;
      let promotionDeltaLeftMinusRightSum = 0;
      let tiePromotionDeltaLeftMinusRightSum = 0;

      for (const trace of traceRows) {
        const left = trace.modes.find((candidate) => candidate.mode === leftMode);
        const right = trace.modes.find((candidate) => candidate.mode === rightMode);
        if (!left || !right) {
          throw new Error(`Missing pairwise trace mode rows for ${leftMode} vs ${rightMode}`);
        }
        if (left.qualityScore > right.qualityScore) {
          traceLeftWins += 1;
        } else if (left.qualityScore < right.qualityScore) {
          traceRightWins += 1;
        } else {
          traceTies += 1;
        }
        const qualityScoreDeltaLeftMinusRight = left.qualityScore - right.qualityScore;
        qualityScoreDeltaLeftMinusRightSum += qualityScoreDeltaLeftMinusRight;
        compileOkDeltaLeftMinusRightSum += left.compileOkCount - right.compileOkCount;
        phraseHitDeltaLeftMinusRightSum += left.phraseHitCount - right.phraseHitCount;
        const promotionDeltaLeftMinusRight = left.promotionCount - right.promotionCount;
        promotionDeltaLeftMinusRightSum += promotionDeltaLeftMinusRight;
        if (qualityScoreDeltaLeftMinusRight === 0) {
          tiePromotionDeltaLeftMinusRightSum += promotionDeltaLeftMinusRight;
        }
      }

      for (const turn of turnRows) {
        const left = turn.modes.find((candidate) => candidate.mode === leftMode);
        const right = turn.modes.find((candidate) => candidate.mode === rightMode);
        if (!left || !right) {
          throw new Error(`Missing pairwise turn mode rows for ${leftMode} vs ${rightMode}`);
        }
        if (left.qualityScore > right.qualityScore) {
          turnLeftWins += 1;
        } else if (left.qualityScore < right.qualityScore) {
          turnRightWins += 1;
        } else {
          turnTies += 1;
        }
      }

      const traceWins = buildWinRateRow(traceLeftWins, traceRightWins, traceTies);
      const turnWins = buildWinRateRow(turnLeftWins, turnRightWins, turnTies);
      pairwiseRows.push({
        leftMode,
        rightMode,
        comparableTraceCount: traceRows.length,
        comparableTurnCount: turnRows.length,
        traceWins,
        traceTieOrBetter: buildTieOrBetterRow(traceWins, traceRows.length),
        turnWins,
        turnTieOrBetter: buildTieOrBetterRow(turnWins, turnRows.length),
        aggregateDeltas: {
          qualityScoreDeltaLeftMinusRightSum,
          qualityScoreDeltaLeftMinusRightMean: toRate(qualityScoreDeltaLeftMinusRightSum, traceRows.length),
          compileOkDeltaLeftMinusRightSum,
          phraseHitDeltaLeftMinusRightSum,
          promotionDeltaLeftMinusRightSum,
          tiePromotionDeltaLeftMinusRightSum,
        },
      });
    }
  }

  return pairwiseRows;
}

function buildPolicy(params: {
  requestedTraceCount: number;
  successfulTraceCount: number;
  failedTraceCount: number;
  runnerStatus: ComparativeEvalStatus;
  modeRows: ComparativeEvalModeScorecardRowV1[];
  pairwiseRows: ComparativeEvalPairwiseScorecardRowV1[];
  thresholds: ComparativeEvalPolicyThresholdsV1;
  issues: string[];
}): ComparativeEvalPolicyV1 {
  const baseline = params.modeRows.find((row) => row.mode === params.thresholds.baselineMode) ?? null;
  const candidate = params.modeRows.find((row) => row.mode === params.thresholds.candidateMode) ?? null;
  const floor = params.modeRows.find((row) => row.mode === params.thresholds.floorMode) ?? null;
  const baselineVsCandidate = params.pairwiseRows.find((row) =>
    row.leftMode === params.thresholds.baselineMode && row.rightMode === params.thresholds.candidateMode
  ) ?? null;
  const candidateMeanQualityRegressionVsBaseline = baseline?.meanQualityScore !== null && baseline?.meanQualityScore !== undefined
    && candidate?.meanQualityScore !== null && candidate?.meanQualityScore !== undefined
    ? round(baseline.meanQualityScore - candidate.meanQualityScore, 6)
    : null;
  const baselineMeanQualityGainVsFloor = baseline?.meanQualityScore !== null && baseline?.meanQualityScore !== undefined
    && floor?.meanQualityScore !== null && floor?.meanQualityScore !== undefined
    ? round(baseline.meanQualityScore - floor.meanQualityScore, 6)
    : null;
  const candidateTieTraceCountVsBaseline = baselineVsCandidate?.traceWins.ties ?? 0;
  const candidateTiePromotionDeltaVsBaseline = baselineVsCandidate
    ? -baselineVsCandidate.aggregateDeltas.tiePromotionDeltaLeftMinusRightSum
    : 0;
  const observed: ComparativeEvalPolicyObservedV1 = {
    requestedTraceCount: params.requestedTraceCount,
    successfulTraceCount: params.successfulTraceCount,
    failedTraceCount: params.failedTraceCount,
    comparableTraceCount: baselineVsCandidate?.comparableTraceCount ?? 0,
    comparableTurnCount: baselineVsCandidate?.comparableTurnCount ?? 0,
    baselineMeanQualityScore: baseline?.meanQualityScore ?? null,
    candidateMeanQualityScore: candidate?.meanQualityScore ?? null,
    floorMeanQualityScore: floor?.meanQualityScore ?? null,
    candidateTraceTieOrBetterCountVsBaseline: baselineVsCandidate?.traceTieOrBetter.right ?? null,
    candidateTraceTieOrBetterRateVsBaseline: baselineVsCandidate?.traceTieOrBetter.rightRate ?? null,
    candidateTurnTieOrBetterCountVsBaseline: baselineVsCandidate?.turnTieOrBetter.right ?? null,
    candidateTurnTieOrBetterRateVsBaseline: baselineVsCandidate?.turnTieOrBetter.rightRate ?? null,
    candidateTieTraceCountVsBaseline,
    candidateTiePromotionDeltaVsBaseline,
    candidateMeanQualityRegressionVsBaseline,
    baselineMeanQualityGainVsFloor,
  };

  if (params.runnerStatus === "blocked") {
    return {
      status: "blocked",
      decisive: false,
      thresholds: params.thresholds,
      observed,
      reasons: params.issues.length > 0 ? [...params.issues] : ["comparative eval runner was blocked before any traces could be compared"],
      checks: [],
    };
  }

  if (observed.comparableTraceCount === 0 || baselineVsCandidate === null || baseline === null || candidate === null || floor === null) {
    return {
      status: "blocked",
      decisive: false,
      thresholds: params.thresholds,
      observed,
      reasons: ["comparative eval produced no complete baseline/candidate comparison set"],
      checks: [],
    };
  }

  const checks: ComparativeEvalPolicyCheckV1[] = [
    {
      id: "trace_coverage_complete",
      status: params.failedTraceCount <= params.thresholds.maxFailedTraceCount ? "pass" : "fail",
      summary: params.failedTraceCount <= params.thresholds.maxFailedTraceCount
        ? "all requested traces produced validated scorecard rows"
        : "comparative eval coverage is incomplete",
      detail: `${params.successfulTraceCount}/${params.requestedTraceCount} traces validated`,
      observed: {
        requestedTraceCount: params.requestedTraceCount,
        successfulTraceCount: params.successfulTraceCount,
        failedTraceCount: params.failedTraceCount,
      },
      threshold: {
        maxFailedTraceCount: params.thresholds.maxFailedTraceCount,
      },
    },
    {
      id: "candidate_trace_tie_or_better_vs_baseline",
      status: observed.candidateTraceTieOrBetterRateVsBaseline !== null
        && observed.candidateTraceTieOrBetterRateVsBaseline >= params.thresholds.minCandidateTraceTieOrBetterRateVsBaseline
        ? "pass"
        : "fail",
      summary: observed.candidateTraceTieOrBetterRateVsBaseline !== null
        && observed.candidateTraceTieOrBetterRateVsBaseline >= params.thresholds.minCandidateTraceTieOrBetterRateVsBaseline
        ? "candidate tied or beat the baseline at the configured per-trace rate"
        : "candidate missed the configured per-trace tie-or-better rate versus the baseline",
      detail: observed.candidateTraceTieOrBetterRateVsBaseline === null
        ? "trace tie-or-better rate could not be computed"
        : `${params.thresholds.candidateMode} tie-or-better vs ${params.thresholds.baselineMode} = ${observed.candidateTraceTieOrBetterRateVsBaseline}`,
      observed: {
        candidateMode: params.thresholds.candidateMode,
        baselineMode: params.thresholds.baselineMode,
        comparableTraceCount: observed.comparableTraceCount,
        candidateTraceTieOrBetterCountVsBaseline: observed.candidateTraceTieOrBetterCountVsBaseline,
        candidateTraceTieOrBetterRateVsBaseline: observed.candidateTraceTieOrBetterRateVsBaseline,
      },
      threshold: {
        minCandidateTraceTieOrBetterRateVsBaseline: params.thresholds.minCandidateTraceTieOrBetterRateVsBaseline,
      },
    },
    {
      id: "candidate_tie_promotion_delta_vs_baseline",
      status: observed.candidateTiePromotionDeltaVsBaseline <= params.thresholds.maxCandidateTiePromotionDeltaVsBaseline
        ? "pass"
        : "fail",
      summary: observed.candidateTiePromotionDeltaVsBaseline <= params.thresholds.maxCandidateTiePromotionDeltaVsBaseline
        ? "candidate did not add promotion churn on tie traces"
        : "candidate added promotion churn on tie traces",
      detail: `${params.thresholds.candidateMode} tie traces=${observed.candidateTieTraceCountVsBaseline} promotion_delta_vs_${params.thresholds.baselineMode}=${observed.candidateTiePromotionDeltaVsBaseline}`,
      observed: {
        candidateMode: params.thresholds.candidateMode,
        baselineMode: params.thresholds.baselineMode,
        candidateTieTraceCountVsBaseline: observed.candidateTieTraceCountVsBaseline,
        candidateTiePromotionDeltaVsBaseline: observed.candidateTiePromotionDeltaVsBaseline,
      },
      threshold: {
        maxCandidateTiePromotionDeltaVsBaseline: params.thresholds.maxCandidateTiePromotionDeltaVsBaseline,
      },
    },
    {
      id: "candidate_mean_quality_regression_vs_baseline",
      status: observed.candidateMeanQualityRegressionVsBaseline !== null
        && observed.candidateMeanQualityRegressionVsBaseline <= params.thresholds.maxCandidateMeanQualityRegressionVsBaseline
        ? "pass"
        : "fail",
      summary: observed.candidateMeanQualityRegressionVsBaseline !== null
        && observed.candidateMeanQualityRegressionVsBaseline <= params.thresholds.maxCandidateMeanQualityRegressionVsBaseline
        ? "candidate mean quality stayed within the allowed regression budget"
        : "candidate mean quality regressed beyond the allowed budget",
      detail: observed.candidateMeanQualityRegressionVsBaseline === null
        ? "mean quality regression could not be computed"
        : `${params.thresholds.baselineMode} - ${params.thresholds.candidateMode} mean quality = ${observed.candidateMeanQualityRegressionVsBaseline}`,
      observed: {
        baselineMeanQualityScore: observed.baselineMeanQualityScore,
        candidateMeanQualityScore: observed.candidateMeanQualityScore,
        candidateMeanQualityRegressionVsBaseline: observed.candidateMeanQualityRegressionVsBaseline,
      },
      threshold: {
        maxCandidateMeanQualityRegressionVsBaseline: params.thresholds.maxCandidateMeanQualityRegressionVsBaseline,
      },
    },
    {
      id: "baseline_mean_quality_gain_vs_floor",
      status: observed.baselineMeanQualityGainVsFloor !== null
        && observed.baselineMeanQualityGainVsFloor >= params.thresholds.minBaselineMeanQualityGainVsFloor
        ? "pass"
        : "fail",
      summary: observed.baselineMeanQualityGainVsFloor !== null
        && observed.baselineMeanQualityGainVsFloor >= params.thresholds.minBaselineMeanQualityGainVsFloor
        ? "baseline clears the floor anchor by the configured mean quality margin"
        : "baseline does not clear the floor anchor by the configured mean quality margin",
      detail: observed.baselineMeanQualityGainVsFloor === null
        ? "baseline floor gain could not be computed"
        : `${params.thresholds.baselineMode} - ${params.thresholds.floorMode} mean quality = ${observed.baselineMeanQualityGainVsFloor}`,
      observed: {
        baselineMeanQualityScore: observed.baselineMeanQualityScore,
        floorMeanQualityScore: observed.floorMeanQualityScore,
        baselineMeanQualityGainVsFloor: observed.baselineMeanQualityGainVsFloor,
      },
      threshold: {
        minBaselineMeanQualityGainVsFloor: params.thresholds.minBaselineMeanQualityGainVsFloor,
      },
    },
  ];

  const coverageCheck = checks[0];
  if (coverageCheck.status === "fail") {
    return {
      status: "partial",
      decisive: false,
      thresholds: params.thresholds,
      observed,
      reasons: [`final gate verdict withheld because only ${params.successfulTraceCount}/${params.requestedTraceCount} traces validated`],
      checks,
    };
  }

  const failingChecks = checks.filter((check) => check.status === "fail");
  return {
    status: failingChecks.length === 0 ? "pass" : "fail",
    decisive: true,
    thresholds: params.thresholds,
    observed,
    reasons: failingChecks.map((check) => `${check.id}: ${check.summary}`),
    checks,
  };
}

function buildScorecard(params: {
  manifestContract: string | null;
  manifestId: string | null;
  requestedTraceCount: number;
  pricingTable: PricingTable;
  traceRows: ComparativeEvalTraceScorecardRowV1[];
  laneDescriptor: RecordedSessionReplayProofLaneDescriptorV1 | null;
  runnerStatus: ComparativeEvalStatus;
  policyThresholds: ComparativeEvalPolicyThresholdsV1;
  issues: string[];
}): ComparativeEvalScorecardV1 {
  const successfulTraceCount = params.traceRows.filter((traceRow) => traceRow.status === "ok" && traceRow.validationOk === true).length;
  const failedTraceCount = params.traceRows.length - successfulTraceCount;
  const modes = buildModeScorecardRows(params.traceRows);
  const pairwise = buildPairwiseScorecardRows({
    laneDescriptor: params.laneDescriptor,
    successfulTraceIds: new Set(
      params.traceRows
        .filter((traceRow) => traceRow.status === "ok" && traceRow.validationOk === true)
        .map((traceRow) => traceRow.traceId),
    ),
  });
  const policy = buildPolicy({
    requestedTraceCount: params.requestedTraceCount,
    successfulTraceCount,
    failedTraceCount,
    runnerStatus: params.runnerStatus,
    modeRows: modes,
    pairwiseRows: pairwise,
    thresholds: params.policyThresholds,
    issues: params.issues,
  });
  const base: Omit<ComparativeEvalScorecardV1, "scorecardHash"> = {
    contract: COMPARATIVE_EVAL_SCORECARD_CONTRACT,
    manifestId: params.manifestId,
    manifestContract: params.manifestContract,
    modeOrder: [...RECORDED_SESSION_REPLAY_PROOF_LANE_MODE_ORDER],
    requestedTraceCount: params.requestedTraceCount,
    successfulTraceCount,
    failedTraceCount,
    pricingTable: params.pricingTable,
    scoringProxyNotes: [
      "qualityScore comes from the deterministic replay proof bundle scoring surface",
      "estimatedPromptTokens is ceil(selectedContextChars / charsPerToken)",
      "estimatedPromptCostUsd uses promptPriceUsdPer1mTokens from scripts/pricing-table.v1.json",
      "pairwise tie-or-better rates compare deterministic qualityScore outcomes across the same validated traces and turns",
      "the scorecard is observational scaffold output; it does not claim long-run task or API economics",
    ],
    modes,
    pairwise,
    policy,
    traces: params.traceRows,
  };
  return {
    ...base,
    scorecardHash: checksumJsonPayload(base),
  };
}

function buildExplainableScorecard(params: {
  generatedAt: string;
  scorecard: ComparativeEvalScorecardV1;
}): OpenClawBrainExplainableEvalScorecardV1 {
  return buildOpenClawBrainExplainableEvalScorecard({
    generatedAt: params.generatedAt,
    manifestId: params.scorecard.manifestId,
    manifestContract: params.scorecard.manifestContract,
    modeOrder: [...params.scorecard.modeOrder],
    requestedTraceCount: params.scorecard.requestedTraceCount,
    successfulTraceCount: params.scorecard.successfulTraceCount,
    failedTraceCount: params.scorecard.failedTraceCount,
    modes: params.scorecard.modes,
    pairwise: params.scorecard.pairwise,
    traces: params.scorecard.traces,
    notes: params.scorecard.scoringProxyNotes,
  });
}

function formatExplainableMetricValue(value: number | null, unit: string): string {
  if (value === null) {
    return "null";
  }
  return unit === "rate"
    ? String(value)
    : unit === "usd"
      ? `$${value}`
      : String(value);
}

function buildSummary(
  report: ComparativeEvalRunnerReportV1,
  scorecard: ComparativeEvalScorecardV1,
  explainableScorecard: OpenClawBrainExplainableEvalScorecardV1,
): string {
  const checkRows = scorecard.policy.checks.map((check) => {
    const observedText = Object.entries(check.observed)
      .map(([key, value]) => `${key}=${value === null ? "null" : String(value)}`)
      .join(", ");
    const thresholdText = Object.entries(check.threshold)
      .map(([key, value]) => `${key}=${value === null ? "null" : String(value)}`)
      .join(", ");
    return `| ${check.id} | ${check.status} | ${observedText} | ${thresholdText} |`;
  });
  const publicMetricRows = explainableScorecard.publicOperatorMetrics.map((metric) =>
    `| ${metric.id} | ${metric.availability} | ${formatExplainableMetricValue(metric.value, metric.unit)} | ${metric.formula.expression} | ${metric.language} |`,
  );
  const internalMetricRows = explainableScorecard.internalMetrics.map((metric) =>
    `| ${metric.id} | ${formatExplainableMetricValue(metric.value, metric.unit)} | ${metric.language} |`,
  );
  const traceRows = scorecard.traces.map((trace) =>
    `| ${trace.traceId} | ${trace.status} | ${trace.validationOk ?? "null"} | ${trace.winnerMode ?? "null"} | ${trace.scoreSpread ?? "null"} | ${trace.error ?? "none"} |`,
  );
  return [
    "# Comparative Eval Runner",
    "",
    `- status: \`${report.status}\``,
    `- gate: \`${report.gateStatus}\``,
    `- gate decisive: \`${report.gateDecisive}\``,
    `- manifest path: \`${report.manifestPath}\``,
    `- manifest contract: \`${report.manifestContract ?? "null"}\``,
    `- manifest id: \`${report.manifestId ?? "null"}\``,
    `- git sha: \`${report.gitSha}\``,
    `- traces: ${report.successfulTraceCount}/${report.requestedTraceCount}`,
    `- scorecard hash: \`${report.scorecardHash}\``,
    `- explainable scorecard hash: \`${report.explainableScorecardHash}\``,
    "",
    "## Public / Operator Headline",
    ...explainableScorecard.headline.map((line) => `- ${line}`),
    "",
    "## Public / Operator Metrics",
    "| metric | availability | value | formula | language |",
    "| --- | --- | ---: | --- | --- |",
    ...(publicMetricRows.length > 0 ? publicMetricRows : ["| none | not_available | null | none | no public metrics were computed |"]),
    "",
    "## Fail-Open Language",
    `- ${explainableScorecard.failOpenLanguage}`,
    "",
    "## Policy",
    `- candidate mode: \`${scorecard.policy.thresholds.candidateMode}\``,
    `- baseline mode: \`${scorecard.policy.thresholds.baselineMode}\``,
    `- floor mode: \`${scorecard.policy.thresholds.floorMode}\``,
    `- comparable traces: ${scorecard.policy.observed.comparableTraceCount}`,
    `- successful traces: ${scorecard.policy.observed.successfulTraceCount}`,
    `- failed traces: ${scorecard.policy.observed.failedTraceCount}`,
    "| check | status | observed | threshold |",
    "| --- | --- | --- | --- |",
    ...(checkRows.length > 0 ? checkRows : ["| none | blocked | none | none |"]),
    "",
    "## Internal Diagnostics",
    `- ${explainableScorecard.diagnosticLanguage}`,
    "| metric | value | language |",
    "| --- | ---: | --- |",
    ...(internalMetricRows.length > 0 ? internalMetricRows : ["| none | null | no internal diagnostics were computed |"]),
    "",
    "## Trace Coverage",
    "| trace | status | validation ok | diagnostic winnerMode | score spread | error |",
    "| --- | --- | --- | --- | ---: | --- |",
    ...(traceRows.length > 0 ? traceRows : ["| none | blocked | null | null | null | none |"]),
    "",
    "## Policy Reasons",
    ...(scorecard.policy.reasons.length > 0 ? scorecard.policy.reasons.map((reason) => `- ${reason}`) : ["- none"]),
    "",
    "## Notes",
    ...report.notes.map((note) => `- ${note}`),
    "",
    "## Assumptions",
    ...report.assumptions.map((assumption) => `- ${assumption}`),
    "",
    ...(report.issues.length === 0
      ? []
      : [
          "## Issues",
          ...report.issues.map((issue) => `- ${issue}`),
          "",
        ]),
  ].join("\n");
}

export function runComparativeEval(input: RunComparativeEvalInput = {}): ComparativeEvalRunnerDescriptor {
  const manifestPath = path.resolve(input.manifestPath ?? DEFAULT_COMPARATIVE_EVAL_MANIFEST_PATH);
  const outputDir = path.resolve(input.outputDir ?? defaultOutputDir(manifestPath));
  const traceRoot = path.join(outputDir, COMPARATIVE_EVAL_RUNNER_LAYOUT.traceDir);
  const generatedAt = new Date().toISOString();
  const manifestLoad = loadManifestInputs(manifestPath);
  const pricingTable = loadPricingTable();
  const policyThresholds = mergePolicyThresholds(input.policy);
  const notes = [
    `default manifest path is ${DEFAULT_COMPARATIVE_EVAL_MANIFEST_PATH}`,
    `mode order is ${RECORDED_SESSION_REPLAY_PROOF_LANE_MODE_ORDER.join(", ")}`,
    "the comparative runner delegates replay execution to writeRecordedSessionReplayProofLane so each trace still runs through the real replay/runtime path",
    ...manifestLoad.notes,
  ];
  const assumptions = [
    `accepted manifest contracts: ${CANONICAL_RECORDED_SESSION_TRACE_SET_MANIFEST_CONTRACT}, ${FROZEN_RECORDED_SESSION_EVAL_MANIFEST_CONTRACT}`,
    "manifest trace paths resolve relative to the manifest file location",
    "traceHash, when present in the manifest, is checksumJsonPayload(trace-json)",
    "scorecard prompt-cost metrics are cheap deterministic proxies derived from selected context chars",
    `${policyThresholds.candidateMode} is the candidate mode, ${policyThresholds.baselineMode} is the baseline mode, and ${policyThresholds.floorMode} is the floor anchor for the explicit comparative policy`,
    "this scaffold does not finalize the frozen trace set or widen proof-bundle generation scope",
  ];

  mkdirSync(outputDir, { recursive: true });
  rmSync(traceRoot, { recursive: true, force: true });

  let laneDescriptor: RecordedSessionReplayProofLaneDescriptorV1 | null = null;
  let status: ComparativeEvalStatus = "blocked";
  let issues = [...manifestLoad.issues];

  if (issues.length === 0 && manifestLoad.manifest !== null) {
    laneDescriptor = writeRecordedSessionReplayProofLane({
      artifactRoot: traceRoot,
      traces: manifestLoad.traces.map((trace) => ({
        trace: trace.trace,
        tracePath: trace.tracePath,
      })),
      sourceManifestPath: manifestPath,
      assumptions,
      ...(input.scratchRootDir ? { scratchRootDir: path.resolve(input.scratchRootDir) } : {}),
      ...(input.workedTraceLimit == null ? {} : { workedTraceLimit: input.workedTraceLimit }),
    });
    const failedEntries = laneDescriptor.generationReport.entries.filter((entry) => entry.result === "failed");
    issues = [
      ...issues,
      ...failedEntries.map((entry) => `${entry.traceId}: ${entry.error ?? "trace replay generation failed"}`),
    ];
    status = failedEntries.length === 0 ? "ok" : "partial";
  }

  const traceRows = buildTraceScorecardRows({
    outputDir,
    manifestTraces: manifestLoad.traces,
    laneDescriptor,
    pricingTable,
  });
  const scorecard = buildScorecard({
    manifestContract: manifestLoad.manifestContract,
    manifestId: manifestLoad.manifestId,
    requestedTraceCount: manifestLoad.traces.length,
    pricingTable,
    traceRows,
    laneDescriptor,
    runnerStatus: status,
    policyThresholds,
    issues,
  });
  const explainableScorecard = buildExplainableScorecard({
    generatedAt,
    scorecard,
  });

  const sourceManifestPath = manifestLoad.manifest === null
    ? null
    : path.join(outputDir, COMPARATIVE_EVAL_RUNNER_LAYOUT.sourceManifest);
  const reportPath = path.join(outputDir, COMPARATIVE_EVAL_RUNNER_LAYOUT.report);
  const scorecardPath = path.join(outputDir, COMPARATIVE_EVAL_RUNNER_LAYOUT.scorecard);
  const explainableScorecardPath = path.join(outputDir, COMPARATIVE_EVAL_RUNNER_LAYOUT.explainableScorecard);
  const summaryPath = path.join(outputDir, COMPARATIVE_EVAL_RUNNER_LAYOUT.summary);
  const report: ComparativeEvalRunnerReportV1 = {
    contract: COMPARATIVE_EVAL_RUNNER_REPORT_CONTRACT,
    status,
    generatedAt,
    repoRoot,
    gitSha: gitShaOrUnknown(),
    manifestPath,
    manifestContract: manifestLoad.manifestContract,
    manifestId: manifestLoad.manifestId,
    outputDir,
    traceRoot,
    requestedTraceCount: manifestLoad.traces.length,
    expectedTraceCount: manifestLoad.expectedTraceCount,
    successfulTraceCount: scorecard.successfulTraceCount,
    failedTraceCount: scorecard.failedTraceCount,
    notes,
    assumptions,
    issues,
    pricingTable,
    scorecardHash: scorecard.scorecardHash,
    explainableScorecardHash: explainableScorecard.scorecardHash,
    gateStatus: scorecard.policy.status,
    gateDecisive: scorecard.policy.decisive,
    gateFailedCheckIds: scorecard.policy.checks.filter((check) => check.status === "fail").map((check) => check.id),
    files: {
      sourceManifest: sourceManifestPath === null ? null : COMPARATIVE_EVAL_RUNNER_LAYOUT.sourceManifest,
      report: COMPARATIVE_EVAL_RUNNER_LAYOUT.report,
      scorecard: COMPARATIVE_EVAL_RUNNER_LAYOUT.scorecard,
      explainableScorecard: COMPARATIVE_EVAL_RUNNER_LAYOUT.explainableScorecard,
      summary: COMPARATIVE_EVAL_RUNNER_LAYOUT.summary,
      traceDir: COMPARATIVE_EVAL_RUNNER_LAYOUT.traceDir,
      laneDir: laneDescriptor ? portableRelativePath(outputDir, laneDescriptor.laneDir) : null,
      laneIndex: laneDescriptor ? portableRelativePath(outputDir, laneDescriptor.indexPath) : null,
      laneSummaryTables: laneDescriptor ? portableRelativePath(outputDir, laneDescriptor.summaryTablesPath) : null,
      lanePairwiseDeltas: laneDescriptor ? portableRelativePath(outputDir, laneDescriptor.pairwiseDeltasPath) : null,
      laneWinRateMatrix: laneDescriptor ? portableRelativePath(outputDir, laneDescriptor.winRateMatrixPath) : null,
      laneWorkedTraces: laneDescriptor ? portableRelativePath(outputDir, laneDescriptor.workedTracesPath) : null,
      laneGenerationReport: laneDescriptor ? portableRelativePath(outputDir, laneDescriptor.generationReportPath) : null,
    },
  };

  if (sourceManifestPath !== null && manifestLoad.manifest !== null) {
    writeJson(sourceManifestPath, manifestLoad.manifest);
  }
  writeJson(reportPath, report);
  writeJson(scorecardPath, scorecard);
  writeJson(explainableScorecardPath, explainableScorecard);
  writeText(summaryPath, buildSummary(report, scorecard, explainableScorecard));

  return {
    outputDir,
    traceRoot,
    sourceManifestPath,
    reportPath,
    scorecardPath,
    explainableScorecardPath,
    summaryPath,
    report,
    scorecard,
    explainableScorecard,
  };
}
