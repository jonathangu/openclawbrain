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
  summary: "summary.md",
  traceDir: "traces",
} as const;

type ComparativeEvalMode = (typeof RECORDED_SESSION_REPLAY_PROOF_LANE_MODE_ORDER)[number];
type ComparativeEvalStatus = "ok" | "partial" | "blocked";

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
  files: {
    sourceManifest: string | null;
    report: string;
    scorecard: string;
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
  summaryPath: string;
  report: ComparativeEvalRunnerReportV1;
  scorecard: ComparativeEvalScorecardV1;
}

export interface RunComparativeEvalInput {
  manifestPath?: string;
  outputDir?: string;
  scratchRootDir?: string;
  workedTraceLimit?: number | null;
}

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

function buildScorecard(params: {
  manifestContract: string | null;
  manifestId: string | null;
  requestedTraceCount: number;
  pricingTable: PricingTable;
  traceRows: ComparativeEvalTraceScorecardRowV1[];
}): ComparativeEvalScorecardV1 {
  const successfulTraceCount = params.traceRows.filter((traceRow) => traceRow.status === "ok" && traceRow.validationOk === true).length;
  const failedTraceCount = params.traceRows.length - successfulTraceCount;
  const modes = buildModeScorecardRows(params.traceRows);
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
      "the scorecard is observational scaffold output; it does not claim long-run task or API economics",
    ],
    modes,
    traces: params.traceRows,
  };
  return {
    ...base,
    scorecardHash: checksumJsonPayload(base),
  };
}

function buildSummary(report: ComparativeEvalRunnerReportV1, scorecard: ComparativeEvalScorecardV1): string {
  const modeRows = scorecard.modes.map((mode) =>
    `| ${mode.mode} | ${mode.traceCount} | ${mode.rankedWinnerCount} | ${mode.meanQualityScore ?? "null"} | ${mode.compileOkRate ?? "null"} | ${mode.phraseHitRate ?? "null"} | ${mode.estimatedPromptTokens} | ${mode.estimatedPromptCostUsd ?? "null"} |`,
  );
  const traceRows = scorecard.traces.map((trace) =>
    `| ${trace.traceId} | ${trace.status} | ${trace.validationOk ?? "null"} | ${trace.winnerMode ?? "null"} | ${trace.scoreSpread ?? "null"} | ${trace.error ?? "none"} |`,
  );
  return [
    "# Comparative Eval Runner",
    "",
    `- status: \`${report.status}\``,
    `- manifest path: \`${report.manifestPath}\``,
    `- manifest contract: \`${report.manifestContract ?? "null"}\``,
    `- manifest id: \`${report.manifestId ?? "null"}\``,
    `- git sha: \`${report.gitSha}\``,
    `- traces: ${report.successfulTraceCount}/${report.requestedTraceCount}`,
    `- scorecard hash: \`${report.scorecardHash}\``,
    "",
    "## Modes",
    "| mode | traces | ranked winners | mean quality | compile ok rate | phrase hit rate | prompt tokens | prompt cost usd |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ...(modeRows.length > 0 ? modeRows : ["| none | 0 | 0 | null | null | null | 0 | null |"]),
    "",
    "## Traces",
    "| trace | status | validation ok | winner | score spread | error |",
    "| --- | --- | --- | --- | ---: | --- |",
    ...(traceRows.length > 0 ? traceRows : ["| none | blocked | null | null | null | none |"]),
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
  const manifestLoad = loadManifestInputs(manifestPath);
  const pricingTable = loadPricingTable();
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
  });

  const sourceManifestPath = manifestLoad.manifest === null
    ? null
    : path.join(outputDir, COMPARATIVE_EVAL_RUNNER_LAYOUT.sourceManifest);
  const reportPath = path.join(outputDir, COMPARATIVE_EVAL_RUNNER_LAYOUT.report);
  const scorecardPath = path.join(outputDir, COMPARATIVE_EVAL_RUNNER_LAYOUT.scorecard);
  const summaryPath = path.join(outputDir, COMPARATIVE_EVAL_RUNNER_LAYOUT.summary);
  const report: ComparativeEvalRunnerReportV1 = {
    contract: COMPARATIVE_EVAL_RUNNER_REPORT_CONTRACT,
    status,
    generatedAt: new Date().toISOString(),
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
    files: {
      sourceManifest: sourceManifestPath === null ? null : COMPARATIVE_EVAL_RUNNER_LAYOUT.sourceManifest,
      report: COMPARATIVE_EVAL_RUNNER_LAYOUT.report,
      scorecard: COMPARATIVE_EVAL_RUNNER_LAYOUT.scorecard,
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
  writeText(summaryPath, buildSummary(report, scorecard));

  return {
    outputDir,
    traceRoot,
    sourceManifestPath,
    reportPath,
    scorecardPath,
    summaryPath,
    report,
    scorecard,
  };
}
