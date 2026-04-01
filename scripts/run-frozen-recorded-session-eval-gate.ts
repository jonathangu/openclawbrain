#!/usr/bin/env tsx

import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import { canonicalJson, checksumJsonPayload } from "@openclawbrain/contracts";
import {
  validateRecordedSessionReplayProofBundle,
  writeRecordedSessionReplayProofBundle,
  type RecordedSessionTraceV1,
} from "../packages/cli/dist/src/index.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..");

export const CANONICAL_RECORDED_SESSION_TRACE_SET_MANIFEST_CONTRACT = "canonical_recorded_session_trace_set_manifest.v1";
export const FROZEN_RECORDED_SESSION_EVAL_MANIFEST_CONTRACT = "frozen_recorded_session_eval_manifest.v1";
export const FROZEN_RECORDED_SESSION_EVAL_REPORT_CONTRACT = "frozen_recorded_session_eval_report.v1";
export const DEFAULT_FROZEN_RECORDED_SESSION_EVAL_MANIFEST_PATH = path.resolve(
  repoRoot,
  "evals",
  "recorded-session-replay",
  "canonical-frozen-20",
  "manifest.json",
);

export const FROZEN_RECORDED_SESSION_EVAL_BUNDLE_LAYOUT = {
  sourceManifest: "source-manifest.json",
  report: "report.json",
  summary: "summary.md",
  traceDir: "traces",
} as const;

const RECORDED_SESSION_TRACE_CONTRACT = "recorded_session_trace.v1";
const MODE_ORDER = ["no_brain", "vector_only", "graph_prior_only", "learned_route"] as const;
const DEFAULT_THRESHOLDS = {
  maxQualityRegression: 5,
  minGraphPriorOnlyQualityVsNoBrain: 5,
  minQualityAdjustedPromptSavingsUsd: 0,
} as const;

type GateStatus = "pass" | "fail" | "blocked";
type GateCheckStatus = "pass" | "fail";
type ModeName = (typeof MODE_ORDER)[number];

export interface FrozenRecordedSessionEvalThresholds {
  maxQualityRegression: number;
  minGraphPriorOnlyQualityVsNoBrain: number;
  minQualityAdjustedPromptSavingsUsd: number;
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
  thresholds?: Partial<FrozenRecordedSessionEvalThresholds>;
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

export interface FrozenRecordedSessionEvalGateCheck {
  id: string;
  status: GateCheckStatus;
  summary: string;
  detail: string;
  observed: Record<string, number | string | boolean | null>;
  threshold: Record<string, number | string | boolean | null>;
}

export interface FrozenRecordedSessionEvalModeSummary {
  mode: ModeName;
  traceCount: number;
  turnCount: number;
  compileOkCount: number;
  compileOkRate: number | null;
  phraseHitCount: number;
  phraseCount: number;
  phraseHitRate: number | null;
  qualityScore: number;
  qualityScoreMean: number | null;
  selectedContextBlockCount: number;
  selectedContextChars: number;
  estimatedPromptTokens: number;
  estimatedPromptCostUsd: number | null;
}

export interface FrozenRecordedSessionEvalTraceModeSummary {
  mode: ModeName;
  turnCount: number;
  compileOkCount: number;
  phraseHitCount: number;
  phraseCount: number;
  qualityScore: number;
  selectedContextBlockCount: number;
  selectedContextChars: number;
  estimatedPromptTokens: number;
  estimatedPromptCostUsd: number | null;
}

export interface FrozenRecordedSessionEvalTraceResult {
  traceId: string;
  traceHash: string;
  tracePath: string;
  relativeTracePath: string;
  outputDir: string;
  status: "pass" | "fail";
  validationOk: boolean | null;
  validationErrors: string[];
  winnerMode: string | null;
  modeSummaries: FrozenRecordedSessionEvalTraceModeSummary[];
  error: string | null;
}

export interface FrozenRecordedSessionEvalGateReportV1 {
  contract: typeof FROZEN_RECORDED_SESSION_EVAL_REPORT_CONTRACT;
  status: GateStatus;
  generatedAt: string;
  repoRoot: string;
  gitSha: string;
  manifestPath: string;
  manifestId: string | null;
  outputDir: string;
  traceCount: number;
  expectedTraceCount: number | null;
  notes: string[];
  assumptions: string[];
  thresholds: FrozenRecordedSessionEvalThresholds;
  pricingTable: {
    version: string | null;
    path: string;
    charsPerToken: number;
    promptPriceUsdPer1mTokens: number;
  };
  issues: string[];
  checks: FrozenRecordedSessionEvalGateCheck[];
  modeSummaries: FrozenRecordedSessionEvalModeSummary[];
  qualityAdjustedPromptSavings: {
    baselineMode: "graph_prior_only";
    candidateMode: "learned_route";
    baselineQualityScore: number | null;
    candidateQualityScore: number | null;
    noBrainQualityScore: number | null;
    baselineQualityGain: number | null;
    candidateQualityGain: number | null;
    baselinePromptCostUsd: number | null;
    candidatePromptCostUsd: number | null;
    rawPromptSavingsUsd: number | null;
    qualityRetention: number | null;
    baselineEquivalentCandidatePromptCostUsd: number | null;
    qualityAdjustedPromptSavingsUsd: number | null;
  };
  traceResults: FrozenRecordedSessionEvalTraceResult[];
}

export interface FrozenRecordedSessionEvalGateDescriptor {
  reportPath: string;
  summaryPath: string;
  sourceManifestPath: string | null;
  report: FrozenRecordedSessionEvalGateReportV1;
}

export interface RunFrozenRecordedSessionEvalGateInput {
  manifestPath?: string;
  outputDir?: string;
  scratchRootDir?: string;
  thresholds?: Partial<FrozenRecordedSessionEvalThresholds>;
}

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
  manifestId: string | null;
  expectedTraceCount: number | null;
  thresholds: Partial<FrozenRecordedSessionEvalThresholds> | undefined;
  notes: string[];
  traces: LoadedTraceInput[];
  issues: string[];
}

function usage() {
  process.stderr.write(
    [
      "Usage: tsx scripts/run-frozen-recorded-session-eval-gate.ts [options]",
      "",
      "Options:",
      `  --manifest <path>                         Manifest path. Defaults to ${DEFAULT_FROZEN_RECORDED_SESSION_EVAL_MANIFEST_PATH}`,
      `                                           Canonical contract: ${CANONICAL_RECORDED_SESSION_TRACE_SET_MANIFEST_CONTRACT}`,
      "  --output-dir <path>                       Output root for gate artifacts.",
      "  --scratch-root-dir <path>                 Scratch parent for per-trace replay runs.",
      `  --max-quality-regression <number>         Max learned_route quality drop vs graph_prior_only. Default ${DEFAULT_THRESHOLDS.maxQualityRegression}`,
      `  --min-no-brain-uplift <number>            Min graph_prior_only quality uplift over no_brain. Default ${DEFAULT_THRESHOLDS.minGraphPriorOnlyQualityVsNoBrain}`,
      `  --min-quality-adjusted-prompt-savings-usd <number>  Min quality-adjusted prompt savings signal. Default ${DEFAULT_THRESHOLDS.minQualityAdjustedPromptSavingsUsd}`,
      "  --help                                    Show this help.",
      "",
      "Outputs:",
      `  ${FROZEN_RECORDED_SESSION_EVAL_BUNDLE_LAYOUT.report}`,
      `  ${FROZEN_RECORDED_SESSION_EVAL_BUNDLE_LAYOUT.summary}`,
      `  ${FROZEN_RECORDED_SESSION_EVAL_BUNDLE_LAYOUT.sourceManifest} (when the source manifest exists)`,
      `  ${FROZEN_RECORDED_SESSION_EVAL_BUNDLE_LAYOUT.traceDir}/<trace-id>/... per-trace replay proof bundles`,
    ].join("\n") + "\n",
  );
}

function normalizeCliString(value: string | undefined): string | null {
  if (typeof value !== "string") {
    return null;
  }
  const trimmed = value.trim();
  return trimmed.length === 0 ? null : trimmed;
}

function parseNumericArg(value: string | null, fieldName: string): number | null {
  if (value === null) {
    return null;
  }
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    throw new Error(`${fieldName} must be a finite number`);
  }
  return numeric;
}

function round(value: number, places = 4): number {
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

function buildQualityScore(turnCount: number, compileOkCount: number, phraseHitCount: number, phraseCount: number): number {
  if (turnCount === 0) {
    return 0;
  }
  const compileScore = (compileOkCount / turnCount) * 40;
  const phraseScore = phraseCount === 0 ? 60 : (phraseHitCount / phraseCount) * 60;
  return round(Math.min(100, compileScore + phraseScore), 4);
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
    "frozen-recorded-session-eval",
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
    path: path.relative(repoRoot, pricingTablePath).split(path.sep).join("/"),
    charsPerToken,
    promptPriceUsdPer1mTokens,
  };
}

function writeJson(filePath: string, payload: unknown): void {
  writeFileSync(filePath, `${canonicalJson(payload)}\n`, "utf8");
}

function writeText(filePath: string, value: string): void {
  writeFileSync(filePath, value.endsWith("\n") ? value : `${value}\n`, "utf8");
}

function toErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function mergeThresholds(
  manifestThresholds: Partial<FrozenRecordedSessionEvalThresholds> | undefined,
  cliThresholds: Partial<FrozenRecordedSessionEvalThresholds> | undefined,
): FrozenRecordedSessionEvalThresholds {
  return {
    maxQualityRegression: cliThresholds?.maxQualityRegression ?? manifestThresholds?.maxQualityRegression ?? DEFAULT_THRESHOLDS.maxQualityRegression,
    minGraphPriorOnlyQualityVsNoBrain: cliThresholds?.minGraphPriorOnlyQualityVsNoBrain
      ?? manifestThresholds?.minGraphPriorOnlyQualityVsNoBrain
      ?? DEFAULT_THRESHOLDS.minGraphPriorOnlyQualityVsNoBrain,
    minQualityAdjustedPromptSavingsUsd: cliThresholds?.minQualityAdjustedPromptSavingsUsd
      ?? manifestThresholds?.minQualityAdjustedPromptSavingsUsd
      ?? DEFAULT_THRESHOLDS.minQualityAdjustedPromptSavingsUsd,
  };
}

function toObjectRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
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

  if ((trace as { contract?: string }).contract !== RECORDED_SESSION_TRACE_CONTRACT) {
    params.issues.push(`trace at ${params.tracePath} must use contract ${RECORDED_SESSION_TRACE_CONTRACT}`);
    return null;
  }

  const traceId = typeof trace.traceId === "string" && trace.traceId.trim().length > 0 ? trace.traceId : null;
  if (traceId === null) {
    params.issues.push(`trace at ${params.tracePath} is missing traceId`);
    return null;
  }
  if (params.expectedTraceId && params.expectedTraceId !== traceId) {
    params.issues.push(`manifest traceId mismatch for ${params.tracePath}: expected ${params.expectedTraceId}, received ${traceId}`);
  }
  if (params.seenTraceIds.has(traceId)) {
    params.issues.push(`duplicate traceId in manifest: ${traceId}`);
    return null;
  }
  params.seenTraceIds.add(traceId);

  const traceHash = checksumJsonPayload(trace);
  if (params.expectedTraceHash && params.expectedTraceHash !== traceHash) {
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
    issues.push(`manifest missing at ${manifestPath}`);
    return {
      manifest: null,
      manifestId: null,
      expectedTraceCount: null,
      thresholds: undefined,
      notes: [],
      traces: [],
      issues,
    };
  }

  let manifest: Record<string, unknown> | null = null;
  try {
    manifest = JSON.parse(readFileSync(manifestPath, "utf8")) as Record<string, unknown>;
  } catch (error) {
    issues.push(`manifest is not valid JSON: ${toErrorMessage(error)}`);
    return {
      manifest: null,
      manifestId: null,
      expectedTraceCount: null,
      thresholds: undefined,
      notes: [],
      traces: [],
      issues,
    };
  }

  const manifestRecord = toObjectRecord(manifest);
  if (!manifestRecord) {
    issues.push("manifest must be a JSON object");
    return {
      manifest: null,
      manifestId: null,
      expectedTraceCount: null,
      thresholds: undefined,
      notes: [],
      traces: [],
      issues,
    };
  }

  const resolvedManifestDir = path.dirname(manifestPath);
  const seenTraceIds = new Set<string>();
  const traces: LoadedTraceInput[] = [];
  const notes: string[] = [];
  let manifestId: string | null = null;
  let expectedTraceCount: number | null = null;
  let thresholds: Partial<FrozenRecordedSessionEvalThresholds> | undefined;

  if (isFrozenRecordedSessionEvalManifest(manifestRecord)) {
    manifestId = normalizeCliString(manifestRecord.manifestId) ?? null;
    if (manifestId === null) {
      issues.push("manifestId is required");
    }
    if (!Array.isArray(manifestRecord.traces)) {
      issues.push("manifest traces must be an array");
      return {
        manifest: manifestRecord,
        manifestId,
        expectedTraceCount,
        thresholds,
        notes,
        traces,
        issues,
      };
    }
    if (manifestRecord.traces.length === 0) {
      issues.push("manifest declares zero traces");
    }
    if (
      manifestRecord.expectedTraceCount !== undefined
      && (!Number.isInteger(manifestRecord.expectedTraceCount) || manifestRecord.expectedTraceCount < 1)
    ) {
      issues.push("expectedTraceCount must be a positive integer when provided");
    }
    expectedTraceCount = Number.isInteger(manifestRecord.expectedTraceCount)
      ? (manifestRecord.expectedTraceCount ?? null)
      : null;
    if (
      expectedTraceCount !== null
      && manifestRecord.traces.length !== expectedTraceCount
    ) {
      issues.push(`manifest expectedTraceCount=${expectedTraceCount} but found ${manifestRecord.traces.length} traces`);
    }
    thresholds = manifestRecord.thresholds;
    if (Array.isArray(manifestRecord.notes)) {
      notes.push(...manifestRecord.notes.map((note) => String(note)));
    }
    for (const [index, entry] of manifestRecord.traces.entries()) {
      if (typeof entry?.tracePath !== "string" || entry.tracePath.trim().length === 0) {
        issues.push(`traces[${index}].tracePath is required`);
        continue;
      }
      const tracePath = path.resolve(resolvedManifestDir, entry.tracePath);
      const loadedTrace = loadTraceInput({
        tracePath,
        relativeTracePath: path.relative(resolvedManifestDir, tracePath).split(path.sep).join("/"),
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
        manifestId,
        expectedTraceCount,
        thresholds,
        notes,
        traces,
        issues,
      };
    }
    if (manifestRecord.entries.length === 0) {
      issues.push("canonical manifest declares zero traces");
    }
    expectedTraceCount = Number.isInteger(manifestRecord.traceCount)
      ? (manifestRecord.traceCount ?? null)
      : null;
    if (
      expectedTraceCount !== null
      && manifestRecord.entries.length !== expectedTraceCount
    ) {
      issues.push(`manifest traceCount=${expectedTraceCount} but found ${manifestRecord.entries.length} entries`);
    }
    const realTraceCoverageSummary = normalizeCliString(
      toObjectRecord(manifestRecord.realTraceCoverage)?.summary as string | undefined,
    );
    if (realTraceCoverageSummary) {
      notes.push(`truth boundary: ${realTraceCoverageSummary}`);
    }
    const redactionPolicySummary = normalizeCliString(
      toObjectRecord(manifestRecord.redactionPolicy)?.summary as string | undefined,
    );
    if (redactionPolicySummary) {
      notes.push(`redaction policy: ${redactionPolicySummary}`);
    }
    for (const [index, entry] of manifestRecord.entries.entries()) {
      const relativeTracePath = normalizeCliString(entry?.path);
      if (relativeTracePath === null) {
        issues.push(`entries[${index}].path is required`);
        continue;
      }
      const tracePath = path.resolve(resolvedManifestDir, relativeTracePath);
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
    manifestId,
    expectedTraceCount,
    thresholds,
    notes,
    traces,
    issues,
  };
}

function summarizeTraceMode(
  mode: {
    mode: string;
    summary: {
      compileOkCount: number;
      phraseHitCount: number;
      phraseCount: number;
      qualityScore: number;
    };
    turns: Array<{
      selectedContextIds?: string[];
      selectedContextTexts?: string[];
    }>;
  },
  pricingTable: PricingTable,
): FrozenRecordedSessionEvalTraceModeSummary {
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
    mode: mode.mode as ModeName,
    turnCount: mode.turns.length,
    compileOkCount: mode.summary.compileOkCount,
    phraseHitCount: mode.summary.phraseHitCount,
    phraseCount: mode.summary.phraseCount,
    qualityScore: mode.summary.qualityScore,
    selectedContextBlockCount,
    selectedContextChars,
    estimatedPromptTokens,
    estimatedPromptCostUsd: estimateUsdFromTokens(estimatedPromptTokens, pricingTable.promptPriceUsdPer1mTokens),
  };
}

function buildModeSummaries(
  traceResults: FrozenRecordedSessionEvalTraceResult[],
): FrozenRecordedSessionEvalModeSummary[] {
  return MODE_ORDER.map((mode) => {
    const entries = traceResults
      .filter((traceResult) => traceResult.status === "pass")
      .map((traceResult) => traceResult.modeSummaries.find((summary) => summary.mode === mode))
      .filter((entry): entry is FrozenRecordedSessionEvalTraceModeSummary => entry !== undefined);

    const traceCount = entries.length;
    const turnCount = entries.reduce((total, entry) => total + entry.turnCount, 0);
    const compileOkCount = entries.reduce((total, entry) => total + entry.compileOkCount, 0);
    const phraseHitCount = entries.reduce((total, entry) => total + entry.phraseHitCount, 0);
    const phraseCount = entries.reduce((total, entry) => total + entry.phraseCount, 0);
    const selectedContextBlockCount = entries.reduce((total, entry) => total + entry.selectedContextBlockCount, 0);
    const selectedContextChars = entries.reduce((total, entry) => total + entry.selectedContextChars, 0);
    const estimatedPromptTokens = entries.reduce((total, entry) => total + entry.estimatedPromptTokens, 0);
    const promptCosts = entries.map((entry) => entry.estimatedPromptCostUsd).filter((value): value is number => value !== null);
    return {
      mode,
      traceCount,
      turnCount,
      compileOkCount,
      compileOkRate: turnCount > 0 ? round(compileOkCount / turnCount, 4) : null,
      phraseHitCount,
      phraseCount,
      phraseHitRate: phraseCount > 0 ? round(phraseHitCount / phraseCount, 4) : null,
      qualityScore: buildQualityScore(turnCount, compileOkCount, phraseHitCount, phraseCount),
      qualityScoreMean: traceCount > 0 ? round(entries.reduce((total, entry) => total + entry.qualityScore, 0) / traceCount, 4) : null,
      selectedContextBlockCount,
      selectedContextChars,
      estimatedPromptTokens,
      estimatedPromptCostUsd: promptCosts.length === traceCount ? round(promptCosts.reduce((total, value) => total + value, 0), 6) : null,
    };
  });
}

function buildChecks(
  modeSummaries: FrozenRecordedSessionEvalModeSummary[],
  traceResults: FrozenRecordedSessionEvalTraceResult[],
  thresholds: FrozenRecordedSessionEvalThresholds,
): {
  checks: FrozenRecordedSessionEvalGateCheck[];
  qualityAdjustedPromptSavings: FrozenRecordedSessionEvalGateReportV1["qualityAdjustedPromptSavings"];
} {
  const traceValidationPassed = traceResults.every((traceResult) => traceResult.status === "pass" && traceResult.validationOk === true);
  const modeByName = new Map(modeSummaries.map((mode) => [mode.mode, mode]));
  const noBrain = modeByName.get("no_brain") ?? null;
  const graphPriorOnly = modeByName.get("graph_prior_only") ?? null;
  const learnedRoute = modeByName.get("learned_route") ?? null;

  const qualityRegression = graphPriorOnly !== null && learnedRoute !== null
    ? round(graphPriorOnly.qualityScore - learnedRoute.qualityScore, 4)
    : null;
  const noBrainUplift = graphPriorOnly !== null && noBrain !== null
    ? round(graphPriorOnly.qualityScore - noBrain.qualityScore, 4)
    : null;
  const rawPromptSavingsUsd = graphPriorOnly?.estimatedPromptCostUsd !== null && graphPriorOnly?.estimatedPromptCostUsd !== undefined
    && learnedRoute?.estimatedPromptCostUsd !== null && learnedRoute?.estimatedPromptCostUsd !== undefined
    ? round(graphPriorOnly.estimatedPromptCostUsd - learnedRoute.estimatedPromptCostUsd, 6)
    : null;
  const baselineQualityGain = noBrain !== null && graphPriorOnly !== null
    ? round(graphPriorOnly.qualityScore - noBrain.qualityScore, 4)
    : null;
  const candidateQualityGain = noBrain !== null && learnedRoute !== null
    ? round(learnedRoute.qualityScore - noBrain.qualityScore, 4)
    : null;
  const qualityRetention = baselineQualityGain !== null && candidateQualityGain !== null && baselineQualityGain > 0
    ? round(candidateQualityGain / baselineQualityGain, 6)
    : null;
  const baselineEquivalentCandidatePromptCostUsd = graphPriorOnly?.estimatedPromptCostUsd !== null
    && graphPriorOnly?.estimatedPromptCostUsd !== undefined
    && qualityRetention !== null
    ? round(graphPriorOnly.estimatedPromptCostUsd * qualityRetention, 6)
    : null;
  const qualityAdjustedPromptSavingsUsd = baselineEquivalentCandidatePromptCostUsd !== null
    && learnedRoute?.estimatedPromptCostUsd !== null
    && learnedRoute?.estimatedPromptCostUsd !== undefined
    ? round(baselineEquivalentCandidatePromptCostUsd - learnedRoute.estimatedPromptCostUsd, 6)
    : null;

  const checks: FrozenRecordedSessionEvalGateCheck[] = [
    {
      id: "trace_replay_proofs_valid",
      status: traceValidationPassed ? "pass" : "fail",
      summary: traceValidationPassed ? "all trace replay proof bundles validated" : "one or more trace replay proof bundles failed validation",
      detail: traceValidationPassed
        ? `${traceResults.length} trace bundles wrote and revalidated cleanly`
        : traceResults
          .filter((traceResult) => traceResult.status !== "pass" || traceResult.validationOk !== true)
          .map((traceResult) => `${traceResult.traceId}: ${traceResult.error ?? traceResult.validationErrors.join("; ")}`)
          .join(" | "),
      observed: {
        passingTraceCount: traceResults.filter((traceResult) => traceResult.status === "pass" && traceResult.validationOk === true).length,
        traceCount: traceResults.length,
      },
      threshold: {
        requiredPassingTraceCount: traceResults.length,
      },
    },
    {
      id: "learned_route_non_inferior_to_graph_prior_only",
      status: qualityRegression !== null && qualityRegression <= thresholds.maxQualityRegression ? "pass" : "fail",
      summary: qualityRegression !== null && qualityRegression <= thresholds.maxQualityRegression
        ? "learned_route quality stayed within the allowed regression margin"
        : "learned_route quality regressed too far versus graph_prior_only",
      detail: qualityRegression === null
        ? "quality regression could not be computed"
        : `graph_prior_only - learned_route = ${qualityRegression}`,
      observed: {
        graphPriorOnlyQualityScore: graphPriorOnly?.qualityScore ?? null,
        learnedRouteQualityScore: learnedRoute?.qualityScore ?? null,
        qualityRegression,
      },
      threshold: {
        maxQualityRegression: thresholds.maxQualityRegression,
      },
    },
    {
      id: "graph_prior_only_clears_no_brain_floor",
      status: noBrainUplift !== null && noBrainUplift >= thresholds.minGraphPriorOnlyQualityVsNoBrain ? "pass" : "fail",
      summary: noBrainUplift !== null && noBrainUplift >= thresholds.minGraphPriorOnlyQualityVsNoBrain
        ? "graph_prior_only clears the no_brain floor"
        : "graph_prior_only does not clear the no_brain floor",
      detail: noBrainUplift === null
        ? "quality floor could not be computed"
        : `graph_prior_only - no_brain = ${noBrainUplift}`,
      observed: {
        graphPriorOnlyQualityScore: graphPriorOnly?.qualityScore ?? null,
        noBrainQualityScore: noBrain?.qualityScore ?? null,
        qualityUpliftVsNoBrain: noBrainUplift,
      },
      threshold: {
        minGraphPriorOnlyQualityVsNoBrain: thresholds.minGraphPriorOnlyQualityVsNoBrain,
      },
    },
    {
      id: "quality_adjusted_prompt_savings_positive",
      status: qualityAdjustedPromptSavingsUsd !== null && qualityAdjustedPromptSavingsUsd > thresholds.minQualityAdjustedPromptSavingsUsd ? "pass" : "fail",
      summary: qualityAdjustedPromptSavingsUsd !== null && qualityAdjustedPromptSavingsUsd > thresholds.minQualityAdjustedPromptSavingsUsd
        ? "quality-adjusted prompt savings signal is positive"
        : "quality-adjusted prompt savings signal is not positive",
      detail: qualityAdjustedPromptSavingsUsd === null
        ? "quality-adjusted prompt savings could not be computed"
        : `quality-adjusted prompt savings = ${qualityAdjustedPromptSavingsUsd}`,
      observed: {
        baselineQualityGain,
        candidateQualityGain,
        graphPriorOnlyPromptCostUsd: graphPriorOnly?.estimatedPromptCostUsd ?? null,
        learnedRoutePromptCostUsd: learnedRoute?.estimatedPromptCostUsd ?? null,
        rawPromptSavingsUsd,
        qualityRetention,
        baselineEquivalentCandidatePromptCostUsd,
        qualityAdjustedPromptSavingsUsd,
      },
      threshold: {
        minQualityAdjustedPromptSavingsUsd: thresholds.minQualityAdjustedPromptSavingsUsd,
      },
    },
  ];

  return {
    checks,
    qualityAdjustedPromptSavings: {
      baselineMode: "graph_prior_only",
      candidateMode: "learned_route",
      baselineQualityScore: graphPriorOnly?.qualityScore ?? null,
      candidateQualityScore: learnedRoute?.qualityScore ?? null,
      noBrainQualityScore: noBrain?.qualityScore ?? null,
      baselineQualityGain,
      candidateQualityGain,
      baselinePromptCostUsd: graphPriorOnly?.estimatedPromptCostUsd ?? null,
      candidatePromptCostUsd: learnedRoute?.estimatedPromptCostUsd ?? null,
      rawPromptSavingsUsd,
      qualityRetention,
      baselineEquivalentCandidatePromptCostUsd,
      qualityAdjustedPromptSavingsUsd,
    },
  };
}

function buildSummary(report: FrozenRecordedSessionEvalGateReportV1): string {
  const checkRows = report.checks.map((check) => {
    const observedText = Object.entries(check.observed)
      .map(([key, value]) => `${key}=${value === null ? "null" : String(value)}`)
      .join(", ");
    const thresholdText = Object.entries(check.threshold)
      .map(([key, value]) => `${key}=${value === null ? "null" : String(value)}`)
      .join(", ");
    return `| ${check.id} | ${check.status} | ${observedText} | ${thresholdText} |`;
  });
  const modeRows = report.modeSummaries.map((mode) => `| ${mode.mode} | ${mode.traceCount} | ${mode.turnCount} | ${mode.qualityScore} | ${mode.compileOkRate ?? "null"} | ${mode.phraseHitRate ?? "null"} | ${mode.estimatedPromptTokens} | ${mode.estimatedPromptCostUsd ?? "null"} |`);
  const traceRows = report.traceResults.map((trace) => `| ${trace.traceId} | ${trace.status} | ${trace.validationOk ?? "null"} | ${trace.winnerMode ?? "null"} | ${trace.error ?? "none"} |`);
  return [
    "# Frozen Recorded Session Eval Gate",
    "",
    `- status: \`${report.status}\``,
    `- manifest path: \`${report.manifestPath}\``,
    `- manifest id: \`${report.manifestId ?? "null"}\``,
    `- git sha: \`${report.gitSha}\``,
    `- traces: ${report.traceCount}${report.expectedTraceCount === null ? "" : `/${report.expectedTraceCount}`}`,
    `- pricing table: \`${report.pricingTable.version ?? "null"}\` from \`${report.pricingTable.path}\``,
    `- quality-adjusted prompt savings usd: ${report.qualityAdjustedPromptSavings.qualityAdjustedPromptSavingsUsd ?? "null"}`,
    "",
    "## Checks",
    "| check | status | observed | threshold |",
    "| --- | --- | --- | --- |",
    ...(checkRows.length > 0 ? checkRows : ["| none | fail | none | none |"]),
    "",
    "## Modes",
    "| mode | traces | turns | quality | compile ok rate | phrase hit rate | prompt tokens | prompt cost usd |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ...(modeRows.length > 0 ? modeRows : ["| none | 0 | 0 | 0 | null | null | 0 | null |"]),
    "",
    "## Traces",
    "| trace | status | validation ok | winner | error |",
    "| --- | --- | --- | --- | --- |",
    ...(traceRows.length > 0 ? traceRows : ["| none | blocked | null | null | none |"]),
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

function writeReportArtifacts(
  outputDir: string,
  report: FrozenRecordedSessionEvalGateReportV1,
  manifest: Record<string, unknown> | null,
): FrozenRecordedSessionEvalGateDescriptor {
  mkdirSync(outputDir, { recursive: true });
  const reportPath = path.join(outputDir, FROZEN_RECORDED_SESSION_EVAL_BUNDLE_LAYOUT.report);
  const summaryPath = path.join(outputDir, FROZEN_RECORDED_SESSION_EVAL_BUNDLE_LAYOUT.summary);
  writeJson(reportPath, report);
  writeText(summaryPath, buildSummary(report));
  let sourceManifestPath: string | null = null;
  if (manifest !== null) {
    sourceManifestPath = path.join(outputDir, FROZEN_RECORDED_SESSION_EVAL_BUNDLE_LAYOUT.sourceManifest);
    writeJson(sourceManifestPath, manifest);
  }
  return {
    reportPath,
    summaryPath,
    sourceManifestPath,
    report,
  };
}

export function runFrozenRecordedSessionEvalGate(
  input: RunFrozenRecordedSessionEvalGateInput = {},
): FrozenRecordedSessionEvalGateDescriptor {
  const manifestPath = path.resolve(input.manifestPath ?? DEFAULT_FROZEN_RECORDED_SESSION_EVAL_MANIFEST_PATH);
  const outputDir = path.resolve(input.outputDir ?? defaultOutputDir(manifestPath));
  const traceDir = path.join(outputDir, FROZEN_RECORDED_SESSION_EVAL_BUNDLE_LAYOUT.traceDir);
  const manifestLoad = loadManifestInputs(manifestPath);
  const pricingTable = loadPricingTable();
  const thresholds = mergeThresholds(manifestLoad.thresholds, input.thresholds);
  const notes = [
    `default manifest path is ${DEFAULT_FROZEN_RECORDED_SESSION_EVAL_MANIFEST_PATH}`,
    "per-trace replay bundles are produced with writeRecordedSessionReplayProofBundle so each trace still runs the real four-mode compile/runtime path",
    ...manifestLoad.notes,
  ];
  const assumptions = [
    `canonical frozen set contract is ${CANONICAL_RECORDED_SESSION_TRACE_SET_MANIFEST_CONTRACT}`,
    "manifest trace paths resolve relative to the manifest file location",
    "traceHash, when present in the manifest, is checksumJsonPayload(trace-json)",
    "quality-adjusted prompt savings use prompt-context cost proxy from selected context chars because replay proof bundles do not model live completion costs",
    "quality adjustment scales graph_prior_only prompt cost to the learned_route quality gain above the no_brain floor",
    "graph_prior_only is the non-inferiority baseline and no_brain is the floor anchor",
  ];

  mkdirSync(outputDir, { recursive: true });
  rmSync(traceDir, { recursive: true, force: true });

  if (manifestLoad.issues.length > 0 || manifestLoad.manifest === null) {
    const report: FrozenRecordedSessionEvalGateReportV1 = {
      contract: FROZEN_RECORDED_SESSION_EVAL_REPORT_CONTRACT,
      status: "blocked",
      generatedAt: new Date().toISOString(),
      repoRoot,
      gitSha: gitShaOrUnknown(),
      manifestPath,
      manifestId: manifestLoad.manifestId,
      outputDir,
      traceCount: manifestLoad.traces.length,
      expectedTraceCount: manifestLoad.expectedTraceCount,
      notes,
      assumptions,
      thresholds,
      pricingTable,
      issues: manifestLoad.issues,
      checks: [],
      modeSummaries: MODE_ORDER.map((mode) => ({
        mode,
        traceCount: 0,
        turnCount: 0,
        compileOkCount: 0,
        compileOkRate: null,
        phraseHitCount: 0,
        phraseCount: 0,
        phraseHitRate: null,
        qualityScore: 0,
        qualityScoreMean: null,
        selectedContextBlockCount: 0,
        selectedContextChars: 0,
        estimatedPromptTokens: 0,
        estimatedPromptCostUsd: null,
      })),
      qualityAdjustedPromptSavings: {
        baselineMode: "graph_prior_only",
        candidateMode: "learned_route",
        baselineQualityScore: null,
        candidateQualityScore: null,
        noBrainQualityScore: null,
        baselineQualityGain: null,
        candidateQualityGain: null,
        baselinePromptCostUsd: null,
        candidatePromptCostUsd: null,
        rawPromptSavingsUsd: null,
        qualityRetention: null,
        baselineEquivalentCandidatePromptCostUsd: null,
        qualityAdjustedPromptSavingsUsd: null,
      },
      traceResults: [],
    };
    return writeReportArtifacts(outputDir, report, manifestLoad.manifest);
  }

  mkdirSync(traceDir, { recursive: true });
  const traceResults: FrozenRecordedSessionEvalTraceResult[] = [];
  for (const traceInput of manifestLoad.traces) {
    const traceOutputDir = path.join(traceDir, traceInput.traceId);
    try {
      const descriptor = writeRecordedSessionReplayProofBundle({
        rootDir: traceOutputDir,
        trace: traceInput.trace,
        ...(input.scratchRootDir ? { scratchRootDir: path.resolve(input.scratchRootDir) } : {}),
      });
      const validation = validateRecordedSessionReplayProofBundle(traceOutputDir);
      traceResults.push({
        traceId: traceInput.traceId,
        traceHash: traceInput.traceHash,
        tracePath: traceInput.tracePath,
        relativeTracePath: traceInput.relativeTracePath,
        outputDir: traceOutputDir,
        status: validation.ok ? "pass" : "fail",
        validationOk: validation.ok,
        validationErrors: [...validation.errors],
        winnerMode: descriptor.bundle.summary.winnerMode,
        modeSummaries: descriptor.bundle.modes.map((mode) => summarizeTraceMode(mode, pricingTable)),
        error: null,
      });
    } catch (error) {
      traceResults.push({
        traceId: traceInput.traceId,
        traceHash: traceInput.traceHash,
        tracePath: traceInput.tracePath,
        relativeTracePath: traceInput.relativeTracePath,
        outputDir: traceOutputDir,
        status: "fail",
        validationOk: null,
        validationErrors: [],
        winnerMode: null,
        modeSummaries: [],
        error: toErrorMessage(error),
      });
    }
  }

  const modeSummaries = buildModeSummaries(traceResults);
  const { checks, qualityAdjustedPromptSavings } = buildChecks(modeSummaries, traceResults, thresholds);
  const issues = traceResults
    .filter((traceResult) => traceResult.status !== "pass")
    .flatMap((traceResult) => {
      const parts = [traceResult.error, ...traceResult.validationErrors].filter((value): value is string => typeof value === "string" && value.length > 0);
      return parts.length > 0 ? [`${traceResult.traceId}: ${parts.join("; ")}`] : [];
    });
  const status: GateStatus = checks.every((check) => check.status === "pass") ? "pass" : "fail";
  const report: FrozenRecordedSessionEvalGateReportV1 = {
    contract: FROZEN_RECORDED_SESSION_EVAL_REPORT_CONTRACT,
    status,
    generatedAt: new Date().toISOString(),
    repoRoot,
    gitSha: gitShaOrUnknown(),
    manifestPath,
    manifestId: manifestLoad.manifestId,
    outputDir,
    traceCount: traceResults.length,
    expectedTraceCount: manifestLoad.expectedTraceCount,
    notes,
    assumptions,
    thresholds,
    pricingTable,
    issues,
    checks,
    modeSummaries,
    qualityAdjustedPromptSavings,
    traceResults,
  };
  return writeReportArtifacts(outputDir, report, manifestLoad.manifest);
}

function parseArgs(argv: string[]): RunFrozenRecordedSessionEvalGateInput {
  const parsed: RunFrozenRecordedSessionEvalGateInput & {
    maxQualityRegression?: number;
    minGraphPriorOnlyQualityVsNoBrain?: number;
    minQualityAdjustedPromptSavingsUsd?: number;
  } = {};

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    switch (arg) {
      case "--manifest":
        parsed.manifestPath = normalizeCliString(argv[index + 1]) ?? undefined;
        index += 1;
        break;
      case "--output-dir":
        parsed.outputDir = normalizeCliString(argv[index + 1]) ?? undefined;
        index += 1;
        break;
      case "--scratch-root-dir":
        parsed.scratchRootDir = normalizeCliString(argv[index + 1]) ?? undefined;
        index += 1;
        break;
      case "--max-quality-regression":
        parsed.maxQualityRegression = parseNumericArg(normalizeCliString(argv[index + 1]), "--max-quality-regression") ?? undefined;
        index += 1;
        break;
      case "--min-no-brain-uplift":
        parsed.minGraphPriorOnlyQualityVsNoBrain = parseNumericArg(normalizeCliString(argv[index + 1]), "--min-no-brain-uplift") ?? undefined;
        index += 1;
        break;
      case "--min-quality-adjusted-prompt-savings-usd":
        parsed.minQualityAdjustedPromptSavingsUsd = parseNumericArg(
          normalizeCliString(argv[index + 1]),
          "--min-quality-adjusted-prompt-savings-usd",
        ) ?? undefined;
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

  return {
    manifestPath: parsed.manifestPath,
    outputDir: parsed.outputDir,
    scratchRootDir: parsed.scratchRootDir,
    thresholds: {
      ...(parsed.maxQualityRegression === undefined ? {} : { maxQualityRegression: parsed.maxQualityRegression }),
      ...(parsed.minGraphPriorOnlyQualityVsNoBrain === undefined
        ? {}
        : { minGraphPriorOnlyQualityVsNoBrain: parsed.minGraphPriorOnlyQualityVsNoBrain }),
      ...(parsed.minQualityAdjustedPromptSavingsUsd === undefined
        ? {}
        : { minQualityAdjustedPromptSavingsUsd: parsed.minQualityAdjustedPromptSavingsUsd }),
    },
  };
}

function printCliSummary(descriptor: FrozenRecordedSessionEvalGateDescriptor): void {
  const lines = [
    `Frozen recorded session eval gate: ${descriptor.report.status}`,
    `manifestPath: ${descriptor.report.manifestPath}`,
    `manifestId: ${descriptor.report.manifestId ?? "null"}`,
    `traceCount: ${descriptor.report.traceCount}${descriptor.report.expectedTraceCount === null ? "" : `/${descriptor.report.expectedTraceCount}`}`,
    `report: ${descriptor.reportPath}`,
    `summary: ${descriptor.summaryPath}`,
  ];
  for (const check of descriptor.report.checks) {
    lines.push(`${check.id}: ${check.status}`);
  }
  process.stdout.write(`${lines.join("\n")}\n`);
}

function exitCodeForStatus(status: GateStatus): number {
  switch (status) {
    case "pass":
      return 0;
    case "blocked":
      return 2;
    case "fail":
    default:
      return 1;
  }
}

function main(): void {
  const args = parseArgs(process.argv.slice(2));
  const descriptor = runFrozenRecordedSessionEvalGate(args);
  printCliSummary(descriptor);
  process.exitCode = exitCodeForStatus(descriptor.report.status);
}

if (import.meta.url === `file://${process.argv[1]}`) {
  try {
    main();
  } catch (error) {
    process.stderr.write(`${toErrorMessage(error)}\n`);
    process.exit(1);
  }
}
