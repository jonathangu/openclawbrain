import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  validateRecordedSessionReplayProofBundle,
  writeRecordedSessionReplayProofBundle,
  type RecordedSessionReplayMode,
  type RecordedSessionReplayModeReportV1,
  type RecordedSessionReplayProofBundleDescriptorV1,
  type RecordedSessionReplayProofBundleValidationV1,
  type RecordedSessionReplayTurnReportV1,
  type RecordedSessionTraceTurnV1,
  type RecordedSessionTraceV1,
} from "../packages/cli/dist/src/index.js";

export const RECORDED_SESSION_REPLAY_PROOF_LANE_MODE_ORDER = [
  "no_brain",
  "vector_only",
  "graph_prior_only",
  "learned_route",
] as const satisfies readonly RecordedSessionReplayMode[];

export const RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT = {
  laneDir: "_lane",
  readme: "README.md",
  summary: "summary.md",
  closeout: "closeout.json",
  index: "index.json",
  summaryTables: "summary-tables.json",
  pairwiseDeltas: "pairwise-deltas.json",
  winRateMatrix: "win-rate-matrix.json",
  workedTraces: "worked-traces.md",
  generationReport: "generation-report.json",
} as const;

const DEFAULT_WORKED_TRACE_LIMIT = 8;
const DEFAULT_WORKED_TURN_LIMIT = 2;
const RECORDED_SESSION_REPLAY_PROOF_LANE_SOURCE_MANIFEST_CONTRACT = "recorded_session_replay_proof_lane_source_manifest.v1";
const RECORDED_SESSION_REPLAY_PROOF_LANE_CLOSEOUT_CONTRACT = "recorded_session_replay_proof_lane_closeout.v1";
const RECORDED_SESSION_REPLAY_PROOF_LANE_SUMMARY_TABLES_CONTRACT = "recorded_session_replay_proof_lane_summary_tables.v1";
const RECORDED_SESSION_REPLAY_PROOF_LANE_PAIRWISE_DELTAS_CONTRACT = "recorded_session_replay_proof_lane_pairwise_deltas.v1";
const RECORDED_SESSION_REPLAY_PROOF_LANE_WIN_RATE_MATRIX_CONTRACT = "recorded_session_replay_proof_lane_win_rate_matrix.v1";
const RECORDED_SESSION_REPLAY_PROOF_LANE_INDEX_CONTRACT = "recorded_session_replay_proof_lane_index.v1";
const RECORDED_SESSION_REPLAY_PROOF_LANE_GENERATION_REPORT_CONTRACT = "recorded_session_replay_proof_lane_generation_report.v1";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..");

type ReplayLaneMode = (typeof RECORDED_SESSION_REPLAY_PROOF_LANE_MODE_ORDER)[number];
type ReplayLaneRelation = "better" | "tied" | "worse";

interface ReplayLanePricingTable {
  version: string | null;
  path: string;
  charsPerToken: number;
  promptPriceUsdPer1mTokens: number;
}

export interface RecordedSessionReplayProofLaneSourceManifestV1 {
  contract: typeof RECORDED_SESSION_REPLAY_PROOF_LANE_SOURCE_MANIFEST_CONTRACT;
  provided: boolean;
  manifestId: string | null;
  manifestContract: string | null;
  manifestDigest: string | null;
}

export interface RecordedSessionReplayProofLaneTraceInputV1 {
  trace: RecordedSessionTraceV1;
  tracePath?: string | null;
  bundleDir?: string | null;
}

export interface WriteRecordedSessionReplayProofLaneInputV1 {
  artifactRoot: string;
  traces: RecordedSessionReplayProofLaneTraceInputV1[];
  scratchRootDir?: string | null;
  workedTraceLimit?: number | null;
  sourceManifestPath?: string | null;
  assumptions?: readonly string[] | null;
}

interface RecordedSessionReplayProofLaneModeTraceRowV1 {
  mode: ReplayLaneMode;
  qualityScore: number;
  compileOkCount: number;
  turnCount: number;
  phraseHitCount: number;
  phraseCount: number;
  promotionCount: number;
  usedLearnedRouteTurnCount: number;
  activationObservedTurnCount: number;
  activationTakenTurnCount: number;
  selectedContextBlockCount: number;
  selectedContextCharCount: number;
  estimatedPromptTokenCount: number;
  estimatedPromptCostUsd: number | null;
  totalLatencyMs: number;
  totalRouteSelectionLatencyMs: number;
  totalPromptAssemblyLatencyMs: number;
  fallbackToStaticContextTurnCount: number;
  hardRequirementViolatedTurnCount: number;
  warningCount: number;
  scoreHash: string;
}

interface RecordedSessionReplayProofLaneTurnModeRowV1 {
  mode: ReplayLaneMode;
  phase: RecordedSessionReplayTurnReportV1["phase"];
  qualityScore: number;
  compileOk: boolean;
  phraseHitCount: number;
  phraseCount: number;
  usedLearnedRouteFn: boolean;
  promoted: boolean;
  modeEffective: string | null;
  activePackId: string | null;
  routerIdentity: string | null;
  selectionDigest: string | null;
  activationTaken: boolean | null;
  activationSource: string | null;
  activationReason: string | null;
  activationConfidence: string | null;
  fallbackToStaticContext: boolean;
  hardRequirementViolated: boolean;
  selectedContextBlockCount: number;
  selectedContextCharCount: number;
  estimatedPromptTokens: number | null;
  estimatedPromptCostUsd: number | null;
  timing: {
    totalMs: number | null;
    routeSelectionMs: number | null;
    promptAssemblyMs: number | null;
  };
  selectedContextPreview: string | null;
}

interface RecordedSessionReplayProofLaneTurnSummaryRowV1 {
  traceId: string;
  bundleDir: string;
  turnId: string;
  userMessagePreview: string;
  expectedContextPhrases: string[];
  feedbackKinds: string[];
  scoreSpread: number;
  topModes: ReplayLaneMode[];
  candidateRelationVsBaseline: ReplayLaneRelation | null;
  candidateRelationVsFloor: ReplayLaneRelation | null;
  candidateTieOrBetterVsBaseline: boolean | null;
  candidateRegressionVsBaseline: boolean | null;
  candidateRegressionVsFloor: boolean | null;
  ranking: Array<{
    mode: ReplayLaneMode;
    qualityScore: number;
  }>;
  modes: RecordedSessionReplayProofLaneTurnModeRowV1[];
}

interface RecordedSessionReplayProofLaneTraceSummaryRowV1 {
  traceId: string;
  bundleDir: string;
  winnerMode: ReplayLaneMode | null;
  topScoreModes: ReplayLaneMode[];
  scoreSpread: number;
  validationOk: boolean;
  candidateRelationVsBaseline: ReplayLaneRelation | null;
  candidateRelationVsFloor: ReplayLaneRelation | null;
  candidateTieOrBetterVsBaseline: boolean | null;
  candidateRegressionVsBaseline: boolean | null;
  candidateRegressionVsFloor: boolean | null;
  bundleHash: string;
  scoreHash: string;
  modes: RecordedSessionReplayProofLaneModeTraceRowV1[];
}

interface RecordedSessionReplayProofLaneModeSummaryRowV1 {
  mode: ReplayLaneMode;
  traceCount: number;
  rankedWinnerCount: number;
  sharedTopScoreTraceCount: number;
  meanQualityScore: number | null;
  totalCompileOkCount: number;
  totalTurnCount: number;
  totalPhraseHitCount: number;
  totalPhraseCount: number;
  totalPromotionCount: number;
  totalUsedLearnedRouteTurnCount: number;
  totalActivationObservedTurnCount: number;
  totalActivationTakenTurnCount: number;
  totalSelectedContextBlockCount: number;
  totalSelectedContextCharCount: number;
  totalEstimatedPromptTokenCount: number;
  totalEstimatedPromptCostUsd: number | null;
  totalLatencyMs: number;
  totalRouteSelectionLatencyMs: number;
  totalPromptAssemblyLatencyMs: number;
  totalFallbackToStaticContextTurnCount: number;
  totalHardRequirementViolatedTurnCount: number;
  totalWarningCount: number;
}

export interface RecordedSessionReplayProofLaneCountRateV1 {
  count: number;
  rate: number | null;
  totalCount: number;
}

export interface RecordedSessionReplayProofLaneOutcomeBreakdownV1 {
  betterCount: number;
  tiedCount: number;
  worseCount: number;
  betterRate: number | null;
  tieRate: number | null;
  worseRate: number | null;
  totalCount: number;
}

export interface RecordedSessionReplayProofLaneRequiredContextRecallV1 {
  available: boolean;
  candidateMode: ReplayLaneMode;
  baselineMode: ReplayLaneMode;
  candidatePhraseHitCount: number | null;
  candidatePhraseCount: number | null;
  candidateRate: number | null;
  baselinePhraseHitCount: number | null;
  baselinePhraseCount: number | null;
  baselineRate: number | null;
  delta: number | null;
  summary: string;
}

export interface RecordedSessionReplayProofLaneCorrectionAbsorptionV1 {
  available: boolean;
  observedFeedbackTurnCount: number;
  observedNonApprovalFeedbackTurnCount: number;
  summary: string;
}

export interface RecordedSessionReplayProofLaneActivationPrecisionProxyV1 {
  available: boolean;
  activationCount: number;
  beneficialActivationCount: number;
  precision: number | null;
  activationDefinition: string;
  summary: string;
  limitations: string[];
}

export interface RecordedSessionReplayProofLaneActivationPrecisionV1 {
  available: boolean;
  observedTurnCount: number;
  activationCount: number;
  beneficialActivationCount: number;
  precision: number | null;
  sourceCounts: Array<{
    source: string;
    count: number;
  }>;
  summary: string;
  limitations: string[];
}

export interface RecordedSessionReplayProofLaneSuccessAdjustedEconomicsV1 {
  available: boolean;
  successUnit: "validated_trace" | null;
  candidateMode: ReplayLaneMode;
  baselineMode: ReplayLaneMode;
  successCount: number;
  candidateEstimatedPromptTokensPerSuccess: number | null;
  baselineEstimatedPromptTokensPerSuccess: number | null;
  candidateEstimatedPromptCostUsdPerSuccess: number | null;
  baselineEstimatedPromptCostUsdPerSuccess: number | null;
  promptTokenDeltaCandidateMinusBaseline: number | null;
  promptCostUsdDeltaCandidateMinusBaseline: number | null;
  candidateServePathLatencyMsPerSuccess: number | null;
  baselineServePathLatencyMsPerSuccess: number | null;
  servePathLatencyMsDeltaCandidateMinusBaseline: number | null;
  summary: string;
  limitations: string[];
}

export interface RecordedSessionReplayProofLaneFailOpenV1 {
  available: boolean;
  degradedTurnCount: number;
  acceptableDegradedTurnCount: number;
  catastrophicDegradedTurnCount: number;
  degradedTurnRate: number | null;
  acceptableDegradedTurnRate: number | null;
  catastrophicDegradedTurnRate: number | null;
  clipRate: number | null;
  failOpenRate: number | null;
  summary: string;
}

export interface RecordedSessionReplayProofLaneExplainableScorecardV1 {
  candidateMode: "learned_route";
  baselineMode: "graph_prior_only";
  floorMode: "no_brain";
  comparableTraceCount: number;
  comparableTurnCount: number;
  traceOutcomeVsBaseline: RecordedSessionReplayProofLaneOutcomeBreakdownV1;
  turnOutcomeVsBaseline: RecordedSessionReplayProofLaneOutcomeBreakdownV1;
  traceTieOrBetterVsBaseline: RecordedSessionReplayProofLaneCountRateV1;
  turnTieOrBetterVsBaseline: RecordedSessionReplayProofLaneCountRateV1;
  regressionVsBaseline: RecordedSessionReplayProofLaneCountRateV1;
  regressionVsFloor: RecordedSessionReplayProofLaneCountRateV1;
  criticalRegressionCount: number;
  requiredContextRecall: RecordedSessionReplayProofLaneRequiredContextRecallV1;
  correctionAbsorption: RecordedSessionReplayProofLaneCorrectionAbsorptionV1;
  activationPrecision: RecordedSessionReplayProofLaneActivationPrecisionV1;
  activationPrecisionProxy: RecordedSessionReplayProofLaneActivationPrecisionProxyV1;
  successAdjustedEconomics: RecordedSessionReplayProofLaneSuccessAdjustedEconomicsV1;
  failOpen: RecordedSessionReplayProofLaneFailOpenV1;
  diagnostics: {
    candidateMeanQualityScore: number | null;
    baselineMeanQualityScore: number | null;
    floorMeanQualityScore: number | null;
    candidateMinusBaselineMeanQualityScore: number | null;
    candidateMinusFloorMeanQualityScore: number | null;
    winnerModeCounts: Array<{
      mode: ReplayLaneMode;
      rankedWinnerCount: number;
      sharedTopScoreTraceCount: number;
    }>;
  };
}

export interface RecordedSessionReplayProofLaneSummaryTablesV1 {
  contract: typeof RECORDED_SESSION_REPLAY_PROOF_LANE_SUMMARY_TABLES_CONTRACT;
  sourceManifest: RecordedSessionReplayProofLaneSourceManifestV1;
  modeOrder: ReplayLaneMode[];
  requestedTraceCount: number;
  successfulTraceCount: number;
  failedTraceCount: number;
  scorecard: RecordedSessionReplayProofLaneExplainableScorecardV1;
  modes: RecordedSessionReplayProofLaneModeSummaryRowV1[];
  traces: RecordedSessionReplayProofLaneTraceSummaryRowV1[];
  turns: RecordedSessionReplayProofLaneTurnSummaryRowV1[];
}

interface RecordedSessionReplayProofLanePairwiseTraceDeltaV1 {
  traceId: string;
  bundleDir: string;
  bundleHash: string;
  scoreHash: string;
  qualityScoreDeltaLeftMinusRight: number;
  compileOkDeltaLeftMinusRight: number;
  phraseHitDeltaLeftMinusRight: number;
  promotionDeltaLeftMinusRight: number;
  turnWins: {
    left: number;
    right: number;
    ties: number;
  };
  maxTurnScoreSpread: number;
}

interface RecordedSessionReplayProofLanePairwiseRowV1 {
  leftMode: ReplayLaneMode;
  rightMode: ReplayLaneMode;
  traceWins: {
    left: number;
    right: number;
    ties: number;
  };
  traceWinRate: {
    left: number | null;
    right: number | null;
    ties: number | null;
  };
  turnWins: {
    left: number;
    right: number;
    ties: number;
  };
  turnWinRate: {
    left: number | null;
    right: number | null;
    ties: number | null;
  };
  aggregateDeltas: {
    qualityScoreDeltaLeftMinusRightSum: number;
    qualityScoreDeltaLeftMinusRightMean: number | null;
    compileOkDeltaLeftMinusRightSum: number;
    phraseHitDeltaLeftMinusRightSum: number;
    promotionDeltaLeftMinusRightSum: number;
  };
  traces: RecordedSessionReplayProofLanePairwiseTraceDeltaV1[];
}

export interface RecordedSessionReplayProofLanePairwiseDeltasV1 {
  contract: typeof RECORDED_SESSION_REPLAY_PROOF_LANE_PAIRWISE_DELTAS_CONTRACT;
  sourceManifest: RecordedSessionReplayProofLaneSourceManifestV1;
  modeOrder: ReplayLaneMode[];
  requestedTraceCount: number;
  successfulTraceCount: number;
  failedTraceCount: number;
  turnComparisonCount: number;
  pairs: RecordedSessionReplayProofLanePairwiseRowV1[];
}

interface RecordedSessionReplayProofLaneMatrixCellV1 {
  mode: ReplayLaneMode;
  wins: number;
  losses: number;
  ties: number;
  winRate: number | null;
  lossRate: number | null;
  tieRate: number | null;
}

interface RecordedSessionReplayProofLaneMatrixRowV1 {
  mode: ReplayLaneMode;
  cells: RecordedSessionReplayProofLaneMatrixCellV1[];
}

export interface RecordedSessionReplayProofLaneWinRateMatrixV1 {
  contract: typeof RECORDED_SESSION_REPLAY_PROOF_LANE_WIN_RATE_MATRIX_CONTRACT;
  sourceManifest: RecordedSessionReplayProofLaneSourceManifestV1;
  modeOrder: ReplayLaneMode[];
  requestedTraceCount: number;
  successfulTraceCount: number;
  failedTraceCount: number;
  traceComparisonCount: number;
  turnComparisonCount: number;
  traceMatrix: RecordedSessionReplayProofLaneMatrixRowV1[];
  turnMatrix: RecordedSessionReplayProofLaneMatrixRowV1[];
}

interface RecordedSessionReplayProofLaneBundleIndexRowV1 {
  traceId: string;
  bundleDir: string;
  validationOk: boolean;
  winnerMode: ReplayLaneMode | null;
  topScoreModes: ReplayLaneMode[];
  scoreSpread: number;
  candidateRelationVsBaseline: ReplayLaneRelation | null;
  candidateRelationVsFloor: ReplayLaneRelation | null;
  candidateTieOrBetterVsBaseline: boolean | null;
  candidateRegressionVsBaseline: boolean | null;
  candidateRegressionVsFloor: boolean | null;
  bundleHash: string;
  scoreHash: string;
}

export interface RecordedSessionReplayProofLaneIndexV1 {
  contract: typeof RECORDED_SESSION_REPLAY_PROOF_LANE_INDEX_CONTRACT;
  laneDir: string;
  sourceManifest: RecordedSessionReplayProofLaneSourceManifestV1;
  modeOrder: ReplayLaneMode[];
  requestedTraceCount: number;
  successfulTraceCount: number;
  failedTraceCount: number;
  failedTraceIds: string[];
  assumptions: string[];
  files: {
    readme: string;
    summary: string;
    closeout: string;
    index: string;
    summaryTables: string;
    pairwiseDeltas: string;
    winRateMatrix: string;
    workedTraces: string;
    generationReport: string;
  };
  traceBundles: RecordedSessionReplayProofLaneBundleIndexRowV1[];
}

interface RecordedSessionReplayProofLaneGenerationEntryV1 {
  traceId: string;
  tracePath: string | null;
  bundleDir: string;
  validationPath: string | null;
  result: "passed" | "failed";
  validation: RecordedSessionReplayProofBundleValidationV1 | null;
  error: string | null;
}

export interface RecordedSessionReplayProofLaneGenerationReportV1 {
  contract: typeof RECORDED_SESSION_REPLAY_PROOF_LANE_GENERATION_REPORT_CONTRACT;
  artifactRoot: string;
  laneDir: string;
  requestedTraceCount: number;
  successfulTraceCount: number;
  failedTraceCount: number;
  sourceManifestPath: string | null;
  sourceManifest: RecordedSessionReplayProofLaneSourceManifestV1;
  assumptions: string[];
  entries: RecordedSessionReplayProofLaneGenerationEntryV1[];
}

export interface RecordedSessionReplayProofLaneCloseoutArtifactV1 {
  role: string;
  path: string;
  digest: string;
  contract: string | null;
}

export interface RecordedSessionReplayProofLaneCloseoutV1 {
  contract: typeof RECORDED_SESSION_REPLAY_PROOF_LANE_CLOSEOUT_CONTRACT;
  hashAlgorithm: "sha256";
  laneDir: string;
  sourceManifest: RecordedSessionReplayProofLaneSourceManifestV1;
  verdict: {
    verdict: "success_and_proven" | "partial_proof" | "no_successful_replays";
    severity: "none" | "warn" | "error";
    why: string;
  };
  requestedTraceCount: number;
  successfulTraceCount: number;
  failedTraceCount: number;
  failedTraceIds: string[];
  modeOrder: ReplayLaneMode[];
  scorecard: RecordedSessionReplayProofLaneExplainableScorecardV1;
  winnerModeCounts: Array<{
    mode: ReplayLaneMode;
    rankedWinnerCount: number;
    sharedTopScoreTraceCount: number;
  }>;
  traceHashes: Array<{
    traceId: string;
    bundleHash: string;
    scoreHash: string;
    winnerMode: ReplayLaneMode | null;
    scoreSpread: number;
    candidateRelationVsBaseline: ReplayLaneRelation | null;
    candidateRelationVsFloor: ReplayLaneRelation | null;
  }>;
  files: RecordedSessionReplayProofLaneCloseoutArtifactV1[];
}

export interface RecordedSessionReplayProofLaneDescriptorV1 {
  artifactRoot: string;
  laneDir: string;
  readmePath: string;
  summaryPath: string;
  closeoutPath: string;
  indexPath: string;
  summaryTablesPath: string;
  pairwiseDeltasPath: string;
  winRateMatrixPath: string;
  workedTracesPath: string;
  generationReportPath: string;
  index: RecordedSessionReplayProofLaneIndexV1;
  summaryTables: RecordedSessionReplayProofLaneSummaryTablesV1;
  pairwiseDeltas: RecordedSessionReplayProofLanePairwiseDeltasV1;
  winRateMatrix: RecordedSessionReplayProofLaneWinRateMatrixV1;
  closeout: RecordedSessionReplayProofLaneCloseoutV1;
  generationReport: RecordedSessionReplayProofLaneGenerationReportV1;
  successfulBundles: RecordedSessionReplayProofBundleDescriptorV1[];
}

interface RecordedSessionReplayProofLaneTraceAnalysisV1 {
  traceId: string;
  bundleDir: string;
  validation: RecordedSessionReplayProofBundleValidationV1;
  descriptor: RecordedSessionReplayProofBundleDescriptorV1;
  topScoreModes: ReplayLaneMode[];
  scoreSpread: number;
  modes: RecordedSessionReplayProofLaneModeTraceRowV1[];
  turns: RecordedSessionReplayProofLaneTurnSummaryRowV1[];
}

function ensureDir(dirPath: string): void {
  mkdirSync(dirPath, { recursive: true });
}

function writeJson(filePath: string, value: unknown): void {
  ensureDir(path.dirname(filePath));
  writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

function writeText(filePath: string, value: string): void {
  ensureDir(path.dirname(filePath));
  writeFileSync(filePath, value.endsWith("\n") ? value : `${value}\n`, "utf8");
}

function loadPricingTable(): ReplayLanePricingTable | null {
  const pricingTablePath = path.resolve(repoRoot, "scripts", "pricing-table.v1.json");
  try {
    const pricingTable = JSON.parse(readFileSync(pricingTablePath, "utf8")) as {
      version?: string;
      charsPerToken?: number;
      promptPriceUsdPer1mTokens?: number;
    };
    const charsPerToken = Number(pricingTable.charsPerToken ?? 4);
    const promptPriceUsdPer1mTokens = Number(pricingTable.promptPriceUsdPer1mTokens ?? 0);
    if (!Number.isFinite(charsPerToken) || charsPerToken <= 0 || !Number.isFinite(promptPriceUsdPer1mTokens)) {
      return null;
    }
    return {
      version: typeof pricingTable.version === "string" ? pricingTable.version : null,
      path: portableRelativePath(repoRoot, pricingTablePath),
      charsPerToken,
      promptPriceUsdPer1mTokens,
    };
  } catch {
    return null;
  }
}

function renderJson(value: unknown): string {
  return `${JSON.stringify(value, null, 2)}\n`;
}

function sha256Text(value: string): string {
  return `sha256-${createHash("sha256").update(value, "utf8").digest("hex")}`;
}

function cloneSourceManifest(
  sourceManifest: RecordedSessionReplayProofLaneSourceManifestV1,
): RecordedSessionReplayProofLaneSourceManifestV1 {
  return { ...sourceManifest };
}

function readSourceManifest(
  sourceManifestPath: string | null | undefined,
): RecordedSessionReplayProofLaneSourceManifestV1 {
  if (!sourceManifestPath) {
    return {
      contract: RECORDED_SESSION_REPLAY_PROOF_LANE_SOURCE_MANIFEST_CONTRACT,
      provided: false,
      manifestId: null,
      manifestContract: null,
      manifestDigest: null,
    };
  }
  try {
    const sourceManifestText = readFileSync(sourceManifestPath, "utf8");
    const sourceManifest = JSON.parse(sourceManifestText) as Record<string, unknown>;
    const manifestId = typeof sourceManifest.manifestId === "string"
      ? sourceManifest.manifestId
      : typeof sourceManifest.setId === "string"
        ? sourceManifest.setId
        : null;
    return {
      contract: RECORDED_SESSION_REPLAY_PROOF_LANE_SOURCE_MANIFEST_CONTRACT,
      provided: true,
      manifestId,
      manifestContract: typeof sourceManifest.contract === "string" ? sourceManifest.contract : null,
      manifestDigest: sha256Text(sourceManifestText),
    };
  } catch {
    return {
      contract: RECORDED_SESSION_REPLAY_PROOF_LANE_SOURCE_MANIFEST_CONTRACT,
      provided: true,
      manifestId: null,
      manifestContract: null,
      manifestDigest: null,
    };
  }
}

function portableRelativePath(fromPath: string, toPath: string): string {
  return path.relative(fromPath, toPath).split(path.sep).join("/");
}

function normalizeStringArray(values: readonly string[] | null | undefined): string[] {
  const seen = new Set<string>();
  const normalized: string[] = [];
  for (const value of values ?? []) {
    const trimmed = String(value ?? "").trim();
    if (trimmed.length === 0 || seen.has(trimmed)) {
      continue;
    }
    seen.add(trimmed);
    normalized.push(trimmed);
  }
  return normalized;
}

function normalizeWorkedTraceLimit(value: number | null | undefined): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return DEFAULT_WORKED_TRACE_LIMIT;
  }
  return Math.max(1, Math.floor(value));
}

function sanitizeTraceBundleDirName(traceId: string): string {
  const trimmed = traceId.trim();
  if (trimmed.length === 0 || trimmed === "." || trimmed === "..") {
    throw new Error(`Invalid traceId for bundle directory: ${traceId}`);
  }
  return trimmed.replaceAll(/[\\/]/g, "__");
}

function roundRate(numerator: number, denominator: number): number | null {
  return denominator > 0 ? Number((numerator / denominator).toFixed(6)) : null;
}

function roundValue(value: number, places = 6): number {
  const factor = 10 ** places;
  return Math.round(value * factor) / factor;
}

function roundAverage(total: number, count: number, places = 6): number | null {
  if (!Number.isFinite(total) || count <= 0) {
    return null;
  }
  return roundValue(total / count, places);
}

function floorAverage(total: number, count: number): number | null {
  if (!Number.isFinite(total) || count <= 0) {
    return null;
  }
  return Math.floor(total / count);
}

function previewText(value: string | null | undefined, maxLength = 96): string | null {
  const normalized = String(value ?? "").replace(/\s+/g, " ").trim();
  if (normalized.length === 0) {
    return null;
  }
  if (normalized.length <= maxLength) {
    return normalized;
  }
  return `${normalized.slice(0, Math.max(1, maxLength - 3))}...`;
}

function shortDigest(value: string | null): string {
  if (value === null) {
    return "none";
  }
  return value.startsWith("sha256-") ? value.slice(7, 19) : value.slice(0, 12);
}

function relationFromScores(left: number, right: number): ReplayLaneRelation {
  if (left > right) {
    return "better";
  }
  if (left < right) {
    return "worse";
  }
  return "tied";
}

function buildCountRate(count: number, totalCount: number): RecordedSessionReplayProofLaneCountRateV1 {
  return {
    count,
    rate: roundRate(count, totalCount),
    totalCount,
  };
}

function buildOutcomeBreakdown(relations: ReplayLaneRelation[]): RecordedSessionReplayProofLaneOutcomeBreakdownV1 {
  const betterCount = relations.filter((relation) => relation === "better").length;
  const tiedCount = relations.filter((relation) => relation === "tied").length;
  const worseCount = relations.filter((relation) => relation === "worse").length;
  return {
    betterCount,
    tiedCount,
    worseCount,
    betterRate: roundRate(betterCount, relations.length),
    tieRate: roundRate(tiedCount, relations.length),
    worseRate: roundRate(worseCount, relations.length),
    totalCount: relations.length,
  };
}

function formatCountRate(value: RecordedSessionReplayProofLaneCountRateV1): string {
  return `${value.count}/${value.totalCount}${value.rate === null ? "" : ` (${value.rate})`}`;
}

function formatOutcomeBreakdown(value: RecordedSessionReplayProofLaneOutcomeBreakdownV1): string {
  return `${value.betterCount} better, ${value.tiedCount} tied, ${value.worseCount} worse`;
}

function buildExplainableScorecard(
  params: {
    modes: RecordedSessionReplayProofLaneModeSummaryRowV1[];
    traces: RecordedSessionReplayProofLaneTraceSummaryRowV1[];
    turns: RecordedSessionReplayProofLaneTurnSummaryRowV1[];
  },
): RecordedSessionReplayProofLaneExplainableScorecardV1 {
  const candidateMode = "learned_route";
  const baselineMode = "graph_prior_only";
  const floorMode = "no_brain";
  const traceRelationsVsBaseline = params.traces
    .map((trace) => trace.candidateRelationVsBaseline)
    .filter((relation): relation is ReplayLaneRelation => relation !== null);
  const turnRelationsVsBaseline = params.turns
    .map((turn) => turn.candidateRelationVsBaseline)
    .filter((relation): relation is ReplayLaneRelation => relation !== null);
  const traceOutcomeVsBaseline = buildOutcomeBreakdown(traceRelationsVsBaseline);
  const turnOutcomeVsBaseline = buildOutcomeBreakdown(turnRelationsVsBaseline);
  const regressionVsBaseline = buildCountRate(
    params.traces.filter((trace) => trace.candidateRegressionVsBaseline === true).length,
    params.traces.length,
  );
  const regressionVsFloor = buildCountRate(
    params.traces.filter((trace) => trace.candidateRegressionVsFloor === true).length,
    params.traces.length,
  );
  const traceTieOrBetterVsBaseline = buildCountRate(
    params.traces.filter((trace) => trace.candidateTieOrBetterVsBaseline === true).length,
    params.traces.length,
  );
  const turnTieOrBetterVsBaseline = buildCountRate(
    params.turns.filter((turn) => turn.candidateTieOrBetterVsBaseline === true).length,
    params.turns.length,
  );
  const candidateRow = params.modes.find((row) => row.mode === candidateMode) ?? null;
  const baselineRow = params.modes.find((row) => row.mode === baselineMode) ?? null;
  const floorRow = params.modes.find((row) => row.mode === floorMode) ?? null;
  const candidateRate = candidateRow ? roundRate(candidateRow.totalPhraseHitCount, candidateRow.totalPhraseCount) : null;
  const baselineRate = baselineRow ? roundRate(baselineRow.totalPhraseHitCount, baselineRow.totalPhraseCount) : null;
  const correctionFeedbackTurnCount = params.turns.filter((turn) => turn.feedbackKinds.length > 0).length;
  const correctionNonApprovalTurnCount = params.turns.filter((turn) =>
    turn.feedbackKinds.some((kind) => kind !== "approval")
  ).length;
  const candidateTurnRows = params.turns
    .map((turn) => turn.modes.find((row) => row.mode === candidateMode) ?? null)
    .filter((row): row is RecordedSessionReplayProofLaneTurnModeRowV1 => row !== null);
  const explicitActivationObservedTurns = candidateTurnRows.filter((row) => row.activationTaken !== null);
  const explicitActivationTurns = explicitActivationObservedTurns.filter((row) => row.activationTaken === true);
  const beneficialExplicitActivationCount = params.turns.filter((turn) => {
    const candidateTurn = turn.modes.find((row) => row.mode === candidateMode);
    return candidateTurn?.activationTaken === true && turn.candidateRelationVsBaseline === "better";
  }).length;
  const activationSourceCounts = [...explicitActivationObservedTurns.reduce((counts, row) => {
    const source = row.activationSource ?? "unknown";
    counts.set(source, (counts.get(source) ?? 0) + 1);
    return counts;
  }, new Map<string, number>()).entries()]
    .map(([source, count]) => ({ source, count }))
    .sort((left, right) => right.count - left.count || left.source.localeCompare(right.source));
  const proxyActivationTurns = params.turns.filter((turn) => {
    const candidateTurn = turn.modes.find((row) => row.mode === candidateMode);
    const baselineTurn = turn.modes.find((row) => row.mode === baselineMode);
    if (!candidateTurn || !baselineTurn) {
      return false;
    }
    return candidateTurn.usedLearnedRouteFn
      || candidateTurn.selectionDigest !== baselineTurn.selectionDigest
      || candidateTurn.activePackId !== baselineTurn.activePackId;
  });
  const beneficialProxyActivationCount = proxyActivationTurns.filter((turn) => turn.candidateRelationVsBaseline === "better").length;
  const degradedCandidateTurnRows = candidateTurnRows.filter((row) =>
    row.compileOk === false || row.fallbackToStaticContext || row.hardRequirementViolated
  );
  const catastrophicDegradedTurnCount = params.turns.filter((turn) => {
    const candidateTurn = turn.modes.find((row) => row.mode === candidateMode);
    if (!candidateTurn) {
      return false;
    }
    const degraded = candidateTurn.compileOk === false || candidateTurn.fallbackToStaticContext || candidateTurn.hardRequirementViolated;
    return degraded && turn.candidateRegressionVsFloor === true;
  }).length;
  const acceptableDegradedTurnCount = params.turns.filter((turn) => {
    const candidateTurn = turn.modes.find((row) => row.mode === candidateMode);
    if (!candidateTurn) {
      return false;
    }
    const degraded = candidateTurn.compileOk === false || candidateTurn.fallbackToStaticContext || candidateTurn.hardRequirementViolated;
    return degraded && turn.candidateRegressionVsFloor === false;
  }).length;
  const winTraces = params.traces.filter((trace) => trace.candidateRelationVsBaseline === "better");
  const candidatePromptTokensOnWins = winTraces.reduce((sum, trace) => {
    const row = trace.modes.find((modeRow) => modeRow.mode === candidateMode);
    return sum + (row?.estimatedPromptTokenCount ?? 0);
  }, 0);
  const baselinePromptTokensOnWins = winTraces.reduce((sum, trace) => {
    const row = trace.modes.find((modeRow) => modeRow.mode === baselineMode);
    return sum + (row?.estimatedPromptTokenCount ?? 0);
  }, 0);
  const candidatePromptCostUsdOnWins = winTraces.reduce((sum, trace) => {
    const row = trace.modes.find((modeRow) => modeRow.mode === candidateMode);
    return sum + (row?.estimatedPromptCostUsd ?? 0);
  }, 0);
  const baselinePromptCostUsdOnWins = winTraces.reduce((sum, trace) => {
    const row = trace.modes.find((modeRow) => modeRow.mode === baselineMode);
    return sum + (row?.estimatedPromptCostUsd ?? 0);
  }, 0);
  const candidateLatencyMsOnWins = winTraces.reduce((sum, trace) => {
    const row = trace.modes.find((modeRow) => modeRow.mode === candidateMode);
    return sum + (row?.totalLatencyMs ?? 0);
  }, 0);
  const baselineLatencyMsOnWins = winTraces.reduce((sum, trace) => {
    const row = trace.modes.find((modeRow) => modeRow.mode === baselineMode);
    return sum + (row?.totalLatencyMs ?? 0);
  }, 0);
  const candidatePromptTokensPerSuccess = roundAverage(candidatePromptTokensOnWins, winTraces.length);
  const baselinePromptTokensPerSuccess = roundAverage(baselinePromptTokensOnWins, winTraces.length);
  const candidatePromptCostUsdPerSuccess = roundAverage(candidatePromptCostUsdOnWins, winTraces.length);
  const baselinePromptCostUsdPerSuccess = roundAverage(baselinePromptCostUsdOnWins, winTraces.length);
  const candidateLatencyMsPerSuccess = floorAverage(candidateLatencyMsOnWins, winTraces.length);
  const baselineLatencyMsPerSuccess = floorAverage(baselineLatencyMsOnWins, winTraces.length);

  return {
    candidateMode,
    baselineMode,
    floorMode,
    comparableTraceCount: params.traces.length,
    comparableTurnCount: params.turns.length,
    traceOutcomeVsBaseline,
    turnOutcomeVsBaseline,
    traceTieOrBetterVsBaseline,
    turnTieOrBetterVsBaseline,
    regressionVsBaseline,
    regressionVsFloor,
    criticalRegressionCount: regressionVsFloor.count,
    requiredContextRecall: {
      available: (candidateRow?.totalPhraseCount ?? 0) > 0 || (baselineRow?.totalPhraseCount ?? 0) > 0,
      candidateMode,
      baselineMode,
      candidatePhraseHitCount: candidateRow?.totalPhraseHitCount ?? null,
      candidatePhraseCount: candidateRow?.totalPhraseCount ?? null,
      candidateRate,
      baselinePhraseHitCount: baselineRow?.totalPhraseHitCount ?? null,
      baselinePhraseCount: baselineRow?.totalPhraseCount ?? null,
      baselineRate,
      delta: candidateRate !== null && baselineRate !== null ? roundValue(candidateRate - baselineRate) : null,
      summary: (candidateRow?.totalPhraseCount ?? 0) > 0 || (baselineRow?.totalPhraseCount ?? 0) > 0
        ? `${candidateMode} recalled ${candidateRow?.totalPhraseHitCount ?? 0}/${candidateRow?.totalPhraseCount ?? 0} required-context phrases vs ${baselineMode} ${baselineRow?.totalPhraseHitCount ?? 0}/${baselineRow?.totalPhraseCount ?? 0}`
        : "required-context recall is unavailable because no expected-context phrases were recorded",
    },
    correctionAbsorption: {
      available: false,
      observedFeedbackTurnCount: correctionFeedbackTurnCount,
      observedNonApprovalFeedbackTurnCount: correctionNonApprovalTurnCount,
      summary: correctionFeedbackTurnCount > 0
        ? `observed ${correctionFeedbackTurnCount} feedback-bearing turns (${correctionNonApprovalTurnCount} non-approval), but replay-lane outputs do not yet measure recurrence after correction`
        : "correction absorption is unavailable in replay-lane outputs because no feedback-bearing turns were recorded here",
    },
    activationPrecision: {
      available: explicitActivationObservedTurns.length > 0,
      observedTurnCount: explicitActivationObservedTurns.length,
      activationCount: explicitActivationTurns.length,
      beneficialActivationCount: beneficialExplicitActivationCount,
      precision: roundRate(beneficialExplicitActivationCount, explicitActivationTurns.length),
      sourceCounts: activationSourceCounts,
      summary: explicitActivationObservedTurns.length === 0
        ? "explicit learned-route activation is unavailable because replay turns did not emit activationTaken"
        : explicitActivationTurns.length === 0
          ? `observed 0/${explicitActivationObservedTurns.length} explicit learned-route activations in this replay lane`
          : `explicit learned-route activation precision is ${beneficialExplicitActivationCount}/${explicitActivationTurns.length} across ${explicitActivationObservedTurns.length} observed candidate turns`,
      limitations: [
        "explicit activation is currently defined by emitted activationTaken on learned_route turns",
        "activation recall is still blocked because replay outputs do not carry an independent beneficial-opportunity oracle",
      ],
    },
    activationPrecisionProxy: {
      available: proxyActivationTurns.length > 0,
      activationCount: proxyActivationTurns.length,
      beneficialActivationCount: beneficialProxyActivationCount,
      precision: roundRate(beneficialProxyActivationCount, proxyActivationTurns.length),
      activationDefinition: "usedLearnedRouteFn OR selectionDigest changed vs graph_prior_only OR activePackId changed vs graph_prior_only",
      summary: proxyActivationTurns.length > 0
        ? `selection-divergence proxy activation precision is ${beneficialProxyActivationCount}/${proxyActivationTurns.length} against graph_prior_only`
        : "no proxy activations were observed against graph_prior_only in this replay lane",
      limitations: [
        "this is a proxy for nontrivial activation, not a true emitted router activation event",
        "selectionDigest divergence can overcount activation when learned_route mode never used a learned route function",
      ],
    },
    successAdjustedEconomics: {
      available: winTraces.length > 0,
      successUnit: winTraces.length > 0 ? "validated_trace" : null,
      candidateMode,
      baselineMode,
      successCount: winTraces.length,
      candidateEstimatedPromptTokensPerSuccess: candidatePromptTokensPerSuccess,
      baselineEstimatedPromptTokensPerSuccess: baselinePromptTokensPerSuccess,
      candidateEstimatedPromptCostUsdPerSuccess: candidatePromptCostUsdPerSuccess,
      baselineEstimatedPromptCostUsdPerSuccess: baselinePromptCostUsdPerSuccess,
      promptTokenDeltaCandidateMinusBaseline: candidatePromptTokensPerSuccess !== null && baselinePromptTokensPerSuccess !== null
        ? roundValue(candidatePromptTokensPerSuccess - baselinePromptTokensPerSuccess)
        : null,
      promptCostUsdDeltaCandidateMinusBaseline: candidatePromptCostUsdPerSuccess !== null && baselinePromptCostUsdPerSuccess !== null
        ? roundValue(candidatePromptCostUsdPerSuccess - baselinePromptCostUsdPerSuccess)
        : null,
      candidateServePathLatencyMsPerSuccess: candidateLatencyMsPerSuccess,
      baselineServePathLatencyMsPerSuccess: baselineLatencyMsPerSuccess,
      servePathLatencyMsDeltaCandidateMinusBaseline: candidateLatencyMsPerSuccess !== null && baselineLatencyMsPerSuccess !== null
        ? candidateLatencyMsPerSuccess - baselineLatencyMsPerSuccess
        : null,
      summary: winTraces.length > 0
        ? `${candidateMode} used ${candidatePromptTokensPerSuccess ?? "n/a"} estimated prompt tokens, ${candidatePromptCostUsdPerSuccess ?? "n/a"} estimated prompt USD, and ${candidateLatencyMsPerSuccess ?? "n/a"} ms serve-path latency per incremental win vs ${baselineMode} ${baselinePromptTokensPerSuccess ?? "n/a"}, ${baselinePromptCostUsdPerSuccess ?? "n/a"}, and ${baselineLatencyMsPerSuccess ?? "n/a"}`
        : "success-adjusted economics are unavailable because learned_route produced no incremental wins vs graph_prior_only in this replay lane",
      limitations: [
        "prompt-token values are estimated from selected-context chars using the default 4 chars/token proxy",
        "prompt USD values are estimated from scripts/pricing-table.v1.json when that pricing table is available",
        "latency values cover the serve-path hot path only and exclude background work",
      ],
    },
    failOpen: {
      available: candidateTurnRows.length > 0,
      degradedTurnCount: degradedCandidateTurnRows.length,
      acceptableDegradedTurnCount,
      catastrophicDegradedTurnCount,
      degradedTurnRate: roundRate(degradedCandidateTurnRows.length, candidateTurnRows.length),
      acceptableDegradedTurnRate: roundRate(acceptableDegradedTurnCount, candidateTurnRows.length),
      catastrophicDegradedTurnRate: roundRate(catastrophicDegradedTurnCount, candidateTurnRows.length),
      clipRate: roundRate(catastrophicDegradedTurnCount, candidateTurnRows.length),
      failOpenRate: roundRate(degradedCandidateTurnRows.length, candidateTurnRows.length),
      summary: degradedCandidateTurnRows.length > 0
        ? `observed ${degradedCandidateTurnRows.length}/${candidateTurnRows.length} degraded learned_route turns, with ${acceptableDegradedTurnCount} acceptable and ${catastrophicDegradedTurnCount} catastrophic vs the no_brain floor`
        : `observed 0/${candidateTurnRows.length} degraded learned_route turns in this replay lane`,
    },
    diagnostics: {
      candidateMeanQualityScore: candidateRow?.meanQualityScore ?? null,
      baselineMeanQualityScore: baselineRow?.meanQualityScore ?? null,
      floorMeanQualityScore: floorRow?.meanQualityScore ?? null,
      candidateMinusBaselineMeanQualityScore: candidateRow?.meanQualityScore !== null
        && candidateRow?.meanQualityScore !== undefined
        && baselineRow?.meanQualityScore !== null
        && baselineRow?.meanQualityScore !== undefined
        ? roundValue(candidateRow.meanQualityScore - baselineRow.meanQualityScore)
        : null,
      candidateMinusFloorMeanQualityScore: candidateRow?.meanQualityScore !== null
        && candidateRow?.meanQualityScore !== undefined
        && floorRow?.meanQualityScore !== null
        && floorRow?.meanQualityScore !== undefined
        ? roundValue(candidateRow.meanQualityScore - floorRow.meanQualityScore)
        : null,
      winnerModeCounts: params.modes.map((row) => ({
        mode: row.mode,
        rankedWinnerCount: row.rankedWinnerCount,
        sharedTopScoreTraceCount: row.sharedTopScoreTraceCount,
      })),
    },
  };
}

function buildCloseoutArtifact(
  role: string,
  artifactPath: string,
  text: string,
  contract: string | null,
): RecordedSessionReplayProofLaneCloseoutArtifactV1 {
  return {
    role,
    path: artifactPath,
    digest: sha256Text(text),
    contract,
  };
}

function compareQualityRows(
  left: { mode: ReplayLaneMode; qualityScore: number },
  right: { mode: ReplayLaneMode; qualityScore: number },
): number {
  return right.qualityScore - left.qualityScore || left.mode.localeCompare(right.mode);
}

function findModeReport(
  descriptor: RecordedSessionReplayProofBundleDescriptorV1,
  mode: ReplayLaneMode,
): RecordedSessionReplayModeReportV1 {
  const report = descriptor.bundle.modes.find((candidate) => candidate.mode === mode);
  if (!report) {
    throw new Error(`Missing mode ${mode} for trace ${descriptor.bundle.traceId}`);
  }
  return report;
}

function findTurnReport(mode: RecordedSessionReplayModeReportV1, turnId: string): RecordedSessionReplayTurnReportV1 {
  const turn = mode.turns.find((candidate) => candidate.turnId === turnId);
  if (!turn) {
    throw new Error(`Missing turn ${turnId} for mode ${mode.mode}`);
  }
  return turn;
}

function buildTopScoreModes(rows: Array<{ mode: ReplayLaneMode; qualityScore: number }>): ReplayLaneMode[] {
  const max = rows.reduce((best, row) => Math.max(best, row.qualityScore), Number.NEGATIVE_INFINITY);
  return rows.filter((row) => row.qualityScore === max).map((row) => row.mode);
}

function scoreSpread(rows: Array<{ qualityScore: number }>): number {
  if (rows.length === 0) {
    return 0;
  }
  const values = rows.map((row) => row.qualityScore);
  return Math.max(...values) - Math.min(...values);
}

function traceFeedbackKinds(turn: RecordedSessionTraceTurnV1): string[] {
  return normalizeStringArray(
    turn.feedback?.flatMap((feedback) => (typeof feedback.kind === "string" ? [feedback.kind] : [])),
  );
}

function countStringChars(values: unknown): number {
  if (!Array.isArray(values)) {
    return 0;
  }
  return values.reduce((total, value) => total + (typeof value === "string" ? value.length : 0), 0);
}

function estimateUsdFromTokens(tokens: number, pricePer1mTokens: number): number | null {
  if (!Number.isFinite(tokens) || !Number.isFinite(pricePer1mTokens)) {
    return null;
  }
  return roundValue((tokens / 1_000_000) * pricePer1mTokens);
}

function normalizeTimingMs(value: number | null | undefined): number | null {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return null;
  }
  return Math.round(value / 5) * 5;
}

function estimateTokensFromChars(chars: number, charsPerToken = 4): number | null {
  if (!Number.isFinite(chars) || !Number.isFinite(charsPerToken) || charsPerToken <= 0) {
    return null;
  }
  return Math.ceil(chars / charsPerToken);
}

function buildTraceAnalysis(
  artifactRoot: string,
  descriptor: RecordedSessionReplayProofBundleDescriptorV1,
  validation: RecordedSessionReplayProofBundleValidationV1,
  pricingTable: ReplayLanePricingTable | null,
): RecordedSessionReplayProofLaneTraceAnalysisV1 {
  const bundleDir = portableRelativePath(artifactRoot, descriptor.rootDir);
  const modes = RECORDED_SESSION_REPLAY_PROOF_LANE_MODE_ORDER.map((mode) => {
    const report = findModeReport(descriptor, mode);
    const normalizedTurnTimings = report.turns.map((turn) => ({
      totalMs: normalizeTimingMs(turn.timing?.totalMs),
      routeSelectionMs: normalizeTimingMs(turn.timing?.routeSelectionMs),
      promptAssemblyMs: normalizeTimingMs(turn.timing?.promptAssemblyMs),
    }));
    const selectedContextBlockCount = report.turns.reduce(
      (sum, turn) => sum + (Array.isArray(turn.selectedContextIds) ? turn.selectedContextIds.length : turn.selectedContextTexts.length),
      0,
    );
    const selectedContextCharCount = report.turns.reduce((sum, turn) => sum + countStringChars(turn.selectedContextTexts), 0);
    const estimatedPromptTokenCount = estimateTokensFromChars(selectedContextCharCount, pricingTable?.charsPerToken ?? 4) ?? 0;
    const estimatedPromptCostUsd = pricingTable
      ? estimateUsdFromTokens(estimatedPromptTokenCount, pricingTable.promptPriceUsdPer1mTokens)
      : null;
    return {
      mode,
      qualityScore: report.summary.qualityScore,
      compileOkCount: report.summary.compileOkCount,
      turnCount: report.turns.length,
      phraseHitCount: report.summary.phraseHitCount,
      phraseCount: report.summary.phraseCount,
      promotionCount: report.summary.promotionCount,
      usedLearnedRouteTurnCount: report.summary.usedLearnedRouteTurnCount,
      activationObservedTurnCount: report.turns.filter((turn) => turn.activationTaken !== null).length,
      activationTakenTurnCount: report.turns.filter((turn) => turn.activationTaken === true).length,
      selectedContextBlockCount,
      selectedContextCharCount,
      estimatedPromptTokenCount,
      estimatedPromptCostUsd,
      totalLatencyMs: normalizedTurnTimings.reduce((sum, timing) => sum + (timing.totalMs ?? 0), 0),
      totalRouteSelectionLatencyMs: normalizedTurnTimings.reduce((sum, timing) => sum + (timing.routeSelectionMs ?? 0), 0),
      totalPromptAssemblyLatencyMs: normalizedTurnTimings.reduce((sum, timing) => sum + (timing.promptAssemblyMs ?? 0), 0),
      fallbackToStaticContextTurnCount: report.turns.filter((turn) => turn.fallbackToStaticContext === true).length,
      hardRequirementViolatedTurnCount: report.turns.filter((turn) => turn.hardRequirementViolated === true).length,
      warningCount: report.summary.scannerEvidence.warnings.length,
      scoreHash: report.summary.scoreHash,
    };
  });
  const turns = descriptor.trace.turns.map((traceTurn) => {
    const traceTurnId = typeof traceTurn.turnId === "string" ? traceTurn.turnId : "";
    if (traceTurnId.length === 0) {
      throw new Error(`Trace ${descriptor.bundle.traceId} contains a turn without turnId`);
    }
    const turnModes = RECORDED_SESSION_REPLAY_PROOF_LANE_MODE_ORDER.map((mode) => {
      const report = findModeReport(descriptor, mode);
      const turn = findTurnReport(report, traceTurnId);
      return {
        mode,
        phase: turn.phase,
        qualityScore: turn.qualityScore,
        compileOk: turn.compileOk,
        phraseHitCount: turn.phraseHits.length,
        phraseCount: turn.expectedContextPhrases.length,
        usedLearnedRouteFn: turn.usedLearnedRouteFn,
        promoted: turn.promoted,
        modeEffective: typeof turn.modeEffective === "string" ? turn.modeEffective : null,
        activePackId: turn.activePackId,
        routerIdentity: typeof turn.routerIdentity === "string" ? turn.routerIdentity : null,
        selectionDigest: turn.selectionDigest,
        activationTaken: typeof turn.activationTaken === "boolean" ? turn.activationTaken : null,
        activationSource: typeof turn.activationSource === "string" ? turn.activationSource : null,
        activationReason: typeof turn.activationReason === "string" ? turn.activationReason : null,
        activationConfidence: typeof turn.activationConfidence === "string" ? turn.activationConfidence : null,
        fallbackToStaticContext: turn.fallbackToStaticContext === true,
        hardRequirementViolated: turn.hardRequirementViolated === true,
        selectedContextBlockCount: Array.isArray(turn.selectedContextIds) ? turn.selectedContextIds.length : turn.selectedContextTexts.length,
        selectedContextCharCount: countStringChars(turn.selectedContextTexts),
        estimatedPromptTokens: estimateTokensFromChars(countStringChars(turn.selectedContextTexts), pricingTable?.charsPerToken ?? 4),
        estimatedPromptCostUsd: pricingTable
          ? estimateUsdFromTokens(
            estimateTokensFromChars(countStringChars(turn.selectedContextTexts), pricingTable.charsPerToken) ?? 0,
            pricingTable.promptPriceUsdPer1mTokens,
          )
          : null,
        timing: {
          totalMs: normalizeTimingMs(turn.timing?.totalMs),
          routeSelectionMs: normalizeTimingMs(turn.timing?.routeSelectionMs),
          promptAssemblyMs: normalizeTimingMs(turn.timing?.promptAssemblyMs),
        },
        selectedContextPreview: previewText(turn.selectedContextTexts.join(" || "), 140),
      };
    });
    const ranking = turnModes
      .map((row) => ({
        mode: row.mode,
        qualityScore: row.qualityScore,
      }))
      .sort(compareQualityRows);
    const candidateTurn = turnModes.find((row) => row.mode === "learned_route");
    const baselineTurn = turnModes.find((row) => row.mode === "graph_prior_only");
    const floorTurn = turnModes.find((row) => row.mode === "no_brain");
    return {
      traceId: descriptor.bundle.traceId,
      bundleDir,
      turnId: traceTurnId,
      userMessagePreview: previewText(traceTurn.userMessage, 140) ?? "",
      expectedContextPhrases: [...(traceTurn.expectedContextPhrases ?? [])],
      feedbackKinds: traceFeedbackKinds(traceTurn),
      scoreSpread: scoreSpread(turnModes),
      topModes: buildTopScoreModes(turnModes),
      candidateRelationVsBaseline: candidateTurn && baselineTurn
        ? relationFromScores(candidateTurn.qualityScore, baselineTurn.qualityScore)
        : null,
      candidateRelationVsFloor: candidateTurn && floorTurn
        ? relationFromScores(candidateTurn.qualityScore, floorTurn.qualityScore)
        : null,
      candidateTieOrBetterVsBaseline: candidateTurn && baselineTurn
        ? candidateTurn.qualityScore >= baselineTurn.qualityScore
        : null,
      candidateRegressionVsBaseline: candidateTurn && baselineTurn
        ? candidateTurn.qualityScore < baselineTurn.qualityScore
        : null,
      candidateRegressionVsFloor: candidateTurn && floorTurn
        ? candidateTurn.qualityScore < floorTurn.qualityScore
        : null,
      ranking,
      modes: turnModes,
    };
  });
  return {
    traceId: descriptor.bundle.traceId,
    bundleDir,
    validation,
    descriptor,
    topScoreModes: buildTopScoreModes(modes),
    scoreSpread: scoreSpread(modes),
    modes,
    turns,
  };
}

function buildSummaryTables(
  analyses: RecordedSessionReplayProofLaneTraceAnalysisV1[],
  sourceManifest: RecordedSessionReplayProofLaneSourceManifestV1,
  requestedTraceCount: number,
  failedTraceCount: number,
): RecordedSessionReplayProofLaneSummaryTablesV1 {
  const modes = RECORDED_SESSION_REPLAY_PROOF_LANE_MODE_ORDER.map((mode) => {
    const rows = analyses.map((analysis) => analysis.modes.find((candidate) => candidate.mode === mode)).filter(Boolean) as RecordedSessionReplayProofLaneModeTraceRowV1[];
    return {
      mode,
      traceCount: rows.length,
      rankedWinnerCount: analyses.filter((analysis) => analysis.descriptor.bundle.summary.winnerMode === mode).length,
      sharedTopScoreTraceCount: analyses.filter((analysis) => analysis.topScoreModes.includes(mode)).length,
      meanQualityScore: roundRate(
        rows.reduce((sum, row) => sum + row.qualityScore, 0),
        rows.length,
      ),
      totalCompileOkCount: rows.reduce((sum, row) => sum + row.compileOkCount, 0),
      totalTurnCount: rows.reduce((sum, row) => sum + row.turnCount, 0),
      totalPhraseHitCount: rows.reduce((sum, row) => sum + row.phraseHitCount, 0),
      totalPhraseCount: rows.reduce((sum, row) => sum + row.phraseCount, 0),
      totalPromotionCount: rows.reduce((sum, row) => sum + row.promotionCount, 0),
      totalUsedLearnedRouteTurnCount: rows.reduce((sum, row) => sum + row.usedLearnedRouteTurnCount, 0),
      totalActivationObservedTurnCount: rows.reduce((sum, row) => sum + row.activationObservedTurnCount, 0),
      totalActivationTakenTurnCount: rows.reduce((sum, row) => sum + row.activationTakenTurnCount, 0),
      totalSelectedContextBlockCount: rows.reduce((sum, row) => sum + row.selectedContextBlockCount, 0),
      totalSelectedContextCharCount: rows.reduce((sum, row) => sum + row.selectedContextCharCount, 0),
      totalEstimatedPromptTokenCount: rows.reduce((sum, row) => sum + row.estimatedPromptTokenCount, 0),
      totalEstimatedPromptCostUsd: rows.every((row) => row.estimatedPromptCostUsd !== null)
        ? roundValue(rows.reduce((sum, row) => sum + (row.estimatedPromptCostUsd ?? 0), 0))
        : null,
      totalLatencyMs: rows.reduce((sum, row) => sum + row.totalLatencyMs, 0),
      totalRouteSelectionLatencyMs: rows.reduce((sum, row) => sum + row.totalRouteSelectionLatencyMs, 0),
      totalPromptAssemblyLatencyMs: rows.reduce((sum, row) => sum + row.totalPromptAssemblyLatencyMs, 0),
      totalFallbackToStaticContextTurnCount: rows.reduce((sum, row) => sum + row.fallbackToStaticContextTurnCount, 0),
      totalHardRequirementViolatedTurnCount: rows.reduce((sum, row) => sum + row.hardRequirementViolatedTurnCount, 0),
      totalWarningCount: rows.reduce((sum, row) => sum + row.warningCount, 0),
    };
  });
  const traces = analyses
    .map((analysis) => {
      const candidateRow = analysis.modes.find((row) => row.mode === "learned_route");
      const baselineRow = analysis.modes.find((row) => row.mode === "graph_prior_only");
      const floorRow = analysis.modes.find((row) => row.mode === "no_brain");
      return {
        traceId: analysis.traceId,
        bundleDir: analysis.bundleDir,
        winnerMode: analysis.descriptor.bundle.summary.winnerMode as ReplayLaneMode | null,
        topScoreModes: [...analysis.topScoreModes],
        scoreSpread: analysis.scoreSpread,
        validationOk: analysis.validation.ok,
        candidateRelationVsBaseline: candidateRow && baselineRow
          ? relationFromScores(candidateRow.qualityScore, baselineRow.qualityScore)
          : null,
        candidateRelationVsFloor: candidateRow && floorRow
          ? relationFromScores(candidateRow.qualityScore, floorRow.qualityScore)
          : null,
        candidateTieOrBetterVsBaseline: candidateRow && baselineRow
          ? candidateRow.qualityScore >= baselineRow.qualityScore
          : null,
        candidateRegressionVsBaseline: candidateRow && baselineRow
          ? candidateRow.qualityScore < baselineRow.qualityScore
          : null,
        candidateRegressionVsFloor: candidateRow && floorRow
          ? candidateRow.qualityScore < floorRow.qualityScore
          : null,
        bundleHash: analysis.descriptor.bundle.bundleHash,
        scoreHash: analysis.descriptor.bundle.scoreHash,
        modes: analysis.modes.map((row) => ({ ...row })),
      };
    })
    .sort((left, right) => left.traceId.localeCompare(right.traceId));
  const turns = analyses
    .flatMap((analysis) => analysis.turns.map((turn) => ({
      ...turn,
      expectedContextPhrases: [...turn.expectedContextPhrases],
      feedbackKinds: [...turn.feedbackKinds],
      topModes: [...turn.topModes],
      ranking: turn.ranking.map((row) => ({ ...row })),
      modes: turn.modes.map((row) => ({ ...row })),
    })))
    .sort((left, right) => left.traceId.localeCompare(right.traceId) || left.turnId.localeCompare(right.turnId));
  return {
    contract: RECORDED_SESSION_REPLAY_PROOF_LANE_SUMMARY_TABLES_CONTRACT,
    sourceManifest: cloneSourceManifest(sourceManifest),
    modeOrder: [...RECORDED_SESSION_REPLAY_PROOF_LANE_MODE_ORDER],
    requestedTraceCount,
    successfulTraceCount: analyses.length,
    failedTraceCount,
    scorecard: buildExplainableScorecard({ modes, traces, turns }),
    modes,
    traces,
    turns,
  };
}

function buildPairwiseDeltas(
  analyses: RecordedSessionReplayProofLaneTraceAnalysisV1[],
  sourceManifest: RecordedSessionReplayProofLaneSourceManifestV1,
  requestedTraceCount: number,
  failedTraceCount: number,
): RecordedSessionReplayProofLanePairwiseDeltasV1 {
  const pairs: RecordedSessionReplayProofLanePairwiseRowV1[] = [];
  let turnComparisonCount = 0;
  for (let leftIndex = 0; leftIndex < RECORDED_SESSION_REPLAY_PROOF_LANE_MODE_ORDER.length; leftIndex += 1) {
    for (let rightIndex = leftIndex + 1; rightIndex < RECORDED_SESSION_REPLAY_PROOF_LANE_MODE_ORDER.length; rightIndex += 1) {
      const leftMode = RECORDED_SESSION_REPLAY_PROOF_LANE_MODE_ORDER[leftIndex];
      const rightMode = RECORDED_SESSION_REPLAY_PROOF_LANE_MODE_ORDER[rightIndex];
      const traceRows = analyses.map((analysis) => {
        const left = analysis.modes.find((candidate) => candidate.mode === leftMode);
        const right = analysis.modes.find((candidate) => candidate.mode === rightMode);
        if (!left || !right) {
          throw new Error(`Missing pairwise mode rows for ${leftMode} vs ${rightMode}`);
        }
        const leftTurns = analysis.turns.map((turn) => turn.modes.find((candidate) => candidate.mode === leftMode));
        const rightTurns = analysis.turns.map((turn) => turn.modes.find((candidate) => candidate.mode === rightMode));
        const turnWins = {
          left: 0,
          right: 0,
          ties: 0,
        };
        let maxTurnScoreSpread = 0;
        for (let index = 0; index < leftTurns.length; index += 1) {
          const leftTurn = leftTurns[index];
          const rightTurn = rightTurns[index];
          if (!leftTurn || !rightTurn) {
            throw new Error(`Missing turn pair for ${analysis.traceId}: ${leftMode} vs ${rightMode}`);
          }
          maxTurnScoreSpread = Math.max(maxTurnScoreSpread, Math.abs(leftTurn.qualityScore - rightTurn.qualityScore));
          if (leftTurn.qualityScore > rightTurn.qualityScore) {
            turnWins.left += 1;
          } else if (leftTurn.qualityScore < rightTurn.qualityScore) {
            turnWins.right += 1;
          } else {
            turnWins.ties += 1;
          }
        }
        turnComparisonCount += analysis.turns.length;
        return {
          traceId: analysis.traceId,
          bundleDir: analysis.bundleDir,
          bundleHash: analysis.descriptor.bundle.bundleHash,
          scoreHash: analysis.descriptor.bundle.scoreHash,
          qualityScoreDeltaLeftMinusRight: left.qualityScore - right.qualityScore,
          compileOkDeltaLeftMinusRight: left.compileOkCount - right.compileOkCount,
          phraseHitDeltaLeftMinusRight: left.phraseHitCount - right.phraseHitCount,
          promotionDeltaLeftMinusRight: left.promotionCount - right.promotionCount,
          turnWins,
          maxTurnScoreSpread,
        };
      });
      const traceWins = {
        left: traceRows.filter((row) => row.qualityScoreDeltaLeftMinusRight > 0).length,
        right: traceRows.filter((row) => row.qualityScoreDeltaLeftMinusRight < 0).length,
        ties: traceRows.filter((row) => row.qualityScoreDeltaLeftMinusRight === 0).length,
      };
      const turnWins = traceRows.reduce(
        (aggregate, row) => ({
          left: aggregate.left + row.turnWins.left,
          right: aggregate.right + row.turnWins.right,
          ties: aggregate.ties + row.turnWins.ties,
        }),
        { left: 0, right: 0, ties: 0 },
      );
      pairs.push({
        leftMode,
        rightMode,
        traceWins,
        traceWinRate: {
          left: roundRate(traceWins.left, traceRows.length),
          right: roundRate(traceWins.right, traceRows.length),
          ties: roundRate(traceWins.ties, traceRows.length),
        },
        turnWins,
        turnWinRate: {
          left: roundRate(turnWins.left, turnWins.left + turnWins.right + turnWins.ties),
          right: roundRate(turnWins.right, turnWins.left + turnWins.right + turnWins.ties),
          ties: roundRate(turnWins.ties, turnWins.left + turnWins.right + turnWins.ties),
        },
        aggregateDeltas: {
          qualityScoreDeltaLeftMinusRightSum: traceRows.reduce((sum, row) => sum + row.qualityScoreDeltaLeftMinusRight, 0),
          qualityScoreDeltaLeftMinusRightMean: roundRate(
            traceRows.reduce((sum, row) => sum + row.qualityScoreDeltaLeftMinusRight, 0),
            traceRows.length,
          ),
          compileOkDeltaLeftMinusRightSum: traceRows.reduce((sum, row) => sum + row.compileOkDeltaLeftMinusRight, 0),
          phraseHitDeltaLeftMinusRightSum: traceRows.reduce((sum, row) => sum + row.phraseHitDeltaLeftMinusRight, 0),
          promotionDeltaLeftMinusRightSum: traceRows.reduce((sum, row) => sum + row.promotionDeltaLeftMinusRight, 0),
        },
        traces: traceRows.sort((left, right) => left.traceId.localeCompare(right.traceId)),
      });
    }
  }
  return {
    contract: RECORDED_SESSION_REPLAY_PROOF_LANE_PAIRWISE_DELTAS_CONTRACT,
    sourceManifest: cloneSourceManifest(sourceManifest),
    modeOrder: [...RECORDED_SESSION_REPLAY_PROOF_LANE_MODE_ORDER],
    requestedTraceCount,
    successfulTraceCount: analyses.length,
    failedTraceCount,
    turnComparisonCount,
    pairs,
  };
}

function buildMatrix(
  analyses: RecordedSessionReplayProofLaneTraceAnalysisV1[],
  selector: (analysis: RecordedSessionReplayProofLaneTraceAnalysisV1, mode: ReplayLaneMode) => number[],
): RecordedSessionReplayProofLaneMatrixRowV1[] {
  return RECORDED_SESSION_REPLAY_PROOF_LANE_MODE_ORDER.map((mode) => ({
    mode,
    cells: RECORDED_SESSION_REPLAY_PROOF_LANE_MODE_ORDER.map((otherMode) => {
      if (mode === otherMode) {
        return {
          mode: otherMode,
          wins: 0,
          losses: 0,
          ties: analyses.reduce((sum, analysis) => sum + selector(analysis, mode).length, 0),
          winRate: null,
          lossRate: null,
          tieRate: null,
        };
      }
      let wins = 0;
      let losses = 0;
      let ties = 0;
      for (const analysis of analyses) {
        const leftScores = selector(analysis, mode);
        const rightScores = selector(analysis, otherMode);
        for (let index = 0; index < leftScores.length; index += 1) {
          const left = leftScores[index];
          const right = rightScores[index];
          if (left > right) {
            wins += 1;
          } else if (left < right) {
            losses += 1;
          } else {
            ties += 1;
          }
        }
      }
      const total = wins + losses + ties;
      return {
        mode: otherMode,
        wins,
        losses,
        ties,
        winRate: roundRate(wins, total),
        lossRate: roundRate(losses, total),
        tieRate: roundRate(ties, total),
      };
    }),
  }));
}

function buildWinRateMatrix(
  analyses: RecordedSessionReplayProofLaneTraceAnalysisV1[],
  sourceManifest: RecordedSessionReplayProofLaneSourceManifestV1,
  requestedTraceCount: number,
  failedTraceCount: number,
): RecordedSessionReplayProofLaneWinRateMatrixV1 {
  return {
    contract: RECORDED_SESSION_REPLAY_PROOF_LANE_WIN_RATE_MATRIX_CONTRACT,
    sourceManifest: cloneSourceManifest(sourceManifest),
    modeOrder: [...RECORDED_SESSION_REPLAY_PROOF_LANE_MODE_ORDER],
    requestedTraceCount,
    successfulTraceCount: analyses.length,
    failedTraceCount,
    traceComparisonCount: analyses.length,
    turnComparisonCount: analyses.reduce((sum, analysis) => sum + analysis.turns.length, 0),
    traceMatrix: buildMatrix(analyses, (analysis, mode) => {
      const row = analysis.modes.find((candidate) => candidate.mode === mode);
      if (!row) {
        throw new Error(`Missing trace matrix mode row for ${mode}`);
      }
      return [row.qualityScore];
    }),
    turnMatrix: buildMatrix(analyses, (analysis, mode) =>
      analysis.turns.map((turn) => {
        const row = turn.modes.find((candidate) => candidate.mode === mode);
        if (!row) {
          throw new Error(`Missing turn matrix mode row for ${mode}`);
        }
        return row.qualityScore;
      }),
    ),
  };
}

function buildWorkedTracesMarkdown(
  analyses: RecordedSessionReplayProofLaneTraceAnalysisV1[],
  sourceManifest: RecordedSessionReplayProofLaneSourceManifestV1,
  workedTraceLimit: number,
): string {
  const sorted = [...analyses].sort(
    (left, right) => right.scoreSpread - left.scoreSpread || left.traceId.localeCompare(right.traceId),
  );
  const selected = (sorted.filter((analysis) => analysis.scoreSpread > 0).length > 0
    ? sorted.filter((analysis) => analysis.scoreSpread > 0)
    : sorted
  ).slice(0, workedTraceLimit);
  const omittedCount = Math.max(0, sorted.length - selected.length);
  const lines: string[] = [
    "# Worked Traces",
    "",
    `- traces included: ${selected.length}/${sorted.length}`,
    `- selection rule: highest bundle score spread first, then trace id; turns ordered by per-turn score spread`,
    "- note: qualityScore and winnerMode are internal deterministic replay diagnostics, not the public/operator scorecard.",
  ];
  if (sourceManifest.manifestId !== null || sourceManifest.manifestDigest !== null) {
    lines.push(
      `- source manifest: \`${sourceManifest.manifestId ?? "unknown"}\` (${sourceManifest.manifestContract ?? "unknown"}, ${shortDigest(sourceManifest.manifestDigest)})`,
    );
  }
  if (omittedCount > 0) {
    lines.push(`- omitted traces: ${omittedCount} (see _lane/summary-tables.json for the complete table)`);
  }
  lines.push("");
  if (selected.length === 0) {
    lines.push("No successful replay bundles were generated.");
    return lines.join("\n");
  }
  for (const analysis of selected) {
    const rankedModes = [...analysis.modes].sort(compareQualityRows);
    const candidateTrace = analysis.modes.find((row) => row.mode === "learned_route");
    const baselineTrace = analysis.modes.find((row) => row.mode === "graph_prior_only");
    const floorTrace = analysis.modes.find((row) => row.mode === "no_brain");
    lines.push(`## ${analysis.traceId}`);
    lines.push("");
    lines.push(`- bundle dir: \`${analysis.bundleDir}\``);
    lines.push(`- learned_route vs approved prior: \`${candidateTrace && baselineTrace ? relationFromScores(candidateTrace.qualityScore, baselineTrace.qualityScore) : "unknown"}\``);
    lines.push(`- learned_route vs no_brain floor: \`${candidateTrace && floorTrace ? relationFromScores(candidateTrace.qualityScore, floorTrace.qualityScore) : "unknown"}\``);
    lines.push(`- diagnostic winner: \`${analysis.descriptor.bundle.summary.winnerMode ?? "none"}\``);
    lines.push(`- diagnostic top score modes: \`${analysis.topScoreModes.join("`, `")}\``);
    lines.push(`- score spread: ${analysis.scoreSpread}`);
    lines.push("");
    lines.push("| mode | diagnostic quality | compile ok | required-context recall | promotions | warnings |");
    lines.push("| --- | ---: | ---: | ---: | ---: | ---: |");
    for (const row of rankedModes) {
      lines.push(
        `| ${row.mode} | ${row.qualityScore} | ${row.compileOkCount}/${row.turnCount} | ${row.phraseHitCount}/${row.phraseCount} | ${row.promotionCount} | ${row.warningCount} |`,
      );
    }
    const traceTurns = [...analysis.turns]
      .sort(
        (left, right) =>
          right.scoreSpread - left.scoreSpread ||
          left.topModes.length - right.topModes.length ||
          left.turnId.localeCompare(right.turnId),
      )
      .slice(0, DEFAULT_WORKED_TURN_LIMIT);
    for (const turn of traceTurns) {
      lines.push("");
      lines.push(`### ${turn.turnId}`);
      lines.push("");
      lines.push(`- user: ${turn.userMessagePreview || "none"}`);
      lines.push(
        `- expected phrases: ${turn.expectedContextPhrases.length > 0 ? turn.expectedContextPhrases.map((phrase) => `\`${phrase}\``).join(", ") : "none"}`,
      );
      lines.push(`- feedback kinds: ${turn.feedbackKinds.length > 0 ? turn.feedbackKinds.map((kind) => `\`${kind}\``).join(", ") : "none"}`);
      lines.push(`- learned_route vs approved prior: \`${turn.candidateRelationVsBaseline ?? "unknown"}\``);
      lines.push(`- diagnostic top modes: ${turn.topModes.map((mode) => `\`${mode}\``).join(", ")} (spread ${turn.scoreSpread})`);
      lines.push("");
      lines.push("| mode | phase | diagnostic quality | compile | required-context recall | activation | source | latency ms | promoted | selection | context preview |");
      lines.push("| --- | --- | ---: | --- | ---: | --- | --- | ---: | --- | --- | --- |");
      for (const row of [...turn.modes].sort(compareQualityRows)) {
        lines.push(
          `| ${row.mode} | ${row.phase} | ${row.qualityScore} | ${row.compileOk ? "yes" : "no"} | ${row.phraseHitCount}/${row.phraseCount} | ${row.activationTaken === null ? "n/a" : row.activationTaken ? "yes" : "no"} | ${row.activationSource ?? "none"} | ${row.timing.totalMs ?? "n/a"} | ${row.promoted ? "yes" : "no"} | ${shortDigest(row.selectionDigest)} | ${row.selectedContextPreview ?? "none"} |`,
        );
      }
    }
    lines.push("");
  }
  return lines.join("\n");
}

function buildLaneReadme(
  summaryTables: RecordedSessionReplayProofLaneSummaryTablesV1,
  pairwiseDeltas: RecordedSessionReplayProofLanePairwiseDeltasV1,
  index: RecordedSessionReplayProofLaneIndexV1,
): string {
  const scorecard = summaryTables.scorecard;
  const lines: string[] = [
    "# Recorded Session Replay Proof Lane",
    "",
    `- requested traces: ${summaryTables.requestedTraceCount}`,
    `- successful traces: ${summaryTables.successfulTraceCount}`,
    `- failed traces: ${summaryTables.failedTraceCount}`,
    `- mode order: \`${summaryTables.modeOrder.join("`, `")}\``,
    "- note: these lane aggregates are internal deterministic replay diagnostics; use the explainable eval scorecard for public/operator reporting.",
  ];
  if (index.sourceManifest.manifestId !== null || index.sourceManifest.manifestDigest !== null) {
    lines.push(
      `- source manifest: \`${index.sourceManifest.manifestId ?? "unknown"}\` (${index.sourceManifest.manifestContract ?? "unknown"}, ${shortDigest(index.sourceManifest.manifestDigest)})`,
    );
  }
  if (index.assumptions.length > 0) {
    lines.push(`- assumptions: ${index.assumptions.map((assumption) => `\`${assumption}\``).join(", ")}`);
  }
  if (index.failedTraceIds.length > 0) {
    lines.push(`- failed trace ids: ${index.failedTraceIds.map((traceId) => `\`${traceId}\``).join(", ")}`);
  }
  lines.push("");
  lines.push("## Explainable Scorecard");
  lines.push(`- learned_route tie-or-better vs graph_prior_only (traces): ${formatCountRate(scorecard.traceTieOrBetterVsBaseline)}`);
  lines.push(`- learned_route vs graph_prior_only (traces): ${formatOutcomeBreakdown(scorecard.traceOutcomeVsBaseline)}`);
  lines.push(`- learned_route tie-or-better vs graph_prior_only (turns): ${formatCountRate(scorecard.turnTieOrBetterVsBaseline)}`);
  lines.push(`- learned_route vs graph_prior_only (turns): ${formatOutcomeBreakdown(scorecard.turnOutcomeVsBaseline)}`);
  lines.push(`- regressions vs graph_prior_only: ${formatCountRate(scorecard.regressionVsBaseline)}`);
  lines.push(`- regressions vs no_brain floor: ${formatCountRate(scorecard.regressionVsFloor)} (critical regressions: ${scorecard.criticalRegressionCount})`);
  lines.push(`- required-context recall: ${scorecard.requiredContextRecall.summary}`);
  lines.push(`- correction absorption: ${scorecard.correctionAbsorption.summary}`);
  lines.push(`- activation precision: ${scorecard.activationPrecision.summary}`);
  lines.push(`- activation precision proxy: ${scorecard.activationPrecisionProxy.summary}`);
  lines.push(`- success-adjusted economics: ${scorecard.successAdjustedEconomics.summary}`);
  lines.push(`- fail-open: ${scorecard.failOpen.summary}`);
  lines.push("");
  lines.push("## Diagnostic Mode Summary");
  lines.push("| mode | traces | diagnostic top-rank | shared top score | mean quality | compile ok | required-context recall | promotions | warnings |");
  lines.push("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |");
  for (const row of summaryTables.modes) {
    lines.push(
      `| ${row.mode} | ${row.traceCount} | ${row.rankedWinnerCount} | ${row.sharedTopScoreTraceCount} | ${row.meanQualityScore ?? "none"} | ${row.totalCompileOkCount}/${row.totalTurnCount} | ${row.totalPhraseHitCount}/${row.totalPhraseCount} | ${row.totalPromotionCount} | ${row.totalWarningCount} |`,
    );
  }
  lines.push("");
  lines.push("## Diagnostic Pairwise Deltas");
  lines.push("| pair | trace outcomes (left/right/tied) | turn outcomes (left/right/tied) | mean quality delta | compile delta sum | required-context delta sum | promotion delta sum |");
  lines.push("| --- | --- | --- | ---: | ---: | ---: | ---: |");
  for (const pair of pairwiseDeltas.pairs) {
    lines.push(
      `| ${pair.leftMode} - ${pair.rightMode} | ${pair.traceWins.left}-${pair.traceWins.right}-${pair.traceWins.ties} | ${pair.turnWins.left}-${pair.turnWins.right}-${pair.turnWins.ties} | ${pair.aggregateDeltas.qualityScoreDeltaLeftMinusRightMean ?? "none"} | ${pair.aggregateDeltas.compileOkDeltaLeftMinusRightSum} | ${pair.aggregateDeltas.phraseHitDeltaLeftMinusRightSum} | ${pair.aggregateDeltas.promotionDeltaLeftMinusRightSum} |`,
      );
  }
  lines.push("");
  lines.push("## Artifacts");
  lines.push(`- summary: \`${RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.summary}\``);
  lines.push(`- closeout: \`${RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.closeout}\``);
  lines.push(`- index: \`${RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.index}\``);
  lines.push(`- summary tables: \`${RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.summaryTables}\``);
  lines.push(`- pairwise deltas: \`${RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.pairwiseDeltas}\``);
  lines.push(`- win-rate matrix: \`${RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.winRateMatrix}\``);
  lines.push(`- worked traces: \`${RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.workedTraces}\``);
  lines.push(`- generation report: \`${RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.generationReport}\``);
  return lines.join("\n");
}

function buildLaneCloseout(
  sourceManifest: RecordedSessionReplayProofLaneSourceManifestV1,
  summaryTables: RecordedSessionReplayProofLaneSummaryTablesV1,
  index: RecordedSessionReplayProofLaneIndexV1,
  artifacts: RecordedSessionReplayProofLaneCloseoutArtifactV1[],
): RecordedSessionReplayProofLaneCloseoutV1 {
  const failedTraceIds = [...index.failedTraceIds];
  const successfulTraceCount = summaryTables.successfulTraceCount;
  let verdict: RecordedSessionReplayProofLaneCloseoutV1["verdict"];
  if (successfulTraceCount === 0) {
    verdict = {
      verdict: "no_successful_replays",
      severity: "error",
      why: "No replay proof bundles were generated successfully, so the aggregate outputs are observationally empty.",
    };
  } else if (failedTraceIds.length > 0) {
    verdict = {
      verdict: "partial_proof",
      severity: "warn",
      why: `${successfulTraceCount}/${summaryTables.requestedTraceCount} replay proof bundles generated successfully; inspect generation-report.json before trusting the aggregate view.`,
    };
  } else {
    verdict = {
      verdict: "success_and_proven",
      severity: "none",
      why: `${successfulTraceCount}/${summaryTables.requestedTraceCount} replay proof bundles generated successfully and produced deterministic aggregate outputs.`,
    };
  }
  return {
    contract: RECORDED_SESSION_REPLAY_PROOF_LANE_CLOSEOUT_CONTRACT,
    hashAlgorithm: "sha256",
    laneDir: RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.laneDir,
    sourceManifest: cloneSourceManifest(sourceManifest),
    verdict,
    requestedTraceCount: summaryTables.requestedTraceCount,
    successfulTraceCount,
    failedTraceCount: summaryTables.failedTraceCount,
    failedTraceIds,
    modeOrder: [...summaryTables.modeOrder],
    scorecard: summaryTables.scorecard,
    winnerModeCounts: summaryTables.modes.map((row) => ({
      mode: row.mode,
      rankedWinnerCount: row.rankedWinnerCount,
      sharedTopScoreTraceCount: row.sharedTopScoreTraceCount,
    })),
    traceHashes: summaryTables.traces.map((trace) => ({
      traceId: trace.traceId,
      bundleHash: trace.bundleHash,
      scoreHash: trace.scoreHash,
      winnerMode: trace.winnerMode,
      scoreSpread: trace.scoreSpread,
      candidateRelationVsBaseline: trace.candidateRelationVsBaseline,
      candidateRelationVsFloor: trace.candidateRelationVsFloor,
    })),
    files: artifacts.map((artifact) => ({ ...artifact })),
  };
}

function buildLaneSummary(
  closeout: RecordedSessionReplayProofLaneCloseoutV1,
  artifacts: RecordedSessionReplayProofLaneCloseoutArtifactV1[],
): string {
  const scorecard = closeout.scorecard;
  const lines: string[] = [
    "# Recorded Session Replay Proof Lane Closeout",
    "",
    `- verdict: **${closeout.verdict.verdict}**`,
    `- severity: **${closeout.verdict.severity}**`,
    `- why: ${closeout.verdict.why}`,
    `- requested traces: ${closeout.requestedTraceCount}`,
    `- successful traces: ${closeout.successfulTraceCount}`,
    `- failed traces: ${closeout.failedTraceCount}`,
    "- note: winner counts below are internal replay diagnostics only.",
  ];
  if (closeout.sourceManifest.manifestId !== null || closeout.sourceManifest.manifestDigest !== null) {
    lines.push(
      `- source manifest: \`${closeout.sourceManifest.manifestId ?? "unknown"}\` (${closeout.sourceManifest.manifestContract ?? "unknown"}, ${shortDigest(closeout.sourceManifest.manifestDigest)})`,
    );
  }
  if (closeout.failedTraceIds.length > 0) {
    lines.push(`- failed trace ids: ${closeout.failedTraceIds.map((traceId) => `\`${traceId}\``).join(", ")}`);
  }
  lines.push("");
  lines.push("## Explainable Scorecard");
  lines.push(`- learned_route tie-or-better vs graph_prior_only (traces): ${formatCountRate(scorecard.traceTieOrBetterVsBaseline)}`);
  lines.push(`- learned_route vs graph_prior_only (traces): ${formatOutcomeBreakdown(scorecard.traceOutcomeVsBaseline)}`);
  lines.push(`- learned_route tie-or-better vs graph_prior_only (turns): ${formatCountRate(scorecard.turnTieOrBetterVsBaseline)}`);
  lines.push(`- learned_route vs graph_prior_only (turns): ${formatOutcomeBreakdown(scorecard.turnOutcomeVsBaseline)}`);
  lines.push(`- regressions vs graph_prior_only: ${formatCountRate(scorecard.regressionVsBaseline)}`);
  lines.push(`- regressions vs no_brain floor: ${formatCountRate(scorecard.regressionVsFloor)} (critical regressions: ${scorecard.criticalRegressionCount})`);
  lines.push(`- required-context recall: ${scorecard.requiredContextRecall.summary}`);
  lines.push(`- correction absorption: ${scorecard.correctionAbsorption.summary}`);
  lines.push(`- activation precision: ${scorecard.activationPrecision.summary}`);
  lines.push(`- activation precision proxy: ${scorecard.activationPrecisionProxy.summary}`);
  lines.push(`- success-adjusted economics: ${scorecard.successAdjustedEconomics.summary}`);
  lines.push(`- fail-open: ${scorecard.failOpen.summary}`);
  lines.push("");
  lines.push("## Diagnostic Tie-Break Counts");
  lines.push("| mode | diagnostic top-rank | shared top score traces |");
  lines.push("| --- | ---: | ---: |");
  for (const row of closeout.winnerModeCounts) {
    lines.push(`| ${row.mode} | ${row.rankedWinnerCount} | ${row.sharedTopScoreTraceCount} |`);
  }
  lines.push("");
  lines.push("## Trace Hashes");
  lines.push("| trace | learned_route vs prior | learned_route vs floor | diagnostic top mode | spread | bundle hash | score hash |");
  lines.push("| --- | --- | --- | --- | ---: | --- | --- |");
  for (const trace of closeout.traceHashes) {
    lines.push(
      `| ${trace.traceId} | ${trace.candidateRelationVsBaseline ?? "unknown"} | ${trace.candidateRelationVsFloor ?? "unknown"} | ${trace.winnerMode ?? "none"} | ${trace.scoreSpread} | ${shortDigest(trace.bundleHash)} | ${shortDigest(trace.scoreHash)} |`,
    );
  }
  lines.push("");
  lines.push("## Deterministic Outputs");
  lines.push("| role | path | contract | digest |");
  lines.push("| --- | --- | --- | --- |");
  for (const artifact of artifacts) {
    lines.push(`| ${artifact.role} | ${artifact.path} | ${artifact.contract ?? "none"} | ${artifact.digest} |`);
  }
  return lines.join("\n");
}

export function writeRecordedSessionReplayProofLane(
  input: WriteRecordedSessionReplayProofLaneInputV1,
): RecordedSessionReplayProofLaneDescriptorV1 {
  const artifactRoot = path.resolve(input.artifactRoot);
  const laneDir = path.join(artifactRoot, RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.laneDir);
  const workedTraceLimit = normalizeWorkedTraceLimit(input.workedTraceLimit);
  const assumptions = normalizeStringArray(input.assumptions);
  const sourceManifest = readSourceManifest(input.sourceManifestPath ?? null);
  const pricingTable = loadPricingTable();
  ensureDir(artifactRoot);
  rmSync(laneDir, { recursive: true, force: true });
  ensureDir(laneDir);
  const successfulBundles: RecordedSessionReplayProofBundleDescriptorV1[] = [];
  const traceAnalyses: RecordedSessionReplayProofLaneTraceAnalysisV1[] = [];
  const generationEntries: RecordedSessionReplayProofLaneGenerationEntryV1[] = [];
  for (const traceInput of input.traces) {
    const traceId = traceInput.trace.traceId;
    const defaultBundleDir = path.join(artifactRoot, sanitizeTraceBundleDirName(traceId));
    const bundleDir = path.resolve(traceInput.bundleDir ?? defaultBundleDir);
    const validationPath = path.join(bundleDir, "validation-report.json");
    try {
      const descriptor = writeRecordedSessionReplayProofBundle({
        rootDir: bundleDir,
        trace: traceInput.trace,
        scratchRootDir: input.scratchRootDir ?? undefined,
      });
      const validation = validateRecordedSessionReplayProofBundle(bundleDir);
      writeJson(validationPath, validation);
      successfulBundles.push(descriptor);
      traceAnalyses.push(buildTraceAnalysis(artifactRoot, descriptor, validation, pricingTable));
      generationEntries.push({
        traceId,
        tracePath: traceInput.tracePath ?? null,
        bundleDir,
        validationPath,
        result: validation.ok ? "passed" : "failed",
        validation,
        error: validation.ok ? null : validation.errors.join("; "),
      });
    } catch (error) {
      generationEntries.push({
        traceId,
        tracePath: traceInput.tracePath ?? null,
        bundleDir,
        validationPath: null,
        result: "failed",
        validation: null,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  }
  const failedTraceIds = generationEntries
    .filter((entry) => entry.result === "failed")
    .map((entry) => entry.traceId)
    .sort((left, right) => left.localeCompare(right));
  const summaryTables = buildSummaryTables(traceAnalyses, sourceManifest, input.traces.length, failedTraceIds.length);
  const pairwiseDeltas = buildPairwiseDeltas(traceAnalyses, sourceManifest, input.traces.length, failedTraceIds.length);
  const winRateMatrix = buildWinRateMatrix(traceAnalyses, sourceManifest, input.traces.length, failedTraceIds.length);
  const index: RecordedSessionReplayProofLaneIndexV1 = {
    contract: RECORDED_SESSION_REPLAY_PROOF_LANE_INDEX_CONTRACT,
    laneDir: RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.laneDir,
    sourceManifest: cloneSourceManifest(sourceManifest),
    modeOrder: [...RECORDED_SESSION_REPLAY_PROOF_LANE_MODE_ORDER],
    requestedTraceCount: input.traces.length,
    successfulTraceCount: successfulBundles.length,
    failedTraceCount: failedTraceIds.length,
    failedTraceIds,
    assumptions,
    files: {
      readme: RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.readme,
      summary: RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.summary,
      closeout: RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.closeout,
      index: RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.index,
      summaryTables: RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.summaryTables,
      pairwiseDeltas: RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.pairwiseDeltas,
      winRateMatrix: RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.winRateMatrix,
      workedTraces: RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.workedTraces,
      generationReport: RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.generationReport,
    },
    traceBundles: traceAnalyses
      .map((analysis) => ({
        ...(() => {
          const candidateRow = analysis.modes.find((row) => row.mode === "learned_route");
          const baselineRow = analysis.modes.find((row) => row.mode === "graph_prior_only");
          const floorRow = analysis.modes.find((row) => row.mode === "no_brain");
          return {
            candidateRelationVsBaseline: candidateRow && baselineRow
              ? relationFromScores(candidateRow.qualityScore, baselineRow.qualityScore)
              : null,
            candidateRelationVsFloor: candidateRow && floorRow
              ? relationFromScores(candidateRow.qualityScore, floorRow.qualityScore)
              : null,
            candidateTieOrBetterVsBaseline: candidateRow && baselineRow
              ? candidateRow.qualityScore >= baselineRow.qualityScore
              : null,
            candidateRegressionVsBaseline: candidateRow && baselineRow
              ? candidateRow.qualityScore < baselineRow.qualityScore
              : null,
            candidateRegressionVsFloor: candidateRow && floorRow
              ? candidateRow.qualityScore < floorRow.qualityScore
              : null,
          };
        })(),
        traceId: analysis.traceId,
        bundleDir: analysis.bundleDir,
        validationOk: analysis.validation.ok,
        winnerMode: analysis.descriptor.bundle.summary.winnerMode as ReplayLaneMode | null,
        topScoreModes: [...analysis.topScoreModes],
        scoreSpread: analysis.scoreSpread,
        bundleHash: analysis.descriptor.bundle.bundleHash,
        scoreHash: analysis.descriptor.bundle.scoreHash,
      }))
      .sort((left, right) => left.traceId.localeCompare(right.traceId)),
  };
  const generationReport: RecordedSessionReplayProofLaneGenerationReportV1 = {
    contract: RECORDED_SESSION_REPLAY_PROOF_LANE_GENERATION_REPORT_CONTRACT,
    artifactRoot,
    laneDir,
    requestedTraceCount: input.traces.length,
    successfulTraceCount: successfulBundles.length,
    failedTraceCount: failedTraceIds.length,
    sourceManifestPath: input.sourceManifestPath ?? null,
    sourceManifest: cloneSourceManifest(sourceManifest),
    assumptions,
    entries: generationEntries,
  };
  const workedTraces = buildWorkedTracesMarkdown(traceAnalyses, sourceManifest, workedTraceLimit);
  const laneReadme = buildLaneReadme(summaryTables, pairwiseDeltas, index);
  const readmeText = laneReadme.endsWith("\n") ? laneReadme : `${laneReadme}\n`;
  const indexText = renderJson(index);
  const summaryTablesText = renderJson(summaryTables);
  const pairwiseDeltasText = renderJson(pairwiseDeltas);
  const winRateMatrixText = renderJson(winRateMatrix);
  const workedTracesText = workedTraces.endsWith("\n") ? workedTraces : `${workedTraces}\n`;
  const generationReportText = renderJson(generationReport);
  const deterministicArtifacts = [
    buildCloseoutArtifact("readme", RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.readme, readmeText, null),
    buildCloseoutArtifact("index", RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.index, indexText, index.contract),
    buildCloseoutArtifact("summary-tables", RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.summaryTables, summaryTablesText, summaryTables.contract),
    buildCloseoutArtifact("pairwise-deltas", RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.pairwiseDeltas, pairwiseDeltasText, pairwiseDeltas.contract),
    buildCloseoutArtifact("win-rate-matrix", RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.winRateMatrix, winRateMatrixText, winRateMatrix.contract),
    buildCloseoutArtifact("worked-traces", RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.workedTraces, workedTracesText, null),
    buildCloseoutArtifact("generation-report", RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.generationReport, generationReportText, generationReport.contract),
  ];
  const provisionalCloseout = buildLaneCloseout(sourceManifest, summaryTables, index, deterministicArtifacts);
  const summaryText = `${buildLaneSummary(provisionalCloseout, deterministicArtifacts)}\n`;
  const summaryArtifact = buildCloseoutArtifact("summary", RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.summary, summaryText, null);
  const closeout = buildLaneCloseout(sourceManifest, summaryTables, index, [...deterministicArtifacts, summaryArtifact]);
  const closeoutText = renderJson(closeout);
  const readmePath = path.join(laneDir, RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.readme);
  const summaryPath = path.join(laneDir, RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.summary);
  const closeoutPath = path.join(laneDir, RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.closeout);
  const indexPath = path.join(laneDir, RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.index);
  const summaryTablesPath = path.join(laneDir, RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.summaryTables);
  const pairwiseDeltasPath = path.join(laneDir, RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.pairwiseDeltas);
  const winRateMatrixPath = path.join(laneDir, RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.winRateMatrix);
  const workedTracesPath = path.join(laneDir, RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.workedTraces);
  const generationReportPath = path.join(laneDir, RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.generationReport);
  writeText(readmePath, readmeText);
  writeText(summaryPath, summaryText);
  writeText(closeoutPath, closeoutText);
  writeText(indexPath, indexText);
  writeText(summaryTablesPath, summaryTablesText);
  writeText(pairwiseDeltasPath, pairwiseDeltasText);
  writeText(winRateMatrixPath, winRateMatrixText);
  writeText(workedTracesPath, workedTracesText);
  writeText(generationReportPath, generationReportText);
  return {
    artifactRoot,
    laneDir,
    readmePath,
    summaryPath,
    closeoutPath,
    indexPath,
    summaryTablesPath,
    pairwiseDeltasPath,
    winRateMatrixPath,
    workedTracesPath,
    generationReportPath,
    index,
    summaryTables,
    pairwiseDeltas,
    winRateMatrix,
    closeout,
    generationReport,
    successfulBundles,
  };
}
