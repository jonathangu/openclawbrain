import { mkdirSync, rmSync, writeFileSync } from "node:fs";
import path from "node:path";
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
  index: "index.json",
  summaryTables: "summary-tables.json",
  pairwiseDeltas: "pairwise-deltas.json",
  winRateMatrix: "win-rate-matrix.json",
  workedTraces: "worked-traces.md",
  generationReport: "generation-report.json",
} as const;

const DEFAULT_WORKED_TRACE_LIMIT = 8;
const DEFAULT_WORKED_TURN_LIMIT = 2;

type ReplayLaneMode = (typeof RECORDED_SESSION_REPLAY_PROOF_LANE_MODE_ORDER)[number];

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
  activePackId: string | null;
  selectionDigest: string | null;
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
  totalWarningCount: number;
}

export interface RecordedSessionReplayProofLaneSummaryTablesV1 {
  contract: "recorded_session_replay_proof_lane_summary_tables.v1";
  modeOrder: ReplayLaneMode[];
  requestedTraceCount: number;
  successfulTraceCount: number;
  failedTraceCount: number;
  modes: RecordedSessionReplayProofLaneModeSummaryRowV1[];
  traces: RecordedSessionReplayProofLaneTraceSummaryRowV1[];
  turns: RecordedSessionReplayProofLaneTurnSummaryRowV1[];
}

interface RecordedSessionReplayProofLanePairwiseTraceDeltaV1 {
  traceId: string;
  bundleDir: string;
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
  contract: "recorded_session_replay_proof_lane_pairwise_deltas.v1";
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
  contract: "recorded_session_replay_proof_lane_win_rate_matrix.v1";
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
  bundleHash: string;
  scoreHash: string;
}

export interface RecordedSessionReplayProofLaneIndexV1 {
  contract: "recorded_session_replay_proof_lane_index.v1";
  laneDir: string;
  modeOrder: ReplayLaneMode[];
  requestedTraceCount: number;
  successfulTraceCount: number;
  failedTraceCount: number;
  failedTraceIds: string[];
  assumptions: string[];
  files: {
    readme: string;
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
  contract: "recorded_session_replay_proof_lane_generation_report.v1";
  artifactRoot: string;
  laneDir: string;
  requestedTraceCount: number;
  successfulTraceCount: number;
  failedTraceCount: number;
  sourceManifestPath: string | null;
  assumptions: string[];
  entries: RecordedSessionReplayProofLaneGenerationEntryV1[];
}

export interface RecordedSessionReplayProofLaneDescriptorV1 {
  artifactRoot: string;
  laneDir: string;
  readmePath: string;
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

function buildTraceAnalysis(
  artifactRoot: string,
  descriptor: RecordedSessionReplayProofBundleDescriptorV1,
  validation: RecordedSessionReplayProofBundleValidationV1,
): RecordedSessionReplayProofLaneTraceAnalysisV1 {
  const bundleDir = portableRelativePath(artifactRoot, descriptor.rootDir);
  const modes = RECORDED_SESSION_REPLAY_PROOF_LANE_MODE_ORDER.map((mode) => {
    const report = findModeReport(descriptor, mode);
    return {
      mode,
      qualityScore: report.summary.qualityScore,
      compileOkCount: report.summary.compileOkCount,
      turnCount: report.turns.length,
      phraseHitCount: report.summary.phraseHitCount,
      phraseCount: report.summary.phraseCount,
      promotionCount: report.summary.promotionCount,
      usedLearnedRouteTurnCount: report.summary.usedLearnedRouteTurnCount,
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
        activePackId: turn.activePackId,
        selectionDigest: turn.selectionDigest,
        selectedContextPreview: previewText(turn.selectedContextTexts.join(" || "), 140),
      };
    });
    const ranking = turnModes
      .map((row) => ({
        mode: row.mode,
        qualityScore: row.qualityScore,
      }))
      .sort(compareQualityRows);
    return {
      traceId: descriptor.bundle.traceId,
      bundleDir,
      turnId: traceTurnId,
      userMessagePreview: previewText(traceTurn.userMessage, 140) ?? "",
      expectedContextPhrases: [...(traceTurn.expectedContextPhrases ?? [])],
      feedbackKinds: traceFeedbackKinds(traceTurn),
      scoreSpread: scoreSpread(turnModes),
      topModes: buildTopScoreModes(turnModes),
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
  requestedTraceCount: number,
  failedTraceCount: number,
): RecordedSessionReplayProofLaneSummaryTablesV1 {
  return {
    contract: "recorded_session_replay_proof_lane_summary_tables.v1",
    modeOrder: [...RECORDED_SESSION_REPLAY_PROOF_LANE_MODE_ORDER],
    requestedTraceCount,
    successfulTraceCount: analyses.length,
    failedTraceCount,
    modes: RECORDED_SESSION_REPLAY_PROOF_LANE_MODE_ORDER.map((mode) => {
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
        totalWarningCount: rows.reduce((sum, row) => sum + row.warningCount, 0),
      };
    }),
    traces: analyses
      .map((analysis) => ({
        traceId: analysis.traceId,
        bundleDir: analysis.bundleDir,
        winnerMode: analysis.descriptor.bundle.summary.winnerMode as ReplayLaneMode | null,
        topScoreModes: [...analysis.topScoreModes],
        scoreSpread: analysis.scoreSpread,
        validationOk: analysis.validation.ok,
        bundleHash: analysis.descriptor.bundle.bundleHash,
        scoreHash: analysis.descriptor.bundle.scoreHash,
        modes: analysis.modes.map((row) => ({ ...row })),
      }))
      .sort((left, right) => left.traceId.localeCompare(right.traceId)),
    turns: analyses
      .flatMap((analysis) => analysis.turns.map((turn) => ({
        ...turn,
        expectedContextPhrases: [...turn.expectedContextPhrases],
        feedbackKinds: [...turn.feedbackKinds],
        topModes: [...turn.topModes],
        ranking: turn.ranking.map((row) => ({ ...row })),
        modes: turn.modes.map((row) => ({ ...row })),
      })))
      .sort((left, right) => left.traceId.localeCompare(right.traceId) || left.turnId.localeCompare(right.turnId)),
  };
}

function buildPairwiseDeltas(
  analyses: RecordedSessionReplayProofLaneTraceAnalysisV1[],
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
    contract: "recorded_session_replay_proof_lane_pairwise_deltas.v1",
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
  requestedTraceCount: number,
  failedTraceCount: number,
): RecordedSessionReplayProofLaneWinRateMatrixV1 {
  return {
    contract: "recorded_session_replay_proof_lane_win_rate_matrix.v1",
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
  ];
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
    lines.push(`## ${analysis.traceId}`);
    lines.push("");
    lines.push(`- bundle dir: \`${analysis.bundleDir}\``);
    lines.push(`- ranked winner: \`${analysis.descriptor.bundle.summary.winnerMode ?? "none"}\``);
    lines.push(`- top score modes: \`${analysis.topScoreModes.join("`, `")}\``);
    lines.push(`- score spread: ${analysis.scoreSpread}`);
    lines.push("");
    lines.push("| mode | quality | compile ok | phrase hits | promotions | warnings |");
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
      lines.push(`- top modes: ${turn.topModes.map((mode) => `\`${mode}\``).join(", ")} (spread ${turn.scoreSpread})`);
      lines.push("");
      lines.push("| mode | phase | quality | compile | phrase hits | learned route | promoted | selection | context preview |");
      lines.push("| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |");
      for (const row of [...turn.modes].sort(compareQualityRows)) {
        lines.push(
          `| ${row.mode} | ${row.phase} | ${row.qualityScore} | ${row.compileOk ? "yes" : "no"} | ${row.phraseHitCount}/${row.phraseCount} | ${row.usedLearnedRouteFn ? "yes" : "no"} | ${row.promoted ? "yes" : "no"} | ${shortDigest(row.selectionDigest)} | ${row.selectedContextPreview ?? "none"} |`,
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
  const lines: string[] = [
    "# Recorded Session Replay Proof Lane",
    "",
    `- requested traces: ${summaryTables.requestedTraceCount}`,
    `- successful traces: ${summaryTables.successfulTraceCount}`,
    `- failed traces: ${summaryTables.failedTraceCount}`,
    `- mode order: \`${summaryTables.modeOrder.join("`, `")}\``,
  ];
  if (index.assumptions.length > 0) {
    lines.push(`- assumptions: ${index.assumptions.map((assumption) => `\`${assumption}\``).join(", ")}`);
  }
  if (index.failedTraceIds.length > 0) {
    lines.push(`- failed trace ids: ${index.failedTraceIds.map((traceId) => `\`${traceId}\``).join(", ")}`);
  }
  lines.push("");
  lines.push("## Mode Summary");
  lines.push("| mode | traces | ranked winners | shared top score | mean quality | compile ok | phrase hits | promotions | warnings |");
  lines.push("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |");
  for (const row of summaryTables.modes) {
    lines.push(
      `| ${row.mode} | ${row.traceCount} | ${row.rankedWinnerCount} | ${row.sharedTopScoreTraceCount} | ${row.meanQualityScore ?? "none"} | ${row.totalCompileOkCount}/${row.totalTurnCount} | ${row.totalPhraseHitCount}/${row.totalPhraseCount} | ${row.totalPromotionCount} | ${row.totalWarningCount} |`,
    );
  }
  lines.push("");
  lines.push("## Pairwise Deltas");
  lines.push("| pair | trace record | turn record | mean quality delta | compile delta sum | phrase-hit delta sum | promotion delta sum |");
  lines.push("| --- | --- | --- | ---: | ---: | ---: | ---: |");
  for (const pair of pairwiseDeltas.pairs) {
    lines.push(
      `| ${pair.leftMode} - ${pair.rightMode} | ${pair.traceWins.left}-${pair.traceWins.right}-${pair.traceWins.ties} | ${pair.turnWins.left}-${pair.turnWins.right}-${pair.turnWins.ties} | ${pair.aggregateDeltas.qualityScoreDeltaLeftMinusRightMean ?? "none"} | ${pair.aggregateDeltas.compileOkDeltaLeftMinusRightSum} | ${pair.aggregateDeltas.phraseHitDeltaLeftMinusRightSum} | ${pair.aggregateDeltas.promotionDeltaLeftMinusRightSum} |`,
    );
  }
  lines.push("");
  lines.push("## Artifacts");
  lines.push(`- index: \`${RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.index}\``);
  lines.push(`- summary tables: \`${RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.summaryTables}\``);
  lines.push(`- pairwise deltas: \`${RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.pairwiseDeltas}\``);
  lines.push(`- win-rate matrix: \`${RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.winRateMatrix}\``);
  lines.push(`- worked traces: \`${RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.workedTraces}\``);
  lines.push(`- generation report: \`${RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.generationReport}\``);
  return lines.join("\n");
}

export function writeRecordedSessionReplayProofLane(
  input: WriteRecordedSessionReplayProofLaneInputV1,
): RecordedSessionReplayProofLaneDescriptorV1 {
  const artifactRoot = path.resolve(input.artifactRoot);
  const laneDir = path.join(artifactRoot, RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.laneDir);
  const workedTraceLimit = normalizeWorkedTraceLimit(input.workedTraceLimit);
  const assumptions = normalizeStringArray(input.assumptions);
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
      traceAnalyses.push(buildTraceAnalysis(artifactRoot, descriptor, validation));
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
  const summaryTables = buildSummaryTables(traceAnalyses, input.traces.length, failedTraceIds.length);
  const pairwiseDeltas = buildPairwiseDeltas(traceAnalyses, input.traces.length, failedTraceIds.length);
  const winRateMatrix = buildWinRateMatrix(traceAnalyses, input.traces.length, failedTraceIds.length);
  const index: RecordedSessionReplayProofLaneIndexV1 = {
    contract: "recorded_session_replay_proof_lane_index.v1",
    laneDir: RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.laneDir,
    modeOrder: [...RECORDED_SESSION_REPLAY_PROOF_LANE_MODE_ORDER],
    requestedTraceCount: input.traces.length,
    successfulTraceCount: successfulBundles.length,
    failedTraceCount: failedTraceIds.length,
    failedTraceIds,
    assumptions,
    files: {
      readme: RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.readme,
      index: RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.index,
      summaryTables: RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.summaryTables,
      pairwiseDeltas: RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.pairwiseDeltas,
      winRateMatrix: RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.winRateMatrix,
      workedTraces: RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.workedTraces,
      generationReport: RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.generationReport,
    },
    traceBundles: traceAnalyses
      .map((analysis) => ({
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
    contract: "recorded_session_replay_proof_lane_generation_report.v1",
    artifactRoot,
    laneDir,
    requestedTraceCount: input.traces.length,
    successfulTraceCount: successfulBundles.length,
    failedTraceCount: failedTraceIds.length,
    sourceManifestPath: input.sourceManifestPath ?? null,
    assumptions,
    entries: generationEntries,
  };
  const workedTraces = buildWorkedTracesMarkdown(traceAnalyses, workedTraceLimit);
  const laneReadme = buildLaneReadme(summaryTables, pairwiseDeltas, index);
  const readmePath = path.join(laneDir, RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.readme);
  const indexPath = path.join(laneDir, RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.index);
  const summaryTablesPath = path.join(laneDir, RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.summaryTables);
  const pairwiseDeltasPath = path.join(laneDir, RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.pairwiseDeltas);
  const winRateMatrixPath = path.join(laneDir, RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.winRateMatrix);
  const workedTracesPath = path.join(laneDir, RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.workedTraces);
  const generationReportPath = path.join(laneDir, RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.generationReport);
  writeText(readmePath, laneReadme);
  writeJson(indexPath, index);
  writeJson(summaryTablesPath, summaryTables);
  writeJson(pairwiseDeltasPath, pairwiseDeltas);
  writeJson(winRateMatrixPath, winRateMatrix);
  writeText(workedTracesPath, workedTraces);
  writeJson(generationReportPath, generationReport);
  return {
    artifactRoot,
    laneDir,
    readmePath,
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
    generationReport,
    successfulBundles,
  };
}
