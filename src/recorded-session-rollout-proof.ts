import { readFileSync, readdirSync } from "node:fs";
import path from "node:path";

const TEST_FIXTURE_PATTERN = /\b(test|fixture|synthetic)\b/i;
const ELIGIBLE_EVIDENCE_CLASSIFICATION = "non_test_recorded_session";

type RecordedSessionTraceV1 = {
  contract: string;
  traceId: string;
  source: string;
  privacy: {
    notes: string[];
  };
  workspace: {
    labels?: string[];
  };
};

type RecordedSessionReplayModeReportV1 = {
  mode: string;
  summary: {
    qualityScore: number;
    evalTurnCount: number;
    usedLearnedRouteTurnCount: number;
  };
};

type RecordedSessionReplayBundleV1 = {
  traceId: string;
  modes: RecordedSessionReplayModeReportV1[];
  summary: {
    winnerMode: string | null;
  };
};

type RecordedSessionReplayProofValidationV1 = {
  ok: boolean;
  errors?: string[];
};

export type RecordedSessionReplayBaselineMode = "no_brain" | "vector_only" | "graph_prior_only";
export type RecordedSessionReplayComparisonRelation = "win" | "tie" | "loss";
export type RecordedSessionReplayEvidenceClassification = "test_fixture" | "non_test_recorded_session";

export type LearnedRouteRolloutBar = {
  minEligibleTraceCount: number;
  minCleanWinRate: number;
  minAverageMarginVsVectorOnly: number;
  minAverageMarginVsGraphPriorOnly: number;
  maxLossCountVsVectorOnly: number;
  maxLossCountVsGraphPriorOnly: number;
};

export type RecordedSessionReplayEvidenceInference = {
  classification: RecordedSessionReplayEvidenceClassification;
  reasons: string[];
};

export type RecordedSessionReplayTraceComparison = {
  baselineMode: RecordedSessionReplayBaselineMode;
  learnedRouteQualityScore: number;
  baselineQualityScore: number;
  margin: number;
  relation: RecordedSessionReplayComparisonRelation;
};

export type RecordedSessionReplayTraceRolloutEvaluation = {
  traceId: string;
  tracePath: string;
  bundleRoot: string;
  source: RecordedSessionTraceV1["source"];
  privacyNotes: string[];
  validationOk: boolean;
  winnerMode: RecordedSessionReplayBundleV1["summary"]["winnerMode"];
  learnedRouteQualityScore: number;
  learnedRouteEvalTurnCount: number;
  learnedRouteUsedLearnedRouteTurnCount: number;
  evidence: RecordedSessionReplayEvidenceInference;
  comparisons: RecordedSessionReplayTraceComparison[];
  cleanWinAgainstRetrievalBaselines: boolean;
};

export type RecordedSessionReplayBaselineAggregate = {
  baselineMode: RecordedSessionReplayBaselineMode;
  traceCount: number;
  strictWinCount: number;
  tieCount: number;
  lossCount: number;
  averageMargin: number | null;
  averageLearnedRouteQualityScore: number | null;
  averageBaselineQualityScore: number | null;
};

export type RecordedSessionReplayRolloutSummary = {
  traceCount: number;
  cleanWinTraceCount: number;
  learnedWinnerTraceCount: number;
  requiredCleanWinCount: number;
  baselineAggregates: RecordedSessionReplayBaselineAggregate[];
};

export type RecordedSessionReplayRolloutVerdict = {
  ok: boolean;
  failureReasons: string[];
  bar: LearnedRouteRolloutBar;
  totalTraceCount: number;
  eligibleTraceCount: number;
  evidenceClassCounts: Record<RecordedSessionReplayEvidenceClassification, number>;
  traces: RecordedSessionReplayTraceRolloutEvaluation[];
  allTraceSummary: RecordedSessionReplayRolloutSummary;
  eligibleTraceSummary: RecordedSessionReplayRolloutSummary;
};

export const DEFAULT_LEARNED_ROUTE_ROLLOUT_BAR: LearnedRouteRolloutBar = {
  minEligibleTraceCount: 3,
  minCleanWinRate: 0.5,
  minAverageMarginVsVectorOnly: 5,
  minAverageMarginVsGraphPriorOnly: 5,
  maxLossCountVsVectorOnly: 0,
  maxLossCountVsGraphPriorOnly: 0,
};

const BASELINE_MODES: RecordedSessionReplayBaselineMode[] = ["no_brain", "vector_only", "graph_prior_only"];
const RETRIEVAL_BASELINE_MODES: Exclude<RecordedSessionReplayBaselineMode, "no_brain">[] = ["vector_only", "graph_prior_only"];

function readJsonFile<T>(filePath: string): T {
  return JSON.parse(readFileSync(filePath, "utf8")) as T;
}

function average(values: readonly number[]): number | null {
  if (values.length === 0) {
    return null;
  }
  const total = values.reduce((sum, value) => sum + value, 0);
  return Number((total / values.length).toFixed(6));
}

function relationFromMargin(margin: number): RecordedSessionReplayComparisonRelation {
  if (margin > 0) {
    return "win";
  }
  if (margin < 0) {
    return "loss";
  }
  return "tie";
}

function ensureRecordedSessionTrace(trace: RecordedSessionTraceV1, tracePath: string): void {
  if (trace.contract !== "recorded_session_trace.v1") {
    throw new Error(`${tracePath} is not a recorded_session_trace.v1 trace`);
  }
}

function walkDirectories(rootDir: string, visit: (entryPath: string, isDirectory: boolean) => void): void {
  for (const entry of readdirSync(rootDir, { withFileTypes: true })) {
    const entryPath = path.join(rootDir, entry.name);
    visit(entryPath, entry.isDirectory());
    if (entry.isDirectory()) {
      walkDirectories(entryPath, visit);
    }
  }
}

function inferEvidenceReasons(trace: RecordedSessionTraceV1): string[] {
  const reasons: string[] = [];
  for (const note of trace.privacy.notes) {
    if (TEST_FIXTURE_PATTERN.test(note)) {
      reasons.push(`privacy_note:${note}`);
    }
  }
  for (const label of trace.workspace.labels ?? []) {
    if (TEST_FIXTURE_PATTERN.test(label)) {
      reasons.push(`workspace_label:${label}`);
    }
  }
  if (TEST_FIXTURE_PATTERN.test(trace.traceId)) {
    reasons.push(`trace_id:${trace.traceId}`);
  }
  return reasons;
}

function findModeQualityScore(bundle: RecordedSessionReplayBundleV1, mode: string): number {
  const report = bundle.modes.find((candidate) => candidate.mode === mode);
  if (report === undefined) {
    throw new Error(`bundle for trace ${bundle.traceId} is missing mode ${mode}`);
  }
  return report.summary.qualityScore;
}

function buildTraceComparisons(bundle: RecordedSessionReplayBundleV1): RecordedSessionReplayTraceComparison[] {
  const learnedRouteQualityScore = findModeQualityScore(bundle, "learned_route");
  return BASELINE_MODES.map((baselineMode) => {
    const baselineQualityScore = findModeQualityScore(bundle, baselineMode);
    const margin = learnedRouteQualityScore - baselineQualityScore;
    return {
      baselineMode,
      learnedRouteQualityScore,
      baselineQualityScore,
      margin,
      relation: relationFromMargin(margin),
    };
  });
}

function summarizeTraceEvaluations(
  traceEvaluations: readonly RecordedSessionReplayTraceRolloutEvaluation[],
  bar: LearnedRouteRolloutBar,
): RecordedSessionReplayRolloutSummary {
  const requiredCleanWinCount =
    traceEvaluations.length === 0 ? 0 : Math.ceil(traceEvaluations.length * bar.minCleanWinRate);
  return {
    traceCount: traceEvaluations.length,
    cleanWinTraceCount: traceEvaluations.filter((trace) => trace.cleanWinAgainstRetrievalBaselines).length,
    learnedWinnerTraceCount: traceEvaluations.filter((trace) => trace.winnerMode === "learned_route").length,
    requiredCleanWinCount,
    baselineAggregates: BASELINE_MODES.map((baselineMode) => {
      const comparisons = traceEvaluations
        .map((trace) => trace.comparisons.find((comparison) => comparison.baselineMode === baselineMode))
        .filter((comparison): comparison is RecordedSessionReplayTraceComparison => comparison !== undefined);
      return {
        baselineMode,
        traceCount: comparisons.length,
        strictWinCount: comparisons.filter((comparison) => comparison.relation === "win").length,
        tieCount: comparisons.filter((comparison) => comparison.relation === "tie").length,
        lossCount: comparisons.filter((comparison) => comparison.relation === "loss").length,
        averageMargin: average(comparisons.map((comparison) => comparison.margin)),
        averageLearnedRouteQualityScore: average(
          comparisons.map((comparison) => comparison.learnedRouteQualityScore),
        ),
        averageBaselineQualityScore: average(comparisons.map((comparison) => comparison.baselineQualityScore)),
      };
    }),
  };
}

function getBaselineAggregate(
  summary: RecordedSessionReplayRolloutSummary,
  baselineMode: RecordedSessionReplayBaselineMode,
): RecordedSessionReplayBaselineAggregate {
  const aggregate = summary.baselineAggregates.find((candidate) => candidate.baselineMode === baselineMode);
  if (aggregate === undefined) {
    throw new Error(`missing baseline aggregate for ${baselineMode}`);
  }
  return aggregate;
}

function countEvidenceClasses(
  traceEvaluations: readonly RecordedSessionReplayTraceRolloutEvaluation[],
): Record<RecordedSessionReplayEvidenceClassification, number> {
  return {
    test_fixture: traceEvaluations.filter((trace) => trace.evidence.classification === "test_fixture").length,
    non_test_recorded_session: traceEvaluations.filter(
      (trace) => trace.evidence.classification === "non_test_recorded_session",
    ).length,
  };
}

export function classifyRecordedSessionReplayEvidence(
  trace: RecordedSessionTraceV1,
): RecordedSessionReplayEvidenceInference {
  const reasons = inferEvidenceReasons(trace);
  return {
    classification: reasons.length > 0 ? "test_fixture" : "non_test_recorded_session",
    reasons,
  };
}

export function discoverRecordedSessionReplayTracePaths(rootDir: string): string[] {
  const resolvedRoot = path.resolve(rootDir);
  const tracePaths: string[] = [];
  walkDirectories(resolvedRoot, (entryPath, isDirectory) => {
    if (isDirectory) {
      return;
    }
    if (path.basename(entryPath) !== "trace.json") {
      return;
    }
    if (!entryPath.includes(`${path.sep}recorded-session-replay${path.sep}`)) {
      return;
    }
    tracePaths.push(entryPath);
  });
  return tracePaths.sort((left, right) => left.localeCompare(right));
}

export function evaluateRecordedSessionReplayTrace(
  tracePath: string,
): RecordedSessionReplayTraceRolloutEvaluation {
  const resolvedTracePath = path.resolve(tracePath);
  const bundleRoot = path.dirname(resolvedTracePath);
  const trace = readJsonFile<RecordedSessionTraceV1>(resolvedTracePath);
  ensureRecordedSessionTrace(trace, resolvedTracePath);
  const bundle = readJsonFile<RecordedSessionReplayBundleV1>(path.join(bundleRoot, "bundle.json"));
  const validation = readJsonFile<RecordedSessionReplayProofValidationV1>(path.join(bundleRoot, "validation-report.json"));
  const learnedRoute = bundle.modes.find((mode) => mode.mode === "learned_route");
  if (learnedRoute === undefined) {
    throw new Error(`bundle for trace ${trace.traceId} is missing learned_route`);
  }
  const comparisons = buildTraceComparisons(bundle);
  return {
    traceId: trace.traceId,
    tracePath: resolvedTracePath,
    bundleRoot,
    source: trace.source,
    privacyNotes: [...trace.privacy.notes],
    validationOk: validation.ok === true,
    winnerMode: bundle.summary.winnerMode,
    learnedRouteQualityScore: learnedRoute.summary.qualityScore,
    learnedRouteEvalTurnCount: learnedRoute.summary.evalTurnCount,
    learnedRouteUsedLearnedRouteTurnCount: learnedRoute.summary.usedLearnedRouteTurnCount,
    evidence: classifyRecordedSessionReplayEvidence(trace),
    comparisons,
    cleanWinAgainstRetrievalBaselines: RETRIEVAL_BASELINE_MODES.every((baselineMode) => {
      const comparison = comparisons.find((candidate) => candidate.baselineMode === baselineMode);
      return comparison?.relation === "win";
    }),
  };
}

export function buildRecordedSessionReplayRolloutVerdict(
  traceEvaluations: readonly RecordedSessionReplayTraceRolloutEvaluation[],
  bar: LearnedRouteRolloutBar = DEFAULT_LEARNED_ROUTE_ROLLOUT_BAR,
): RecordedSessionReplayRolloutVerdict {
  const traces = [...traceEvaluations].sort((left, right) => left.tracePath.localeCompare(right.tracePath));
  const eligibleTraces = traces.filter(
    (trace) => trace.evidence.classification === ELIGIBLE_EVIDENCE_CLASSIFICATION && trace.validationOk,
  );
  const allTraceSummary = summarizeTraceEvaluations(traces, bar);
  const eligibleTraceSummary = summarizeTraceEvaluations(eligibleTraces, bar);
  const vectorOnly = getBaselineAggregate(eligibleTraceSummary, "vector_only");
  const graphPriorOnly = getBaselineAggregate(eligibleTraceSummary, "graph_prior_only");
  const failureReasons: string[] = [];

  if (traces.length > 0 && traces.every((trace) => trace.evidence.classification === "test_fixture")) {
    failureReasons.push("only_test_fixture_evidence_available");
  }
  if (
    traces.some((trace) => trace.evidence.classification === ELIGIBLE_EVIDENCE_CLASSIFICATION && trace.validationOk === false)
  ) {
    failureReasons.push("invalid_non_test_proof_bundle_present");
  }
  if (eligibleTraces.length < bar.minEligibleTraceCount) {
    failureReasons.push("insufficient_eligible_trace_count");
  }
  if (eligibleTraceSummary.cleanWinTraceCount < eligibleTraceSummary.requiredCleanWinCount) {
    failureReasons.push("clean_win_rate_below_bar");
  }
  if ((vectorOnly.averageMargin ?? Number.NEGATIVE_INFINITY) < bar.minAverageMarginVsVectorOnly) {
    failureReasons.push("average_margin_vs_vector_only_below_bar");
  }
  if ((graphPriorOnly.averageMargin ?? Number.NEGATIVE_INFINITY) < bar.minAverageMarginVsGraphPriorOnly) {
    failureReasons.push("average_margin_vs_graph_prior_only_below_bar");
  }
  if (vectorOnly.lossCount > bar.maxLossCountVsVectorOnly) {
    failureReasons.push("vector_only_loss_count_above_bar");
  }
  if (graphPriorOnly.lossCount > bar.maxLossCountVsGraphPriorOnly) {
    failureReasons.push("graph_prior_only_loss_count_above_bar");
  }

  return {
    ok: failureReasons.length === 0,
    failureReasons,
    bar: { ...bar },
    totalTraceCount: traces.length,
    eligibleTraceCount: eligibleTraces.length,
    evidenceClassCounts: countEvidenceClasses(traces),
    traces,
    allTraceSummary,
    eligibleTraceSummary,
  };
}

export function evaluateRecordedSessionReplayRollout(
  tracePaths: readonly string[],
  options?: {
    bar?: LearnedRouteRolloutBar;
  },
): RecordedSessionReplayRolloutVerdict {
  const traceEvaluations = tracePaths.map((tracePath) => evaluateRecordedSessionReplayTrace(tracePath));
  return buildRecordedSessionReplayRolloutVerdict(traceEvaluations, options?.bar);
}

export function formatRecordedSessionReplayRolloutVerdict(
  verdict: RecordedSessionReplayRolloutVerdict,
): string {
  const eligibleVectorOnly = getBaselineAggregate(verdict.eligibleTraceSummary, "vector_only");
  const eligibleGraphPriorOnly = getBaselineAggregate(verdict.eligibleTraceSummary, "graph_prior_only");
  const allVectorOnly = getBaselineAggregate(verdict.allTraceSummary, "vector_only");
  const allGraphPriorOnly = getBaselineAggregate(verdict.allTraceSummary, "graph_prior_only");
  const lines = [
    `learned-route rollout verdict: ${verdict.ok ? "pass" : "fail"}`,
    `total traces: ${verdict.totalTraceCount}`,
    `eligible non-test traces: ${verdict.eligibleTraceCount}`,
    `eligible required clean wins: ${verdict.eligibleTraceSummary.requiredCleanWinCount}`,
    `eligible clean wins: ${verdict.eligibleTraceSummary.cleanWinTraceCount}`,
    `all-trace clean wins: ${verdict.allTraceSummary.cleanWinTraceCount}`,
    `eligible vs vector_only: wins=${eligibleVectorOnly.strictWinCount} ties=${eligibleVectorOnly.tieCount} losses=${eligibleVectorOnly.lossCount} avg_margin=${eligibleVectorOnly.averageMargin ?? "none"}`,
    `eligible vs graph_prior_only: wins=${eligibleGraphPriorOnly.strictWinCount} ties=${eligibleGraphPriorOnly.tieCount} losses=${eligibleGraphPriorOnly.lossCount} avg_margin=${eligibleGraphPriorOnly.averageMargin ?? "none"}`,
    `all vs vector_only: wins=${allVectorOnly.strictWinCount} ties=${allVectorOnly.tieCount} losses=${allVectorOnly.lossCount} avg_margin=${allVectorOnly.averageMargin ?? "none"}`,
    `all vs graph_prior_only: wins=${allGraphPriorOnly.strictWinCount} ties=${allGraphPriorOnly.tieCount} losses=${allGraphPriorOnly.lossCount} avg_margin=${allGraphPriorOnly.averageMargin ?? "none"}`,
    `failure reasons: ${verdict.failureReasons.length === 0 ? "none" : verdict.failureReasons.join(", ")}`,
    "",
    "trace details:",
    ...verdict.traces.map((trace) => {
      const vectorOnly = trace.comparisons.find((comparison) => comparison.baselineMode === "vector_only");
      const graphPriorOnly = trace.comparisons.find((comparison) => comparison.baselineMode === "graph_prior_only");
      return [
        `- ${trace.traceId}`,
        `  class=${trace.evidence.classification}`,
        `  validation=${trace.validationOk ? "ok" : "failed"}`,
        `  winner=${trace.winnerMode ?? "none"}`,
        `  learned_score=${trace.learnedRouteQualityScore}`,
        `  learned_used_turns=${trace.learnedRouteUsedLearnedRouteTurnCount}/${trace.learnedRouteEvalTurnCount}`,
        `  vs_vector=${vectorOnly?.relation ?? "missing"}(${vectorOnly?.margin ?? "none"})`,
        `  vs_graph=${graphPriorOnly?.relation ?? "missing"}(${graphPriorOnly?.margin ?? "none"})`,
        `  reasons=${trace.evidence.reasons.length === 0 ? "none" : trace.evidence.reasons.join(";")}`,
      ].join(" ");
    }),
  ];
  return `${lines.join("\n")}\n`;
}
