import { checksumJsonPayload } from "@openclawbrain/contracts";

export const OPENCLAWBRAIN_EXPLAINABLE_EVAL_SCORECARD_CONTRACT = "openclawbrain_explainable_eval_scorecard.v1";
export const OPENCLAWBRAIN_REPLAY_TRACE_SUCCESS_PROXY_ID = "validated_replay_trace_success_proxy";

export const OPENCLAWBRAIN_EXPLAINABLE_EVAL_MODE_ORDER = [
  "no_brain",
  "vector_only",
  "graph_prior_only",
  "learned_route",
] as const;

export type OpenClawBrainExplainableEvalModeV1 = (typeof OPENCLAWBRAIN_EXPLAINABLE_EVAL_MODE_ORDER)[number];
export type OpenClawBrainExplainableEvalAudienceV1 = "public_operator" | "internal";
export type OpenClawBrainExplainableEvalAvailabilityV1 = "measured" | "proxy" | "not_available";
export type OpenClawBrainExplainableEvalMetricKindV1 = "rate" | "count" | "delta" | "scalar";
export type OpenClawBrainExplainableEvalMetricCategoryV1 =
  | "regression_safety"
  | "comparison"
  | "required_context"
  | "economics"
  | "fail_open"
  | "diagnostic";
export type OpenClawBrainExplainableEvalUnitV1 = "rate" | "count" | "score" | "tokens" | "usd";

export interface OpenClawBrainExplainableEvalMetricComponentV1 {
  id: string;
  label: string;
  role: "numerator" | "denominator" | "input";
  value: number | null;
  unit: OpenClawBrainExplainableEvalUnitV1;
  mode?: OpenClawBrainExplainableEvalModeV1 | null;
}

export interface OpenClawBrainExplainableEvalMetricFormulaV1 {
  expression: string;
  components: OpenClawBrainExplainableEvalMetricComponentV1[];
}

export interface OpenClawBrainExplainableEvalMetricV1 {
  id: string;
  label: string;
  category: OpenClawBrainExplainableEvalMetricCategoryV1;
  audience: OpenClawBrainExplainableEvalAudienceV1;
  availability: OpenClawBrainExplainableEvalAvailabilityV1;
  kind: OpenClawBrainExplainableEvalMetricKindV1;
  unit: OpenClawBrainExplainableEvalUnitV1;
  value: number | null;
  leftMode: OpenClawBrainExplainableEvalModeV1 | null;
  rightMode: OpenClawBrainExplainableEvalModeV1 | null;
  formula: OpenClawBrainExplainableEvalMetricFormulaV1;
  language: string;
  notes: string[];
}

export interface OpenClawBrainExplainableEvalTraceSuccessProxyV1 {
  id: typeof OPENCLAWBRAIN_REPLAY_TRACE_SUCCESS_PROXY_ID;
  label: string;
  definition: string;
  formula: string;
}

export interface OpenClawBrainExplainableEvalModeInputV1 {
  mode: OpenClawBrainExplainableEvalModeV1;
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

export interface OpenClawBrainExplainableEvalTraceModeInputV1 {
  mode: OpenClawBrainExplainableEvalModeV1;
  qualityScore: number;
  compileOkCount: number;
  turnCount: number;
  phraseHitCount: number;
  phraseCount: number;
  selectedContextBlockCount: number;
  selectedContextChars: number;
  estimatedPromptTokens: number;
  estimatedPromptCostUsd: number | null;
}

export interface OpenClawBrainExplainableEvalTraceInputV1 {
  traceId: string;
  status: "ok" | "failed";
  validationOk: boolean | null;
  winnerMode: OpenClawBrainExplainableEvalModeV1 | null;
  topScoreModes: OpenClawBrainExplainableEvalModeV1[];
  scoreSpread: number | null;
  modes: OpenClawBrainExplainableEvalTraceModeInputV1[];
}

export interface OpenClawBrainExplainableEvalWinRateInputV1 {
  left: number;
  right: number;
  ties: number;
  leftRate: number | null;
  rightRate: number | null;
  tieRate: number | null;
}

export interface OpenClawBrainExplainableEvalTieOrBetterInputV1 {
  left: number;
  right: number;
  leftRate: number | null;
  rightRate: number | null;
}

export interface OpenClawBrainExplainableEvalPairwiseInputV1 {
  leftMode: OpenClawBrainExplainableEvalModeV1;
  rightMode: OpenClawBrainExplainableEvalModeV1;
  comparableTraceCount: number;
  comparableTurnCount: number;
  traceWins: OpenClawBrainExplainableEvalWinRateInputV1;
  traceTieOrBetter: OpenClawBrainExplainableEvalTieOrBetterInputV1;
  turnWins: OpenClawBrainExplainableEvalWinRateInputV1;
  turnTieOrBetter: OpenClawBrainExplainableEvalTieOrBetterInputV1;
  aggregateDeltas: {
    qualityScoreDeltaLeftMinusRightSum: number;
    qualityScoreDeltaLeftMinusRightMean: number | null;
    compileOkDeltaLeftMinusRightSum: number;
    phraseHitDeltaLeftMinusRightSum: number;
    promotionDeltaLeftMinusRightSum: number;
    tiePromotionDeltaLeftMinusRightSum: number;
  };
}

export interface BuildOpenClawBrainExplainableEvalScorecardInputV1 {
  generatedAt: string;
  manifestId: string | null;
  manifestContract: string | null;
  modeOrder: OpenClawBrainExplainableEvalModeV1[];
  requestedTraceCount: number;
  successfulTraceCount: number;
  failedTraceCount: number;
  modes: OpenClawBrainExplainableEvalModeInputV1[];
  pairwise: OpenClawBrainExplainableEvalPairwiseInputV1[];
  traces: OpenClawBrainExplainableEvalTraceInputV1[];
  notes?: readonly string[] | null;
}

export interface OpenClawBrainExplainableEvalScorecardV1 {
  contract: typeof OPENCLAWBRAIN_EXPLAINABLE_EVAL_SCORECARD_CONTRACT;
  generatedAt: string;
  manifestId: string | null;
  manifestContract: string | null;
  modeOrder: OpenClawBrainExplainableEvalModeV1[];
  requestedTraceCount: number;
  successfulTraceCount: number;
  failedTraceCount: number;
  comparableTraceCount: number;
  traceSuccessProxy: OpenClawBrainExplainableEvalTraceSuccessProxyV1;
  headline: string[];
  failOpenLanguage: string;
  diagnosticLanguage: string;
  publicOperatorMetrics: OpenClawBrainExplainableEvalMetricV1[];
  internalMetrics: OpenClawBrainExplainableEvalMetricV1[];
  notes: string[];
  scorecardHash: string;
}

function round(value: number, places = 6): number {
  const factor = 10 ** places;
  return Math.round(value * factor) / factor;
}

function toRate(numerator: number, denominator: number): number | null {
  return denominator > 0 ? round(numerator / denominator, 6) : null;
}

function toObjectRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function describeDelta(value: number | null, unitLabel: string): string {
  if (value === null) {
    return `not available (${unitLabel})`;
  }
  if (value === 0) {
    return `no change (${unitLabel})`;
  }
  if (value < 0) {
    return `${Math.abs(value)} lower ${unitLabel}`;
  }
  return `${value} higher ${unitLabel}`;
}

function findModeRow(
  rows: readonly OpenClawBrainExplainableEvalModeInputV1[],
  mode: OpenClawBrainExplainableEvalModeV1,
): OpenClawBrainExplainableEvalModeInputV1 | null {
  return rows.find((row) => row.mode === mode) ?? null;
}

function findTraceModeRow(
  trace: OpenClawBrainExplainableEvalTraceInputV1,
  mode: OpenClawBrainExplainableEvalModeV1,
): OpenClawBrainExplainableEvalTraceModeInputV1 | null {
  return trace.modes.find((row) => row.mode === mode) ?? null;
}

function findPairwiseRow(
  rows: readonly OpenClawBrainExplainableEvalPairwiseInputV1[],
  leftMode: OpenClawBrainExplainableEvalModeV1,
  rightMode: OpenClawBrainExplainableEvalModeV1,
): OpenClawBrainExplainableEvalPairwiseInputV1 | null {
  return rows.find((row) => row.leftMode === leftMode && row.rightMode === rightMode) ?? null;
}

function successfulTraceProxyCount(
  traces: readonly OpenClawBrainExplainableEvalTraceInputV1[],
  mode: OpenClawBrainExplainableEvalModeV1,
): number {
  return traces.filter((trace) => {
    if (trace.status !== "ok" || trace.validationOk !== true) {
      return false;
    }
    const row = findTraceModeRow(trace, mode);
    return row !== null
      && row.compileOkCount === row.turnCount
      && row.phraseHitCount === row.phraseCount;
  }).length;
}

function modePerSuccessfulTrace(
  modeRow: OpenClawBrainExplainableEvalModeInputV1 | null,
  successCount: number,
  field: "estimatedPromptTokens" | "estimatedPromptCostUsd",
): number | null {
  if (modeRow === null || successCount <= 0) {
    return null;
  }
  const rawValue = modeRow[field];
  if (rawValue === null) {
    return null;
  }
  return round(rawValue / successCount, 6);
}

function metric(params: Omit<OpenClawBrainExplainableEvalMetricV1, "notes"> & { notes?: readonly string[] | null }): OpenClawBrainExplainableEvalMetricV1 {
  return {
    ...params,
    notes: [...(params.notes ?? [])],
  };
}

export function buildOpenClawBrainExplainableEvalScorecard(
  input: BuildOpenClawBrainExplainableEvalScorecardInputV1,
): OpenClawBrainExplainableEvalScorecardV1 {
  const baselineMode: OpenClawBrainExplainableEvalModeV1 = "graph_prior_only";
  const candidateMode: OpenClawBrainExplainableEvalModeV1 = "learned_route";
  const floorMode: OpenClawBrainExplainableEvalModeV1 = "no_brain";
  const validatedTraces = input.traces.filter((trace) => trace.status === "ok" && trace.validationOk === true);
  const baselineVsCandidate = findPairwiseRow(input.pairwise, baselineMode, candidateMode);
  const floorVsCandidate = findPairwiseRow(input.pairwise, floorMode, candidateMode);
  const candidateRow = findModeRow(input.modes, candidateMode);
  const baselineRow = findModeRow(input.modes, baselineMode);
  const floorRow = findModeRow(input.modes, floorMode);
  const comparableTraceCount = baselineVsCandidate?.comparableTraceCount ?? validatedTraces.length;
  const candidateSuccessProxyCount = successfulTraceProxyCount(validatedTraces, candidateMode);
  const baselineSuccessProxyCount = successfulTraceProxyCount(validatedTraces, baselineMode);
  const floorSuccessProxyCount = successfulTraceProxyCount(validatedTraces, floorMode);

  let compileRegressionTraceCount = 0;
  let requiredContextRegressionTraceCount = 0;
  let criticalRegressionTraceCount = 0;
  for (const trace of validatedTraces) {
    const baselineTraceRow = findTraceModeRow(trace, baselineMode);
    const candidateTraceRow = findTraceModeRow(trace, candidateMode);
    if (!baselineTraceRow || !candidateTraceRow) {
      continue;
    }
    const compileRegression = candidateTraceRow.compileOkCount < baselineTraceRow.compileOkCount;
    const requiredContextRegression = candidateTraceRow.phraseHitCount < baselineTraceRow.phraseHitCount;
    if (compileRegression) {
      compileRegressionTraceCount += 1;
    }
    if (requiredContextRegression) {
      requiredContextRegressionTraceCount += 1;
    }
    if (compileRegression || requiredContextRegression) {
      criticalRegressionTraceCount += 1;
    }
  }

  const candidatePromptTokensPerSuccess = modePerSuccessfulTrace(candidateRow, candidateSuccessProxyCount, "estimatedPromptTokens");
  const baselinePromptTokensPerSuccess = modePerSuccessfulTrace(baselineRow, baselineSuccessProxyCount, "estimatedPromptTokens");
  const floorPromptTokensPerSuccess = modePerSuccessfulTrace(floorRow, floorSuccessProxyCount, "estimatedPromptTokens");
  const candidatePromptCostPerSuccess = modePerSuccessfulTrace(candidateRow, candidateSuccessProxyCount, "estimatedPromptCostUsd");
  const baselinePromptCostPerSuccess = modePerSuccessfulTrace(baselineRow, baselineSuccessProxyCount, "estimatedPromptCostUsd");
  const floorPromptCostPerSuccess = modePerSuccessfulTrace(floorRow, floorSuccessProxyCount, "estimatedPromptCostUsd");

  const promptTokensDeltaVsPrior = candidatePromptTokensPerSuccess !== null && baselinePromptTokensPerSuccess !== null
    ? round(candidatePromptTokensPerSuccess - baselinePromptTokensPerSuccess, 6)
    : null;
  const promptTokensDeltaVsNoBrain = candidatePromptTokensPerSuccess !== null && floorPromptTokensPerSuccess !== null
    ? round(candidatePromptTokensPerSuccess - floorPromptTokensPerSuccess, 6)
    : null;
  const promptCostDeltaVsPrior = candidatePromptCostPerSuccess !== null && baselinePromptCostPerSuccess !== null
    ? round(candidatePromptCostPerSuccess - baselinePromptCostPerSuccess, 6)
    : null;
  const promptCostDeltaVsNoBrain = candidatePromptCostPerSuccess !== null && floorPromptCostPerSuccess !== null
    ? round(candidatePromptCostPerSuccess - floorPromptCostPerSuccess, 6)
    : null;

  const publicOperatorMetrics: OpenClawBrainExplainableEvalMetricV1[] = [
    metric({
      id: "brain_on_regression_rate_vs_prior",
      label: "Brain-on regression rate vs prior",
      category: "regression_safety",
      audience: "public_operator",
      availability: "proxy",
      kind: "rate",
      unit: "rate",
      value: baselineVsCandidate ? baselineVsCandidate.traceWins.leftRate : null,
      leftMode: candidateMode,
      rightMode: baselineMode,
      formula: {
        expression: "worseThanPriorCount / comparableTraceCount",
        components: [
          {
            id: "worse_than_prior_count",
            label: "Traces where learned_route scored below graph_prior_only",
            role: "numerator",
            value: baselineVsCandidate?.traceWins.left ?? null,
            unit: "count",
            mode: candidateMode,
          },
          {
            id: "comparable_trace_count",
            label: "Validated traces comparable across learned_route and graph_prior_only",
            role: "denominator",
            value: baselineVsCandidate?.comparableTraceCount ?? comparableTraceCount,
            unit: "count",
          },
        ],
      },
      language: baselineVsCandidate === null
        ? "Regression versus the approved prior could not be computed from this comparative replay bundle."
        : `learned_route was worse than graph_prior_only on ${baselineVsCandidate.traceWins.left}/${baselineVsCandidate.comparableTraceCount} validated traces.`,
      notes: [
        "Uses deterministic replay trace quality ordering as the comparison surface.",
        "This is a replay metric, not a live serve-path fail-open proof.",
      ],
    }),
    metric({
      id: "brain_on_regression_rate_vs_no_brain",
      label: "Brain-on regression rate vs no-brain floor",
      category: "regression_safety",
      audience: "public_operator",
      availability: "proxy",
      kind: "rate",
      unit: "rate",
      value: floorVsCandidate ? floorVsCandidate.traceWins.leftRate : null,
      leftMode: candidateMode,
      rightMode: floorMode,
      formula: {
        expression: "worseThanNoBrainCount / comparableTraceCount",
        components: [
          {
            id: "worse_than_no_brain_count",
            label: "Traces where learned_route scored below no_brain",
            role: "numerator",
            value: floorVsCandidate?.traceWins.left ?? null,
            unit: "count",
            mode: candidateMode,
          },
          {
            id: "comparable_trace_count",
            label: "Validated traces comparable across learned_route and no_brain",
            role: "denominator",
            value: floorVsCandidate?.comparableTraceCount ?? input.successfulTraceCount,
            unit: "count",
          },
        ],
      },
      language: floorVsCandidate === null
        ? "Regression versus the no-brain floor could not be computed from this comparative replay bundle."
        : `learned_route was worse than no_brain on ${floorVsCandidate.traceWins.left}/${floorVsCandidate.comparableTraceCount} validated traces.`,
      notes: [
        "The no_brain mode is a floor anchor, not proof of live fail-open behavior.",
      ],
    }),
    metric({
      id: "critical_regression_rate_vs_prior",
      label: "Critical regression rate vs prior",
      category: "regression_safety",
      audience: "public_operator",
      availability: "proxy",
      kind: "rate",
      unit: "rate",
      value: toRate(criticalRegressionTraceCount, comparableTraceCount),
      leftMode: candidateMode,
      rightMode: baselineMode,
      formula: {
        expression: "criticalRegressionCount / comparableTraceCount",
        components: [
          {
            id: "critical_regression_count",
            label: "Traces where learned_route lost compile-ok turns or required-context hits versus graph_prior_only",
            role: "numerator",
            value: criticalRegressionTraceCount,
            unit: "count",
            mode: candidateMode,
          },
          {
            id: "compile_regression_trace_count",
            label: "Traces where learned_route compiled fewer turns than graph_prior_only",
            role: "input",
            value: compileRegressionTraceCount,
            unit: "count",
            mode: candidateMode,
          },
          {
            id: "required_context_regression_trace_count",
            label: "Traces where learned_route hit fewer required phrases than graph_prior_only",
            role: "input",
            value: requiredContextRegressionTraceCount,
            unit: "count",
            mode: candidateMode,
          },
          {
            id: "comparable_trace_count",
            label: "Validated traces comparable across learned_route and graph_prior_only",
            role: "denominator",
            value: comparableTraceCount,
            unit: "count",
          },
        ],
      },
      language: `critical regressions were observed on ${criticalRegressionTraceCount}/${comparableTraceCount} validated traces when compile coverage or required-context hits worsened versus graph_prior_only.`,
      notes: [
        "Critical here means a replay-observed drop in compile-ok coverage or required-context hits.",
      ],
    }),
    metric({
      id: "tie_or_better_rate_vs_prior",
      label: "Tie-or-better rate vs prior",
      category: "comparison",
      audience: "public_operator",
      availability: "proxy",
      kind: "rate",
      unit: "rate",
      value: baselineVsCandidate?.traceTieOrBetter.rightRate ?? null,
      leftMode: candidateMode,
      rightMode: baselineMode,
      formula: {
        expression: "(betterThanPriorCount + tiedWithPriorCount) / comparableTraceCount",
        components: [
          {
            id: "better_than_prior_count",
            label: "Traces where learned_route scored above graph_prior_only",
            role: "input",
            value: baselineVsCandidate?.traceWins.right ?? null,
            unit: "count",
            mode: candidateMode,
          },
          {
            id: "tied_with_prior_count",
            label: "Traces where learned_route tied graph_prior_only",
            role: "input",
            value: baselineVsCandidate?.traceWins.ties ?? null,
            unit: "count",
            mode: candidateMode,
          },
          {
            id: "worse_than_prior_count",
            label: "Traces where learned_route scored below graph_prior_only",
            role: "input",
            value: baselineVsCandidate?.traceWins.left ?? null,
            unit: "count",
            mode: candidateMode,
          },
          {
            id: "comparable_trace_count",
            label: "Validated traces comparable across learned_route and graph_prior_only",
            role: "denominator",
            value: baselineVsCandidate?.comparableTraceCount ?? comparableTraceCount,
            unit: "count",
          },
        ],
      },
      language: baselineVsCandidate === null
        ? "Tie-or-better versus the approved prior could not be computed from this comparative replay bundle."
        : `learned_route tied or beat graph_prior_only on ${baselineVsCandidate.traceTieOrBetter.right}/${baselineVsCandidate.comparableTraceCount} validated traces (better ${baselineVsCandidate.traceWins.right}, tied ${baselineVsCandidate.traceWins.ties}, worse ${baselineVsCandidate.traceWins.left}).`,
      notes: [
        "The strict better/tied/worse decomposition is surfaced directly instead of only reporting a winner mode.",
      ],
    }),
    metric({
      id: "required_context_recall",
      label: "Required-context recall",
      category: "required_context",
      audience: "public_operator",
      availability: "measured",
      kind: "rate",
      unit: "rate",
      value: candidateRow?.phraseHitRate ?? null,
      leftMode: candidateMode,
      rightMode: null,
      formula: {
        expression: "retrievedRequiredEvidenceCount / totalRequiredEvidenceCount",
        components: [
          {
            id: "retrieved_required_evidence_count",
            label: "Required phrases hit by learned_route",
            role: "numerator",
            value: candidateRow?.totalPhraseHitCount ?? null,
            unit: "count",
            mode: candidateMode,
          },
          {
            id: "total_required_evidence_count",
            label: "Required phrases requested by the validated replay traces",
            role: "denominator",
            value: candidateRow?.totalPhraseCount ?? null,
            unit: "count",
            mode: candidateMode,
          },
          {
            id: "prior_required_context_recall",
            label: "Required-context recall for graph_prior_only",
            role: "input",
            value: baselineRow?.phraseHitRate ?? null,
            unit: "rate",
            mode: baselineMode,
          },
        ],
      },
      language: candidateRow === null
        ? "Required-context recall could not be computed for learned_route."
        : `learned_route retrieved required replay phrases at ${candidateRow.phraseHitRate ?? "null"} recall versus ${baselineRow?.phraseHitRate ?? "null"} for graph_prior_only.`,
      notes: [
        "Uses recorded-session expected context phrases as the required-evidence surface.",
      ],
    }),
    metric({
      id: "missing_required_context_rate",
      label: "Missing required-context rate",
      category: "required_context",
      audience: "public_operator",
      availability: "measured",
      kind: "rate",
      unit: "rate",
      value: candidateRow && candidateRow.totalPhraseCount > 0
        ? round((candidateRow.totalPhraseCount - candidateRow.totalPhraseHitCount) / candidateRow.totalPhraseCount, 6)
        : null,
      leftMode: candidateMode,
      rightMode: null,
      formula: {
        expression: "missingRequiredEvidenceCount / totalRequiredEvidenceCount",
        components: [
          {
            id: "missing_required_evidence_count",
            label: "Required phrases missed by learned_route",
            role: "numerator",
            value: candidateRow === null ? null : candidateRow.totalPhraseCount - candidateRow.totalPhraseHitCount,
            unit: "count",
            mode: candidateMode,
          },
          {
            id: "total_required_evidence_count",
            label: "Required phrases requested by the validated replay traces",
            role: "denominator",
            value: candidateRow?.totalPhraseCount ?? null,
            unit: "count",
            mode: candidateMode,
          },
        ],
      },
      language: candidateRow === null
        ? "Missing required-context rate could not be computed for learned_route."
        : `learned_route missed ${candidateRow.totalPhraseCount - candidateRow.totalPhraseHitCount}/${candidateRow.totalPhraseCount} required replay phrases.`,
      notes: [
        "Lower is better.",
      ],
    }),
    metric({
      id: "estimated_prompt_tokens_per_successful_trace_delta_vs_prior",
      label: "Estimated prompt tokens per successful trace delta vs prior",
      category: "economics",
      audience: "public_operator",
      availability: "proxy",
      kind: "delta",
      unit: "tokens",
      value: promptTokensDeltaVsPrior,
      leftMode: candidateMode,
      rightMode: baselineMode,
      formula: {
        expression: "(candidatePromptTokens / candidateSuccessfulTraceProxyCount) - (priorPromptTokens / priorSuccessfulTraceProxyCount)",
        components: [
          {
            id: "candidate_prompt_tokens",
            label: "Estimated prompt tokens for learned_route",
            role: "input",
            value: candidateRow?.estimatedPromptTokens ?? null,
            unit: "tokens",
            mode: candidateMode,
          },
          {
            id: "candidate_successful_trace_proxy_count",
            label: "Validated replay trace success proxy count for learned_route",
            role: "input",
            value: candidateSuccessProxyCount,
            unit: "count",
            mode: candidateMode,
          },
          {
            id: "prior_prompt_tokens",
            label: "Estimated prompt tokens for graph_prior_only",
            role: "input",
            value: baselineRow?.estimatedPromptTokens ?? null,
            unit: "tokens",
            mode: baselineMode,
          },
          {
            id: "prior_successful_trace_proxy_count",
            label: "Validated replay trace success proxy count for graph_prior_only",
            role: "input",
            value: baselineSuccessProxyCount,
            unit: "count",
            mode: baselineMode,
          },
        ],
      },
      language: `learned_route used ${describeDelta(promptTokensDeltaVsPrior, "estimated prompt tokens per replay-successful trace")} versus graph_prior_only.`,
      notes: [
        "Success is a replay success proxy: every turn compiled and no required phrase was missed on that trace.",
        "Prompt tokens are a deterministic proxy derived from selected context chars.",
      ],
    }),
    metric({
      id: "estimated_prompt_cost_per_successful_trace_delta_vs_prior",
      label: "Estimated prompt cost per successful trace delta vs prior",
      category: "economics",
      audience: "public_operator",
      availability: "proxy",
      kind: "delta",
      unit: "usd",
      value: promptCostDeltaVsPrior,
      leftMode: candidateMode,
      rightMode: baselineMode,
      formula: {
        expression: "(candidatePromptCostUsd / candidateSuccessfulTraceProxyCount) - (priorPromptCostUsd / priorSuccessfulTraceProxyCount)",
        components: [
          {
            id: "candidate_prompt_cost_usd",
            label: "Estimated prompt cost for learned_route",
            role: "input",
            value: candidateRow?.estimatedPromptCostUsd ?? null,
            unit: "usd",
            mode: candidateMode,
          },
          {
            id: "candidate_successful_trace_proxy_count",
            label: "Validated replay trace success proxy count for learned_route",
            role: "input",
            value: candidateSuccessProxyCount,
            unit: "count",
            mode: candidateMode,
          },
          {
            id: "prior_prompt_cost_usd",
            label: "Estimated prompt cost for graph_prior_only",
            role: "input",
            value: baselineRow?.estimatedPromptCostUsd ?? null,
            unit: "usd",
            mode: baselineMode,
          },
          {
            id: "prior_successful_trace_proxy_count",
            label: "Validated replay trace success proxy count for graph_prior_only",
            role: "input",
            value: baselineSuccessProxyCount,
            unit: "count",
            mode: baselineMode,
          },
        ],
      },
      language: `learned_route used ${describeDelta(promptCostDeltaVsPrior, "estimated prompt cost per replay-successful trace")} versus graph_prior_only.`,
      notes: [
        "Prompt cost is a deterministic proxy; this comparative replay does not measure end-to-end API or latency economics.",
      ],
    }),
    metric({
      id: "estimated_prompt_tokens_per_successful_trace_delta_vs_no_brain",
      label: "Estimated prompt tokens per successful trace delta vs no-brain floor",
      category: "economics",
      audience: "public_operator",
      availability: "proxy",
      kind: "delta",
      unit: "tokens",
      value: promptTokensDeltaVsNoBrain,
      leftMode: candidateMode,
      rightMode: floorMode,
      formula: {
        expression: "(candidatePromptTokens / candidateSuccessfulTraceProxyCount) - (noBrainPromptTokens / noBrainSuccessfulTraceProxyCount)",
        components: [
          {
            id: "candidate_prompt_tokens",
            label: "Estimated prompt tokens for learned_route",
            role: "input",
            value: candidateRow?.estimatedPromptTokens ?? null,
            unit: "tokens",
            mode: candidateMode,
          },
          {
            id: "candidate_successful_trace_proxy_count",
            label: "Validated replay trace success proxy count for learned_route",
            role: "input",
            value: candidateSuccessProxyCount,
            unit: "count",
            mode: candidateMode,
          },
          {
            id: "no_brain_prompt_tokens",
            label: "Estimated prompt tokens for no_brain",
            role: "input",
            value: floorRow?.estimatedPromptTokens ?? null,
            unit: "tokens",
            mode: floorMode,
          },
          {
            id: "no_brain_successful_trace_proxy_count",
            label: "Validated replay trace success proxy count for no_brain",
            role: "input",
            value: floorSuccessProxyCount,
            unit: "count",
            mode: floorMode,
          },
        ],
      },
      language: `learned_route used ${describeDelta(promptTokensDeltaVsNoBrain, "estimated prompt tokens per replay-successful trace")} versus no_brain.`,
      notes: [
        "The no_brain comparison is a floor anchor, not proof of live fail-open serving.",
      ],
    }),
    metric({
      id: "estimated_prompt_cost_per_successful_trace_delta_vs_no_brain",
      label: "Estimated prompt cost per successful trace delta vs no-brain floor",
      category: "economics",
      audience: "public_operator",
      availability: "proxy",
      kind: "delta",
      unit: "usd",
      value: promptCostDeltaVsNoBrain,
      leftMode: candidateMode,
      rightMode: floorMode,
      formula: {
        expression: "(candidatePromptCostUsd / candidateSuccessfulTraceProxyCount) - (noBrainPromptCostUsd / noBrainSuccessfulTraceProxyCount)",
        components: [
          {
            id: "candidate_prompt_cost_usd",
            label: "Estimated prompt cost for learned_route",
            role: "input",
            value: candidateRow?.estimatedPromptCostUsd ?? null,
            unit: "usd",
            mode: candidateMode,
          },
          {
            id: "candidate_successful_trace_proxy_count",
            label: "Validated replay trace success proxy count for learned_route",
            role: "input",
            value: candidateSuccessProxyCount,
            unit: "count",
            mode: candidateMode,
          },
          {
            id: "no_brain_prompt_cost_usd",
            label: "Estimated prompt cost for no_brain",
            role: "input",
            value: floorRow?.estimatedPromptCostUsd ?? null,
            unit: "usd",
            mode: floorMode,
          },
          {
            id: "no_brain_successful_trace_proxy_count",
            label: "Validated replay trace success proxy count for no_brain",
            role: "input",
            value: floorSuccessProxyCount,
            unit: "count",
            mode: floorMode,
          },
        ],
      },
      language: `learned_route used ${describeDelta(promptCostDeltaVsNoBrain, "estimated prompt cost per replay-successful trace")} versus no_brain.`,
      notes: [
        "This is a replay prompt-cost proxy rather than a full API or latency economics measure.",
      ],
    }),
    metric({
      id: "safe_fallback_rate",
      label: "Safe fallback rate",
      category: "fail_open",
      audience: "public_operator",
      availability: "not_available",
      kind: "rate",
      unit: "rate",
      value: null,
      leftMode: candidateMode,
      rightMode: null,
      formula: {
        expression: "safeFallbackCount / degradedBrainInvocationCount",
        components: [
          {
            id: "safe_fallback_count",
            label: "Served responses that degraded safely instead of hard-failing",
            role: "numerator",
            value: null,
            unit: "count",
          },
          {
            id: "degraded_brain_invocation_count",
            label: "Live invocations where the brain degraded or was unavailable",
            role: "denominator",
            value: null,
            unit: "count",
          },
        ],
      },
      language: "Comparative replay does not observe live safe-fallback invocations, so safe fallback rate is not computed here.",
      notes: [
        "Requires live serve-path instrumentation rather than recorded-session replay.",
      ],
    }),
    metric({
      id: "worker_down_safe_serve_rate",
      label: "Worker-down safe serve rate",
      category: "fail_open",
      audience: "public_operator",
      availability: "not_available",
      kind: "rate",
      unit: "rate",
      value: null,
      leftMode: candidateMode,
      rightMode: null,
      formula: {
        expression: "workerDownSafeServeCount / workerDownInvocationCount",
        components: [
          {
            id: "worker_down_safe_serve_count",
            label: "Served responses that stayed safe while the worker was down",
            role: "numerator",
            value: null,
            unit: "count",
          },
          {
            id: "worker_down_invocation_count",
            label: "Live invocations observed while the worker was down",
            role: "denominator",
            value: null,
            unit: "count",
          },
        ],
      },
      language: "Comparative replay does not simulate worker-down serving, so worker-down safe serve rate is not computed here.",
      notes: [
        "Requires live or dedicated fail-open probe evidence.",
      ],
    }),
    metric({
      id: "brain_disabled_comparable_success_rate",
      label: "Brain-disabled comparable success rate",
      category: "fail_open",
      audience: "public_operator",
      availability: "proxy",
      kind: "rate",
      unit: "rate",
      value: toRate(floorSuccessProxyCount, validatedTraces.length),
      leftMode: floorMode,
      rightMode: null,
      formula: {
        expression: "successfulBrainDisabledTasks / comparableBrainDisabledTasks",
        components: [
          {
            id: "successful_brain_disabled_tasks",
            label: "Validated no_brain traces that met the replay success proxy",
            role: "numerator",
            value: floorSuccessProxyCount,
            unit: "count",
            mode: floorMode,
          },
          {
            id: "comparable_brain_disabled_tasks",
            label: "Validated traces comparable under the no_brain floor anchor",
            role: "denominator",
            value: validatedTraces.length,
            unit: "count",
            mode: floorMode,
          },
        ],
      },
      language: `the no_brain floor met the replay success proxy on ${floorSuccessProxyCount}/${validatedTraces.length} validated traces.`,
      notes: [
        "This is a no_brain replay proxy, not a live fail-open serve-path proof.",
      ],
    }),
  ];

  const internalMetrics: OpenClawBrainExplainableEvalMetricV1[] = [
    metric({
      id: "diagnostic_quality_score_mean_by_mode",
      label: "Diagnostic qualityScore mean by mode",
      category: "diagnostic",
      audience: "internal",
      availability: "proxy",
      kind: "scalar",
      unit: "score",
      value: candidateRow?.meanQualityScore ?? null,
      leftMode: candidateMode,
      rightMode: null,
      formula: {
        expression: "mean(qualityScore) per mode over validated traces",
        components: input.modes.map((row) => ({
          id: `${row.mode}_mean_quality_score`,
          label: `Mean qualityScore for ${row.mode}`,
          role: "input" as const,
          value: row.meanQualityScore,
          unit: "score" as const,
          mode: row.mode,
        })),
      },
      language: "qualityScore remains an internal deterministic replay composite for smoke comparisons and tuning only.",
      notes: [
        "Do not use qualityScore as the public definition of victory.",
      ],
    }),
    metric({
      id: "diagnostic_ranked_winner_count_by_mode",
      label: "Diagnostic winnerMode count by mode",
      category: "diagnostic",
      audience: "internal",
      availability: "proxy",
      kind: "count",
      unit: "count",
      value: candidateRow?.rankedWinnerCount ?? null,
      leftMode: candidateMode,
      rightMode: null,
      formula: {
        expression: "sum(trace.winnerMode === mode) over validated traces",
        components: input.modes.map((row) => ({
          id: `${row.mode}_ranked_winner_count`,
          label: `Validated traces where ${row.mode} ranked first by qualityScore`,
          role: "input" as const,
          value: row.rankedWinnerCount,
          unit: "count" as const,
          mode: row.mode,
        })),
      },
      language: "winnerMode is retained only as an internal tie-break and ranking surface.",
      notes: [
        "Public/operator reporting should lead with strict better/tied/worse counts instead.",
      ],
    }),
    metric({
      id: "diagnostic_shared_top_score_trace_count_by_mode",
      label: "Diagnostic shared-top trace count by mode",
      category: "diagnostic",
      audience: "internal",
      availability: "proxy",
      kind: "count",
      unit: "count",
      value: candidateRow?.sharedTopScoreTraceCount ?? null,
      leftMode: candidateMode,
      rightMode: null,
      formula: {
        expression: "sum(trace.topScoreModes includes mode) over validated traces",
        components: input.modes.map((row) => ({
          id: `${row.mode}_shared_top_score_trace_count`,
          label: `Validated traces where ${row.mode} shared the top qualityScore`,
          role: "input" as const,
          value: row.sharedTopScoreTraceCount,
          unit: "count" as const,
          mode: row.mode,
        })),
      },
      language: "Shared-top counts remain an internal replay diagnostic for tie analysis.",
      notes: [
        "They are useful for tuning but should not replace explainable public metrics.",
      ],
    }),
  ];

  const regressionMetric = publicOperatorMetrics.find((entry) => entry.id === "brain_on_regression_rate_vs_prior");
  const tieOrBetterMetric = publicOperatorMetrics.find((entry) => entry.id === "tie_or_better_rate_vs_prior");
  const requiredContextMetric = publicOperatorMetrics.find((entry) => entry.id === "required_context_recall");
  const promptCostMetric = publicOperatorMetrics.find((entry) => entry.id === "estimated_prompt_cost_per_successful_trace_delta_vs_prior");
  const headline = [
    regressionMetric?.language ?? "Regression versus the approved prior was not computed.",
    tieOrBetterMetric?.language ?? "Tie-or-better versus the approved prior was not computed.",
    requiredContextMetric?.language ?? "Required-context recall was not computed.",
    promptCostMetric?.language ?? "Success-adjusted economics deltas were not computed.",
  ];
  const failOpenLanguage = `Comparative replay does not prove live safe-fallback or worker-down serving. It does expose a no_brain floor anchor: ${floorSuccessProxyCount}/${validatedTraces.length} validated traces met the replay success proxy under no_brain.`;
  const diagnosticLanguage = "qualityScore and winnerMode are preserved only as internal deterministic replay diagnostics; they are not the public/operator definition of success.";
  const notes = [
    "Validated replay trace success proxy means every replay turn compiled and no required context phrase was missed on that trace.",
    "Success-adjusted economics in this artifact use prompt-token and prompt-cost proxies derived from replay context selection, not full API/tool/latency telemetry.",
    "Live fail-open-safe rates require runtime instrumentation and are intentionally left unavailable in comparative replay outputs.",
    ...(input.notes ?? []),
  ];

  const base: Omit<OpenClawBrainExplainableEvalScorecardV1, "scorecardHash"> = {
    contract: OPENCLAWBRAIN_EXPLAINABLE_EVAL_SCORECARD_CONTRACT,
    generatedAt: input.generatedAt,
    manifestId: input.manifestId,
    manifestContract: input.manifestContract,
    modeOrder: [...input.modeOrder],
    requestedTraceCount: input.requestedTraceCount,
    successfulTraceCount: input.successfulTraceCount,
    failedTraceCount: input.failedTraceCount,
    comparableTraceCount,
    traceSuccessProxy: {
      id: OPENCLAWBRAIN_REPLAY_TRACE_SUCCESS_PROXY_ID,
      label: "Validated replay trace success proxy",
      definition: "A trace counts as successful for a mode when every replay turn compiled and the mode missed no required context phrases on that trace.",
      formula: "compileOkCount == turnCount && phraseHitCount == phraseCount",
    },
    headline,
    failOpenLanguage,
    diagnosticLanguage,
    publicOperatorMetrics,
    internalMetrics,
    notes,
  };

  return {
    ...base,
    scorecardHash: checksumJsonPayload(base),
  };
}

export function isOpenClawBrainExplainableEvalScorecard(
  value: unknown,
): value is OpenClawBrainExplainableEvalScorecardV1 {
  const record = toObjectRecord(value);
  return record?.contract === OPENCLAWBRAIN_EXPLAINABLE_EVAL_SCORECARD_CONTRACT
    && Array.isArray(record.publicOperatorMetrics)
    && Array.isArray(record.internalMetrics)
    && Array.isArray(record.headline);
}
