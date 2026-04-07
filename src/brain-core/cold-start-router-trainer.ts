import { createHash } from "node:crypto";
import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import type {
  ColdStartPackTypeV1,
  ColdStartStopLabelV1,
  DataRegistryEntryV1,
  RouteCandidateV1,
  RouteDecisionRowV1,
  RouterArtifactManifestV1,
} from "./cold-start-router-contracts.ts";
import {
  buildColdStartRouterLivePolicyInitializerV1,
  materializeColdStartRouterLivePolicyGraphV1,
  type ColdStartRouterLivePolicyInitializerV1,
} from "./graph.js";
import { scoreAction, softmaxPolicy } from "./policy.js";
import {
  COLD_START_CONTRACT_VERSION_V1,
  COLD_START_STOP_LABELS_V1,
  summarizeRouterArtifactManifestV1,
  validateDataRegistryEntryV1,
  validateRouteDecisionRowV1,
  validateRouterArtifactManifestV1,
} from "./cold-start-router-contracts.ts";
import type { TrajectoryCandidateScoreBreakdown, TraversalState } from "./types.js";

export const COLD_START_ROUTER_BASE_MODEL_CONTRACT_V1 = "cold_start_router_base_model.v1";
export const COLD_START_ROUTER_WEIGHTS_CONTRACT_V1 = "cold_start_router_weights.v1";
export const COLD_START_ROUTER_CALIBRATION_CONTRACT_V1 = "cold_start_router_calibration.v1";
export const COLD_START_ROUTER_FEATURE_NORMALIZERS_CONTRACT_V1 = "cold_start_router_feature_normalizers.v1";
export const COLD_START_ROUTER_SOURCE_PRIORS_CONTRACT_V1 = "cold_start_router_source_priors.v1";
export const COLD_START_ROUTER_SAFETY_RULES_CONTRACT_V1 = "cold_start_router_safety_rules.v1";

export const COLD_START_ROUTER_ARTIFACT_LAYOUT_V1 = {
  baseModel: "base-model.json",
  weights: "weights.json",
  calibration: "calibration.json",
  featureNormalizers: "feature-normalizers.json",
  sourcePriors: "source-priors.json",
  safetyRules: "safety-rules.json",
  manifest: "manifest.json",
} as const;

export type ColdStartRouterStopBucketFieldV1 =
  | "candidate_count"
  | "evidence_span_count"
  | "hard_negative_count"
  | "outcome_gain";

export const COLD_START_ROUTER_STOP_BUCKET_FIELDS_V1: ColdStartRouterStopBucketFieldV1[] = [
  "candidate_count",
  "evidence_span_count",
  "hard_negative_count",
  "outcome_gain",
];

export interface ColdStartRouterFeatureWeightV1 {
  positive: number;
  negative: number;
  support: number;
  weight: number;
}

export interface ColdStartRouterRowSkipV1 {
  rowId: string;
  datasetId: string;
  reason: string;
}

export interface ColdStartRouterTrainingSummaryV1 {
  totalRows: number;
  eligibleRows: number;
  usedRows: number;
  skippedRows: number;
  candidatePositiveCount: number;
  candidateNegativeCount: number;
  usedDatasetIds: string[];
  skippedRowDetails: ColdStartRouterRowSkipV1[];
  featureWeightCount: number;
  toolActionPriorCount: number;
  toolActionSetCount: number;
}

export interface ColdStartRouterCalibrationV1 {
  contract: typeof COLD_START_ROUTER_CALIBRATION_CONTRACT_V1;
  smoothing: number;
  supportDampening: number;
  labelOrder: ColdStartStopLabelV1[];
}

export interface ColdStartRouterFeatureNormalizersV1 {
  contract: typeof COLD_START_ROUTER_FEATURE_NORMALIZERS_CONTRACT_V1;
  candidateFeatures: {
    categorical: string[];
    tokenCostBuckets: string[];
    scoreHintBuckets: string[];
  };
  stopFeatures: Record<ColdStartRouterStopBucketFieldV1, string[]>;
}

export interface ColdStartRouterSourceDatasetPriorV1 {
  datasetId: string;
  sourceFamily: DataRegistryEntryV1["source_family"];
  approvalStatus: DataRegistryEntryV1["approval_status"];
  benchmarkSplitStatus: DataRegistryEntryV1["benchmark_split_status"];
  commercialUseStatus: DataRegistryEntryV1["commercial_use_status"];
  redistributionStatus: DataRegistryEntryV1["redistribution_status"];
  piiRisk: DataRegistryEntryV1["pii_risk"];
  exactFileCount: number;
  rowCount: number;
  usedRowCount: number;
  skippedRowCount: number;
}

export interface ColdStartRouterSourcePriorsV1 {
  contract: typeof COLD_START_ROUTER_SOURCE_PRIORS_CONTRACT_V1;
  datasetCount: number;
  usedDatasetCount: number;
  datasets: Record<string, ColdStartRouterSourceDatasetPriorV1>;
  counts: {
    sourceFamily: Record<string, number>;
    approvalStatus: Record<string, number>;
    benchmarkSplitStatus: Record<string, number>;
    commercialUseStatus: Record<string, number>;
    redistributionStatus: Record<string, number>;
    piiRisk: Record<string, number>;
  };
}

export interface ColdStartRouterSafetyRulesV1 {
  contract: typeof COLD_START_ROUTER_SAFETY_RULES_CONTRACT_V1;
  requireCandidateMembership: boolean;
  preferExplicitCorrections: boolean;
  allowedApprovalStatuses: Array<DataRegistryEntryV1["approval_status"]>;
  allowedCommercialUseStatuses: Array<DataRegistryEntryV1["commercial_use_status"]>;
  allowedRedistributionStatuses: Array<DataRegistryEntryV1["redistribution_status"]>;
  allowedPiiRisks: Array<DataRegistryEntryV1["pii_risk"]>;
  stopLabelOrder: ColdStartStopLabelV1[];
  skipEvalOnlyDatasets: boolean;
}

export interface ColdStartRouterToolActionPriorV1 {
  sourceNodeId: string;
  toolNodeId: string;
  positive: number;
  negative: number;
  support: number;
  prior: number;
  weight: number;
}

export interface ColdStartRouterToolActionSetV1 {
  sourceNodeId: string;
  rowIds: string[];
  teacherToolNodeIds: string[];
  candidateIds: string[];
  candidates: RouteCandidateV1[];
  support: number;
}

export interface ColdStartRouterModelV1 {
  contract: typeof COLD_START_ROUTER_WEIGHTS_CONTRACT_V1;
  artifactId: string;
  artifactVersion: string;
  packType: ColdStartPackTypeV1;
  compatibleRuntimeVersion: string;
  routerIdentity: string | null;
  createdAt: string;
  training: ColdStartRouterTrainingSummaryV1;
  candidateFeatureWeights: Record<string, ColdStartRouterFeatureWeightV1>;
  stopLabelCounts: Record<ColdStartStopLabelV1, number>;
  stopBucketCounts: Record<ColdStartRouterStopBucketFieldV1, Record<string, Record<ColdStartStopLabelV1, number>>>;
  calibration: ColdStartRouterCalibrationV1;
  featureNormalizers: ColdStartRouterFeatureNormalizersV1;
  sourcePriors: ColdStartRouterSourcePriorsV1;
  safetyRules: ColdStartRouterSafetyRulesV1;
  toolActionPriors: ColdStartRouterToolActionPriorV1[];
  toolActionSets: ColdStartRouterToolActionSetV1[];
  livePolicyInitializer: ColdStartRouterLivePolicyInitializerV1;
}

export interface ColdStartRouterBaseModelV1 {
  contract: typeof COLD_START_ROUTER_BASE_MODEL_CONTRACT_V1;
  artifactId: string;
  artifactVersion: string;
  packType: ColdStartPackTypeV1;
  compatibleRuntimeVersion: string;
  routerIdentity: string | null;
  createdAt: string;
  training: ColdStartRouterTrainingSummaryV1;
  weightsRef: string;
  calibrationRef: string;
  featureNormalizersRef: string;
  sourcePriorsRef: string;
  safetyRulesRef: string;
}

export interface ColdStartRouterRankedCandidateV1 {
  candidate: RouteCandidateV1;
  score: number;
  contributions: Array<{
    featureKey: string;
    featureValue: string;
    weight: number;
    support: number;
    contribution: number;
  }>;
}

export interface ColdStartRouterStopPredictionV1 {
  label: ColdStartStopLabelV1;
  scores: Record<ColdStartStopLabelV1, number>;
  contributingBuckets: Array<{
    field: ColdStartRouterStopBucketFieldV1;
    bucket: string;
    labelScores: Record<ColdStartStopLabelV1, number>;
  }>;
}

export interface ColdStartRouterPolicyTraverseActionV1 {
  type: "traverse";
  targetNodeId?: string;
  candidate?: RouteCandidateV1;
}

export interface ColdStartRouterPolicyStopActionV1 {
  type: "stop_local";
}

export type ColdStartRouterPolicyActionV1 =
  | ColdStartRouterPolicyTraverseActionV1
  | ColdStartRouterPolicyStopActionV1;

export interface ColdStartRouterPolicyActionScoreV1 {
  action: ColdStartRouterPolicyActionV1;
  score: number;
  probability: number;
  contributions?: ColdStartRouterRankedCandidateV1["contributions"];
}

export interface ColdStartRouterPolicyDistributionV1 {
  actions: ColdStartRouterPolicyActionScoreV1[];
  stopAction: ColdStartRouterPolicyActionScoreV1;
}

export interface ColdStartRouterScoringResultV1 {
  rankedCandidates: ColdStartRouterRankedCandidateV1[];
  stopPrediction: ColdStartRouterStopPredictionV1;
  policyDistribution: ColdStartRouterPolicyDistributionV1;
}

export interface ColdStartRouterTrainingInputV1 {
  artifactId: string;
  artifactVersion: string;
  packType: ColdStartPackTypeV1;
  compatibleRuntimeVersion: string;
  registryEntries: DataRegistryEntryV1[];
  routeRows: RouteDecisionRowV1[];
  outputDir: string;
  routerIdentity?: string | null;
  createdAt?: string;
  trainingDataRefs?: string[];
  replayGateRefs?: string[];
  baseModelRef?: string;
}

export interface ColdStartRouterTrainingResultV1 {
  outputDir: string;
  manifestPath: string;
  manifest: RouterArtifactManifestV1;
  baseModelPath: string;
  weightsPath: string;
  calibrationPath: string;
  featureNormalizersPath: string;
  sourcePriorsPath: string;
  safetyRulesPath: string;
  model: ColdStartRouterModelV1;
}

const CANDIDATE_TOKEN_COST_BUCKETS = ["0", "1-8", "9-32", "33-128", "129+"] as const;
const CANDIDATE_SCORE_HINT_BUCKETS = ["missing", "<0", "0-0.25", "0.25-0.5", "0.5-0.75", "0.75-1.0", ">1.0"] as const;
const STOP_CANDIDATE_COUNT_BUCKETS = ["1", "2-3", "4-7", "8+"] as const;
const STOP_EVIDENCE_SPAN_BUCKETS = ["1", "2", "3+"] as const;
const STOP_HARD_NEGATIVE_BUCKETS = ["0", "1", "2+"] as const;
const STOP_OUTCOME_GAIN_BUCKETS = ["loss", "low", "medium", "high"] as const;

const DEFAULT_STOP_LABEL_ORDER: ColdStartStopLabelV1[] = ["CONTINUE", "STOP_LOCAL", "STOP"];

function sha256Text(value: string): string {
  return `sha256:${createHash("sha256").update(value, "utf8").digest("hex")}`;
}

function writeJsonArtifact(filePath: string, value: unknown): { path: string; digest: string } {
  const text = `${JSON.stringify(value, null, 2)}\n`;
  mkdirSync(path.dirname(filePath), { recursive: true });
  writeFileSync(filePath, text, "utf8");
  return {
    path: filePath,
    digest: sha256Text(text),
  };
}

function sortedRecord<T>(record: Record<string, T>): Record<string, T> {
  return Object.fromEntries(Object.keys(record).sort().map((key) => [key, record[key]]));
}

function normalizeText(value: unknown, fallback = ""): string {
  if (typeof value !== "string") {
    return fallback;
  }
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : fallback;
}

function clampFinite(value: number, lower: number, upper: number): number {
  if (!Number.isFinite(value)) {
    return lower;
  }
  return Math.min(upper, Math.max(lower, value));
}

function toRowWeight(outcomeGain: number): number {
  const magnitude = Math.abs(Number(outcomeGain));
  if (!Number.isFinite(magnitude) || magnitude === 0) {
    return 0.25;
  }
  return clampFinite(magnitude, 0.25, 2);
}

function bucketTokenCost(tokenCost: number | null | undefined): string {
  const value = Number(tokenCost ?? 0);
  if (!Number.isFinite(value) || value <= 0) {
    return "0";
  }
  if (value <= 8) {
    return "1-8";
  }
  if (value <= 32) {
    return "9-32";
  }
  if (value <= 128) {
    return "33-128";
  }
  return "129+";
}

function bucketScoreHint(scoreHint: number | null | undefined): string {
  if (scoreHint === null || scoreHint === undefined || !Number.isFinite(Number(scoreHint))) {
    return "missing";
  }
  const value = Number(scoreHint);
  if (value < 0) {
    return "<0";
  }
  if (value < 0.25) {
    return "0-0.25";
  }
  if (value < 0.5) {
    return "0.25-0.5";
  }
  if (value < 0.75) {
    return "0.5-0.75";
  }
  if (value <= 1) {
    return "0.75-1.0";
  }
  return ">1.0";
}

function bucketCandidateCount(candidateCount: number): string {
  if (candidateCount <= 1) {
    return "1";
  }
  if (candidateCount <= 3) {
    return "2-3";
  }
  if (candidateCount <= 7) {
    return "4-7";
  }
  return "8+";
}

function bucketEvidenceSpanCount(count: number): string {
  if (count <= 1) {
    return "1";
  }
  if (count === 2) {
    return "2";
  }
  return "3+";
}

function bucketHardNegativeCount(count: number): string {
  if (count <= 0) {
    return "0";
  }
  if (count === 1) {
    return "1";
  }
  return "2+";
}

function bucketOutcomeGain(outcomeGain: number): string {
  if (!Number.isFinite(outcomeGain) || outcomeGain <= 0) {
    return "loss";
  }
  if (outcomeGain < 0.35) {
    return "low";
  }
  if (outcomeGain < 0.7) {
    return "medium";
  }
  return "high";
}

function candidateFeatureKeys(candidate: RouteCandidateV1): string[] {
  return [
    `candidate_type:${candidate.candidate_type}`,
    `authority:${normalizeText(candidate.authority, "none")}`,
    `freshness:${normalizeText(candidate.freshness, "none")}`,
    `token_cost_bucket:${bucketTokenCost(candidate.token_cost)}`,
    `score_hint_bucket:${bucketScoreHint(candidate.score_hint)}`,
  ];
}

function resolvePositiveCandidateIds(row: RouteDecisionRowV1): Set<string> {
  const teacherAction = row.teacher_action;
  if (teacherAction.kind === "traverse") {
    return new Set(teacherAction.target_ids);
  }

  const toolTeacherAction = teacherAction as Extract<RouteDecisionRowV1["teacher_action"], { kind: "tool" }>;

  const explicitToolMatches = row.candidate_set
    .filter((candidate) => candidate.candidate_type === "tool" && candidate.candidate_id === toolTeacherAction.tool_name)
    .map((candidate) => candidate.candidate_id);
  if (explicitToolMatches.length > 0) {
    return new Set(explicitToolMatches);
  }

  const toolCandidates = row.candidate_set.filter((candidate) => candidate.candidate_type === "tool").map((candidate) => candidate.candidate_id);
  return toolCandidates.length === 1 ? new Set(toolCandidates) : new Set();
}

function isEligibleRegistryEntry(entry: DataRegistryEntryV1): boolean {
  return entry.approval_status === "approved_train"
    && entry.commercial_use_status === "allowed"
    && entry.redistribution_status === "allowed"
    && (entry.pii_risk === "none" || entry.pii_risk === "low");
}

function bumpFeatureCount(map: Map<string, { positive: number; negative: number; support: number }>, featureKey: string, isPositive: boolean, weight: number): void {
  const next = map.get(featureKey) ?? { positive: 0, negative: 0, support: 0 };
  if (isPositive) {
    next.positive += weight;
  } else {
    next.negative += weight;
  }
  next.support += weight;
  map.set(featureKey, next);
}

function labelCounts(): Record<ColdStartStopLabelV1, number> {
  return {
    CONTINUE: 0,
    STOP_LOCAL: 0,
    STOP: 0,
  };
}

function createStopBucketCounts(): Record<ColdStartRouterStopBucketFieldV1, Record<string, Record<ColdStartStopLabelV1, number>>> {
  return {
    candidate_count: {},
    evidence_span_count: {},
    hard_negative_count: {},
    outcome_gain: {},
  };
}

function bumpStopBucket(
  bucketCounts: Record<ColdStartRouterStopBucketFieldV1, Record<string, Record<ColdStartStopLabelV1, number>>>,
  field: ColdStartRouterStopBucketFieldV1,
  bucket: string,
  label: ColdStartStopLabelV1,
  weight: number,
): void {
  const fieldBuckets = bucketCounts[field];
  const next = fieldBuckets[bucket] ?? labelCounts();
  next[label] += weight;
  fieldBuckets[bucket] = next;
}

function calcWeight(positive: number, negative: number, support: number, smoothing: number, supportDampening: number): number {
  const odds = Math.log((positive + smoothing) / (negative + smoothing));
  const supportScale = support / (support + supportDampening);
  return clampFinite(odds * supportScale, -8, 8);
}

function normalizeFeatureWeights(
  featureCounts: Map<string, { positive: number; negative: number; support: number }>,
  smoothing: number,
  supportDampening: number,
): Record<string, ColdStartRouterFeatureWeightV1> {
  const entries = [...featureCounts.entries()].sort(([left], [right]) => left.localeCompare(right));
  return Object.fromEntries(entries.map(([featureKey, counts]) => [
    featureKey,
    {
      positive: counts.positive,
      negative: counts.negative,
      support: counts.support,
      weight: calcWeight(counts.positive, counts.negative, counts.support, smoothing, supportDampening),
    },
  ]));
}

function totalLabelCount(counts: Record<ColdStartStopLabelV1, number>): number {
  return COLD_START_STOP_LABELS_V1.reduce((sum, label) => sum + counts[label], 0);
}

function logSumExp(values: number[]): number {
  const finiteValues = values.filter((value) => Number.isFinite(value));
  if (finiteValues.length === 0) {
    return Number.NEGATIVE_INFINITY;
  }

  const max = Math.max(...finiteValues);
  const sum = finiteValues.reduce((accumulator, value) => accumulator + Math.exp(value - max), 0);
  return max + Math.log(sum);
}

function scoreLabelFromCounts(
  counts: Record<ColdStartStopLabelV1, number>,
  smoothing: number,
  labelOrder: ColdStartStopLabelV1[],
): Record<ColdStartStopLabelV1, number> {
  const total = totalLabelCount(counts);
  const denominator = total + (labelOrder.length * smoothing);
  return labelOrder.reduce((scores, label) => {
    scores[label] = Math.log((counts[label] + smoothing) / denominator);
    return scores;
  }, labelCounts());
}

function softmaxScores<T extends { score: number }>(items: T[]): Array<T & { probability: number }> {
  if (items.length === 0) {
    return [];
  }

  const maxScore = Math.max(...items.map((item) => item.score));
  const expScores = items.map((item) => ({
    ...item,
    expScore: Math.exp(item.score - maxScore),
  }));
  const sumExp = expScores.reduce((sum, item) => sum + item.expScore, 0);

  return expScores.map(({ expScore, ...item }) => ({
    ...item,
    probability: sumExp > 0 ? expScore / sumExp : 1 / items.length,
  })) as unknown as Array<T & { probability: number }>;
}

function computeStopPrediction(
  model: ColdStartRouterModelV1,
  params: {
    candidateCount: number;
    evidenceSpanCount: number;
    hardNegativeCount: number;
    outcomeGain: number;
  },
): ColdStartRouterStopPredictionV1 {
  const labelOrder = model.calibration.labelOrder;
  const scores = scoreLabelFromCounts(model.stopLabelCounts, model.calibration.smoothing, labelOrder);
  const contributingBuckets: ColdStartRouterStopPredictionV1["contributingBuckets"] = [];

  const bucketMap: Array<{ field: ColdStartRouterStopBucketFieldV1; bucket: string }> = [
    { field: "candidate_count", bucket: bucketCandidateCount(params.candidateCount) },
    { field: "evidence_span_count", bucket: bucketEvidenceSpanCount(params.evidenceSpanCount) },
    { field: "hard_negative_count", bucket: bucketHardNegativeCount(params.hardNegativeCount) },
    { field: "outcome_gain", bucket: bucketOutcomeGain(params.outcomeGain) },
  ];

  for (const entry of bucketMap) {
    const fieldCounts = model.stopBucketCounts[entry.field];
    const populatedBucketCount = Object.values(fieldCounts).filter((bucketCounts) => totalLabelCount(bucketCounts) > 0).length;
    const bucketCounts = fieldCounts[entry.bucket] ?? null;
    const labelScores = bucketCounts
      ? scoreLabelFromCounts(bucketCounts, model.calibration.smoothing, labelOrder)
      : labelCounts();
    contributingBuckets.push({ field: entry.field, bucket: entry.bucket, labelScores });
    if (populatedBucketCount < 2) {
      continue;
    }
    for (const label of labelOrder) {
      scores[label] += labelScores[label];
    }
  }

  let bestLabel = labelOrder[0];
  for (const label of labelOrder.slice(1)) {
    if (scores[label] > scores[bestLabel]) {
      bestLabel = label;
    }
  }

  return {
    label: bestLabel,
    scores,
    contributingBuckets,
  };
}

function computeStopActionLogit(stopPrediction: ColdStartRouterStopPredictionV1): number {
  const terminalStopScore = logSumExp([
    stopPrediction.scores.STOP_LOCAL,
    stopPrediction.scores.STOP,
  ]);
  return terminalStopScore - stopPrediction.scores.CONTINUE;
}

function candidateScoreBreakdown(
  model: ColdStartRouterModelV1,
  candidate: RouteCandidateV1,
): ColdStartRouterRankedCandidateV1 {
  const contributions: ColdStartRouterRankedCandidateV1["contributions"] = [];
  let score = 0;
  for (const featureKey of candidateFeatureKeys(candidate)) {
    const feature = model.candidateFeatureWeights[featureKey];
    if (!feature) {
      continue;
    }
    score += feature.weight;
    contributions.push({
      featureKey: featureKey.split(":")[0] ?? featureKey,
      featureValue: featureKey.includes(":") ? featureKey.slice(featureKey.indexOf(":") + 1) : featureKey,
      weight: feature.weight,
      support: feature.support,
      contribution: feature.weight,
    });
  }

  contributions.sort((left, right) => Math.abs(right.contribution) - Math.abs(left.contribution));
  return {
    candidate,
    score,
    contributions,
  };
}

function buildPolicyDistribution(params: {
  rankedCandidates: ColdStartRouterRankedCandidateV1[];
  stopPrediction: ColdStartRouterStopPredictionV1;
}): ColdStartRouterPolicyDistributionV1 {
  const stopActionScore = computeStopActionLogit(params.stopPrediction);
  const scoredActions = [
    ...params.rankedCandidates.map((candidate) => ({
      action: { type: "traverse" as const, candidate: candidate.candidate },
      score: candidate.score,
      contributions: candidate.contributions,
    })),
    {
      action: { type: "stop_local" as const },
      score: stopActionScore,
    },
  ];

  const scoredWithProbabilities = softmaxScores(scoredActions);
  const stopAction = scoredWithProbabilities[scoredWithProbabilities.length - 1];

  if (!stopAction) {
    return {
      actions: [],
      stopAction: {
        action: { type: "stop_local" },
        score: Number.NEGATIVE_INFINITY,
        probability: 1,
      },
    };
  }

  return {
    actions: scoredWithProbabilities,
    stopAction,
  };
}

function validateTrainingInputs(params: ColdStartRouterTrainingInputV1): void {
  if (params.artifactId.trim().length === 0) {
    throw new Error("artifactId is required");
  }
  if (params.artifactVersion.trim().length === 0) {
    throw new Error("artifactVersion is required");
  }
  if (!params.outputDir.trim()) {
    throw new Error("outputDir is required");
  }
  if (params.registryEntries.length === 0) {
    throw new Error("registryEntries must not be empty");
  }
  if (params.routeRows.length === 0) {
    throw new Error("routeRows must not be empty");
  }

  for (const entry of params.registryEntries) {
    const validation = validateDataRegistryEntryV1(entry);
    if (!validation.valid) {
      throw new Error(`invalid registry entry ${entry.dataset_id}: ${validation.issues.join("; ")}`);
    }
  }
  for (const row of params.routeRows) {
    const validation = validateRouteDecisionRowV1(row);
    if (!validation.valid) {
      throw new Error(`invalid route row ${row.row_id}: ${validation.issues.join("; ")}`);
    }
  }
}

function buildSourcePriors(
  registryEntries: DataRegistryEntryV1[],
  rowStats: Map<string, { total: number; used: number; skipped: number }>,
): ColdStartRouterSourcePriorsV1 {
  const counts = {
    sourceFamily: {} as Record<string, number>,
    approvalStatus: {} as Record<string, number>,
    benchmarkSplitStatus: {} as Record<string, number>,
    commercialUseStatus: {} as Record<string, number>,
    redistributionStatus: {} as Record<string, number>,
    piiRisk: {} as Record<string, number>,
  };

  const datasets: Record<string, ColdStartRouterSourceDatasetPriorV1> = {};
  for (const entry of registryEntries) {
    counts.sourceFamily[entry.source_family] = (counts.sourceFamily[entry.source_family] ?? 0) + 1;
    counts.approvalStatus[entry.approval_status] = (counts.approvalStatus[entry.approval_status] ?? 0) + 1;
    counts.benchmarkSplitStatus[entry.benchmark_split_status] = (counts.benchmarkSplitStatus[entry.benchmark_split_status] ?? 0) + 1;
    counts.commercialUseStatus[entry.commercial_use_status] = (counts.commercialUseStatus[entry.commercial_use_status] ?? 0) + 1;
    counts.redistributionStatus[entry.redistribution_status] = (counts.redistributionStatus[entry.redistribution_status] ?? 0) + 1;
    counts.piiRisk[entry.pii_risk] = (counts.piiRisk[entry.pii_risk] ?? 0) + 1;

    const stats = rowStats.get(entry.dataset_id) ?? { total: 0, used: 0, skipped: 0 };
    datasets[entry.dataset_id] = {
      datasetId: entry.dataset_id,
      sourceFamily: entry.source_family,
      approvalStatus: entry.approval_status,
      benchmarkSplitStatus: entry.benchmark_split_status,
      commercialUseStatus: entry.commercial_use_status,
      redistributionStatus: entry.redistribution_status,
      piiRisk: entry.pii_risk,
      exactFileCount: entry.exact_files.length,
      rowCount: stats.total,
      usedRowCount: stats.used,
      skippedRowCount: stats.skipped,
    };
  }

  return {
    contract: COLD_START_ROUTER_SOURCE_PRIORS_CONTRACT_V1,
    datasetCount: registryEntries.length,
    usedDatasetCount: Object.values(datasets).filter((dataset) => dataset.usedRowCount > 0).length,
    datasets: sortedRecord(datasets),
    counts: {
      sourceFamily: sortedRecord(counts.sourceFamily),
      approvalStatus: sortedRecord(counts.approvalStatus),
      benchmarkSplitStatus: sortedRecord(counts.benchmarkSplitStatus),
      commercialUseStatus: sortedRecord(counts.commercialUseStatus),
      redistributionStatus: sortedRecord(counts.redistributionStatus),
      piiRisk: sortedRecord(counts.piiRisk),
    },
  };
}

function buildFeatureNormalizers(): ColdStartRouterFeatureNormalizersV1 {
  return {
    contract: COLD_START_ROUTER_FEATURE_NORMALIZERS_CONTRACT_V1,
    candidateFeatures: {
      categorical: ["candidate_type", "authority", "freshness"],
      tokenCostBuckets: [...CANDIDATE_TOKEN_COST_BUCKETS],
      scoreHintBuckets: [...CANDIDATE_SCORE_HINT_BUCKETS],
    },
    stopFeatures: {
      candidate_count: [...STOP_CANDIDATE_COUNT_BUCKETS],
      evidence_span_count: [...STOP_EVIDENCE_SPAN_BUCKETS],
      hard_negative_count: [...STOP_HARD_NEGATIVE_BUCKETS],
      outcome_gain: [...STOP_OUTCOME_GAIN_BUCKETS],
    },
  };
}

function buildCalibration(): ColdStartRouterCalibrationV1 {
  return {
    contract: COLD_START_ROUTER_CALIBRATION_CONTRACT_V1,
    smoothing: 1,
    supportDampening: 2,
    labelOrder: [...DEFAULT_STOP_LABEL_ORDER],
  };
}

function buildSafetyRules(): ColdStartRouterSafetyRulesV1 {
  return {
    contract: COLD_START_ROUTER_SAFETY_RULES_CONTRACT_V1,
    requireCandidateMembership: true,
    preferExplicitCorrections: true,
    allowedApprovalStatuses: ["approved_train"],
    allowedCommercialUseStatuses: ["allowed"],
    allowedRedistributionStatuses: ["allowed"],
    allowedPiiRisks: ["none", "low"],
    stopLabelOrder: [...DEFAULT_STOP_LABEL_ORDER],
    skipEvalOnlyDatasets: true,
  };
}

function buildBaseModel(params: {
  artifactId: string;
  artifactVersion: string;
  packType: ColdStartPackTypeV1;
  compatibleRuntimeVersion: string;
  routerIdentity: string | null;
  createdAt: string;
  training: ColdStartRouterTrainingSummaryV1;
  weightsRef: string;
  calibrationRef: string;
  featureNormalizersRef: string;
  sourcePriorsRef: string;
  safetyRulesRef: string;
}): ColdStartRouterBaseModelV1 {
  return {
    contract: COLD_START_ROUTER_BASE_MODEL_CONTRACT_V1,
    artifactId: params.artifactId,
    artifactVersion: params.artifactVersion,
    packType: params.packType,
    compatibleRuntimeVersion: params.compatibleRuntimeVersion,
    routerIdentity: params.routerIdentity,
    createdAt: params.createdAt,
    training: params.training,
    weightsRef: params.weightsRef,
    calibrationRef: params.calibrationRef,
    featureNormalizersRef: params.featureNormalizersRef,
    sourcePriorsRef: params.sourcePriorsRef,
    safetyRulesRef: params.safetyRulesRef,
  };
}

function buildArtifactChecksum(params: {
  manifestCore: Omit<RouterArtifactManifestV1, "artifact_checksum">;
  trainingSummary: ColdStartRouterTrainingSummaryV1;
  modelRefDigests: Record<string, string>;
}): string {
  return sha256Text(JSON.stringify({
    ...params.manifestCore,
    trainingSummary: params.trainingSummary,
    modelRefDigests: params.modelRefDigests,
  }));
}

function sortStopBucketCounts(
  stopBucketCounts: Record<ColdStartRouterStopBucketFieldV1, Record<string, Record<ColdStartStopLabelV1, number>>>,
): Record<ColdStartRouterStopBucketFieldV1, Record<string, Record<ColdStartStopLabelV1, number>>> {
  const result = createStopBucketCounts();
  for (const field of COLD_START_ROUTER_STOP_BUCKET_FIELDS_V1) {
    const fieldCounts = stopBucketCounts[field];
    const sortedBuckets = Object.keys(fieldCounts).sort().reduce((acc, bucket) => {
      acc[bucket] = fieldCounts[bucket];
      return acc;
    }, {} as Record<string, Record<ColdStartStopLabelV1, number>>);
    result[field] = sortedBuckets;
  }
  return result;
}

function createSkippedRowDetails(rows: ColdStartRouterRowSkipV1[]): ColdStartRouterRowSkipV1[] {
  return rows.slice().sort((left, right) => left.rowId.localeCompare(right.rowId));
}

function uniqueSortedStrings(values: readonly string[]): string[] {
  return [...new Set(values.map((value) => value.trim()).filter((value) => value.length > 0))].sort();
}

export function trainColdStartRouterArtifactV1(params: ColdStartRouterTrainingInputV1): ColdStartRouterTrainingResultV1 {
  validateTrainingInputs(params);

  const createdAt = params.createdAt ?? new Date().toISOString();
  const registryByDataset = new Map(params.registryEntries.map((entry) => [entry.dataset_id, entry] as const));
  const usedDatasetIds = new Set<string>();
  const usedRouteRows: RouteDecisionRowV1[] = [];
  const skippedRows: ColdStartRouterRowSkipV1[] = [];
  const rowStats = new Map<string, { total: number; used: number; skipped: number }>();
  const featureCounts = new Map<string, { positive: number; negative: number; support: number }>();
  const toolActionCounts = new Map<string, Map<string, { positive: number; negative: number; support: number }>>();
  const toolActionSets = new Map<string, {
    rowIds: Set<string>;
    teacherToolNodeIds: Set<string>;
    candidateIds: Set<string>;
    candidates: Map<string, RouteCandidateV1>;
    support: number;
  }>();
  const stopLabelCounts = labelCounts();
  const stopBucketCounts = createStopBucketCounts();
  let eligibleRows = 0;
  let usedRows = 0;
  let positiveCandidateCount = 0;
  let negativeCandidateCount = 0;

  for (const row of params.routeRows) {
    const registryEntry = registryByDataset.get(row.dataset_id) ?? null;
    const stats = rowStats.get(row.dataset_id) ?? { total: 0, used: 0, skipped: 0 };
    stats.total += 1;
    rowStats.set(row.dataset_id, stats);

    if (!registryEntry) {
      stats.skipped += 1;
      skippedRows.push({ rowId: row.row_id, datasetId: row.dataset_id, reason: "missing registry entry" });
      continue;
    }

    if (!isEligibleRegistryEntry(registryEntry)) {
      stats.skipped += 1;
      skippedRows.push({
        rowId: row.row_id,
        datasetId: row.dataset_id,
        reason: `dataset ${row.dataset_id} is not eligible for training (${registryEntry.approval_status}, ${registryEntry.commercial_use_status}, ${registryEntry.redistribution_status}, ${registryEntry.pii_risk})`,
      });
      continue;
    }

    eligibleRows += 1;
    usedRows += 1;
    stats.used += 1;
    usedDatasetIds.add(row.dataset_id);
    usedRouteRows.push(row);

    const rowWeight = toRowWeight(row.outcome_gain);
    const positiveIds = resolvePositiveCandidateIds(row);
    const canTrainRanking = positiveIds.size > 0;
    const toolCandidates = row.candidate_set.filter((candidate) => candidate.candidate_type === "tool");
    const toolActionSet = toolCandidates.length > 0
      ? (toolActionSets.get(normalizeText(row.cursor_path[row.cursor_path.length - 1] ?? "", "__START__")) ?? {
          rowIds: new Set<string>(),
          teacherToolNodeIds: new Set<string>(),
          candidateIds: new Set<string>(),
          candidates: new Map<string, RouteCandidateV1>(),
          support: 0,
        })
      : null;

    if (toolActionSet) {
      const sourceNodeId = normalizeText(row.cursor_path[row.cursor_path.length - 1] ?? "", "__START__");
      toolActionSet.rowIds.add(row.row_id);
      toolActionSet.support += rowWeight;
      for (const candidate of toolCandidates) {
        toolActionSet.candidateIds.add(candidate.candidate_id);
        if (positiveIds.has(candidate.candidate_id)) {
          toolActionSet.teacherToolNodeIds.add(candidate.candidate_id);
        }
        const existing = toolActionSet.candidates.get(candidate.candidate_id);
        if (!existing || (candidate.score_hint ?? Number.NEGATIVE_INFINITY) > (existing.score_hint ?? Number.NEGATIVE_INFINITY)) {
          toolActionSet.candidates.set(candidate.candidate_id, {
            candidate_id: candidate.candidate_id,
            candidate_type: candidate.candidate_type,
            ...(candidate.authority ? { authority: candidate.authority } : {}),
            ...(candidate.freshness ? { freshness: candidate.freshness } : {}),
            ...(candidate.token_cost !== undefined ? { token_cost: candidate.token_cost } : {}),
            ...(candidate.score_hint !== undefined ? { score_hint: candidate.score_hint } : {}),
          });
        }

        const sourceMap = toolActionCounts.get(sourceNodeId) ?? new Map<string, { positive: number; negative: number; support: number }>();
        const bucket = sourceMap.get(candidate.candidate_id) ?? { positive: 0, negative: 0, support: 0 };
        if (positiveIds.has(candidate.candidate_id)) {
          bucket.positive += rowWeight;
        } else {
          bucket.negative += rowWeight;
        }
        bucket.support += rowWeight;
        sourceMap.set(candidate.candidate_id, bucket);
        toolActionCounts.set(sourceNodeId, sourceMap);
      }
      toolActionSets.set(normalizeText(row.cursor_path[row.cursor_path.length - 1] ?? "", "__START__"), toolActionSet);
    }

    if (canTrainRanking) {
      for (const candidate of row.candidate_set) {
        const isPositive = positiveIds.has(candidate.candidate_id);
        for (const featureKey of candidateFeatureKeys(candidate)) {
          bumpFeatureCount(featureCounts, featureKey, isPositive, rowWeight);
        }
        if (isPositive) {
          positiveCandidateCount += 1;
        } else {
          negativeCandidateCount += 1;
        }
      }
    }

    stopLabelCounts[row.stop_label] += rowWeight;
    const rowFeatureBuckets = [
      { field: "candidate_count" as const, bucket: bucketCandidateCount(row.candidate_set.length) },
      { field: "evidence_span_count" as const, bucket: bucketEvidenceSpanCount(row.evidence_spans.length) },
      { field: "hard_negative_count" as const, bucket: bucketHardNegativeCount(row.hard_negatives.length) },
      { field: "outcome_gain" as const, bucket: bucketOutcomeGain(row.outcome_gain) },
    ];
    for (const entry of rowFeatureBuckets) {
      bumpStopBucket(stopBucketCounts, entry.field, entry.bucket, row.stop_label, rowWeight);
    }
  }

  const skippedRowDetails = createSkippedRowDetails(skippedRows);
  const featureWeightMap = normalizeFeatureWeights(featureCounts, 1, 2);
  const toolActionPriorEntries = [...toolActionCounts.entries()]
    .flatMap(([sourceNodeId, toolMap]) => [...toolMap.entries()].map(([toolNodeId, bucket]) => ({
      sourceNodeId,
      toolNodeId,
      positive: bucket.positive,
      negative: bucket.negative,
      support: bucket.support,
      prior: bucket.support > 0 ? bucket.positive / bucket.support : 0,
      weight: calcWeight(bucket.positive, bucket.negative, bucket.support, 1, 2),
    })))
    .sort((left, right) => {
      const bySource = left.sourceNodeId.localeCompare(right.sourceNodeId);
      if (bySource !== 0) {
        return bySource;
      }
      return left.toolNodeId.localeCompare(right.toolNodeId);
    });
  const toolActionSetEntries = [...toolActionSets.entries()]
    .map(([sourceNodeId, entry]) => ({
      sourceNodeId,
      rowIds: [...entry.rowIds].sort(),
      teacherToolNodeIds: [...entry.teacherToolNodeIds].sort(),
      candidateIds: [...entry.candidateIds].sort(),
      candidates: [...entry.candidates.values()].sort((left, right) => left.candidate_id.localeCompare(right.candidate_id)),
      support: entry.support,
    }))
    .sort((left, right) => left.sourceNodeId.localeCompare(right.sourceNodeId));
  const trainingSummary: ColdStartRouterTrainingSummaryV1 = {
    totalRows: params.routeRows.length,
    eligibleRows,
    usedRows,
    skippedRows: params.routeRows.length - usedRows,
    candidatePositiveCount: positiveCandidateCount,
    candidateNegativeCount: negativeCandidateCount,
    usedDatasetIds: [...usedDatasetIds].sort(),
    skippedRowDetails,
    featureWeightCount: Object.keys(featureWeightMap).length,
    toolActionPriorCount: toolActionPriorEntries.length,
    toolActionSetCount: toolActionSetEntries.length,
  };

  const calibration = buildCalibration();
  const featureNormalizers = buildFeatureNormalizers();
  const sourcePriors = buildSourcePriors(params.registryEntries, rowStats);
  const safetyRules = buildSafetyRules();
  const livePolicyInitializer = buildColdStartRouterLivePolicyInitializerV1({
    routeRows: usedRouteRows,
  });
  const model: ColdStartRouterModelV1 = {
    contract: COLD_START_ROUTER_WEIGHTS_CONTRACT_V1,
    artifactId: params.artifactId,
    artifactVersion: params.artifactVersion,
    packType: params.packType,
    compatibleRuntimeVersion: params.compatibleRuntimeVersion,
    routerIdentity: params.routerIdentity ?? null,
    createdAt,
    training: trainingSummary,
    candidateFeatureWeights: featureWeightMap,
    stopLabelCounts,
    stopBucketCounts: sortStopBucketCounts(stopBucketCounts),
    calibration,
    featureNormalizers,
    sourcePriors,
    safetyRules,
    toolActionPriors: toolActionPriorEntries,
    toolActionSets: toolActionSetEntries,
    livePolicyInitializer,
  };

  if (trainingSummary.usedRows === 0) {
    throw new Error("no eligible training rows were available after registry filtering");
  }

  mkdirSync(params.outputDir, { recursive: true });
  const weightsPath = path.join(params.outputDir, COLD_START_ROUTER_ARTIFACT_LAYOUT_V1.weights);
  const calibrationPath = path.join(params.outputDir, COLD_START_ROUTER_ARTIFACT_LAYOUT_V1.calibration);
  const featureNormalizersPath = path.join(params.outputDir, COLD_START_ROUTER_ARTIFACT_LAYOUT_V1.featureNormalizers);
  const sourcePriorsPath = path.join(params.outputDir, COLD_START_ROUTER_ARTIFACT_LAYOUT_V1.sourcePriors);
  const safetyRulesPath = path.join(params.outputDir, COLD_START_ROUTER_ARTIFACT_LAYOUT_V1.safetyRules);
  const baseModelPath = path.join(params.outputDir, COLD_START_ROUTER_ARTIFACT_LAYOUT_V1.baseModel);
  const manifestPath = path.join(params.outputDir, COLD_START_ROUTER_ARTIFACT_LAYOUT_V1.manifest);

  const weightsFile = writeJsonArtifact(weightsPath, model);
  const calibrationFile = writeJsonArtifact(calibrationPath, calibration);
  const featureNormalizersFile = writeJsonArtifact(featureNormalizersPath, featureNormalizers);
  const sourcePriorsFile = writeJsonArtifact(sourcePriorsPath, sourcePriors);
  const safetyRulesFile = writeJsonArtifact(safetyRulesPath, safetyRules);

  const trainingDataRefs = (params.trainingDataRefs?.length ?? 0) > 0
    ? uniqueSortedStrings(params.trainingDataRefs ?? [])
    : [...usedDatasetIds].sort();
  const replayGateRefs = uniqueSortedStrings(params.replayGateRefs ?? []);
  const logicalBaseModelDigest = sha256Text(JSON.stringify({
    contract: COLD_START_ROUTER_BASE_MODEL_CONTRACT_V1,
    artifactId: params.artifactId,
    artifactVersion: params.artifactVersion,
    packType: params.packType,
    compatibleRuntimeVersion: params.compatibleRuntimeVersion,
    routerIdentity: params.routerIdentity ?? null,
    createdAt,
    training: trainingSummary,
  }));
  const baseModelRef = params.baseModelRef?.trim().length
    ? params.baseModelRef.trim()
    : `${COLD_START_ROUTER_ARTIFACT_LAYOUT_V1.baseModel}#${logicalBaseModelDigest}`;

  const manifestCore: Omit<RouterArtifactManifestV1, "artifact_checksum"> = {
    schema_version: COLD_START_CONTRACT_VERSION_V1,
    artifact_id: params.artifactId,
    artifact_version: params.artifactVersion,
    pack_type: params.packType,
    base_model_ref: baseModelRef,
    weights_ref: `${COLD_START_ROUTER_ARTIFACT_LAYOUT_V1.weights}#${weightsFile.digest}`,
    calibration_ref: `${COLD_START_ROUTER_ARTIFACT_LAYOUT_V1.calibration}#${calibrationFile.digest}`,
    feature_normalizers_ref: `${COLD_START_ROUTER_ARTIFACT_LAYOUT_V1.featureNormalizers}#${featureNormalizersFile.digest}`,
    source_priors_ref: `${COLD_START_ROUTER_ARTIFACT_LAYOUT_V1.sourcePriors}#${sourcePriorsFile.digest}`,
    safety_rules_ref: `${COLD_START_ROUTER_ARTIFACT_LAYOUT_V1.safetyRules}#${safetyRulesFile.digest}`,
    compatible_runtime_version: params.compatibleRuntimeVersion,
    training_data_refs: trainingDataRefs,
    replay_gate_refs: replayGateRefs,
    created_at: createdAt,
    router_identity: params.routerIdentity ?? undefined,
  };
  const artifactChecksum = buildArtifactChecksum({
    manifestCore,
    trainingSummary,
    modelRefDigests: {
      [COLD_START_ROUTER_ARTIFACT_LAYOUT_V1.baseModel]: sha256Text(JSON.stringify({
        contract: COLD_START_ROUTER_BASE_MODEL_CONTRACT_V1,
        artifactId: params.artifactId,
        artifactVersion: params.artifactVersion,
        packType: params.packType,
        compatibleRuntimeVersion: params.compatibleRuntimeVersion,
        routerIdentity: params.routerIdentity ?? null,
        createdAt,
        training: trainingSummary,
      })),
      [COLD_START_ROUTER_ARTIFACT_LAYOUT_V1.weights]: weightsFile.digest,
      [COLD_START_ROUTER_ARTIFACT_LAYOUT_V1.calibration]: calibrationFile.digest,
      [COLD_START_ROUTER_ARTIFACT_LAYOUT_V1.featureNormalizers]: featureNormalizersFile.digest,
      [COLD_START_ROUTER_ARTIFACT_LAYOUT_V1.sourcePriors]: sourcePriorsFile.digest,
      [COLD_START_ROUTER_ARTIFACT_LAYOUT_V1.safetyRules]: safetyRulesFile.digest,
    },
  });

  const manifest: RouterArtifactManifestV1 = {
    ...manifestCore,
    artifact_checksum: artifactChecksum,
  };
  const manifestValidation = validateRouterArtifactManifestV1(manifest);
  if (!manifestValidation.valid) {
    throw new Error(`generated router manifest failed validation: ${manifestValidation.issues.join("; ")}`);
  }

  const baseModelFile = writeJsonArtifact(baseModelPath, buildBaseModel({
    artifactId: params.artifactId,
    artifactVersion: params.artifactVersion,
    packType: params.packType,
    compatibleRuntimeVersion: params.compatibleRuntimeVersion,
    routerIdentity: params.routerIdentity ?? null,
    createdAt,
    training: trainingSummary,
    weightsRef: manifest.weights_ref,
    calibrationRef: manifest.calibration_ref,
    featureNormalizersRef: manifest.feature_normalizers_ref,
    sourcePriorsRef: manifest.source_priors_ref,
    safetyRulesRef: manifest.safety_rules_ref,
  }));

  // Regenerate the manifest file after the final base model file has been written,
  // so the artifact output directory contains a self-consistent bundle.
  writeJsonArtifact(manifestPath, manifest);
  void baseModelFile;

  return {
    outputDir: params.outputDir,
    manifestPath,
    manifest,
    baseModelPath,
    weightsPath,
    calibrationPath,
    featureNormalizersPath,
    sourcePriorsPath,
    safetyRulesPath,
    model,
  };
}

export function rankColdStartRouteCandidatesV1(params: {
  model: ColdStartRouterModelV1;
  candidates: RouteCandidateV1[];
}): ColdStartRouterRankedCandidateV1[] {
  return [...params.candidates]
    .map((candidate) => candidateScoreBreakdown(params.model, candidate))
    .sort((left, right) => {
      const byScore = right.score - left.score;
      if (Math.abs(byScore) > 1e-9) {
        return byScore;
      }
      return left.candidate.candidate_id.localeCompare(right.candidate.candidate_id);
    });
}

export function predictColdStartStopLabelV1(params: {
  model: ColdStartRouterModelV1;
  candidateCount: number;
  evidenceSpanCount: number;
  hardNegativeCount: number;
  outcomeGain: number;
}): ColdStartRouterStopPredictionV1 {
  return computeStopPrediction(params.model, params);
}

function scoreBreakdownToContributions(scoreBreakdown: Partial<TrajectoryCandidateScoreBreakdown> | undefined): Array<{
  featureKey: string;
  featureValue: string;
  weight: number;
  support: number;
  contribution: number;
}> {
  if (!scoreBreakdown) {
    return [];
  }

  const entries: Array<{
    featureKey: string;
    featureValue: string;
    weight: number;
    support: number;
    contribution: number;
  }> = [];

  const fields: Array<[string, unknown]> = [
    ["seedPrior", scoreBreakdown.seedPrior],
    ["learnedSeedWeight", scoreBreakdown.learnedSeedWeight],
    ["edgeScore", scoreBreakdown.edgeScore],
    ["relevance", scoreBreakdown.relevance],
    ["kindBias", scoreBreakdown.kindBias],
    ["evidenceQualityBonus", scoreBreakdown.evidenceQualityBonus],
    ["toolActionPrior", scoreBreakdown.toolActionPrior],
    ["opportunityCostPenalty", scoreBreakdown.opportunityCostPenalty],
    ["redundancyPenalty", scoreBreakdown.redundancyPenalty],
    ["learnedStopWeight", scoreBreakdown.learnedStopWeight],
    ["stopBias", scoreBreakdown.stopBias],
    ["budgetPressureContribution", scoreBreakdown.budgetPressureContribution],
    ["hopPressureContribution", scoreBreakdown.hopPressureContribution],
    ["frontierPressureContribution", scoreBreakdown.frontierPressureContribution],
  ];

  for (const [featureKey, value] of fields) {
    if (typeof value !== "number" || !Number.isFinite(value) || Math.abs(value) < 1e-12) {
      continue;
    }
    entries.push({
      featureKey,
      featureValue: featureKey,
      weight: value,
      support: 1,
      contribution: value,
    });
  }

  return entries.sort((left, right) => Math.abs(right.contribution) - Math.abs(left.contribution));
}

function buildTraversalStateForRow(row: RouteDecisionRowV1, sourceNodeId: string | null): TraversalState {
  const visited = new Set<string>();
  for (const entry of row.cursor_path) {
    const normalized = entry.trim();
    if (normalized.length > 0) {
      visited.add(normalized);
    }
  }
  if (sourceNodeId) {
    visited.delete(sourceNodeId);
  }

  return {
    sourceNodeId,
    queryEmbedding: new Float32Array(0),
    frontier: [],
    visited,
    fired: [],
    budgetRemaining: 1000,
    initialBudget: 1000,
    reservedTokenCost: 0,
    expansionCount: Math.max(0, row.cursor_path.length - 1),
    maxHops: 8,
  };
}

export function scoreColdStartRouteRowV1(params: {
  model: ColdStartRouterModelV1;
  row: RouteDecisionRowV1;
}): ColdStartRouterScoringResultV1 {
  const liveFamily = materializeColdStartRouterLivePolicyGraphV1({
    initializer: params.model.livePolicyInitializer,
    row: params.row,
  });
  const state = buildTraversalStateForRow(params.row, liveFamily.sourceNodeId);
  const actions = [
    ...params.row.candidate_set.map((candidate: RouteCandidateV1) => ({
      type: "traverse" as const,
      candidate,
      targetNodeId: candidate.candidate_id,
      seedScore: candidate.score_hint,
    })),
    { type: "stop_local" as const },
  ];
  const scoredActions = softmaxPolicy(actions, state, liveFamily.graph, liveFamily.policyParams);
  const stopAction = scoredActions.find((entry) => entry.action.type === "stop_local") ?? {
    action: { type: "stop_local" as const },
    score: Number.NEGATIVE_INFINITY,
    probability: 0,
  };

  const rankedCandidates = scoredActions
    .filter((entry): entry is typeof entry & { action: { type: "traverse"; targetNodeId: string } } => entry.action.type === "traverse")
    .map((entry) => {
      const candidate = params.row.candidate_set.find((item: RouteCandidateV1) => item.candidate_id === entry.action.targetNodeId);
      if (!candidate) {
        throw new Error(`live policy candidate ${entry.action.targetNodeId} missing from row candidate_set`);
      }
      return {
        candidate,
        score: entry.score,
        contributions: scoreBreakdownToContributions(entry.scoreBreakdown),
      };
    })
    .sort((left, right) => {
      const byScore = right.score - left.score;
      if (Math.abs(byScore) > 1e-9) {
        return byScore;
      }
      return left.candidate.candidate_id.localeCompare(right.candidate.candidate_id);
    });

  const stopPrediction = predictColdStartStopLabelV1({
    model: params.model,
    candidateCount: params.row.candidate_set.length,
    evidenceSpanCount: params.row.evidence_spans.length,
    hardNegativeCount: params.row.hard_negatives.length,
    outcomeGain: params.row.outcome_gain,
  });

  const policyDistribution = {
    actions: scoredActions.map((entry) => ({
      action: entry.action,
      score: entry.score,
      probability: entry.probability,
      contributions: scoreBreakdownToContributions(entry.scoreBreakdown),
    })),
    stopAction: {
      action: { type: "stop_local" as const },
      score: stopAction.score,
      probability: stopAction.probability,
      contributions: scoreBreakdownToContributions(stopAction.scoreBreakdown),
    },
  };

  return {
    rankedCandidates,
    stopPrediction,
    policyDistribution,
  };
}

export function summarizeColdStartRouterTrainingResultV1(result: ColdStartRouterTrainingResultV1): Record<string, unknown> {
  return {
    manifest: summarizeRouterArtifactManifestV1(result.manifest),
    outputDir: result.outputDir,
    files: {
      baseModelPath: result.baseModelPath,
      weightsPath: result.weightsPath,
      calibrationPath: result.calibrationPath,
      featureNormalizersPath: result.featureNormalizersPath,
      sourcePriorsPath: result.sourcePriorsPath,
      safetyRulesPath: result.safetyRulesPath,
      manifestPath: result.manifestPath,
    },
    training: result.model.training,
  };
}
