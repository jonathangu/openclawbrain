import type { RouteKind } from './memory-types.js';
import type { RoutingModeV3 } from './route-policy-v3-routing-mode.js';

export interface RouteCalibrationPredictionV3 {
  rawScore: number;
  route: RouteKind;
  observedSuccess: boolean;
  comparable?: boolean;
}

export interface RoutePolicyV3CalibrationBucket {
  minScore: number;
  maxScore: number;
  successRate: number;
  count: number;
}

export interface RoutePolicyV3CalibrationSummary {
  method: 'histogram_binning_v1';
  holdoutFrames: number;
  comparableFrames: number;
  globalThreshold: number;
  abstainMargin: number;
  globalBuckets: RoutePolicyV3CalibrationBucket[];
  routeThresholds: Partial<Record<RouteKind, number>>;
  routeBuckets: Partial<Record<RouteKind, RoutePolicyV3CalibrationBucket[]>>;
}

export function buildCalibrationSummaryV3(predictions: RouteCalibrationPredictionV3[], config: any = {}): RoutePolicyV3CalibrationSummary | undefined {
  const comparable = predictions.filter((prediction) => prediction.comparable !== false);
  if (comparable.length === 0) return undefined;
  const bucketCount = clampInt(config.routeLearning?.policyV3?.calibrationBuckets, 5, 3, 10);
  const abstainMargin = clamp01(Number(config.routeLearning?.policyV3?.abstainMargin ?? 0.05));
  const minThreshold = clamp01(Number(config.routeLearning?.policyV3?.minCalibratedConfidence ?? 0.62));
  const minRouteSamples = clampInt(config.routeLearning?.policyV3?.minRouteThresholdSamples, 3, 1, 50);
  const globalBuckets = buildBuckets(comparable, bucketCount);
  const routeThresholds: Partial<Record<RouteKind, number>> = {};
  const routeBuckets: Partial<Record<RouteKind, RoutePolicyV3CalibrationBucket[]>> = {};

  for (const route of uniqueRoutes(comparable.map((prediction) => prediction.route))) {
    const routeRows = comparable.filter((prediction) => prediction.route === route);
    if (routeRows.length < minRouteSamples) continue;
    const buckets = buildBuckets(routeRows, bucketCount);
    routeBuckets[route] = buckets;
    routeThresholds[route] = chooseThreshold(routeRows, buckets, minThreshold);
  }

  return {
    method: 'histogram_binning_v1',
    holdoutFrames: predictions.length,
    comparableFrames: comparable.length,
    globalThreshold: chooseThreshold(comparable, globalBuckets, minThreshold),
    abstainMargin,
    globalBuckets,
    routeThresholds,
    routeBuckets,
  };
}

export function applyCalibrationV3(rawScore: number, route: RouteKind, calibration?: RoutePolicyV3CalibrationSummary, mode?: RoutingModeV3) {
  if (!calibration) {
    return {
      rawScore,
      calibratedScore: rawScore,
      threshold: 0,
      abstained: false,
    };
  }
  const routeThreshold = calibration.routeThresholds?.[route] ?? calibration.globalThreshold;
  const adjustedThreshold = clamp01(routeThreshold + modeThresholdAdjustment(mode));
  const buckets = calibration.routeBuckets?.[route]?.length ? calibration.routeBuckets[route] : calibration.globalBuckets;
  const bucket = findBucketOrNearest(rawScore, buckets);
  const bucketRate = bucket?.successRate ?? rawScore;
  const calibratedScore = clamp01((rawScore * 0.35) + (bucketRate * 0.65));
  return {
    rawScore,
    calibratedScore,
    threshold: adjustedThreshold,
    abstained: calibratedScore + (calibration.abstainMargin || 0) < adjustedThreshold,
  };
}

function buildBuckets(predictions: RouteCalibrationPredictionV3[], bucketCount: number): RoutePolicyV3CalibrationBucket[] {
  const buckets: RoutePolicyV3CalibrationBucket[] = [];
  const width = 1 / bucketCount;
  for (let i = 0; i < bucketCount; i += 1) {
    const minScore = Number((i * width).toFixed(6));
    const maxScore = i === bucketCount - 1 ? 1 : Number(((i + 1) * width).toFixed(6));
    const rows = predictions.filter((prediction) => prediction.rawScore >= minScore && (i === bucketCount - 1 ? prediction.rawScore <= maxScore : prediction.rawScore < maxScore));
    const count = rows.length;
    buckets.push({
      minScore,
      maxScore,
      count,
      successRate: count ? Number((rows.filter((row) => row.observedSuccess).length / count).toFixed(4)) : 0,
    });
  }
  return buckets;
}

function chooseThreshold(predictions: RouteCalibrationPredictionV3[], buckets: RoutePolicyV3CalibrationBucket[], minThreshold: number): number {
  const candidates = unique([
    minThreshold,
    ...predictions.map((prediction) => Number(prediction.rawScore.toFixed(2))),
  ]).sort((a, b) => a - b);
  let bestThreshold = minThreshold;
  let bestUtility = -Infinity;
  for (const threshold of candidates) {
    const kept = predictions.filter((prediction) => prediction.rawScore >= threshold);
    if (kept.length === 0) continue;
    const precision = kept.filter((prediction) => prediction.observedSuccess).length / kept.length;
    const recall = kept.filter((prediction) => prediction.observedSuccess).length / Math.max(1, predictions.filter((prediction) => prediction.observedSuccess).length);
    const bucket = findBucket(threshold, buckets);
    const bucketSupport = bucket?.count || 0;
    const utility = precision * 0.7 + recall * 0.25 + Math.min(0.05, bucketSupport / 200);
    if (utility > bestUtility) {
      bestUtility = utility;
      bestThreshold = threshold;
    }
  }
  return clamp01(bestThreshold);
}

function findBucket(score: number, buckets: RoutePolicyV3CalibrationBucket[] = []) {
  return buckets.find((bucket, index) => score >= bucket.minScore && (index === buckets.length - 1 ? score <= bucket.maxScore : score < bucket.maxScore));
}

function findBucketOrNearest(score: number, buckets: RoutePolicyV3CalibrationBucket[] = []) {
  const direct = findBucket(score, buckets);
  if (direct || buckets.length === 0) return direct;
  return [...buckets].sort((a, b) => distanceToBucket(score, a) - distanceToBucket(score, b))[0];
}

function distanceToBucket(score: number, bucket: RoutePolicyV3CalibrationBucket) {
  if (score < bucket.minScore) return bucket.minScore - score;
  if (score > bucket.maxScore) return score - bucket.maxScore;
  return 0;
}

function modeThresholdAdjustment(mode?: RoutingModeV3) {
  if (mode === 'exact_correction') return 0.04;
  if (mode === 'semantic_planning') return 0.02;
  if (mode === 'casual_silence') return -0.03;
  return 0;
}

function unique(values: number[]) {
  return [...new Set(values.map((value) => Number(value.toFixed(2))))];
}

function uniqueRoutes(values: RouteKind[]) {
  return [...new Set(values)];
}

function clamp01(value: number) {
  return Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0));
}

function clampInt(value: any, fallback: number, min: number, max: number) {
  const n = Number.isFinite(Number(value)) ? Math.trunc(Number(value)) : fallback;
  return Math.max(min, Math.min(max, n));
}
