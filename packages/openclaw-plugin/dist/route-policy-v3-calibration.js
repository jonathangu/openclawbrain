export function buildCalibrationSummaryV3(predictions, config = {}) {
    const comparable = predictions.filter((prediction) => prediction.comparable !== false);
    if (comparable.length === 0)
        return undefined;
    const bucketCount = clampInt(config.routeLearning?.policyV3?.calibrationBuckets, 5, 3, 10);
    const abstainMargin = clamp01(Number(config.routeLearning?.policyV3?.abstainMargin ?? 0.05));
    const minThreshold = clamp01(Number(config.routeLearning?.policyV3?.minCalibratedConfidence ?? 0.62));
    const minRouteSamples = clampInt(config.routeLearning?.policyV3?.minRouteThresholdSamples, 3, 1, 50);
    const globalBuckets = buildBuckets(comparable, bucketCount);
    const routeThresholds = {};
    const routeBuckets = {};
    for (const route of uniqueRoutes(comparable.map((prediction) => prediction.route))) {
        const routeRows = comparable.filter((prediction) => prediction.route === route);
        if (routeRows.length < minRouteSamples)
            continue;
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
export function applyCalibrationV3(rawScore, route, calibration, mode) {
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
function buildBuckets(predictions, bucketCount) {
    const buckets = [];
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
function chooseThreshold(predictions, buckets, minThreshold) {
    const candidates = unique([
        minThreshold,
        ...predictions.map((prediction) => Number(prediction.rawScore.toFixed(2))),
    ]).sort((a, b) => a - b);
    let bestThreshold = minThreshold;
    let bestUtility = -Infinity;
    for (const threshold of candidates) {
        const kept = predictions.filter((prediction) => prediction.rawScore >= threshold);
        if (kept.length === 0)
            continue;
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
function findBucket(score, buckets = []) {
    return buckets.find((bucket, index) => score >= bucket.minScore && (index === buckets.length - 1 ? score <= bucket.maxScore : score < bucket.maxScore));
}
function findBucketOrNearest(score, buckets = []) {
    const direct = findBucket(score, buckets);
    if (direct || buckets.length === 0)
        return direct;
    return [...buckets].sort((a, b) => distanceToBucket(score, a) - distanceToBucket(score, b))[0];
}
function distanceToBucket(score, bucket) {
    if (score < bucket.minScore)
        return bucket.minScore - score;
    if (score > bucket.maxScore)
        return score - bucket.maxScore;
    return 0;
}
function modeThresholdAdjustment(mode) {
    if (mode === 'exact_correction')
        return 0.04;
    if (mode === 'semantic_planning')
        return 0.02;
    if (mode === 'casual_silence')
        return -0.03;
    return 0;
}
function unique(values) {
    return [...new Set(values.map((value) => Number(value.toFixed(2))))];
}
function uniqueRoutes(values) {
    return [...new Set(values)];
}
function clamp01(value) {
    return Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0));
}
function clampInt(value, fallback, min, max) {
    const n = Number.isFinite(Number(value)) ? Math.trunc(Number(value)) : fallback;
    return Math.max(min, Math.min(max, n));
}
