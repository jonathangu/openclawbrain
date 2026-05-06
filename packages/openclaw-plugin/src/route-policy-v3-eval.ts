import type { RouteFrameV3, RouteKind } from './memory-types.js';
import type { RoutePolicyV3CalibrationSummary } from './route-policy-v3-calibration.js';
import { routeModeDiagnosticFamilyV3, type RoutingModeV3 } from './route-policy-v3-routing-mode.js';

export interface RouteReplayPredictionV3 {
  frameId: string;
  route: RouteKind | null;
  rawScore: number;
  calibratedScore: number;
  abstained: boolean;
  comparable: boolean;
  matchedObservedRoute: boolean;
  reward: number;
  mode: RoutingModeV3;
}

export interface RoutePolicyV3ReplaySummary {
  frames: number;
  comparableFrames: number;
  matchedFrames: number;
  abstainRate: number;
  routeAgreement: number;
  rewardWeightedAgreement: number;
  projectedValue: number;
  baselineProjectedValue: number;
  estimatedImprovement: number;
  modeBreakdown: Record<string, {
    frames: number;
    matchedFrames: number;
    abstained: number;
    projectedValue: number;
  }>;
  calibration?: Pick<RoutePolicyV3CalibrationSummary, 'holdoutFrames' | 'comparableFrames' | 'globalThreshold' | 'abstainMargin'>;
}

export function summarizeReplayEvaluationV3(
  predictions: RouteReplayPredictionV3[],
  baselinePredictions: RouteReplayPredictionV3[] = [],
  calibration?: RoutePolicyV3CalibrationSummary,
): RoutePolicyV3ReplaySummary {
  const comparableFrames = predictions.filter((prediction) => prediction.comparable).length;
  const matchedFrames = predictions.filter((prediction) => prediction.matchedObservedRoute).length;
  const abstained = predictions.filter((prediction) => prediction.abstained).length;
  const rewardDenom = predictions.reduce((sum, prediction) => sum + Math.abs(prediction.reward), 0) || 1;
  const projectedValue = predictions.reduce((sum, prediction) => sum + proxyValueForPrediction(prediction), 0);
  const baselineProjectedValue = baselinePredictions.reduce((sum, prediction) => sum + proxyValueForPrediction(prediction), 0);
  const modeBreakdown: RoutePolicyV3ReplaySummary['modeBreakdown'] = {};

  for (const prediction of predictions) {
    const key = routeModeDiagnosticFamilyV3(prediction.mode, prediction.route || 'no_memory');
    modeBreakdown[key] ||= { frames: 0, matchedFrames: 0, abstained: 0, projectedValue: 0 };
    modeBreakdown[key].frames += 1;
    modeBreakdown[key].matchedFrames += prediction.matchedObservedRoute ? 1 : 0;
    modeBreakdown[key].abstained += prediction.abstained ? 1 : 0;
    modeBreakdown[key].projectedValue += proxyValueForPrediction(prediction);
  }

  return {
    frames: predictions.length,
    comparableFrames,
    matchedFrames,
    abstainRate: predictions.length ? Number((abstained / predictions.length).toFixed(4)) : 0,
    routeAgreement: comparableFrames ? Number((matchedFrames / comparableFrames).toFixed(4)) : 0,
    rewardWeightedAgreement: Number((predictions.reduce((sum, prediction) => sum + (prediction.matchedObservedRoute ? Math.abs(prediction.reward) : 0), 0) / rewardDenom).toFixed(4)),
    projectedValue: Number(projectedValue.toFixed(4)),
    baselineProjectedValue: Number(baselineProjectedValue.toFixed(4)),
    estimatedImprovement: Number((projectedValue - baselineProjectedValue).toFixed(4)),
    modeBreakdown,
    calibration: calibration ? {
      holdoutFrames: calibration.holdoutFrames,
      comparableFrames: calibration.comparableFrames,
      globalThreshold: calibration.globalThreshold,
      abstainMargin: calibration.abstainMargin,
    } : undefined,
  };
}

function proxyValueForPrediction(prediction: RouteReplayPredictionV3) {
  if (prediction.abstained) return prediction.matchedObservedRoute ? prediction.reward * 0.15 : 0;
  if (prediction.matchedObservedRoute) return prediction.reward;
  if (prediction.reward < 0 && prediction.route === 'no_memory') return Math.abs(prediction.reward) * 0.35;
  if (prediction.reward < 0 && prediction.route && prediction.route !== 'no_memory') return Math.abs(prediction.reward) * 0.1;
  return 0;
}
