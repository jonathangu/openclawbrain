import type { RouteKind } from './memory-types.js';
import type { RoutePolicyV3CalibrationSummary } from './route-policy-v3-calibration.js';
import { type RoutingModeV3 } from './route-policy-v3-routing-mode.js';
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
export declare function summarizeReplayEvaluationV3(predictions: RouteReplayPredictionV3[], baselinePredictions?: RouteReplayPredictionV3[], calibration?: RoutePolicyV3CalibrationSummary): RoutePolicyV3ReplaySummary;
