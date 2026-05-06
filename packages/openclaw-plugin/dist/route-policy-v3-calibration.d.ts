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
export declare function buildCalibrationSummaryV3(predictions: RouteCalibrationPredictionV3[], config?: any): RoutePolicyV3CalibrationSummary | undefined;
export declare function applyCalibrationV3(rawScore: number, route: RouteKind, calibration?: RoutePolicyV3CalibrationSummary, mode?: RoutingModeV3): {
    rawScore: number;
    calibratedScore: number;
    threshold: number;
    abstained: boolean;
};
