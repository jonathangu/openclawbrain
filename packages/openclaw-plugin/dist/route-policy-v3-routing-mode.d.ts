import type { RouteActionPrototypeV3, RouteFrameV3, RouteKind, TurnFrame } from './memory-types.js';
export type RoutingModeV3 = 'exact_correction' | 'workflow_exact' | 'semantic_planning' | 'casual_silence' | 'mixed';
export interface RoutingWeightsV3 {
    sparse: number;
    dense: number;
    bandit: number;
    risk: number;
    signalBonus: number;
}
export declare function detectRoutingModeV3(frame: Pick<RouteFrameV3, 'taskType' | 'turnSignals' | 'routeHintFlags' | 'redactedTurnSummary'> | TurnFrame, message?: string): RoutingModeV3;
export declare function hybridWeightsForRoutingModeV3(mode: RoutingModeV3): RoutingWeightsV3;
export declare function calibrationThresholdAdjustmentForModeV3(mode: RoutingModeV3): number;
export declare function prototypeRiskPenaltyV3(prototype: Pick<RouteActionPrototypeV3, 'route' | 'memoryTypes' | 'graphDepth' | 'syncPlanner' | 'harmPrior'>, mode: RoutingModeV3): number;
export declare function routeModeDiagnosticFamilyV3(mode: RoutingModeV3, route: RouteKind): string;
