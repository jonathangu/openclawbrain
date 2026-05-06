import type { RouteActionPrototypeV3, RouteBanditStateV3, RouteCounterfactual, RouteDecision, RouteFrameV2, RouteFrameV3, RoutePairExampleV3, RoutePolicyRuleV3, RoutePolicySnapshotV3, RouteTeacherRun, RouteTrainingExampleV2, TurnFrame } from './memory-types.js';
export interface RoutePolicyV3ValidationReport {
    ok: boolean;
    status: 'active' | 'shadow' | 'rejected';
    reason: string;
    errors: string[];
    warnings: string[];
    projectedSyncPlannerRate: number;
    noisyActionRate: number;
    harmRate: number;
}
export interface RoutePolicyV3MatchResult {
    matched: boolean;
    rule?: RoutePolicyRuleV3;
    score: number;
    rawScore?: number;
    calibratedScore?: number;
    threshold?: number;
    abstained?: boolean;
    reasonCode: string;
}
interface RoutePolicyV3ScoreOptions {
    requireActive?: boolean;
}
export interface RoutePolicyV3DistillationReport {
    snapshot?: RoutePolicySnapshotV3;
    validation?: RoutePolicyV3ValidationReport;
    framesConsidered: number;
    pairExamplesConsidered: number;
    prototypesConsidered: number;
    rulesGenerated: number;
}
export interface RouteLearningV3IngestReport {
    frameId: string;
    chosenActionId: string;
    prototypeIds: string[];
    pairExamples: number;
    banditFeedbackId: string;
}
export declare function ingestRouteLearningArtifactsV3(store: any, agentId: string, decision: RouteDecision, routeFrame: RouteFrameV2 | null | undefined, teacherRun: RouteTeacherRun, counterfactuals: RouteCounterfactual[], lessons: RouteTrainingExampleV2[], config: any): RouteLearningV3IngestReport;
export declare function maybeDistillAndStorePolicyV3(store: any, agentId: string, config: any): RoutePolicyV3DistillationReport;
export declare function distillPolicyRulesV3(frames: RouteFrameV3[], pairs: RoutePairExampleV3[], prototypes: RouteActionPrototypeV3[], banditState: RouteBanditStateV3 | null, config: any): RoutePolicyRuleV3[];
export declare function validatePolicySnapshotV3(snapshot: Partial<RoutePolicySnapshotV3>, config?: any, existing?: RoutePolicySnapshotV3 | null): RoutePolicyV3ValidationReport;
export declare function scorePolicySnapshotV3(snapshot: RoutePolicySnapshotV3 | null | undefined, turnFrame: TurnFrame, message?: string, options?: RoutePolicyV3ScoreOptions): RoutePolicyV3MatchResult;
export declare function rankActionPrototypesV3(frame: Pick<RouteFrameV3, 'taskType' | 'turnSignals' | 'projectHint' | 'repoHint' | 'toolHints' | 'routeHintFlags' | 'redactedTurnSummary'>, prototypes: RouteActionPrototypeV3[], banditState: RouteBanditStateV3 | null): {
    prototype: RouteActionPrototypeV3;
    score: number;
    sparse: number;
    dense: number;
    bonus: number;
    riskPenalty: number;
    mode: import("./route-policy-v3-routing-mode.js").RoutingModeV3;
}[];
export {};
