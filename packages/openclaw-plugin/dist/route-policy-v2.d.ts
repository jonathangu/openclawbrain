import type { RoutePolicyRuleV2, RoutePolicySnapshotV2, RouteTrainingExampleV2, TurnFrame } from './memory-types.js';
export interface RoutePolicyValidationReport {
    ok: boolean;
    status: 'active' | 'shadow' | 'rejected';
    reason: string;
    errors: string[];
    warnings: string[];
    projectedSyncPlannerRate: number;
    noisyInjectionRate: number;
    harmRate: number;
}
export interface RoutePolicyMatchResult {
    matched: boolean;
    rule?: RoutePolicyRuleV2;
    score: number;
    reasonCode: string;
}
export interface RoutePolicyDistillationReport {
    snapshot?: RoutePolicySnapshotV2;
    validation?: RoutePolicyValidationReport;
    examplesConsidered: number;
    rulesGenerated: number;
}
export declare function maybeDistillAndStorePolicyV2(store: any, agentId: string, config: any): RoutePolicyDistillationReport;
export declare function distillPolicyRulesV2(examples: RouteTrainingExampleV2[], config: any): RoutePolicyRuleV2[];
export declare function validatePolicySnapshotV2(snapshot: Partial<RoutePolicySnapshotV2>, config?: any, existing?: RoutePolicySnapshotV2 | null): RoutePolicyValidationReport;
export declare function scorePolicySnapshotV2(snapshot: RoutePolicySnapshotV2 | null | undefined, turnFrame: TurnFrame, message?: string): RoutePolicyMatchResult;
