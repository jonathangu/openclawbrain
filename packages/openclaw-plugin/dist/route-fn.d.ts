import type { InjectionPlan, RetrievalPlan, RouteKind, TurnFrame } from './memory-types.js';
import type { TurnEventPacket } from './capture.js';
import { type CaptureIntentResult, type RetrievalIntentResult } from './capture-intent.js';
export interface RouteFingerprint {
    agentId: string;
    scopeKey?: string;
    taskTypeHint?: string;
    topicKeys: string[];
    explicitMemoryReference: boolean;
    explicitCorrectionCue: boolean;
    captureIntent?: string;
    retrievalIntent?: string;
}
export interface CachedRoutePlan {
    route: RouteKind;
    retrievalPlan: RetrievalPlan;
    injectionPlan: InjectionPlan;
    confidence: number;
    expiresAt: string;
    sourceRouteDecisionId?: string;
    policySnapshotId?: string;
    matchedPolicyRuleId?: string;
    routingMode?: string;
    rawPolicyScore?: number;
    calibratedPolicyScore?: number;
    policyThreshold?: number;
    abstained?: boolean;
    fallbackSource?: string;
    retrievalIntent?: RetrievalIntentResult;
    captureIntent?: CaptureIntentResult;
}
export interface RoutePlan {
    route: RouteKind;
    confidence: number;
    turnFrame: TurnFrame;
    retrievalPlan: RetrievalPlan;
    injectionPlan: InjectionPlan;
    shouldRetrieve: boolean;
    enqueueCapture: boolean;
    retrievalIntent: RetrievalIntentResult;
    captureIntent: CaptureIntentResult;
    latencyReason: string;
    policySnapshotId?: string;
    matchedPolicyRuleId?: string;
    routingMode?: string;
    rawPolicyScore?: number;
    calibratedPolicyScore?: number;
    policyThreshold?: number;
    abstained?: boolean;
    fallbackSource?: string;
    reasonCode?: string;
}
export declare class RouteCache {
    private cache;
    get(fingerprint: RouteFingerprint): CachedRoutePlan | null;
    set(fingerprint: RouteFingerprint, plan: CachedRoutePlan): void;
    invalidate(predicate?: (key: string, value: CachedRoutePlan) => boolean): void;
}
export declare class RouteFn {
    private config;
    private cache;
    private store?;
    constructor(options: {
        config: any;
        cache?: RouteCache;
        store?: any;
    });
    fingerprint(packet: TurnEventPacket): RouteFingerprint;
    plan(packet: TurnEventPacket): RoutePlan;
    private loadPolicySnapshot;
}
