export interface LatencyDecisionInput {
    agentId: string;
    sessionId?: string;
    latestUserMessage: string;
    recentRouteCacheHit?: boolean;
    recentPolicyMatch?: boolean;
    candidateCount?: number;
    candidateAmbiguity?: number;
    hasHighConfidenceCorrectionCandidate?: boolean;
    userExplicitlyReferencesMemory?: boolean;
    taskValueEstimate: 'low' | 'medium' | 'high';
    configMode: 'conservative' | 'balanced' | 'aggressive' | string;
    syncBudgetAvailable?: boolean;
}
export interface LatencyTierDecision {
    kind: 'no_extra_llm' | 'cached_route' | 'sync_memory_planner' | 'enqueue_async_only';
    maxSyncMs: number;
    reason: string;
    fallback: 'no_memory' | 'cached_route' | 'high_confidence_corrections_only';
}
export declare class LatencyController {
    private config;
    constructor(config: any);
    chooseTier(input: LatencyDecisionInput): LatencyTierDecision;
    private decision;
}
