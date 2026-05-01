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

export class LatencyController {
  private config: any;

  constructor(config: any) {
    this.config = config;
  }

  chooseTier(input: LatencyDecisionInput): LatencyTierDecision {
    const budgetAvailable = input.syncBudgetAvailable !== false;
    const shortTurn = input.latestUserMessage.trim().length < 40;
    const candidateCount = input.candidateCount ?? 0;
    const candidateAmbiguity = input.candidateAmbiguity ?? 0;

    if (budgetAvailable === false) {
      return this.decision('enqueue_async_only', 'sync planner budget exhausted', this.config.latency.syncPlannerSoftTimeoutMs, this.config.latency.fallbackOnTimeout);
    }

    if (input.recentRouteCacheHit === true) {
      return this.decision('cached_route', 'route cache hit', 0, 'cached_route');
    }

    if (input.recentPolicyMatch === true && !input.userExplicitlyReferencesMemory && candidateCount <= 5) {
      return this.decision('no_extra_llm', 'policy snapshot confident', 0, 'cached_route');
    }

    if (input.hasHighConfidenceCorrectionCandidate === true && this.config.latency.syncPlannerEnabled === true) {
      return this.decision('sync_memory_planner', 'high-signal correction', this.config.latency.syncPlannerSoftTimeoutMs, this.config.latency.fallbackOnTimeout);
    }

    if (input.userExplicitlyReferencesMemory === true && this.config.latency.syncPlannerEnabled === true) {
      return this.decision('sync_memory_planner', 'explicit memory reference', this.config.latency.syncPlannerSoftTimeoutMs, this.config.latency.fallbackOnTimeout);
    }

    if (input.taskValueEstimate === 'high' && (candidateCount >= 20 || candidateAmbiguity >= 0.5) && this.config.latency.syncPlannerEnabled === true) {
      return this.decision('sync_memory_planner', 'high-value ambiguous retrieval', this.config.latency.syncPlannerSoftTimeoutMs, this.config.latency.fallbackOnTimeout);
    }

    if (candidateCount === 0 && shortTurn) {
      return this.decision('no_extra_llm', 'short low-signal turn', 0, 'no_memory');
    }

    if (candidateCount > 0) {
      return this.decision('cached_route', 'local retrieval sufficient', 0, 'cached_route');
    }

    return this.decision('enqueue_async_only', 'no immediate memory work needed', 0, 'no_memory');
  }

  private decision(kind: LatencyTierDecision['kind'], reason: string, maxSyncMs: number, fallback: LatencyTierDecision['fallback']): LatencyTierDecision {
    return { kind, reason, maxSyncMs, fallback };
  }
}
