export type MemoryType = 'correction' | 'preference' | 'workflow' | 'context';
export interface MemoryNode {
    id: string;
    agentId: string;
    type: MemoryType;
    content: string;
    positive?: string;
    negative?: string;
    scopeKind: 'global_user' | 'agent' | 'repo' | 'project' | 'session' | 'tool';
    scopeKey?: string;
    normalizedKey: string;
    tags: string[];
    importance: number;
    freshness: number;
    confidence: number;
    useCount: number;
    usefulCount: number;
    captureCount: number;
    distilledByModel?: string;
    distillerPromptVersion?: string;
    distillationConfidence?: number;
    evidenceKind?: string;
    evidenceHash?: string;
    sourceHook?: string;
    sourceTurnId?: string;
    sourceSessionId?: string;
    createdAt: string;
    updatedAt: string;
    lastSeenAt: string;
    lastUsedAt?: string;
    supersededBy?: string;
    deletedAt?: string;
}
export type EdgeRelation = 'related' | 'contradicts' | 'supersedes' | 'extends' | 'used_with' | 'supports_workflow';
export interface MemoryEdge {
    id: string;
    agentId: string;
    fromId: string;
    toId: string;
    relation: EdgeRelation;
    weight: number;
    evidenceCount: number;
    createdAt: string;
    updatedAt: string;
}
export type TaskType = 'coding' | 'planning' | 'debugging' | 'writing' | 'preference_update' | 'correction' | 'general_question' | 'other';
export interface ActiveObject {
    kind: 'repo' | 'file' | 'tool' | 'preference' | 'plan' | 'person' | 'concept';
    value: string;
}
export interface TurnFrame {
    summary: string;
    userGoal: string;
    taskType: TaskType;
    activeObjects: ActiveObject[];
    impliedNeeds: string[];
    memoryQuestions: string[];
    constraints: string[];
    routeHints: {
        likelyNeedsCorrections: boolean;
        likelyNeedsPreferences: boolean;
        likelyNeedsWorkflow: boolean;
        likelyNeedsProjectContext: boolean;
    };
}
export type RouteKind = 'no_memory' | 'capture_only' | 'retrieve_memory' | 'retrieve_and_distill' | 'high_confidence_correction_only';
export interface RetrievalPlan {
    queries: string[];
    memoryTypes: MemoryType[];
    requiredTags: string[];
    excludedTags: string[];
    graphDepth: 0 | 1 | 2;
    maxCandidates: number;
}
export interface InjectionPlan {
    maxItems: number;
    maxChars: number;
    preferredFormat: 'bullets' | 'rules' | 'workflow_steps' | 'do_dont' | 'none';
}
export interface CapturePlan {
    shouldDistillFeedbackNow: boolean;
    likelyFeedbackType?: 'correction' | 'preference' | 'workflow' | 'outcome' | 'none';
}
export interface LatencyPlan {
    syncLlmAllowed: boolean;
    reason: string;
    fallback: 'no_memory' | 'cached_route' | 'high_confidence_corrections_only';
}
export interface RouteDecision {
    id: string;
    agentId: string;
    sessionId?: string;
    turnId?: string;
    runId?: string;
    route: RouteKind;
    confidence: number;
    latencyTier: string;
    syncLlmUsed: boolean;
    syncLatencyMs?: number;
    fallbackUsed: boolean;
    turnFrame: TurnFrame;
    retrievalPlan: RetrievalPlan;
    injectionPlan: InjectionPlan;
    selectedMemoryIds: string[];
    omittedMemoryIds: string[];
    model?: string;
    promptVersion?: string;
    policySnapshotId?: string;
    outcome?: string;
    reward: number;
    createdAt: string;
    resolvedAt?: string;
}
export interface ContextSelection {
    shouldInject: boolean;
    confidence: number;
    selectedMemoryIds: string[];
    distilledContext: string;
    selected: Array<{
        memoryId: string;
        reason: 'directly_relevant_correction' | 'matching_user_preference' | 'repo_workflow' | 'tool_guidance' | 'contradiction_resolution' | 'supporting_context';
        useHow: 'must_follow' | 'prefer' | 'consider' | 'avoid';
        confidence: number;
    }>;
    omitted: Array<{
        memoryId: string;
        reason: 'irrelevant' | 'too_general' | 'superseded' | 'low_confidence' | 'would_pollute_prompt' | 'budget';
    }>;
    audit: {
        promptBudgetUsedChars: number;
        risk: 'low' | 'medium' | 'high';
    };
}
export type InjectionOutcome = 'pending' | 'helped' | 'accepted' | 'ignored' | 'assistant_failed_to_use' | 'user_corrected' | 'harmful' | 'tool_success' | 'tool_failure' | 'unknown';
export interface InjectionEvent {
    id: string;
    agentId: string;
    memoryId: string;
    routeDecisionId?: string;
    runId?: string;
    turnId?: string;
    sessionId?: string;
    query: string;
    rank: number;
    score: number;
    injectedAt: string;
    resolvedAt?: string;
    outcome: InjectionOutcome;
    correctionSignal?: string;
}
export type FeedbackType = 'correction' | 'preference' | 'standing_instruction' | 'workflow' | 'context' | 'outcome' | 'delete_or_suppress' | 'none';
export interface ContradictionAction {
    existingMemoryId?: string;
    reason: string;
    action: 'supersede_existing' | 'merge' | 'keep_both';
}
export interface MemoryCandidate {
    type: 'correction' | 'preference' | 'workflow' | 'context';
    distilledText: string;
    subject: string;
    scope: {
        kind: string;
        key?: string;
    };
    positive?: string;
    negative?: string;
    normalizedKey: string;
    tags: string[];
    confidence: number;
    importanceHint: number;
    retention: 'durable' | 'medium_term' | 'short_term' | 'ephemeral';
    contradictions: ContradictionAction[];
}
export interface InjectionFeedback {
    injectionId: string;
    memoryId: string;
    outcome: InjectionOutcome;
    confidence: number;
    evidence: string;
}
export interface WorkflowCandidate {
    distilledWorkflow: string;
    prerequisites: string[];
    steps: string[];
    successSignal: string;
    failureSignal?: string;
    confidence: number;
}
export interface FeedbackDistillation {
    version: 1;
    shouldStore: boolean;
    confidence: number;
    feedbackType: FeedbackType;
    memoryCandidates: MemoryCandidate[];
    injectionFeedback: InjectionFeedback[];
    workflowCandidates: WorkflowCandidate[];
    audit: {
        modelReasonCode: string;
        storeRawTranscript: false;
        redactionNeeded: boolean;
    };
}
export type MemoryOperationKind = 'create' | 'update' | 'supersede' | 'reinforce' | 'delete_or_suppress' | 'ignore';
export interface MemoryOperation {
    kind: MemoryOperationKind;
    candidate?: MemoryCandidate;
    memoryId?: string;
    reason?: string;
}
export type JobStatus = 'pending' | 'running' | 'completed' | 'failed' | 'dead';
export type JobKind = 'feedback_distillation' | 'route_learning' | 'outcome_classification' | 'consolidation' | 'pruning' | 'score_update';
export interface BackgroundJob {
    id: string;
    agentId: string;
    kind: JobKind;
    status: JobStatus;
    priority: number;
    payload: Record<string, unknown>;
    attempts: number;
    maxAttempts: number;
    availableAt: string;
    startedAt?: string;
    finishedAt?: string;
    error?: string;
    createdAt: string;
    updatedAt: string;
}
export interface RouteExample {
    id: string;
    agentId: string;
    turnFrame: TurnFrame;
    routeDecision: Partial<RouteDecision>;
    outcome: string;
    reward: number;
    lesson: string;
    tags: string[];
    createdAt: string;
}
export interface RoutePolicySnapshot {
    id: string;
    agentId: string;
    policyText: string;
    examples: string[];
    model?: string;
    promptVersion?: string;
    createdAt: string;
    active: boolean;
}
export type DistillationPhase = 'immediate_feedback' | 'agent_end_feedback' | 'route_turn_frame' | 'context_selection' | 'memory_planner' | 'route_learning' | 'outcome_classification';
export interface DistillationRun {
    id: string;
    agentId: string;
    sessionId?: string;
    turnId?: string;
    runId?: string;
    phase: DistillationPhase;
    model: string;
    promptVersion: string;
    inputHash: string;
    redactedInputSummary?: string;
    outputJson: string;
    validationStatus: 'valid' | 'invalid' | 'repaired';
    validationError?: string;
    latencyMs?: number;
    createdAt: string;
}
export interface ProofEvent {
    id: string;
    agentId: string;
    kind: string;
    createdAt: string;
    sourceHook?: string;
    turnId?: string;
    sessionId?: string;
    runId?: string;
    memoryId?: string;
    injectionId?: string;
    routeDecisionId?: string;
    distillationRunId?: string;
    rawTranscriptStored: boolean;
    payload: Record<string, unknown>;
}
