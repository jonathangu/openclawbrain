export type MemoryType = 'correction' | 'preference' | 'workflow' | 'project_fact' | 'tool_convention' | 'routing_rule' | 'agent_assignment' | 'recall_rule' | 'outcome' | 'context';
export interface MemoryNode {
    id: string;
    agentId: string;
    type: MemoryType;
    content: string;
    positive?: string;
    negative?: string;
    scopeKind: 'global_user' | 'agent' | 'repo' | 'project' | 'app' | 'person' | 'channel' | 'session' | 'task' | 'tool';
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
export type MemoryRetentionState = 'stored' | 'redacted' | 'soft_deleted' | 'tombstoned' | 'hard_deleted';
export type MemoryBehavioralAvailability = 'injectable' | 'confirm_before_use' | 'explicit_request_only' | 'never_use';
export type MemoryTemporalValidity = 'current' | 'stale' | 'expired' | 'unknown';
export type MemoryPrivacyClass = 'normal' | 'sensitive' | 'recall_only' | 'do_not_restore' | 'do_not_reveal_proactively';
export type MemoryValidationStrategy = 'none' | 'user_confirm' | 'environment_check' | 'tool_success' | 'document_check' | 'time_expiry' | 'never_proactive';
export interface MemoryValidity {
    memoryId: string;
    retentionState: MemoryRetentionState;
    behavioralAvailability: MemoryBehavioralAvailability;
    temporalValidity: MemoryTemporalValidity;
    privacyClass: MemoryPrivacyClass;
    decayPolicy: string;
    validationStrategy: MemoryValidationStrategy;
    validFrom?: string;
    validUntil?: string;
    expiresAt?: string;
    lastConfirmedAt?: string;
    lastVerifiedAt?: string;
    lastSuccessfulUseAt?: string;
    lastFailedUseAt?: string;
    lastContradictedAt?: string;
    revalidateAfter?: string;
    halfLifeDays?: number;
    evidenceConfidence: number;
    currentValidityScore: number;
    behavioralAuthorityScore: number;
    stateReason?: string;
    updatedAt: string;
}
export type MemoryAuthorityDecisionKind = 'inject' | 'weak_context' | 'verify_before_use' | 'confirm_before_use' | 'abstain' | 'audit_only' | 'never_use';
export interface MemoryAuthorityResolution {
    memoryId: string;
    decision: MemoryAuthorityDecisionKind;
    authorityScore: number;
    relevanceScore: number;
    evidenceConfidence: number;
    currentValidity: number;
    behavioralAuthority: number;
    validationStrategy: MemoryValidationStrategy;
    reasons: string[];
    requiredAction?: 'verify_environment' | 'ask_user' | 'explicit_recall' | 'none';
    risk: 'low' | 'medium' | 'high';
}
export type MemoryAuthorityEventType = 'captured' | 'reinforced' | 'used' | 'weak_context_used' | 'verification_requested' | 'verification_failed' | 'confirmation_requested' | 'confirmation_declined' | 'abstained' | 'audit_only' | 'suppressed' | 'soft_deleted' | 'tombstoned' | 'hard_deleted' | 'superseded' | 'expired' | 'revived' | 'user_corrected' | 'overridden_by_current_instruction';
export interface MemoryAuthorityEvent {
    id: string;
    agentId: string;
    memoryId: string;
    eventType: MemoryAuthorityEventType;
    source: string;
    turnId?: string;
    routeId?: string;
    evidenceId?: string;
    oldValue?: string;
    newValue?: string;
    reason?: string;
    createdAt: string;
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
export type GraphMaintenanceRunMode = 'health' | 'dry_run' | 'safe_auto' | 'manual_apply';
export type GraphMaintenanceRunStatus = 'running' | 'completed' | 'failed';
export type GraphMaintenanceProposalStatus = 'drafted' | 'guard_rejected' | 'pending_review' | 'approved' | 'applied' | 'rejected' | 'expired' | 'superseded' | 'failed_apply' | 'stale';
export type GraphMaintenanceProposalType = 'merge_exact_duplicate_nodes' | 'retire_bad_edge' | 'mark_stale_high_authority' | 'block_tombstone_recapture' | 'propose_scoped_exception' | 'record_feedback_edge_observation';
export type GraphMaintenanceRisk = 'low' | 'medium' | 'high';
export type GraphMaintenanceEdgeFamily = 'epistemic' | 'temporal' | 'scope' | 'behavioral' | 'lineage' | 'retention';
export type GraphMaintenanceEdgeState = 'candidate' | 'active' | 'weakened' | 'retired' | 'blocked' | 'structural';
export interface GraphMaintenanceRun {
    id: string;
    agentId: string;
    mode: GraphMaintenanceRunMode;
    status: GraphMaintenanceRunStatus;
    startedAt: string;
    finishedAt?: string;
    inputWindowStart?: string;
    inputWindowEnd?: string;
    nodesScanned: number;
    edgesScanned: number;
    proposalsCreated: number;
    proposalsApplied: number;
    proposalsRejected: number;
    riskSummary: Record<string, unknown>;
    metrics: Record<string, unknown>;
}
export interface GraphMaintenanceProposal {
    id: string;
    runId: string;
    agentId: string;
    proposalType: GraphMaintenanceProposalType;
    targetKind: 'node' | 'edge' | 'graph' | 'validity' | 'tombstone';
    targetIds: string[];
    proposedPatch: Record<string, unknown>;
    evidence: Record<string, unknown>;
    preconditions: Record<string, unknown>;
    appliedDiff: Record<string, unknown>;
    rollbackPatch?: Record<string, unknown>;
    riskFactors: Record<string, unknown>;
    confidence: number;
    risk: GraphMaintenanceRisk;
    status: GraphMaintenanceProposalStatus;
    reason: string;
    reviewRequiredReason?: string;
    reviewedBy?: string;
    reviewedAt?: string;
    proposalHash: string;
    expiresAt?: string;
    graphSnapshotId?: string;
    graphVersion: number;
    createdAt: string;
    appliedAt?: string;
    rejectedAt?: string;
}
export interface MemoryNodeLineage {
    id: string;
    agentId: string;
    childMemoryId: string;
    parentMemoryId: string;
    relation: 'duplicate_of' | 'canonical_for' | 'merged_into' | 'derived_from' | 'split_from' | 'superseded_by';
    proposalId?: string;
    evidence: Record<string, unknown>;
    createdAt: string;
}
export interface MemoryEdgeObservation {
    id: string;
    agentId: string;
    edgeId?: string;
    fromId?: string;
    toId?: string;
    relation: string;
    edgeFamily: GraphMaintenanceEdgeFamily;
    edgeState: GraphMaintenanceEdgeState;
    observationType: string;
    delta: number;
    sourceType: 'user_statement' | 'user_correction' | 'tool_result' | 'environment_check' | 'route_teacher' | 'inferred_outcome' | 'llm_judgment' | 'maintenance';
    sourceIndependence: 'independent' | 'same_origin' | 'derived' | 'self_reinforcing';
    signalStrength: 'explicit' | 'implicit' | 'inferred' | 'weak';
    polarity: 'supports' | 'contradicts' | 'scopes' | 'supersedes' | 'irrelevant';
    causalAttribution: 'memory_helped' | 'memory_harmed' | 'memory_irrelevant' | 'unknown';
    routeId?: string;
    proofEventId?: string;
    reason: string;
    createdAt: string;
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
export interface RouteFrameV2 {
    id: string;
    agentId: string;
    sessionKeyHash?: string;
    turnHash: string;
    redactedTurnSummary: string;
    taskType: TaskType;
    turnSignals: string[];
    intentSignals: string[];
    safetySignals: string[];
    projectHint?: string;
    repoHint?: string;
    latencyBudgetMs: number;
    createdAt: string;
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
    routeFrameId?: string;
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
    retrievalIntent?: any;
    captureIntent?: any;
    selectedMemoryIds: string[];
    omittedMemoryIds: string[];
    model?: string;
    promptVersion?: string;
    policySnapshotId?: string;
    policyRuleId?: string;
    routingMode?: string;
    rawPolicyScore?: number;
    calibratedPolicyScore?: number;
    policyThreshold?: number;
    abstained?: boolean;
    fallbackSource?: string;
    candidateCount?: number;
    reasonCode?: string;
    injectionPayloadHash?: string;
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
        authorityDecision?: MemoryAuthorityDecisionKind;
        authorityReasons?: string[];
        requiredAction?: MemoryAuthorityResolution['requiredAction'];
    }>;
    omitted: Array<{
        memoryId: string;
        reason: 'irrelevant' | 'too_general' | 'superseded' | 'low_confidence' | 'would_pollute_prompt' | 'budget' | 'stale' | 'expired' | 'confirm_needed' | 'verify_needed' | 'privacy' | 'tombstoned' | 'current_instruction_override' | 'never_use' | 'audit_only';
        authorityDecision?: MemoryAuthorityDecisionKind;
        authorityReasons?: string[];
    }>;
    audit: {
        promptBudgetUsedChars: number;
        risk: 'low' | 'medium' | 'high';
        authority?: MemoryAuthorityResolution[];
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
    type: MemoryType;
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
    riskClass?: 'ordinary' | 'private' | 'sensitive_recall' | 'credential_secret' | 'unsafe';
    disclosure?: 'normal' | 'on_explicit_user_request_only' | 'never';
    proactiveInjectionAllowed?: boolean;
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
        rejectionReasons?: string[];
        safeCandidatePreview?: string;
    };
}
export interface CaptureAuditRow {
    id: string;
    agentId: string;
    turnId?: string;
    sessionId?: string;
    runId?: string;
    createdAt: string;
    retrievalIntent: any;
    captureIntent: any;
    captureJobCreated: boolean;
    distillerRan: boolean;
    distillerModel?: string;
    distillerLatencyMs?: number;
    fallbackRan: boolean;
    candidateCount: number;
    storedCount: number;
    rejectedCount: number;
    rejectionReasons: string[];
    safeCandidatePreview?: string;
    evidenceHash?: string;
}
export type MemoryOperationKind = 'create' | 'update' | 'supersede' | 'reinforce' | 'delete_or_suppress' | 'ignore';
export interface MemoryOperation {
    kind: MemoryOperationKind;
    candidate?: MemoryCandidate;
    memoryId?: string;
    reason?: string;
}
export type JobStatus = 'pending' | 'running' | 'completed' | 'failed' | 'dead';
export type JobKind = 'feedback_distillation' | 'route_learning' | 'route_teacher' | 'outcome_classification' | 'consolidation' | 'pruning' | 'score_update';
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
export type RouteTeacherVerdict = 'correct_route' | 'missed_recall' | 'over_injected' | 'should_stay_silent' | 'wrong_memory_type' | 'latency_waste' | 'unsafe' | 'unknown';
export type RouteCounterfactualKind = 'no_memory' | 'actual_injection' | 'top_k_alternate' | 'broader_graph' | 'correction_only' | 'workflow_only' | 'preference_only' | 'context_only' | 'stay_silent' | 'sync_planner';
export type RouteCounterfactualOutcome = 'likely_helpful' | 'likely_neutral' | 'likely_noise' | 'likely_harmful' | 'likely_missed' | 'unknown';
export type RouteTrainingExampleKind = 'prefer_route' | 'avoid_route' | 'missed_recall' | 'correct_silence' | 'avoid_sync_planner' | 'prefer_sync_planner' | 'prefer_memory_type' | 'avoid_memory_type' | 'prefer_graph_depth' | 'avoid_graph_depth';
export interface RouteGraphSnapshot {
    id: string;
    agentId: string;
    routeDecisionId: string;
    querySet: string[];
    candidateMemoryIds: string[];
    candidateSummaries: Array<{
        id: string;
        type: MemoryType;
        scope: string;
        redactedContent: string;
        score: number;
        freshness: number;
        graphDistance: number;
        linkedMemoryIds: string[];
    }>;
    graphStats: {
        nodeCountSeen: number;
        edgeCountSeen: number;
        maxDepth: number;
    };
    createdAt: string;
}
export interface RouteTeacherRun {
    id: string;
    agentId: string;
    routeDecisionId: string;
    model: string;
    promptVersion: string;
    inputHash: string;
    outputHash: string;
    verdict: RouteTeacherVerdict;
    teacherRoute: RouteKind;
    teacherMemoryIds: string[];
    teacherQueries: string[];
    teacherGraphDepth: 0 | 1 | 2;
    syncPlannerWorthIt: boolean;
    confidence: number;
    rationale: string;
    validated: boolean;
    rejectionReason?: string;
    createdAt: string;
}
export interface RouteCounterfactual {
    id: string;
    agentId: string;
    routeTeacherRunId: string;
    routeDecisionId: string;
    kind: RouteCounterfactualKind;
    memoryIds: string[];
    memoryTypes: MemoryType[];
    graphDepth: 0 | 1 | 2;
    estimatedOutcome: RouteCounterfactualOutcome;
    confidence: number;
    rationale: string;
    createdAt: string;
}
export interface RouteTrainingExampleV2 {
    id: string;
    agentId: string;
    routeDecisionId: string;
    routeTeacherRunId?: string;
    exampleKind: RouteTrainingExampleKind;
    taskType: TaskType;
    turnSignals: string[];
    route: RouteKind;
    memoryTypes: MemoryType[];
    queryTemplates: string[];
    graphDepth: 0 | 1 | 2;
    confidence: number;
    supportCount: number;
    harmCount: number;
    source: 'actual_outcome' | 'teacher' | 'counterfactual' | 'manual_eval';
    evidenceIds: string[];
    createdAt: string;
}
export interface RoutePolicyRuleV2 {
    id: string;
    priority?: number;
    match: {
        taskType?: TaskType | string;
        turnSignals?: string[];
        projectHint?: string;
        repoHintPresent?: boolean;
        safetySignalsAbsent?: string[];
    };
    route: RouteKind;
    memoryTypes: MemoryType[];
    queries: string[];
    graphDepth: 0 | 1 | 2;
    syncPlanner: 'no' | 'never_unless_ambiguous' | 'allowed' | 'prefer';
    confidence: number;
    evidenceIds: string[];
    stats?: {
        support?: number;
        harm?: number;
        harmRate?: number;
        kind?: string;
    };
    reason?: string;
}
export interface RoutePolicySnapshotV2 {
    id: string;
    agentId: string;
    version: 'route-policy-v2';
    status: 'candidate' | 'active' | 'rejected' | 'shadow';
    rules: RoutePolicyRuleV2[];
    globalBudgets: {
        maxSyncPlannerRate: number;
        maxInjectedMemories: number;
        maxInjectedChars: number;
        defaultGraphDepth: 0 | 1 | 2;
    };
    evalSummary?: {
        cases: number;
        wins: number;
        ties: number;
        misses: number;
        noisyInjections: number;
        harms: number;
        p95LatencyMs: number;
        activationDecision?: string;
        activationStatusReason?: string;
        validationErrors?: string[];
        validationWarnings?: string[];
        projectedSyncPlannerRate?: number;
        noisyInjectionRate?: number;
        harmRate?: number;
    };
    exampleIds: string[];
    model?: string;
    promptVersion?: string;
    createdAt: string;
}
export type RouteActionSyncPlannerMode = 'no' | 'never_unless_ambiguous' | 'allowed' | 'prefer';
export type RouteActionPrototypeStatus = 'active' | 'shadow' | 'retired' | 'cold_start';
export type RouteActionPrototypeProvenance = 'handwritten' | 'distilled' | 'learned';
export type RoutePairLabelSource = 'teacher' | 'counterfactual' | 'manual' | 'outcome' | 'bandit';
export type RouteBanditOutcomeLabel = 'accepted' | 'rejected' | 'ambiguous';
export type RouteCalibrationExampleSplitV3 = 'holdout' | 'replay_eval' | 'shadow_audit';
export type RouteEvalCaseQualityV3 = 'trusted' | 'usable' | 'weak' | 'ambiguous';
export type RouteEvalCaseLabelSourceV3 = 'outcome' | 'teacher' | 'manual' | 'consensus';
export interface RouteFrameV3 {
    id: string;
    agentId: string;
    routeDecisionId: string;
    routeFrameId?: string;
    redactedTurnSummary: string;
    taskType: TaskType;
    turnSignals: string[];
    projectHint?: string;
    repoHint?: string;
    toolHints: string[];
    routeHintFlags: string[];
    chosenActionId: string;
    chosenRoute: RouteKind;
    chosenMemoryTypes: MemoryType[];
    chosenGraphDepth: 0 | 1 | 2;
    chosenSyncPlanner: RouteActionSyncPlannerMode;
    policySnapshotId?: string;
    policyRuleId?: string;
    routingMode?: string;
    rawPolicyScore?: number;
    calibratedPolicyScore?: number;
    policyThreshold?: number;
    abstained?: boolean;
    fallbackSource?: string;
    outcome?: string;
    reward: number;
    rewardComponents?: Record<string, number>;
    payloadHash: string;
    createdAt: string;
}
export interface RouteActionPrototypeV3 {
    id: string;
    agentId: string;
    route: RouteKind;
    memoryTypes: MemoryType[];
    graphDepth: 0 | 1 | 2;
    syncPlanner: RouteActionSyncPlannerMode;
    queryTemplateFamily: string[];
    sparseSignature: string[];
    denseEmbedding: number[];
    supportPrior: number;
    harmPrior: number;
    status: RouteActionPrototypeStatus;
    provenance: RouteActionPrototypeProvenance;
    sourceExampleIds: string[];
    createdAt: string;
    updatedAt: string;
}
export interface RoutePairExampleV3 {
    id: string;
    agentId: string;
    frameId: string;
    positiveActionId: string;
    negativeActionId: string;
    labelSource: RoutePairLabelSource;
    marginWeight: number;
    evidenceIds: string[];
    createdAt: string;
}
export interface RouteBanditFeedbackV3 {
    id: string;
    agentId: string;
    frameId: string;
    chosenActionId: string;
    reward: number;
    rewardComponents: Record<string, number>;
    cost: number;
    latencyMs: number;
    outcomeLabel: RouteBanditOutcomeLabel;
    learningBucket: boolean;
    createdAt: string;
}
export interface RouteBanditStateV3 {
    agentId: string;
    learnerVersion: string;
    featureSchemaVersion: string;
    explorationAlpha: number;
    sharedWeights: number[];
    actionStats: Record<string, {
        count: number;
        rewardSum: number;
        rewardMean: number;
        rewardVariance: number;
        lastReward: number;
        positiveCount: number;
        negativeCount: number;
        updatedAt: string;
    }>;
    updatedAt: string;
}
export interface RouteShadowDecisionV3 {
    id: string;
    agentId: string;
    routeDecisionId: string;
    snapshotId: string;
    snapshotStatus: 'candidate' | 'active' | 'rejected' | 'shadow';
    proposedRoute: RouteKind;
    proposedActionId?: string;
    proposedRuleId?: string;
    rawScore: number;
    calibratedScore: number;
    threshold: number;
    abstained: boolean;
    routingMode: string;
    reasonCode: string;
    matchedObservedRoute?: boolean;
    reward?: number;
    createdAt: string;
}
export interface RouteCalibrationExampleV3 {
    id: string;
    agentId: string;
    snapshotId: string;
    frameId: string;
    route: RouteKind;
    actionId?: string;
    ruleId?: string;
    routingMode: string;
    rawScore: number;
    calibratedScore: number;
    observedSuccess: boolean;
    comparable: boolean;
    split: RouteCalibrationExampleSplitV3;
    createdAt: string;
}
export interface RouteActionFamilyStatsV3 {
    familyKey: string;
    agentId: string;
    route: RouteKind;
    memoryTypes: MemoryType[];
    graphDepth: 0 | 1 | 2;
    syncPlanner: RouteActionSyncPlannerMode;
    supportCount: number;
    harmCount: number;
    meanReward: number;
    rewardVariance: number;
    pairWinRate: number;
    banditMeanReward: number;
    banditCount: number;
    shadowAgreementRate: number;
    updatedAt: string;
}
export interface RoutePolicyCandidateReportV3 {
    id: string;
    agentId: string;
    snapshotId: string;
    previousSnapshotId?: string;
    status: 'candidate' | 'active' | 'rejected' | 'shadow';
    bodyHash: string;
    ruleCount: number;
    compactnessBefore: number;
    compactnessAfter: number;
    duplicateGroups: number;
    mergedAway: number;
    dominatedPruned: number;
    estimatedImprovement: number;
    projectedSyncPlannerRate: number;
    noisyActionRate: number;
    harmRate: number;
    calibrationHoldoutFrames: number;
    shadowDecisionCount: number;
    retiredPrototypeIds: string[];
    activationReason: string;
    createdAt: string;
}
export interface RouteEvalCaseV3 {
    id: string;
    agentId: string;
    snapshotId: string;
    frameId: string;
    routingMode: string;
    observedRoute: RouteKind;
    expectedRoute: RouteKind;
    reward: number;
    quality: RouteEvalCaseQualityV3;
    humanReviewed: boolean;
    promotionSafe: boolean;
    notes?: string;
    split: RouteCalibrationExampleSplitV3;
    createdAt: string;
}
export interface RouteEvalCaseLabelV3 {
    id: string;
    agentId: string;
    caseId: string;
    source: RouteEvalCaseLabelSourceV3;
    preferredRoute: RouteKind;
    confidence: number;
    notes?: string;
    createdAt: string;
}
export interface RoutePolicyRuleV3 {
    id: string;
    priority?: number;
    actionId: string;
    family?: 'silence' | 'workflow' | 'correction' | 'project_context' | 'general_retrieval' | 'sync_enabling';
    match: {
        taskType?: TaskType | string;
        turnSignals?: string[];
        projectHint?: string;
        repoHintPresent?: boolean;
        safetySignalsAbsent?: string[];
    };
    route: RouteKind;
    memoryTypes: MemoryType[];
    queries: string[];
    graphDepth: 0 | 1 | 2;
    syncPlanner: RouteActionSyncPlannerMode;
    confidence: number;
    rawConfidence?: number;
    matchSpecificityScore?: number;
    dominanceGroupKey?: string;
    canonicalActionKey?: string;
    evidenceIds: string[];
    priors?: {
        support?: number;
        harm?: number;
        banditMeanReward?: number;
        banditCount?: number;
        pairWinRate?: number;
        teacherConfidenceMean?: number;
        validatorConfidenceMean?: number;
        ambiguityPenaltyMean?: number;
    };
    calibration?: {
        sampleCount?: number;
        successRate?: number;
        calibratedConfidence?: number;
        threshold?: number;
    };
    riskFlags?: string[];
    diagnosticNotes?: string[];
    reason?: string;
}
export interface RoutePolicySnapshotV3 {
    id: string;
    agentId: string;
    version: 'route-policy-v3';
    status: 'candidate' | 'active' | 'rejected' | 'shadow';
    rules: RoutePolicyRuleV3[];
    actionPriors: Record<string, {
        support: number;
        harm: number;
        banditMeanReward: number;
        banditCount: number;
        pairWinRate: number;
    }>;
    globalBudgets: {
        maxSyncPlannerRate: number;
        maxInjectedMemories: number;
        maxInjectedChars: number;
        defaultGraphDepth: 0 | 1 | 2;
        minCalibratedConfidence?: number;
        abstainMargin?: number;
    };
    evalSummary?: {
        frames: number;
        pairExamples: number;
        prototypes: number;
        projectedSyncPlannerRate: number;
        noisyActionRate: number;
        harmRate: number;
        activationDecision?: string;
        activationStatusReason?: string;
        validationErrors?: string[];
        validationWarnings?: string[];
        activationSummary?: {
            mode?: string;
            status?: string;
            reason?: string;
        };
        rollbackRecommendation?: {
            shouldRollback: boolean;
            reason: string;
            shadowDisagreementRate: number;
            shadowSampleCount: number;
        };
        thresholds?: {
            global?: number;
            byRoute?: Partial<Record<RouteKind, number>>;
            byFamily?: Record<string, number>;
        };
        compactness?: {
            beforeMerge: number;
            afterMerge: number;
            afterPrune: number;
            duplicateGroups: number;
            mergedAway: number;
            dominatedPruned: number;
            avgSignalsPerRule: number;
            avgQueriesPerRule: number;
            maxRulesPerRoute: number;
        };
        replay?: {
            frames: number;
            comparableFrames: number;
            matchedFrames: number;
            abstainRate: number;
            routeAgreement: number;
            rewardWeightedAgreement: number;
            projectedValue: number;
            baselineProjectedValue: number;
            estimatedImprovement: number;
            modeBreakdown?: Record<string, {
                frames: number;
                matchedFrames: number;
                abstained: number;
                projectedValue: number;
            }>;
            calibration?: {
                holdoutFrames: number;
                comparableFrames: number;
                globalThreshold: number;
                abstainMargin: number;
            };
        };
    };
    calibration?: {
        method: 'histogram_binning_v1';
        holdoutFrames: number;
        comparableFrames: number;
        globalThreshold: number;
        abstainMargin: number;
        globalBuckets: Array<{
            minScore: number;
            maxScore: number;
            successRate: number;
            count: number;
        }>;
        routeThresholds: Partial<Record<RouteKind, number>>;
        routeBuckets: Partial<Record<RouteKind, Array<{
            minScore: number;
            maxScore: number;
            successRate: number;
            count: number;
        }>>>;
    };
    lineage?: {
        previousSnapshotId?: string;
        comparedAgainstSnapshotId?: string;
        retiredPrototypeIds?: string[];
    };
    sourceFrameIds: string[];
    sourcePrototypeIds: string[];
    model?: string;
    promptVersion?: string;
    createdAt: string;
}
export type DistillationPhase = 'immediate_feedback' | 'agent_end_feedback' | 'route_turn_frame' | 'context_selection' | 'memory_planner' | 'route_learning' | 'outcome_classification' | 'route_teacher';
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
    validationStatus: 'valid' | 'invalid' | 'repaired' | 'fallback';
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
