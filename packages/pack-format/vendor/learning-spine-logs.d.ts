import { type ActivationPointerSlot, type BrainServeHotPathTimingV1, type FeedbackEventKind, type InteractionEventKind, type RouteMode, type RouterCollectedLabelCountsV1, type RuntimeCompileStructuralSignalsV1, type ServedArtifactProofV1, type RouterSupervisionKind, type RoutingChannelScoreSummaryV1, type RoutingChannelV1 } from "@openclawbrain/contracts";
export declare const LEARNING_SPINE_LOG_LAYOUT: {
    readonly dir: "logs/learning-spine";
    readonly serveTimeRouteDecisions: "logs/learning-spine/serve-time-route-decisions.jsonl";
    readonly supervisionLabelBindings: "logs/learning-spine/supervision-label-bindings.jsonl";
    readonly pgRouteUpdates: "logs/learning-spine/pg-route-updates.jsonl";
    readonly promotionActivationEvents: "logs/learning-spine/promotion-activation-events.jsonl";
    readonly trajectories: "logs/learning-spine/trajectories.jsonl";
};
declare const LEARNING_SPINE_LOG_STREAM_PATHS: {
    readonly serveTimeRouteDecisions: "logs/learning-spine/serve-time-route-decisions.jsonl";
    readonly supervisionLabelBindings: "logs/learning-spine/supervision-label-bindings.jsonl";
    readonly pgRouteUpdates: "logs/learning-spine/pg-route-updates.jsonl";
    readonly promotionActivationEvents: "logs/learning-spine/promotion-activation-events.jsonl";
    readonly trajectories: "logs/learning-spine/trajectories.jsonl";
};
export type LearningSpineLogStream = keyof typeof LEARNING_SPINE_LOG_STREAM_PATHS;
export interface LearningSpineActivationPackSnapshotV1 {
    slot: ActivationPointerSlot;
    packId: string;
    routePolicy: string;
    routerIdentity: string | null;
    routerChecksum: string | null;
    graphChecksum: string | null;
    eventExportDigest: string | null;
    builtAt: string;
    updatedAt: string;
}
export interface LearningSpineActivationPointersSnapshotV1 {
    active: LearningSpineActivationPackSnapshotV1 | null;
    candidate: LearningSpineActivationPackSnapshotV1 | null;
    previous: LearningSpineActivationPackSnapshotV1 | null;
}
export interface LearningSpinePromotionLinkV1 {
    matchedPromotionEventId: string | null;
    servedAfterPromotion: boolean;
    matchedPackId: string | null;
    matchedRouterChecksum: string | null;
}
export interface LearningSpineServeRouteCandidateScoreV1 {
    blockId: string;
    source: string;
    selected: boolean;
    compactedFrom: string[];
    matchedTokens: string[];
    routingChannels: RoutingChannelV1[];
    channelScores: RoutingChannelScoreSummaryV1;
    routeFnScore: number;
    actionScore: number;
    actionProbability: number;
    actionLogProbability: number | null;
    traversalScore: number;
    priority: number;
}
export interface LearningSpineServeRouteBreadcrumbsV1 {
    entrypoint: "compileRuntimeContext" | "runRuntimeTurn";
    invocationSurface: "direct_compile_call" | "installed_extension_before_prompt_build" | "runtime_turn_helper";
    hostEvent: "before_prompt_build" | null;
    installedEntryPath: string | null;
    syntheticTurn: boolean;
}
export interface LearningSpineServeRouteDecisionLogEntryV1 {
    recordType: "serve_time_route_decision";
    recordId: string;
    recordedAt: string;
    activationRoot: string;
    breadcrumbs: LearningSpineServeRouteBreadcrumbsV1;
    sessionId: string | null;
    channel: string | null;
    userMessage: string;
    turnSequenceStart: number | null;
    turnCompileEventId: string | null;
    turnCreatedAt: string | null;
    activePackId: string | null;
    activePackBuiltAt: string | null;
    activePackEventExportDigest: string | null;
    activePackRouterChecksum: string | null;
    activePackGraphChecksum: string | null;
    routerIdentity: string | null;
    usedLearnedRouteFn: boolean | null;
    servedArtifact: ServedArtifactProofV1 | null;
    selectionDigest: string | null;
    requestedBudget: {
        modeRequested: RouteMode | null;
        maxContextBlocks: number | null;
        maxContextChars: number | null;
    };
    actualBudget: {
        modeEffective: RouteMode | null;
        selectedCount: number;
        selectedCharCount: number;
        selectedTokenCount: number;
    };
    candidateSetIds: string[];
    chosenContextIds: string[];
    candidateScores: LearningSpineServeRouteCandidateScoreV1[];
    structuralSignals: RuntimeCompileStructuralSignalsV1 | null;
    fallbackReason: string | null;
    hotPathTiming: BrainServeHotPathTimingV1;
    kernelContextCount: number;
    brainContextCount: number;
    selectedKernelContextIds: string[];
    selectedBrainContextIds: string[];
    promotionLink: LearningSpinePromotionLinkV1 | null;
}
export interface LearningSpineSupervisionLabelBindingLogEntryV1 {
    recordType: "supervision_label_binding";
    recordId: string;
    recordedAt: string;
    activationRoot: string;
    updateId: string;
    packId: string;
    routerIdentity: string | null;
    eventExportDigest: string | null;
    sourceEventId: string;
    feedbackEventId: string | null;
    linkedTurnId: string | null;
    sourceContract: string;
    sourceKind: FeedbackEventKind | InteractionEventKind;
    supervisionKind: RouterSupervisionKind;
    rewardAssigned: number;
    boundBlockIds: string[];
    boundEdgeIds: string[];
    boundTargetIds: string[];
    dedupOrDiscardReason: string | null;
}
export interface LearningSpinePgRouteUpdateLogEntryV1 {
    recordType: "pg_route_update";
    recordId: string;
    updateId: string;
    recordedAt: string;
    activationRoot: string;
    previousPackId: string | null;
    nextPackId: string;
    previousRouterIdentity: string | null;
    nextRouterIdentity: string | null;
    previousRouterChecksum: string | null;
    nextRouterChecksum: string | null;
    previousUpdateCount: number | null;
    nextUpdateCount: number | null;
    updateCountIncrement: number | null;
    routeTraceCount: number;
    supervisionCount: number;
    labelSourceCounts: RouterCollectedLabelCountsV1;
    sourceCounts: {
        routeTrace: number;
        humanFeedback: number;
        operatorOverride: number;
        selfMemory: number;
        teacherSupervision: number;
    };
    updateOutcome: "updated" | "no_supervision" | "disabled";
    noOpReason: string | null;
    objectiveSummary: {
        updateMechanism: string | null;
        updateVersion: string | null;
        objective: string | null;
        objectiveChecksum: string | null;
        profile: unknown;
        materializedLoss: false;
        lossSummary: string;
    };
    resultingArtifacts: {
        packId: string;
        routerIdentity: string | null;
        manifestPath: string | null;
        routerPath: string | null;
        weightsChecksum: string | null;
        freshnessChecksum: string | null;
        eventExportDigest: string | null;
    };
}
export interface LearningSpineLaterServedTurnProofV1 {
    turnCompileEventId: string | null;
    sessionId: string | null;
    channel: string | null;
    servedAt: string;
    servedArtifact: ServedArtifactProofV1 | null;
    servedPackId: string | null;
    servedRouterChecksum: string | null;
    matchedPromotionEventId: string | null;
    switchedToPromotedPack: boolean;
}
export interface LearningSpinePromotionActivationLogEntryV1 {
    recordType: "promotion_activation";
    recordId: string;
    operation: "activate_pack" | "stage_candidate_pack" | "promote_candidate_pack" | "rollback_active_pack" | "served_after_promotion";
    occurredAt: string;
    activationRoot: string;
    reason: string;
    promotionReason: string | null;
    rollbackTarget: string | null;
    before: LearningSpineActivationPointersSnapshotV1;
    after: LearningSpineActivationPointersSnapshotV1;
    laterServedTurn: LearningSpineLaterServedTurnProofV1 | null;
}
export interface TrajectoryStepV1 {
    stepIndex: number;
    nodeBlockId: string;
    actionBlockId: string | null;
    actionScore: number;
    actionLogProbability: number;
    candidateNeighborIds: string[];
    candidateScores: Record<string, number>;
    candidateProbabilities: Record<string, number>;
}
export interface TrajectoryV1 {
    trajectoryId: string;
    steps: TrajectoryStepV1[];
    outcome: number;
    baselineValue: number;
    sessionId: string | null;
    turnId: string | null;
    createdAt: string;
}
export interface BaselineStateV1 {
    movingAverage: number;
    count: number;
    alpha: number;
    lastUpdatedAt: string;
}
export interface GraphLocalActionSet {
    nodeBlockId: string;
    neighborBlockIds: string[];
    includesStop: boolean;
    logits: Record<string, number>;
    probabilities: Record<string, number>;
}
export interface LearningSpineTrajectoryLogEntryV1 {
    recordType: "trajectory";
    recordId: string;
    recordedAt: string;
    activationRoot: string;
    sourceDecisionRecordId: string;
    trajectory: TrajectoryV1;
}
export declare function resolveLearningSpineLogPath(rootDir: string, stream: LearningSpineLogStream): string;
export declare function buildLearningSpineLogId(prefix: string, payload: unknown): string;
export declare function appendLearningSpineLogEntry(rootDir: string, stream: LearningSpineLogStream, entry: unknown): string;
export declare function readLearningSpineLogEntries<T>(rootDir: string, stream: LearningSpineLogStream): T[];
export {};
