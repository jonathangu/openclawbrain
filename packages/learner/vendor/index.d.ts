import { type ArtifactManifestV1, type FeedbackEventV1, type InteractionEventV1, type NormalizedEventExportV1, type NormalizedEventV1, type PrincipalPriorityClassV1, type PrincipalRoleV1, type PackGraphPayloadV1, type PackVectorsPayloadV1, type RouterArtifactV1, type RouterPolicyUpdateV1, type RuntimeCompileStructuralSignalsV1, type RuntimeGraphPlasticityStateV1, type SparseFeedbackPolicyV1, type TeacherSupervisionArtifactV1 } from "@openclawbrain/contracts";
import { type EventExportCursorV1, type EventExportLaneV1, type NormalizedEventExportBridgeV1, type NormalizedEventExportSliceV1 } from "@openclawbrain/event-export";
import type { TextEmbedder } from "@openclawbrain/compiler";
import { type PackDescriptor, type GraphEvolutionLogV1, type LearningSpineServeRouteDecisionLogEntryV1 } from "@openclawbrain/pack-format";
import { type WorkspaceMetadataInput } from "@openclawbrain/workspace-metadata";
export interface CandidatePackEventExports {
    interactionEvents: InteractionEventV1[];
    feedbackEvents: FeedbackEventV1[];
}
export interface CandidatePackBuildInput {
    packLabel: string;
    workspace: WorkspaceMetadataInput;
    eventRange: {
        start: number;
        end: number;
    };
    eventExports?: CandidatePackEventExports;
    teacherSupervisionArtifacts?: readonly TeacherSupervisionArtifactV1[];
    learnedRouting: boolean;
    builtAt?: string;
    offlineArtifacts?: string[];
    structuralOps?: Partial<ArtifactManifestV1["graphDynamics"]["structuralOps"]>;
    runtimeGraph?: PackGraphPayloadV1;
    sparseFeedback?: Partial<SparseFeedbackPolicyV1>;
    principalBacklog?: PrincipalBacklogSummaryV1;
    /** PG algorithm version: "v1" (default flat softmax) or "v2" (paper-aligned graph-local). */
    pgVersion?: "v1" | "v2";
    /** Serve-time route decision logs for V2 trajectory reconstruction. */
    serveTimeDecisions?: LearningSpineServeRouteDecisionLogEntryV1[];
    /** Baseline state for V2 variance reduction. */
    baselineState?: BaselineStateV1;
}
export interface CandidatePackFromNormalizedEventExportInput {
    packLabel: string;
    workspace: WorkspaceMetadataInput;
    normalizedEventExport: NormalizedEventExportV1;
    teacherSupervisionArtifacts?: readonly TeacherSupervisionArtifactV1[];
    learnedRouting: boolean;
    builtAt?: string;
    offlineArtifacts?: string[];
    structuralOps?: Partial<ArtifactManifestV1["graphDynamics"]["structuralOps"]>;
    runtimeGraph?: PackGraphPayloadV1;
    sparseFeedback?: Partial<SparseFeedbackPolicyV1>;
    principalBacklog?: PrincipalBacklogSummaryV1;
    pgVersion?: "v1" | "v2";
    serveTimeDecisions?: LearningSpineServeRouteDecisionLogEntryV1[];
    baselineState?: BaselineStateV1;
}
export interface BuildTeacherSupervisionArtifactsInput {
    normalizedEventExport: NormalizedEventExportV1;
    observedAt?: string;
    staleAfterMs?: number;
    sparseFeedback?: Partial<SparseFeedbackPolicyV1>;
}
interface CandidatePackBridgeInputBase {
    packLabel: string;
    workspace: WorkspaceMetadataInput;
    teacherSupervisionArtifacts?: readonly TeacherSupervisionArtifactV1[];
    learnedRouting: boolean;
    builtAt?: string;
    offlineArtifacts?: string[];
    structuralOps?: Partial<ArtifactManifestV1["graphDynamics"]["structuralOps"]>;
    runtimeGraph?: PackGraphPayloadV1;
    sparseFeedback?: Partial<SparseFeedbackPolicyV1>;
    principalBacklog?: PrincipalBacklogSummaryV1;
    pgVersion?: "v1" | "v2";
    serveTimeDecisions?: LearningSpineServeRouteDecisionLogEntryV1[];
    baselineState?: BaselineStateV1;
}
export interface CandidatePackFromNormalizedEventExportSliceInput extends CandidatePackBridgeInputBase {
    normalizedEventExportSlice: NormalizedEventExportSliceV1;
}
export interface CandidatePackBundleFromNormalizedEventExportBridgeInput extends CandidatePackBridgeInputBase {
    normalizedEventExportBridge: NormalizedEventExportBridgeV1;
}
export interface CandidatePackPayloads {
    graph: PackGraphPayloadV1;
    vectors: PackVectorsPayloadV1;
    router: RouterArtifactV1 | null;
}
export interface CandidatePackBuildResult {
    manifest: ArtifactManifestV1;
    payloads: CandidatePackPayloads;
    routingBuild: {
        learnedRoutingPath: "disabled" | "policy_gradient_v1" | "policy_gradient_v2";
        pgVersionRequested: "v1" | "v2" | null;
        pgVersionUsed: "v1" | "v2" | null;
        decisionLogCount: number;
        fallbackReason: string | null;
        updatedBaseline: BaselineStateV1 | null;
    };
    summary: {
        packId: string;
        immutable: true;
        routePolicy: ArtifactManifestV1["routePolicy"];
        workspaceSnapshot: string;
        eventRange: ArtifactManifestV1["provenance"]["eventRange"];
        eventExportDigest: string | null;
        learningSurface: ArtifactManifestV1["provenance"]["learningSurface"];
        bootstrapping: ArtifactManifestV1["graphDynamics"]["bootstrapping"];
        workspaceInit: PointerAwareWorkingSetSummaryV1;
        runtimePlasticity: RuntimeGraphPlasticityStateV1;
        graphEvolutionLog: GraphEvolutionLogV1;
        learnedRouter: {
            routerIdentity: string | null;
            trainingMethod: RouterArtifactV1["training"]["method"] | null;
            refreshStatus: RouterArtifactV1["training"]["status"] | null;
            updateCount: number;
            supervisionCount: number;
            weightsChecksum: string | null;
            visibleDelta: string[];
            noOpReason: string | null;
        };
    };
}
export type PointerGraphHintV1 = "memory_index" | "markdown_link" | "bare_path";
export type PointerAwareWorkingSetLayerV1 = "anchor" | "working_set" | "passive_expansion";
export interface PointerGraphEdgeRecordV1 {
    sourcePath: string;
    targetPath: string;
    depth: number;
    hint: PointerGraphHintV1;
}
export interface PointerAwareWorkingSetFileRecordV1 {
    path: string;
    layer: PointerAwareWorkingSetLayerV1;
    priority: number;
    excerpt: string;
    inboundCount: number;
    outboundCount: number;
}
export interface PointerAwareWorkingSetInput {
    rootDir: string;
    observedAt?: string;
    workingSetLimit?: number;
    passiveExpansionLimit?: number;
}
export interface PointerAwareWorkingSetResultV1 {
    rootDir: string;
    observedAt: string;
    memoryPath: string | null;
    activeTasksPath: string | null;
    todaysMemoryPath: string | null;
    anchorPaths: string[];
    bootInputs: string[];
    workingSet: string[];
    passiveExpansion: string[];
    files: PointerAwareWorkingSetFileRecordV1[];
    pointers: PointerGraphEdgeRecordV1[];
    graphDigest: string | null;
}
export interface PointerAwareWorkingSetSummaryV1 {
    pointerAware: boolean;
    memoryPath: string | null;
    bootInputs: string[];
    workingSet: string[];
    passiveExpansion: string[];
    graphDigest: string | null;
}
export interface CandidatePackBundleEntry {
    lane: EventExportLaneV1;
    sliceId: string;
    packLabel: string;
    normalizedEventExport: NormalizedEventExportV1;
    nextCursor: EventExportCursorV1;
    watermark: NormalizedEventExportSliceV1["watermark"];
    build: CandidatePackBuildResult;
}
export interface CandidatePackBundleBuildResult {
    runtimeOwner: "openclaw";
    bridgeDigest: string;
    bundleDigest: string;
    cursor: EventExportCursorV1;
    dedupedInputCount: number;
    duplicateIdentityCount: number;
    entries: CandidatePackBundleEntry[];
}
export interface MaterializedCandidatePackBundleEntry extends CandidatePackBundleEntry {
    rootDir: string;
    descriptor: PackDescriptor;
}
export interface CandidatePackBundleMaterializationResult {
    runtimeOwner: "openclaw";
    bridgeDigest: string;
    bundleDigest: string;
    cursor: EventExportCursorV1;
    dedupedInputCount: number;
    duplicateIdentityCount: number;
    entries: MaterializedCandidatePackBundleEntry[];
}
export declare const DEFAULT_ALWAYS_ON_LEARNING_LIVE_SLICES_PER_CYCLE = 1;
export declare const DEFAULT_ALWAYS_ON_LEARNING_BACKFILL_SLICES_PER_CYCLE = 1;
export declare const DEFAULT_TEACHER_SUPERVISION_STALE_AFTER_MS: number;
export declare const DEFAULT_POINTER_AWARE_WORKING_SET_LIMIT = 6;
export declare const DEFAULT_POINTER_AWARE_PASSIVE_EXPANSION_LIMIT = 12;
export declare const DEFAULT_ALWAYS_ON_STRUCTURAL_PLASTICITY_OPS: Required<ArtifactManifestV1["graphDynamics"]["structuralOps"]>;
export declare const ALWAYS_ON_STRUCTURAL_PLASTICITY_OP_CEILING: Required<ArtifactManifestV1["graphDynamics"]["structuralOps"]>;
export declare const ALWAYS_ON_STRUCTURAL_PLASTICITY_MIN_INTERACTIONS = 2;
export declare const ALWAYS_ON_STRUCTURAL_PLASTICITY_MIN_FEEDBACK = 1;
export declare const DEFAULT_SPARSE_FEEDBACK_POLICY: SparseFeedbackPolicyV1;
export interface SparseFeedbackRuntimeDiagnosticsV1 extends SparseFeedbackPolicyV1 {
    eligibleFeedbackCount: number;
    maskedFeedbackCount: number;
    delayedFeedbackCount: number;
    budgetedOutFeedbackCount: number;
    amplifiedBackgroundLabelCount: number;
    retainedFeedbackCount?: number;
    selectionCursor?: number;
    processedFeedbackEventIds?: string[];
}
export type SparseFeedbackEventDispositionReasonV1 = "masked" | "delayed" | "budgeted_out";
export interface SparseFeedbackEventDispositionV1 {
    eventId: string;
    selected: boolean;
    reason: SparseFeedbackEventDispositionReasonV1 | null;
}
export interface AlwaysOnLearningCadenceV1 {
    liveSlicesPerCycle: number;
    backfillSlicesPerCycle: number;
}
export interface AlwaysOnLearningPendingSlicesV1 {
    live: NormalizedEventExportSliceV1[];
    backfill: NormalizedEventExportSliceV1[];
}
export interface AlwaysOnLearningRuntimeStateV1 {
    runtimeOwner: "openclaw";
    hotPathLearning: false;
    attachBlocksOnFullReplay: false;
    cursor: EventExportCursorV1;
    pending: AlwaysOnLearningPendingSlicesV1;
    learnedEventExport: NormalizedEventExportV1 | null;
    runtimeGraph: PackGraphPayloadV1 | null;
    runtimePlasticity: RuntimeGraphPlasticityStateV1 | null;
    learnedGraph: PackGraphPayloadV1 | null;
    structuralController: AlwaysOnLearningStructuralControllerStateV1;
    sparseFeedback: SparseFeedbackRuntimeDiagnosticsV1;
    lastMaterializedAt: string | null;
    materializationCount: number;
}
export type AlwaysOnLearningStructuralControlStrategyV1 = "fixed_v1" | "empirical_v1";
export type AlwaysOnLearningStructuralControlSourceV1 = "caller_override" | "fixed_default" | "no_compile_signal_evidence_fallback" | "compile_structural_signals_empirical_v1";
export interface AlwaysOnLearningCompileStructuralSignalsV1 extends Pick<RuntimeCompileStructuralSignalsV1, "matchedCandidateCount" | "selectedMatchedCount" | "overlapPrunedCount" | "traversalActivatedCount"> {
}
export interface AlwaysOnLearningStructuralControllerStateV1 {
    requestedStrategy: AlwaysOnLearningStructuralControlStrategyV1;
    effectiveStrategy: AlwaysOnLearningStructuralControlStrategyV1;
    source: AlwaysOnLearningStructuralControlSourceV1;
    compileSignals: AlwaysOnLearningCompileStructuralSignalsV1 | null;
    structuralOps: Required<ArtifactManifestV1["graphDynamics"]["structuralOps"]>;
}
export type AlwaysOnLearningSchedulerBucketV1 = "principal_immediate" | "principal_backfill" | "live" | "backfill";
export interface PrincipalLearningCheckpointV1 {
    teacherIdentity: string;
    teacherRole: PrincipalRoleV1 | null;
    priorityClass: PrincipalPriorityClassV1 | null;
    learnedThroughSequence: number | null;
    learnedThroughCreatedAt: string | null;
    pendingEventCount: number;
    pendingLiveEventCount: number;
    pendingBackfillEventCount: number;
    oldestPendingSequence: number | null;
    oldestPendingCreatedAt: string | null;
    newestPendingSequence: number | null;
    newestPendingCreatedAt: string | null;
}
export interface PendingPrincipalEventV1 {
    teacherIdentity: string;
    teacherRole: PrincipalRoleV1 | null;
    priorityClass: PrincipalPriorityClassV1 | null;
    eventId: string;
    kind: NormalizedEventV1["kind"];
    sequence: number;
    createdAt: string;
    lane: EventExportLaneV1;
    sourceStream: string;
}
export interface PrincipalBacklogSummaryV1 {
    principalCount: number;
    pendingEventCount: number;
    checkpoints: PrincipalLearningCheckpointV1[];
    oldestUnlearnedEvent: PendingPrincipalEventV1 | null;
    newestPendingEvent: PendingPrincipalEventV1 | null;
}
export interface AlwaysOnLearningRuntimePlanV1 {
    runtimeOwner: "openclaw";
    hotPathLearning: false;
    attachBlocksOnFullReplay: false;
    bootstrapped: boolean;
    mode: "cold_start" | "live_priority" | "background_catchup" | "caught_up";
    nextPriorityLane: EventExportLaneV1 | "none";
    nextPriorityBucket: AlwaysOnLearningSchedulerBucketV1 | "none";
    pending: {
        live: number;
        backfill: number;
        total: number;
        freshLivePriority: boolean;
        byBucket: Record<AlwaysOnLearningSchedulerBucketV1, number>;
    };
    principalBacklog: PrincipalBacklogSummaryV1;
    learnedRange: NormalizedEventExportV1["range"] | null;
    materialization: {
        count: number;
        lastMaterializedAt: string | null;
        lastJobId: string | null;
        lastReason: AlwaysOnLearningMaterializationJobV1["reason"] | null;
        lastLane: AlwaysOnLearningMaterializationJobV1["lane"] | null;
        lastPriority: AlwaysOnLearningMaterializationJobV1["priority"] | null;
        lastSchedulerBucket: AlwaysOnLearningMaterializationJobV1["schedulerBucket"] | null;
    };
}
export interface AdvanceAlwaysOnLearningRuntimeInput {
    packLabel: string;
    workspace: WorkspaceMetadataInput;
    interactionEvents: readonly InteractionEventV1[];
    feedbackEvents: readonly FeedbackEventV1[];
    teacherSupervisionArtifacts?: readonly TeacherSupervisionArtifactV1[];
    learnedRouting: boolean;
    state?: AlwaysOnLearningRuntimeStateV1;
    builtAt?: string;
    offlineArtifacts?: string[];
    structuralOps?: Partial<ArtifactManifestV1["graphDynamics"]["structuralOps"]>;
    structuralControlStrategy?: AlwaysOnLearningStructuralControlStrategyV1;
    compileStructuralSignals?: AlwaysOnLearningCompileStructuralSignalsV1 | null;
    sparseFeedback?: Partial<SparseFeedbackPolicyV1>;
    liveSliceSize?: number;
    backfillSliceSize?: number;
    cadence?: Partial<AlwaysOnLearningCadenceV1>;
    pgVersion?: "v1" | "v2";
    serveTimeDecisions?: LearningSpineServeRouteDecisionLogEntryV1[];
    baselineState?: BaselineStateV1;
}
export interface AlwaysOnLearningMaterializationJobV1 {
    jobId: string;
    lane: EventExportLaneV1;
    priority: "immediate" | "background";
    schedulerBucket: AlwaysOnLearningSchedulerBucketV1;
    reason: "attach_bootstrap" | "fresh_live_events" | "passive_history_catchup";
    selectedSliceIds: string[];
    selectedEventRange: NormalizedEventExportV1["range"];
    normalizedEventExport: NormalizedEventExportV1;
    candidateInput: CandidatePackFromNormalizedEventExportInput;
    candidate: CandidatePackBuildResult;
}
export interface AdvanceAlwaysOnLearningRuntimeResultV1 {
    runtimeOwner: "openclaw";
    hotPathLearning: false;
    attachBlocksOnFullReplay: false;
    bridge: NormalizedEventExportBridgeV1;
    selectedSlices: NormalizedEventExportSliceV1[];
    deferred: {
        live: number;
        backfill: number;
    };
    materialization: AlwaysOnLearningMaterializationJobV1 | null;
    state: AlwaysOnLearningRuntimeStateV1;
}
export interface DrainAlwaysOnLearningRuntimeInput extends AdvanceAlwaysOnLearningRuntimeInput {
    maxCycles?: number;
}
export interface AlwaysOnLearningRuntimeCycleV1 extends AdvanceAlwaysOnLearningRuntimeResultV1 {
    cycle: number;
}
export interface DrainAlwaysOnLearningRuntimeResultV1 {
    runtimeOwner: "openclaw";
    drained: boolean;
    stopReason: "idle" | "max_cycles" | "no_progress";
    cycles: AlwaysOnLearningRuntimeCycleV1[];
    materializations: AlwaysOnLearningMaterializationJobV1[];
    state: AlwaysOnLearningRuntimeStateV1;
}
export declare function buildTeacherSupervisionArtifactsFromNormalizedEventExport(input: BuildTeacherSupervisionArtifactsInput): TeacherSupervisionArtifactV1[];
export declare function createAlwaysOnLearningRuntimeState(): AlwaysOnLearningRuntimeStateV1;
export declare function describeAlwaysOnLearningRuntimeState(state: AlwaysOnLearningRuntimeStateV1, lastMaterialization?: AlwaysOnLearningMaterializationJobV1 | null): AlwaysOnLearningRuntimePlanV1;
export declare function advanceAlwaysOnLearningRuntime(input: AdvanceAlwaysOnLearningRuntimeInput): AdvanceAlwaysOnLearningRuntimeResultV1;
export declare function drainAlwaysOnLearningRuntime(input: DrainAlwaysOnLearningRuntimeInput): DrainAlwaysOnLearningRuntimeResultV1;
export declare function materializeAlwaysOnLearningCandidatePack(rootDir: string, job: AlwaysOnLearningMaterializationJobV1): PackDescriptor;
export declare function materializeAlwaysOnLearningCandidatePackWithEmbedder(rootDir: string, job: AlwaysOnLearningMaterializationJobV1, embedder: TextEmbedder): Promise<PackDescriptor>;
export declare function buildCandidatePackFromNormalizedEventExportSlice(input: CandidatePackFromNormalizedEventExportSliceInput): CandidatePackBuildResult;
export declare function buildCandidatePackBundleFromNormalizedEventExportBridge(input: CandidatePackBundleFromNormalizedEventExportBridgeInput): CandidatePackBundleBuildResult;
export declare function buildPointerAwareWorkingSet(input: PointerAwareWorkingSetInput): PointerAwareWorkingSetResultV1;
export declare function describeSparseFeedbackEventDispositions(feedbackEvents: readonly FeedbackEventV1[], observedAt: string, sparseFeedback: Partial<SparseFeedbackPolicyV1> | undefined): SparseFeedbackEventDispositionV1[];
export declare function reindexCandidatePackBuildResultWithEmbedder(result: CandidatePackBuildResult, embedder: TextEmbedder): Promise<CandidatePackBuildResult>;
/**
 * Build an adjacency map from the graph's block edges.
 *
 * For each block, collects the targetBlockIds from its edges, filtering out:
 * - Self-loops (edges pointing back to the same block)
 * - Edges targeting blocks that don't exist in the graph
 *
 * Blocks with no (valid) outgoing edges get an empty neighbor array.
 */
export declare function buildAdjacencyMap(graph: PackGraphPayloadV1): Map<string, string[]>;
export interface GraphLocalActionSet {
    nodeBlockId: string;
    neighborBlockIds: string[];
    includesStop: boolean;
    logits: Map<string, number>;
    probabilities: Map<string, number>;
}
export declare function buildGraphLocalActionSet(nodeBlockId: string, neighborBlockIds: string[], graph: PackGraphPayloadV1, vectors: PackVectorsPayloadV1, queryContext: {
    queryTokens: string[];
    queryVector: Record<string, number>;
}, tau: number, stopBias?: number): GraphLocalActionSet;
export declare const STOP_ACTION_ID = "__STOP__";
export declare function createDefaultBaselineState(alpha?: number): BaselineStateV1;
/**
 * EMA update: on first observation use the raw outcome; thereafter blend.
 */
export declare function updateBaseline(current: BaselineStateV1, outcome: number): BaselineStateV1;
/**
 * Corrected tail-sum policy gradient update for a single trajectory.
 *
 * From Gu (2016):
 *   ∂v(s_t)/∂W = E[ z_T · Σ_{l=t}^{T-1} ∇_W log π(a_l | s_l) ]
 *
 * At each node i, ∇_{w_i} log π(a|i) = (1/τ)(e_a − π_i),
 * so Σ_j gradient_j = 0 (mass redistribution, not inflation).
 */
export declare function computeTrajectoryPolicyGradient(trajectory: TrajectoryV1, adjacency: Map<string, string[]>, graph: PackGraphPayloadV1, vectors: PackVectorsPayloadV1, tau: number, pgScale: number): Map<string, {
    delta: number;
    evidenceCount: number;
    rewardSum: number;
}>;
/**
 * Aggregate policy gradient updates from multiple trajectories into
 * RouterPolicyUpdateV1[] format.
 */
export declare function aggregateTrajectoryUpdates(trajectories: TrajectoryV1[], adjacency: Map<string, string[]>, graph: PackGraphPayloadV1, vectors: PackVectorsPayloadV1, tau: number, pgScale: number): RouterPolicyUpdateV1[];
/**
 * Baseline state for variance reduction (exponential moving average of returns).
 */
export interface BaselineStateV1 {
    movingAverage: number;
    count: number;
    alpha: number;
    lastUpdatedAt: string;
}
/**
 * A single step in a trajectory through the graph.
 */
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
/**
 * A full trajectory: a sequence of (state, action) pairs with an outcome.
 */
export interface TrajectoryV1 {
    trajectoryId: string;
    sessionId: string | null;
    turnId: string | null;
    createdAt: string;
    steps: TrajectoryStepV1[];
    outcome: number;
    baselineValue: number;
}
/**
 * Join serve-time decisions with feedback events to assign outcome rewards.
 * Returns decisionRecordId → outcome (z_T).
 */
export declare function joinDecisionsWithFeedback(decisions: LearningSpineServeRouteDecisionLogEntryV1[], eventExport: NormalizedEventExportV1 | null, maxDelayMs?: number): Map<string, number>;
/**
 * Reconstruct a trajectory from a serve-time decision log entry.
 * Traces through the graph starting from the highest-scoring selected block.
 */
export declare function reconstructTrajectoryFromServeDecision(decision: LearningSpineServeRouteDecisionLogEntryV1, graph: PackGraphPayloadV1, vectors: PackVectorsPayloadV1, adjacency: Map<string, string[]>, tau: number, outcome: number, baselineValue: number): TrajectoryV1;
export declare function buildCandidatePack(input: CandidatePackBuildInput): CandidatePackBuildResult;
export declare function buildCandidatePackFromNormalizedEventExport(input: CandidatePackFromNormalizedEventExportInput): CandidatePackBuildResult;
export declare function buildCandidatePackWithEmbedder(input: CandidatePackBuildInput, embedder: TextEmbedder): Promise<CandidatePackBuildResult>;
export declare function buildCandidatePackFromNormalizedEventExportWithEmbedder(input: CandidatePackFromNormalizedEventExportInput, embedder: TextEmbedder): Promise<CandidatePackBuildResult>;
export declare function materializeCandidatePack(rootDir: string, input: CandidatePackBuildInput): PackDescriptor;
export declare function materializeCandidatePackWithEmbedder(rootDir: string, input: CandidatePackBuildInput, embedder: TextEmbedder): Promise<PackDescriptor>;
export declare function materializeCandidatePackFromNormalizedEventExport(rootDir: string, input: CandidatePackFromNormalizedEventExportInput): PackDescriptor;
export declare function materializeCandidatePackFromNormalizedEventExportWithEmbedder(rootDir: string, input: CandidatePackFromNormalizedEventExportInput, embedder: TextEmbedder): Promise<PackDescriptor>;
export declare function materializeCandidatePackFromNormalizedEventExportSlice(rootDir: string, input: CandidatePackFromNormalizedEventExportSliceInput): PackDescriptor;
export declare function materializeCandidatePackBundleFromNormalizedEventExportBridge(rootDir: string, input: CandidatePackBundleFromNormalizedEventExportBridgeInput): CandidatePackBundleMaterializationResult;
/**
 * Load baseline state from `<activationRoot>/baseline-state.json`.
 * If the file is missing or unparseable, returns a fresh zero-initialised state.
 */
export declare function loadOrInitBaseline(activationRoot: string): BaselineStateV1;
/**
 * Persist baseline state to `<activationRoot>/baseline-state.json`.
 * Creates the directory tree if it doesn't already exist.
 */
export declare function persistBaseline(activationRoot: string, state: BaselineStateV1): void;
export {};
