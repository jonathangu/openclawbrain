import { type CompileSelectionMode, type LearnedRouteSelectionOverride } from "@openclawbrain/compiler";
import { CONTRACT_IDS, type ArtifactManifestV1, type ActivationPointerRecordV1, type ActivationPointerSlot, type RuntimeTurnBrainAttachmentPolicyV1, type ContextCompactionMode, type ContextContributionEvidenceStateV1, type CurrentProfileBrainStatusAnswerV1, type CurrentProfileActivationStateV1, type CurrentProfilePassiveLearningDeltaSummaryV1, type CurrentProfilePassiveLearningWatchStateV1, type CurrentProfileAttachmentStateV1, type CurrentProfileAttachmentProofStateV1, type CurrentProfileHookInstallStateV1, type CurrentProfileHookLoadabilityV1, type CurrentProfileHookLoadProofV1, type BrainServeHotPathTimingV1, type CurrentProfileStructuralDecisionV1, type FeedbackEventKind, type PackGraphConnectDiagnosticsV1, type PackGraphEvolutionV1, type EventSemanticSurfaceV1, type FeedbackEventV1, type KernelSurfaceValidationResultV1, type LearningBootProfile, type LearningCadence, type LearningScanPolicy, type InteractionEventV1, type NormalizedEventExportV1, type NormalizedEventV1, type PrincipalPriorityClassV1, type PrincipalRoleV1, type RouteMode, type RuntimeCompileResponseV1, type RuntimeCompileStructuralSignalsV1, type RuntimeCompileTargetV1, type RuntimeGraphPlasticityStateV1, type RuntimePlasticitySourceV1, type SparseFeedbackPolicyV1, type TeacherAuthorityV1, type TeacherSupervisionArtifactV1, type WorkspaceInjectionSurfaceV1 } from "@openclawbrain/contracts";
import { type EventExportLaneV1 } from "@openclawbrain/event-export";
import { type AdvanceAlwaysOnLearningRuntimeInput, type AlwaysOnLearningCadenceV1, type AlwaysOnLearningMaterializationJobV1, type AlwaysOnLearningRuntimePlanV1, type AlwaysOnLearningRuntimeStateV1, type BaselineStateV1, type PendingPrincipalEventV1, type PrincipalLearningCheckpointV1, type LearningSpineServeRouteDecisionLogEntryV1 } from "./local-learner.js";
import { type ActivationInspection, type ActivationObservabilityReport, type GraphEvolutionLogV1, type LearningSpineServeRouteBreadcrumbsV1, type ActivationSlotInspection, type InitHandoffState } from "@openclawbrain/pack-format";
export { clearOpenClawProfileRuntimeLoadProof, listOpenClawProfileRuntimeLoadProofs, recordOpenClawProfileRuntimeLoadProof, resolveAttachmentRuntimeLoadProofsPath, type OpenClawProfileRuntimeLoadProofRecordV1, type OpenClawProfileRuntimeLoadProofSetV1, type OpenClawProfileRuntimeLoadProofsV1 } from "./attachment-truth.js";
import { type AsyncTeacherLabelerConfigV1 } from "./teacher-labeler.js";
export { createHttpOllamaTeacherLabelerClient, createOllamaTeacherLabeler, createTeacherLabeler, summarizeTeacherLabelerOpportunity, type AsyncTeacherLabelerConfigV1, type AsyncTeacherNoopLabelerConfigV1, type AsyncTeacherOllamaLabelerConfigV1, type OllamaTeacherLabelerClient, type TeacherLabeler, type TeacherLabelerOpportunityInputV1, type TeacherLabelerOpportunityV1, type TeacherLabelerResultV1, type TeacherLabelerRunInputV1 } from "./teacher-labeler.js";
export type OperatorPassiveLearningWatchState = CurrentProfilePassiveLearningWatchStateV1 | "lagging";
export interface OperatorPassiveLearningWatchSummary {
    state: OperatorPassiveLearningWatchState;
    detail: string;
    lastHeartbeatAt: string | null;
    lagSeconds: number | null;
    intervalSeconds: number | null;
    healthyWithinSeconds: number | null;
    staleAfterSeconds: number | null;
}
export declare const DEFAULT_ASYNC_TEACHER_QUEUE_CAPACITY = 8;
declare const RECORDED_SESSION_TRACE_CONTRACT: "recorded_session_trace.v1";
declare const RECORDED_SESSION_FIXTURE_CONTRACT: "recorded_session_replay_fixture.v1";
declare const RECORDED_SESSION_BUNDLE_CONTRACT: "recorded_session_replay_bundle.v1";
declare const RUNTIME_EVENT_EXPORT_BUNDLE_CONTRACT: "normalized_event_export_bundle.v1";
export declare const RUNTIME_EVENT_EXPORT_BUNDLE_LAYOUT: {
    readonly manifest: "manifest.json";
    readonly payload: "normalized-event-export.json";
};
export declare const RECORDED_SESSION_REPLAY_PROOF_BUNDLE_LAYOUT: {
    readonly manifest: "manifest.json";
    readonly trace: "trace.json";
    readonly fixture: "fixture.json";
    readonly bundle: "bundle.json";
    readonly environment: "environment.json";
    readonly summary: "summary.md";
    readonly summaryTables: "summary-tables.json";
    readonly coverageSnapshot: "coverage-snapshot.json";
    readonly hardeningSnapshot: "hardening-snapshot.json";
    readonly hashes: "hashes.json";
    readonly modeDir: "modes";
};
export interface CompileServeRouteBreadcrumbInput {
    invocationSurface: LearningSpineServeRouteBreadcrumbsV1["invocationSurface"];
    hostEvent: LearningSpineServeRouteBreadcrumbsV1["hostEvent"];
    installedEntryPath?: string | null;
}
export interface RuntimeEventExportBundleSummaryV1 {
    runtimeOwner: "openclaw";
    sessionId: string | null;
    channel: string | null;
    eventRange: Pick<NormalizedEventExportV1["range"], "start" | "end" | "count">;
    interactionCount: number;
    feedbackCount: number;
    sourceStreams: string[];
    contracts: NormalizedEventExportV1["provenance"]["contracts"];
    semanticSurface?: EventSemanticSurfaceV1;
}
export type ScannerExportStatusV1 = "complete" | "partial" | "failed";
export interface ScannerExportManifestV1 {
    scannerId: string;
    lane: string;
    status: ScannerExportStatusV1;
    producedAt: string;
    sourceManifestPath: string | null;
    sourceManifestDigest: string | null;
    warnings: string[];
    failures: string[];
}
export interface ScannedEventExportInputV1 {
    interactionEvents: readonly InteractionEventV1[];
    feedbackEvents: readonly FeedbackEventV1[];
    scanner: ScannerExportManifestV1;
}
export type ScannedEventExportNoopReasonV1 = "scan_failed" | "no_events" | "invalid_scanner_manifest";
export interface ScannedEventExportBuildSuccessV1 {
    ok: true;
    normalizedEventExport: NormalizedEventExportV1;
    scanner: ScannerExportManifestV1;
    warnings: string[];
}
export interface ScannedEventExportBuildFailureV1 {
    ok: false;
    normalizedEventExport: null;
    scanner: ScannerExportManifestV1;
    warnings: string[];
    reason: ScannedEventExportNoopReasonV1;
    error: string;
}
export type ScannedEventExportBuildResultV1 = ScannedEventExportBuildSuccessV1 | ScannedEventExportBuildFailureV1;
export interface RuntimeEventExportBundleManifestV1 {
    contract: typeof RUNTIME_EVENT_EXPORT_BUNDLE_CONTRACT;
    exportName: string;
    exportedAt: string;
    payloadPath: string;
    payloadDigest: string;
    summary: RuntimeEventExportBundleSummaryV1;
    scanner?: ScannerExportManifestV1 | null;
}
export interface RuntimeEventExportBundleDescriptor {
    rootDir: string;
    manifestPath: string;
    payloadPath: string;
    manifest: RuntimeEventExportBundleManifestV1;
    normalizedEventExport: NormalizedEventExportV1;
}
export type CompileRuntimeBudgetStrategy = "fixed_v1" | "empirical_v1";
export type RuntimeComparativeReplayMode = "vector_only" | "graph_prior_only" | "learned_route";
export interface FrozenReplayEvalIdentityV1 {
    packId: string;
    routerIdentity?: string | null;
}
export interface CompileRuntimeContextInput {
    activationRoot: string;
    message: string;
    agentId?: string;
    maxContextBlocks?: number;
    budgetStrategy?: CompileRuntimeBudgetStrategy;
    maxContextChars?: number;
    mode?: RouteMode | RuntimeComparativeReplayMode;
    selectionMode?: CompileSelectionMode;
    compactionMode?: ContextCompactionMode;
    runtimeHints?: readonly string[];
    /** Optional session ID for serve-time route decision logging (extension callers). */
    sessionId?: string;
    /** Optional channel identifier for serve-time route decision logging (extension callers). */
    channel?: string;
    /** @internal Suppress serve-time logging inside compileRuntimeContext when called from runRuntimeTurn (which logs separately). */
    _suppressServeLog?: boolean;
    /** @internal Preserve explicit serve-route breadcrumbs for installed/runtime hook paths. */
    _serveRouteBreadcrumbs?: CompileServeRouteBreadcrumbInput;
    /** @internal Freeze the expected active pack/router identity for replay eval scoring. */
    _frozenReplayEvalIdentity?: FrozenReplayEvalIdentityV1;
    /** @internal Replay-only learned-route selection override. */
    _learnedRouteSelectionOverride?: LearnedRouteSelectionOverride | null;
}
export interface ActiveCompileTarget {
    activationRoot: string;
    activePointer: ActivationPointerRecordV1;
    inspection: ActivationSlotInspection;
}
export interface RuntimeCompileSuccess {
    ok: true;
    fallbackToStaticContext: false;
    hardRequirementViolated: false;
    activationRoot: string;
    activePackId: string;
    packRootDir: string;
    compileResponse: RuntimeCompileResponseV1;
    brainContext: string;
    timing: BrainServeHotPathTimingV1;
}
export interface RuntimeCompileFailOpenFailure {
    ok: false;
    fallbackToStaticContext: true;
    hardRequirementViolated: false;
    activationRoot: string;
    error: string;
    brainContext: string;
    timing: BrainServeHotPathTimingV1;
}
export interface RuntimeCompileHardFailure {
    ok: false;
    fallbackToStaticContext: false;
    hardRequirementViolated: true;
    activationRoot: string;
    error: string;
    brainContext: string;
    timing: BrainServeHotPathTimingV1;
}
export type RuntimeCompileFailure = RuntimeCompileFailOpenFailure | RuntimeCompileHardFailure;
export type RuntimeCompileResult = RuntimeCompileSuccess | RuntimeCompileFailure;
export interface AttachStatusInput {
    activationRoot: string;
    compile?: false | Omit<CompileRuntimeContextInput, "activationRoot">;
}
export type ContextContributionEvidenceState = ContextContributionEvidenceStateV1;
export interface ContextAttributionSummaryV1 {
    selectedContextCount: number;
    stableKernelBlockCount: number;
    brainCompiledBlockCount: number;
    stableKernelSources: string[];
    brainCompiledSources: string[];
    selectionTiers: string | null;
    evidence: ContextContributionEvidenceState;
    detail: string;
}
export interface AttachCompileStatusV1 {
    ok: boolean;
    fallbackToStaticContext: boolean;
    hardRequirementViolated: boolean;
    activePackId: string | null;
    usedLearnedRouteFn: boolean | null;
    routerIdentity: string | null;
    selectionDigest: string | null;
    initMode: string | null;
    handoffState: string | null;
    seedSources: string[];
    contextAttribution: ContextAttributionSummaryV1;
    timing: BrainServeHotPathTimingV1;
    notes: string[];
    error: string | null;
}
export interface OpenClawCompileBoundaryV1 {
    contract: typeof CONTRACT_IDS.runtimeCompile;
    activationSlot: "active";
    entrypoint: "compileRuntimeContext";
    servedFromCandidateBeforePromotion: false;
    learnedRouteEvidenceRequiredWhenManifestRequiresIt: true;
}
export interface OpenClawEventExportBoundaryV1 {
    emittedContracts: readonly [typeof CONTRACT_IDS.interactionEvents, typeof CONTRACT_IDS.feedbackEvents];
    entrypoint: "runRuntimeTurn";
    bundleWriteOptional: true;
    writeFailuresEraseSuccessfulCompile: false;
    learningHandoffStaysOffHotPath: true;
}
export interface OpenClawActivePackBoundaryV1 {
    servingSlot: "active";
    inspectableSlots: readonly ActivationPointerSlot[];
    candidateServedBeforePromotion: false;
    previousSlotUsedForRollback: true;
}
export interface OpenClawPromotionBoundaryV1 {
    candidateSlot: "candidate";
    activeSlot: "active";
    previousSlot: "previous";
    requiresActivationReadyCandidate: true;
    compileSeesCandidateOnlyAfterPromotion: true;
    promotionHappensOffHotPath: true;
}
export interface OpenClawFailOpenSemanticsV1 {
    missingActivePackFallsBackToStaticContext: true;
    learnedRequiredRouteArtifactDriftHardFails: true;
    hardFailuresDisableStaticFallback: true;
    eventExportWriteFailurePreservesCompile: true;
}
export interface OpenClawLandingBoundariesV1 {
    compileBoundary: OpenClawCompileBoundaryV1;
    eventExportBoundary: OpenClawEventExportBoundaryV1;
    activePackBoundary: OpenClawActivePackBoundaryV1;
    promotionBoundary: OpenClawPromotionBoundaryV1;
    failOpenSemantics: OpenClawFailOpenSemanticsV1;
    runtimeResponsibilities: string[];
    brainResponsibilities: string[];
}
export interface AttachStatusSnapshotV1 {
    runtimeOwner: "openclaw";
    activationRoot: string;
    inspection: ActivationInspection;
    activeObservability: ActivationObservabilityReport | null;
    compile: AttachCompileStatusV1 | null;
    landingBoundaries: OpenClawLandingBoundariesV1;
    successSignals: string[];
}
export interface RollbackRuntimeAttachInput {
    activationRoot: string;
    updatedAt?: string;
    dryRun?: boolean;
}
export interface RollbackRuntimeAttachResult {
    runtimeOwner: "openclaw";
    activationRoot: string;
    updatedAt: string;
    dryRun: boolean;
    allowed: boolean;
    findings: string[];
    before: {
        activePackId: string | null;
        candidatePackId: string | null;
        previousPackId: string | null;
    };
    after: {
        activePackId: string | null;
        candidatePackId: string | null;
        previousPackId: string | null;
    } | null;
    restoredPackId: string | null;
    parkedCandidatePackId: string | null;
}
export interface BootstrapRuntimeAttachInput {
    profileSelector?: string | null;
    profileId?: string | null;
    brainAttachmentPolicy?: RuntimeTurnBrainAttachmentPolicyV1 | null;
    activationRoot: string;
    packRoot: string;
    packLabel: string;
    workspace: AdvanceAlwaysOnLearningRuntimeInput["workspace"];
    normalizedEventExport?: NormalizedEventExportV1;
    interactionEvents?: readonly InteractionEventV1[];
    feedbackEvents?: readonly FeedbackEventV1[];
    learnedRouting?: boolean;
    builtAt?: string;
    activatedAt?: string;
    offlineArtifacts?: string[];
    structuralOps?: Partial<ArtifactManifestV1["graphDynamics"]["structuralOps"]>;
    sparseFeedback?: Partial<SparseFeedbackPolicyV1>;
    compile?: false | Omit<CompileRuntimeContextInput, "activationRoot">;
}
export interface BootstrapRuntimeAttachNextStepV1 {
    id: "inspect_current_profile_status" | "preview_rollback_readiness" | "record_next_current_profile_turn" | "continue_current_profile_learning_loop";
    detail: string;
    command: string | null;
}
export interface BootstrapRuntimeAttachResult {
    runtimeOwner: "openclaw";
    profileSelector: string;
    operatorReadScope: "current_profile_only";
    activationRoot: string;
    packRoot: string;
    packId: string;
    normalizedEventExport: NormalizedEventExportV1;
    status: AttachStatusSnapshotV1;
    currentProfile: CurrentProfileBrainStatusV1;
    nextSteps: BootstrapRuntimeAttachNextStepV1[];
}
export declare function formatBootstrapRuntimeAttachReport(result: BootstrapRuntimeAttachResult): string;
export interface RuntimeTurnCompileInput {
    createdAt?: string | null;
    sequence?: number | null;
    eventId?: string | null;
}
export interface RuntimeTurnDeliveryInput {
    createdAt?: string | null;
    sequence?: number | null;
    eventId?: string | null;
    messageId?: string | null;
}
export interface RuntimeTurnFeedbackInput {
    content: string;
    createdAt?: string | null;
    sequence?: number | null;
    eventId?: string | null;
    kind?: FeedbackEventKind | null;
    messageId?: string | null;
    relatedInteractionId?: string | null;
    actorName?: string | null;
    priorityHint?: PrincipalPriorityClassV1 | null;
}
export interface RuntimeTurnExportInput {
    rootDir: string;
    exportName?: string | null;
    exportedAt?: string | null;
}
export interface RuntimeTurnContextFingerprintInputV1 {
    promptContextFingerprints?: readonly string[] | null;
    workspaceInjectionSurface?: WorkspaceInjectionSurfaceV1 | null;
    profileLineage?: readonly string[] | null;
    sessionLineage?: readonly string[] | null;
}
export interface OpenClawRuntimeTurnInput {
    activationRoot?: string | null;
    agentId?: string | null;
    profileSelector?: string | null;
    profileId?: string | null;
    sessionId: string;
    channel: string;
    userId?: string | null;
    sourceStream?: string | null;
    userMessage: string;
    createdAt?: string | null;
    sequenceStart?: number | null;
    maxContextBlocks?: number;
    budgetStrategy?: CompileRuntimeBudgetStrategy;
    mode?: RouteMode | RuntimeComparativeReplayMode;
    selectionMode?: CompileSelectionMode;
    runtimeHints?: readonly string[];
    brainAttachmentPolicy?: RuntimeTurnBrainAttachmentPolicyV1 | null;
    contextFingerprint?: RuntimeTurnContextFingerprintInputV1 | null;
    compile?: RuntimeTurnCompileInput | null;
    delivery?: boolean | RuntimeTurnDeliveryInput | null;
    feedback?: readonly (RuntimeTurnFeedbackInput | null)[] | null;
    export?: RuntimeTurnExportInput | null;
}
export interface RuntimeEventExportNoWrite {
    ok: true;
    wroteBundle: false;
    normalizedEventExport: NormalizedEventExportV1;
}
export interface RuntimeEventExportWriteSuccess {
    ok: true;
    wroteBundle: true;
    normalizedEventExport: NormalizedEventExportV1;
    rootDir: string;
    manifestPath: string;
    payloadPath: string;
    manifest: RuntimeEventExportBundleManifestV1;
}
export interface RuntimeEventExportFailure {
    ok: false;
    wroteBundle: false;
    error: string;
}
export type RuntimeEventExportResult = RuntimeEventExportNoWrite | RuntimeEventExportWriteSuccess | RuntimeEventExportFailure;
export interface RunRuntimeTurnOptions {
    activationRoot?: string;
    failOpen?: boolean;
    /** @internal Freeze the expected active pack/router identity for replay eval scoring. */
    _frozenReplayEvalIdentity?: FrozenReplayEvalIdentityV1;
    /** @internal Replay-only learned-route selection override. */
    _learnedRouteSelectionOverride?: LearnedRouteSelectionOverride | null;
}
export type RuntimeTurnResult = RuntimeCompileResult & {
    eventExport: RuntimeEventExportResult;
    warnings: string[];
};
export type TeacherLoopNoOpReason = "none" | "duplicate_export" | "queue_full" | "no_teacher_artifacts" | "empty_scan";
export interface AsyncTeacherLiveLoopInput extends Pick<AdvanceAlwaysOnLearningRuntimeInput, "packLabel" | "workspace" | "learnedRouting" | "builtAt" | "offlineArtifacts" | "structuralOps" | "sparseFeedback" | "liveSliceSize" | "backfillSliceSize" | "cadence"> {
    maxQueuedExports?: number;
    staleAfterMs?: number;
    resumeFromSnapshot?: AsyncTeacherLiveLoopSnapshotV1 | null;
    resolveLearnedRoutingState?: () => {
        pgVersion?: AdvanceAlwaysOnLearningRuntimeInput["pgVersion"];
        serveTimeDecisions?: LearningSpineServeRouteDecisionLogEntryV1[];
        baselineState?: BaselineStateV1;
    };
    persistUpdatedBaseline?: (state: BaselineStateV1) => void;
    teacherLabeler?: AsyncTeacherLabelerConfigV1 | null;
}
export interface TeacherNoArtifactCycleSummaryV1 {
    shouldWarn: boolean;
    detail: string;
}
export interface AsyncTeacherQueuedExportJobV1 {
    jobId: string;
    exportDigest: string;
    observedAt: string;
    normalizedEventExport: NormalizedEventExportV1;
}
export interface AsyncTeacherLiveLoopDiagnosticsV1 {
    acceptedExportCount: number;
    processedExportCount: number;
    duplicateExportCount: number;
    droppedExportCount: number;
    emittedArtifactCount: number;
    dedupedArtifactCount: number;
    lastProcessedAt: string | null;
    latestFreshness: TeacherSupervisionArtifactV1["freshness"]["status"] | "none";
    lastNoOpReason: TeacherLoopNoOpReason;
    notes: string[];
}
export interface AsyncTeacherLiveLoopSnapshotV1 {
    runtimeOwner: "openclaw";
    queue: {
        capacity: number;
        depth: number;
        running: boolean;
    };
    teacher: {
        artifactCount: number;
        artifacts: TeacherSupervisionArtifactV1[];
        latestFreshness: TeacherSupervisionArtifactV1["freshness"]["status"] | "none";
    };
    learner: {
        state: AlwaysOnLearningRuntimeStateV1;
        lastMaterialization: AlwaysOnLearningMaterializationJobV1 | null;
    };
    diagnostics: AsyncTeacherLiveLoopDiagnosticsV1;
    state?: {
        interactionEvents: InteractionEventV1[];
        feedbackEvents: FeedbackEventV1[];
        seenExportDigests: string[];
    };
    runtime?: {
        startedAt: string | null;
        lastHeartbeatAt: string | null;
        lastScanAt: string | null;
        scanRoot: string | null;
        lastAppliedMaterializationJobId: string | null;
    };
}
export interface AsyncTeacherEnqueueResultV1 {
    accepted: boolean;
    exportDigest: string;
    queueDepth: number;
    notes: string[];
    reason: Exclude<TeacherLoopNoOpReason, "none"> | null;
}
export interface AsyncTeacherScannedEnqueueResultV1 {
    accepted: boolean;
    exportDigest: string | null;
    queueDepth: number;
    notes: string[];
    reason: Exclude<TeacherLoopNoOpReason, "none"> | ScannedEventExportNoopReasonV1 | null;
    warnings: string[];
    error: string | null;
    scanner: ScannerExportManifestV1;
}
export type AsyncTeacherScannerScanNoOpReasonV1 = "none" | "empty_scan" | "duplicate_exports" | "queue_full";
export interface AsyncTeacherScannerScanHitResultV1 {
    lane: EventExportLaneV1;
    exportDigest: string;
    exportName: string;
    exportedAt: string;
    eventRange: Pick<NormalizedEventExportV1["range"], "start" | "end" | "count">;
    accepted: boolean;
    queueDepth: number;
    reason: AsyncTeacherEnqueueResultV1["reason"];
}
export interface AsyncTeacherScannerScanResultV1 {
    runtimeOwner: "openclaw";
    scanRoot: string;
    scannedAt: string;
    selectedCount: number;
    acceptedCount: number;
    duplicateCount: number;
    droppedCount: number;
    liveAcceptedCount: number;
    backfillAcceptedCount: number;
    duplicateScannerDigestCount: number;
    staleSkippedCount: number;
    invalidBundleCount: number;
    noOpReason: AsyncTeacherScannerScanNoOpReasonV1;
    notes: string[];
    results: AsyncTeacherScannerScanHitResultV1[];
    snapshot: AsyncTeacherLiveLoopSnapshotV1;
}
export interface CanonicalSupervisionFeedbackRecordV1 {
    eventId: string;
    kind: FeedbackEventKind;
    sequence: number;
    createdAt: string;
    content: string;
    relatedInteractionId: string | null;
}
export interface CanonicalSupervisionV1 {
    runtimeOwner: "openclaw";
    exportDigest: string;
    supervisionDigest: string;
    sessionId: string | null;
    channel: string | null;
    eventRange: Pick<NormalizedEventExportV1["range"], "start" | "end" | "count">;
    sourceStreams: string[];
    humanLabelCount: number;
    selfLabelCount: number;
    feedbackCounts: {
        corrections: number;
        teachings: number;
        approvals: number;
        suppressions: number;
    };
    compilePackIds: string[];
    relatedInteractionIds: string[];
    feedback: CanonicalSupervisionFeedbackRecordV1[];
}
export interface ContinuousProductLoopPackVersionV1 {
    version: number;
    packId: string;
    routePolicy: RuntimeCompileTargetV1["routePolicy"];
    routerIdentity: string | null;
    workspaceSnapshot: string;
    workspaceRevision: string | null;
    eventRange: RuntimeCompileTargetV1["eventRange"];
    eventExportDigest: string | null;
    builtAt: string;
}
export interface ContinuousProductLoopStateV1 {
    runtimeOwner: "openclaw";
    activationRoot: string;
    loopRoot: string;
    interactionEvents: InteractionEventV1[];
    feedbackEvents: FeedbackEventV1[];
    learner: AlwaysOnLearningRuntimeStateV1;
    runtimePlasticity: RuntimeGraphPlasticityStateV1 | null;
    activePackVersion: number;
    currentActivePack: ContinuousProductLoopPackVersionV1 | null;
    candidatePack: ContinuousProductLoopPackVersionV1 | null;
    packLineage: ContinuousProductLoopPackVersionV1[];
    nextPackVersion: number;
    promotionCount: number;
    lastSupervision: CanonicalSupervisionV1 | null;
}
export interface ContinuousProductLoopLearningUpdateV1 {
    warnings: string[];
    supervisionDigest: string | null;
    bridgeDigest: string | null;
    selectedSliceIds: string[];
    materializationJobId: string | null;
    materializationReason: AlwaysOnLearningMaterializationJobV1["reason"] | null;
    materializationLane: AlwaysOnLearningMaterializationJobV1["lane"] | null;
    candidateRootDir: string | null;
    candidatePack: ContinuousProductLoopPackVersionV1 | null;
    runtimePlasticity: RuntimeGraphPlasticityStateV1 | null;
    promotionAllowed: boolean;
    promotionFindings: string[];
    promoted: boolean;
}
export interface RunContinuousProductLoopTurnInput {
    activationRoot: string;
    loopRoot: string;
    packLabel: string;
    workspace: AdvanceAlwaysOnLearningRuntimeInput["workspace"];
    turn: OpenClawRuntimeTurnInput;
    state?: ContinuousProductLoopStateV1;
    learnedRouting?: boolean;
    failOpen?: boolean;
    autoPromote?: boolean;
    candidateBuiltAt?: string | null;
    stageUpdatedAt?: string | null;
    promoteUpdatedAt?: string | null;
    offlineArtifacts?: string[];
    structuralOps?: Partial<ArtifactManifestV1["graphDynamics"]["structuralOps"]>;
    sparseFeedback?: Partial<SparseFeedbackPolicyV1>;
    liveSliceSize?: number;
    backfillSliceSize?: number;
    cadence?: Partial<AlwaysOnLearningCadenceV1>;
}
export interface ContinuousProductLoopTurnResultV1 {
    runtimeOwner: "openclaw";
    compileActiveVersion: number;
    compileActivePackId: string | null;
    turn: RuntimeTurnResult;
    supervision: CanonicalSupervisionV1 | null;
    learning: ContinuousProductLoopLearningUpdateV1;
    state: ContinuousProductLoopStateV1;
}
export declare function buildCanonicalSupervision(normalizedEventExport: NormalizedEventExportV1): CanonicalSupervisionV1;
export declare function createContinuousProductLoopState(input: {
    activationRoot: string;
    loopRoot: string;
}): ContinuousProductLoopStateV1;
export declare class AsyncTeacherLiveLoop {
    private readonly input;
    private readonly queueCapacity;
    private readonly staleAfterMs;
    private readonly teacherLabeler;
    private readonly queuedExportDigests;
    private readonly seenExportDigests;
    private queue;
    private drainPromise;
    private interactionEvents;
    private feedbackEvents;
    private teacherArtifacts;
    private learnerState;
    private lastMaterialization;
    private lastTeacherLabelerResult;
    private diagnostics;
    constructor(input: AsyncTeacherLiveLoopInput);
    enqueueNormalizedEventExport(normalizedEventExport: NormalizedEventExportV1, options?: {
        observedAt?: string;
    }): AsyncTeacherEnqueueResultV1;
    enqueueScannedEventExport(scannedEventExport: ScannedEventExportInputV1, options?: {
        observedAt?: string;
    }): AsyncTeacherScannedEnqueueResultV1;
    ingestRuntimeEventExportScannerScan(scan: RuntimeEventExportScannerScanResultV1): Promise<AsyncTeacherScannerScanResultV1>;
    flush(): Promise<AsyncTeacherLiveLoopSnapshotV1>;
    snapshot(): AsyncTeacherLiveLoopSnapshotV1;
    private ensureDrain;
    private drain;
    private refreshNotes;
}
export declare function createAsyncTeacherLiveLoop(input: AsyncTeacherLiveLoopInput): AsyncTeacherLiveLoop;
export interface ScannerLiveWorkspaceInputV1 {
    workspaceId: string;
    snapshotId: string;
    capturedAt: string;
    rootDir: string;
    branch?: string | null;
    revision?: string | null;
    dirty?: boolean;
    manifestDigest?: string | null;
    labels?: readonly string[];
    files?: readonly string[];
}
export interface ScanRecordedSessionInputV1 {
    rootDir: string;
    trace: RecordedSessionTraceV1;
}
export interface ScanRecordedSessionResultV1 {
    runtimeOwner: "openclaw";
    scanMode: "session";
    rootDir: string;
    fixtureHash: string;
    bundle: RecordedSessionReplayBundleV1;
}
export interface ScanLiveEventExportInputV1 {
    normalizedEventExport: NormalizedEventExportV1;
    workspace: ScannerLiveWorkspaceInputV1;
    packLabel?: string | null;
    observedAt?: string | null;
    builtAt?: string | null;
    learnedRouting?: boolean;
    staleAfterMs?: number;
    liveSliceSize?: number;
    backfillSliceSize?: number;
}
export interface ScanLiveEventExportResultV1 {
    runtimeOwner: "openclaw";
    scanMode: "live";
    observedAt: string;
    packLabel: string;
    supervision: CanonicalSupervisionV1;
    snapshot: AsyncTeacherLiveLoopSnapshotV1;
    labelFlow: OperatorLabelFlowSummary;
    learningPath: OperatorLearningPathSummary;
}
export declare function scanRecordedSession(input: ScanRecordedSessionInputV1): ScanRecordedSessionResultV1;
export declare function scanLiveEventExport(input: ScanLiveEventExportInputV1): ScanLiveEventExportResultV1;
export declare function resolveAsyncTeacherLiveLoopSnapshotPath(activationRoot: string): string;
export declare const WATCH_STATE_DIRNAME = "watch";
export declare const WATCH_SESSION_TAIL_CURSOR_BASENAME = "session-tail-cursor.json";
export declare const WATCH_TEACHER_SNAPSHOT_BASENAME = "teacher-snapshot.json";
export declare const DEFAULT_WATCH_POLL_INTERVAL_SECONDS = 30;
export declare function summarizeTeacherNoArtifactCycle(notes: readonly string[] | null | undefined): TeacherNoArtifactCycleSummaryV1;
export interface WatchTeacherSnapshotFailureV1 {
    mode: "materialization_failed" | "teacher_fail_open";
    detail: string;
    at: string;
}
export interface WatchTeacherSnapshotTeacherSummaryV1 {
    artifactCount: number;
    latestFreshness: AsyncTeacherLiveLoopDiagnosticsV1["latestFreshness"] | "unavailable";
    acceptedExportCount: number;
    processedExportCount: number;
    duplicateExportCount: number;
    droppedExportCount: number;
    emittedArtifactCount: number;
    dedupedArtifactCount: number;
    lastProcessedAt: string | null;
    lastNoOpReason: TeacherLoopNoOpReason;
    queueDepth: number;
    queueCapacity: number;
    running: boolean;
    lastAppliedMaterializationJobId: string | null;
    lastMaterializedPackId: string | null;
}
export interface WatchTeacherSnapshotLearningSummaryV1 {
    bootstrapped: boolean;
    mode: AlwaysOnLearningRuntimePlanV1["mode"];
    nextPriorityLane: AlwaysOnLearningRuntimePlanV1["nextPriorityLane"];
    nextPriorityBucket: AlwaysOnLearningRuntimePlanV1["nextPriorityBucket"];
    pendingLive: number;
    pendingBackfill: number;
    pendingTotal: number;
    pendingByBucket: AlwaysOnLearningRuntimePlanV1["pending"]["byBucket"];
    materializationCount: number;
    lastMaterializedAt: string | null;
    lastMaterializationReason: AlwaysOnLearningMaterializationJobV1["reason"] | null;
    lastMaterializationLane: AlwaysOnLearningMaterializationJobV1["lane"] | null;
    lastMaterializedPackId: string | null;
    lastHandledMaterializationPackId: string | null;
}
export interface WatchTeacherSnapshotLabelingSummaryV1 {
    learningCadence: LearningCadence;
    scanPolicy: LearningScanPolicy;
    liveSlicesPerCycle: number;
    backfillSlicesPerCycle: number;
    teacherBudget: number;
    teacherDelayMs: number;
    backgroundLabelAmplification: number;
}
export interface WatchEmbedInstrumentationPointV1 {
    slot: "candidate" | "active" | null;
    packId: string | null;
    runtimeEmbedderPresent: boolean;
    runtimeEmbedderModel: string | null;
    vectorEntryCount: number | null;
    numericEmbeddingEntryCount: number | null;
    embeddingModels: string[];
    error: string | null;
}
export interface WatchEmbedInstrumentationTraceV1 {
    observedAt: string;
    candidatePackId: string | null;
    promotionAllowed: boolean | null;
    promotionFindings: string[];
    beforeCandidateMaterialization: WatchEmbedInstrumentationPointV1;
    afterCandidateMaterialization: WatchEmbedInstrumentationPointV1 | null;
    afterStage: WatchEmbedInstrumentationPointV1 | null;
    afterPromote: WatchEmbedInstrumentationPointV1 | null;
}
export interface WatchTeacherSnapshotFileV1 {
    contract: "openclaw_watch_teacher_snapshot.v1";
    runtimeOwner: "openclaw";
    updatedAt: string;
    lastRunAt: string;
    pollIntervalSeconds: number;
    scanRoot: string;
    sessionTailCursorPath: string;
    sessionTailCursorUpdatedAt: string;
    sessionTailSessionsTracked: number;
    sessionTailBridgedEventCount: number;
    scannerCheckpointPath: string;
    scannerCheckpoint: RuntimeEventExportScannerCheckpointV1;
    replayedBundleCount: number;
    replayedEventCount: number;
    exportedBundleCount: number;
    exportedEventCount: number;
    startupWarnings: string[];
    lastTeacherError: string | null;
    localSessionTailNoopReason: string | null;
    lastHandledMaterializationPackId: string | null;
    teacher: WatchTeacherSnapshotTeacherSummaryV1;
    learning: WatchTeacherSnapshotLearningSummaryV1;
    labeling: WatchTeacherSnapshotLabelingSummaryV1;
    lastObservedDelta: CurrentProfilePassiveLearningDeltaSummaryV1;
    embedInstrumentation: WatchEmbedInstrumentationTraceV1 | null;
    failure: WatchTeacherSnapshotFailureV1 | null;
    snapshot: AsyncTeacherLiveLoopSnapshotV1;
}
export interface LoadedTeacherSurfaceV1 {
    sourcePath: string;
    sourceKind: "watch_snapshot" | "async_snapshot";
    snapshot: AsyncTeacherLiveLoopSnapshotV1;
    watchSnapshot: WatchTeacherSnapshotFileV1 | null;
}
export declare function resolveWatchStateRoot(activationRoot: string): string;
export declare function resolveWatchSessionTailCursorPath(activationRoot: string): string;
export declare function resolveWatchTeacherSnapshotPath(activationRoot: string): string;
export declare function resolveOperatorTeacherSnapshotPath(activationRoot: string, explicitPath: string | null | undefined): string | null;
export declare function loadTeacherSurface(snapshotPath: string): LoadedTeacherSurfaceV1 | null;
export declare function loadWatchTeacherSnapshotState(snapshotPath: string): {
    lastHandledMaterializationPackId: string | null;
    lastObservedDelta: CurrentProfilePassiveLearningDeltaSummaryV1;
    embedInstrumentation: WatchEmbedInstrumentationTraceV1 | null;
    snapshot: AsyncTeacherLiveLoopSnapshotV1 | null;
    error: string | null;
};
export declare function persistWatchTeacherSnapshot(snapshotPath: string, input: {
    lastRunAt: string;
    pollIntervalSeconds: number;
    scanRoot: string;
    sessionTailCursorPath: string;
    sessionTailCursorUpdatedAt: string;
    sessionTailSessionsTracked: number;
    sessionTailBridgedEventCount: number;
    scannerCheckpointPath: string;
    scannerCheckpoint: RuntimeEventExportScannerCheckpointV1;
    replayedBundleCount: number;
    replayedEventCount: number;
    exportedBundleCount: number;
    exportedEventCount: number;
    startupWarnings: readonly string[];
    lastTeacherError: string | null;
    localSessionTailNoopReason: string | null;
    lastHandledMaterializationPackId: string | null;
    lastObservedDelta: CurrentProfilePassiveLearningDeltaSummaryV1;
    embedInstrumentation?: WatchEmbedInstrumentationTraceV1 | null;
    failure: WatchTeacherSnapshotFailureV1 | null;
    snapshot: AsyncTeacherLiveLoopSnapshotV1;
}): WatchTeacherSnapshotFileV1;
export declare function loadAsyncTeacherLiveLoopSnapshot(snapshotPath: string): AsyncTeacherLiveLoopSnapshotV1;
export declare function buildRuntimeEventExportBundleManifest(input: {
    exportName: string;
    exportedAt: string;
    payloadPath: string;
    normalizedEventExport: NormalizedEventExportV1;
    scanner?: ScannerExportManifestV1 | null;
}): RuntimeEventExportBundleManifestV1;
export declare function validateRuntimeEventExportBundleManifest(value: RuntimeEventExportBundleManifestV1, normalizedEventExport?: NormalizedEventExportV1): string[];
export declare function loadRuntimeEventExportBundle(rootDir: string): RuntimeEventExportBundleDescriptor;
export declare function buildNormalizedEventExportFromScannedEvents(input: ScannedEventExportInputV1): ScannedEventExportBuildResultV1;
declare const RUNTIME_EVENT_EXPORT_SCANNER_CHECKPOINT_CONTRACT: "runtime_event_export_scanner_checkpoint.v1";
export declare const DEFAULT_RUNTIME_EVENT_EXPORT_SCANNER_LIVE_TAIL_BUNDLES = 2;
export declare const DEFAULT_RUNTIME_EVENT_EXPORT_SCANNER_BACKFILL_BUNDLES_PER_PASS = 1;
export declare const DEFAULT_RUNTIME_EVENT_EXPORT_SCANNER_STALE_HISTORY_MS: number;
export declare const DEFAULT_RUNTIME_EVENT_EXPORT_SCANNER_CHECKPOINT_BASENAME = ".openclawbrain-scanner-checkpoint.json";
export interface RuntimeEventExportScannerBundleCursorV1 {
    exportDigest: string;
    exportName: string;
    exportedAt: string;
    eventRange: Pick<NormalizedEventExportV1["range"], "start" | "end" | "count">;
}
export interface RuntimeEventExportScannerCheckpointV1 {
    contract: typeof RUNTIME_EVENT_EXPORT_SCANNER_CHECKPOINT_CONTRACT;
    runtimeOwner: "openclaw";
    scanRoot: string;
    updatedAt: string;
    live: {
        after: RuntimeEventExportScannerBundleCursorV1 | null;
    };
    backfill: {
        before: RuntimeEventExportScannerBundleCursorV1 | null;
        exhausted: boolean;
        staleBefore: string | null;
    };
    processedExportDigests: string[];
    stats: {
        scanPasses: number;
        liveBundlesScanned: number;
        backfillBundlesScanned: number;
        duplicateBundlesSkipped: number;
        staleBundlesSkipped: number;
        invalidBundlesSkipped: number;
    };
}
export interface RuntimeEventExportScannerInput {
    scanRoot: string;
    checkpointPath?: string;
    liveTailBundles?: number;
    backfillBundlesPerPass?: number;
    staleHistoryMs?: number;
}
export interface RuntimeEventExportScannerInvalidBundleV1 {
    rootDir: string;
    error: string;
}
export type RuntimeEventExportScannerPriorityBucketV1 = "live" | "principal_backfill" | "backfill" | "stale_history";
export interface RuntimeEventExportScannerQueueEntryV1 {
    lane: EventExportLaneV1;
    rootDir: string;
    exportDigest: string;
    exportName: string;
    exportedAt: string;
    eventRange: Pick<NormalizedEventExportV1["range"], "start" | "end" | "count">;
    priorityBucket: RuntimeEventExportScannerPriorityBucketV1;
    priorityScore: number;
    priorityReasons: string[];
    humanLabelCount: number;
    feedbackCount: number;
    teacherRoles: PrincipalRoleV1[];
    teacherAuthorities: TeacherAuthorityV1[];
    priorityClasses: PrincipalPriorityClassV1[];
    scopedPrincipalEventCount: number;
    supersedingPrincipalEventCount: number;
    staleHistory: boolean;
    ageMsFromLatest: number | null;
}
export interface RuntimeEventExportScannerHitV1 {
    lane: EventExportLaneV1;
    rootDir: string;
    exportDigest: string;
    exportName: string;
    exportedAt: string;
    eventRange: Pick<NormalizedEventExportV1["range"], "start" | "end" | "count">;
    priorityBucket: RuntimeEventExportScannerPriorityBucketV1;
    priorityScore: number;
    priorityReasons: string[];
    humanLabelCount: number;
    feedbackCount: number;
    teacherRoles: PrincipalRoleV1[];
    teacherAuthorities: TeacherAuthorityV1[];
    priorityClasses: PrincipalPriorityClassV1[];
    scopedPrincipalEventCount: number;
    supersedingPrincipalEventCount: number;
    staleHistory: boolean;
    ageMsFromLatest: number | null;
    normalizedEventExport: NormalizedEventExportV1;
}
export interface RuntimeEventExportScannerQueueV1 {
    ageFloor: {
        newestExportedAt: string | null;
        staleBefore: string | null;
        staleHistoryMs: number;
    };
    live: RuntimeEventExportScannerQueueEntryV1[];
    backfill: RuntimeEventExportScannerQueueEntryV1[];
    staleHistory: RuntimeEventExportScannerQueueEntryV1[];
}
export interface RuntimeEventExportScannerScanResultV1 {
    runtimeOwner: "openclaw";
    scanRoot: string;
    checkpointPath: string;
    scannedAt: string;
    live: RuntimeEventExportScannerHitV1[];
    backfill: RuntimeEventExportScannerHitV1[];
    selected: RuntimeEventExportScannerHitV1[];
    queue: RuntimeEventExportScannerQueueV1;
    duplicateExportDigests: string[];
    staleSkippedExportDigests: string[];
    invalidBundles: RuntimeEventExportScannerInvalidBundleV1[];
    idle: boolean;
    checkpoint: RuntimeEventExportScannerCheckpointV1;
}
export interface RuntimeEventExportScannerLoopOptionsV1 {
    pollIntervalMs?: number;
    maxPasses?: number;
    stopWhenIdle?: boolean;
    signal?: AbortSignal;
    onPass?: (result: RuntimeEventExportScannerScanResultV1) => void | Promise<void>;
}
export interface RuntimeEventExportScannerLoopResultV1 {
    runtimeOwner: "openclaw";
    passCount: number;
    liveBundlesScanned: number;
    backfillBundlesScanned: number;
    stoppedReason: "idle" | "max_passes" | "aborted";
    lastScan: RuntimeEventExportScannerScanResultV1 | null;
    checkpoint: RuntimeEventExportScannerCheckpointV1;
}
export declare function createRuntimeEventExportScannerCheckpoint(input: {
    scanRoot: string;
    updatedAt?: string;
}): RuntimeEventExportScannerCheckpointV1;
export declare function validateRuntimeEventExportScannerCheckpoint(value: RuntimeEventExportScannerCheckpointV1): string[];
export declare function loadRuntimeEventExportScannerCheckpoint(checkpointPath: string): RuntimeEventExportScannerCheckpointV1;
export declare class RuntimeEventExportScanner {
    readonly scanRoot: string;
    readonly checkpointPath: string;
    readonly liveTailBundles: number;
    readonly backfillBundlesPerPass: number;
    readonly staleHistoryMs: number;
    private checkpoint;
    constructor(input: RuntimeEventExportScannerInput);
    snapshot(): RuntimeEventExportScannerCheckpointV1;
    restoreCheckpoint(checkpoint: RuntimeEventExportScannerCheckpointV1): void;
    scanOnce(options?: {
        scannedAt?: string;
    }): RuntimeEventExportScannerScanResultV1;
    runLoop(options?: RuntimeEventExportScannerLoopOptionsV1): Promise<RuntimeEventExportScannerLoopResultV1>;
}
export declare function createRuntimeEventExportScanner(input: RuntimeEventExportScannerInput): RuntimeEventExportScanner;
export declare function classifyFeedbackKind(content: string): FeedbackEventKind;
export declare function formatPromptContext(compileResponse: RuntimeCompileResponseV1): string;
export interface EmpiricalStructuralBudgetEvidenceV1 {
    split: number;
    merge: number;
    prune: number;
    connect: number;
}
export interface ResolvedCompileBudgetV1 {
    requestedStrategy: CompileRuntimeBudgetStrategy;
    effectiveStrategy: CompileRuntimeBudgetStrategy;
    maxContextBlocks: number;
    defaultMaxContextBlocks: number;
    evidence: EmpiricalStructuralBudgetEvidenceV1;
    evidenceTotal: number;
    tendencies: EmpiricalStructuralBudgetEvidenceV1;
    notes: string[];
}
export interface CompileStructuralBudgetSignalEvidenceV1 {
    expansionCandidates: number;
    traversalActivations: number;
    overlapPruned: number;
}
export declare function deriveEmpiricalStructuralBudget(input: {
    requestedStrategy?: CompileRuntimeBudgetStrategy;
    requestedMaxContextBlocks?: number;
    evolution?: Pick<PackGraphEvolutionV1, "structuralOps" | "prunedBlockIds"> | null;
    defaultMaxContextBlocks?: number;
    minimumMaxContextBlocks?: number;
    maximumMaxContextBlocks?: number;
}): ResolvedCompileBudgetV1;
export declare function deriveEmpiricalStructuralBudgetFromCompileSignals(input: {
    requestedStrategy?: CompileRuntimeBudgetStrategy;
    requestedMaxContextBlocks?: number;
    structuralSignals?: Pick<RuntimeCompileStructuralSignalsV1, "matchedCandidateCount" | "selectedMatchedCount" | "overlapPrunedCount" | "traversalActivatedCount"> | null;
    defaultMaxContextBlocks?: number;
    minimumMaxContextBlocks?: number;
    maximumMaxContextBlocks?: number;
}): ResolvedCompileBudgetV1;
export declare function resolveActivePackForCompile(activationRoot: string): ActiveCompileTarget;
export declare function compileRuntimeContext(input: CompileRuntimeContextInput): RuntimeCompileResult;
export declare function describeAttachStatus(input: AttachStatusInput): AttachStatusSnapshotV1;
export declare function rollbackRuntimeAttach(input: RollbackRuntimeAttachInput): RollbackRuntimeAttachResult;
export declare function bootstrapRuntimeAttach(input: BootstrapRuntimeAttachInput): BootstrapRuntimeAttachResult;
export declare function buildNormalizedRuntimeEventExport(turn: OpenClawRuntimeTurnInput, compileResult: RuntimeCompileResult): NormalizedEventExportV1;
export interface WriteScannedEventExportBundleInputV1 {
    rootDir: string;
    exportName?: string | null;
    exportedAt?: string | null;
    scannedEventExport: ScannedEventExportInputV1;
}
export type WriteScannedEventExportBundleResultV1 = RuntimeEventExportWriteSuccess | ScannedEventExportBuildFailureV1;
export declare function writeRuntimeEventExportBundle(turn: OpenClawRuntimeTurnInput, normalizedEventExport: NormalizedEventExportV1): RuntimeEventExportNoWrite | RuntimeEventExportWriteSuccess;
export declare function writeScannedEventExportBundle(input: WriteScannedEventExportBundleInputV1): WriteScannedEventExportBundleResultV1;
export declare function runRuntimeTurn(turn: OpenClawRuntimeTurnInput, options?: RunRuntimeTurnOptions): RuntimeTurnResult;
export declare function runContinuousProductLoopTurn(input: RunContinuousProductLoopTurnInput): ContinuousProductLoopTurnResultV1;
export type RecordedSessionReplayMode = "no_brain" | RuntimeComparativeReplayMode;
export interface RecordedSessionReplayWorkspaceV1 {
    workspaceId: string;
    snapshotId: string;
    capturedAt: string;
    rootDir: string;
    branch?: string;
    revision: string;
    labels?: string[];
}
export interface RecordedSessionSeedCueV1 {
    cueId: string;
    createdAt: string;
    content: string;
    kind?: FeedbackEventKind;
}
export interface RecordedSessionTraceFeedbackV1 {
    createdAt: string;
    content: string;
    kind?: FeedbackEventKind | null;
}
export interface RecordedSessionTraceTurnV1 {
    turnId?: string;
    createdAt: string;
    deliveredAt?: string | null;
    userMessage: string;
    runtimeHints?: readonly string[];
    feedback?: readonly RecordedSessionTraceFeedbackV1[];
    expectedContextPhrases: readonly string[];
    minimumPhraseHits?: number;
}
export interface RecordedSessionTraceV1 {
    contract: typeof RECORDED_SESSION_TRACE_CONTRACT;
    traceId: string;
    source: "sanitized_recorded_session";
    recordedAt: string;
    bundleBuiltAt: string;
    agentId?: string | null;
    sessionId: string;
    channel: string;
    sourceStream: string;
    privacy: {
        sanitized: true;
        notes: string[];
    };
    workspace: RecordedSessionReplayWorkspaceV1;
    evalTurnCount?: number;
    seedBuiltAt: string;
    seedActivatedAt: string;
    seedCues: readonly RecordedSessionSeedCueV1[];
    turns: readonly RecordedSessionTraceTurnV1[];
}
export interface RecordedSessionReplayTurnFixtureV1 {
    turnId: string;
    turn: OpenClawRuntimeTurnInput;
    expectedContextPhrases: string[];
    minimumPhraseHits: number;
}
export interface RecordedSessionReplayFixtureV1 {
    contract: typeof RECORDED_SESSION_FIXTURE_CONTRACT;
    traceId: string;
    source: RecordedSessionTraceV1["source"];
    recordedAt: string;
    bundleBuiltAt: string;
    traceHash: string;
    fixtureHash: string;
    privacy: RecordedSessionTraceV1["privacy"];
    workspace: RecordedSessionReplayWorkspaceV1;
    evalTurnCount?: number;
    seedBuiltAt: string;
    seedActivatedAt: string;
    seedExport: NormalizedEventExportV1;
    turns: RecordedSessionReplayTurnFixtureV1[];
}
export type RecordedSessionReplayTurnPhase = "train" | "eval";
export interface RecordedSessionReplayTurnReportV1 {
    turnId: string;
    replayMode: RecordedSessionReplayMode;
    phase: RecordedSessionReplayTurnPhase;
    compileOk: boolean;
    fallbackToStaticContext: boolean;
    hardRequirementViolated: boolean;
    activePackId: string | null;
    modeRequested: RouteMode | null;
    modeEffective: RouteMode | null;
    selectionEngine: CompileSelectionMode | null;
    usedLearnedRouteFn: boolean;
    activationTaken: boolean | null;
    activationSource: string | null;
    activationReason: string | null;
    activationConfidence: string | null;
    routerIdentity: string | null;
    selectionDigest: string | null;
    selectedContextIds: string[];
    selectedContextTexts: string[];
    eventExportDigest: string | null;
    expectedContextPhrases: string[];
    minimumPhraseHits: number;
    phraseHits: string[];
    missedPhrases: string[];
    qualityScore: number;
    compileActiveVersion: number | null;
    promoted: boolean;
    timing: BrainServeHotPathTimingV1 | null;
    observability: {
        scanPolicy: LearningScanPolicy | null;
        scanSurfaces: string[];
        humanLabelCount: number;
        selfLabelCount: number;
        totalEventCount: number;
        attributedEventCount: number;
        selectionDigestCount: number;
        freshestSourceStream: string | null;
        freshestCreatedAt: string | null;
    };
    warnings: string[];
}
export interface RecordedSessionReplayScannerEvidenceV1 {
    exportTurnCount: number;
    scanPolicy: LearningScanPolicy | null;
    scanSurfaceCount: number;
    scanSurfaces: string[];
    humanLabelCount: number;
    selfLabelCount: number;
    totalEventCount: number;
    attributedEventCount: number;
    attributedTurnCount: number;
    selectionDigestCount: number;
    selectionDigestTurnCount: number;
    activePackChangeCount: number;
    freshestSourceStream: string | null;
    freshestCreatedAt: string | null;
    warnings: string[];
}
export interface RecordedSessionReplayModeSummaryV1 {
    mode: RecordedSessionReplayMode;
    activationStrategy: "no_brain" | "seed_pack" | "continuous_learned_loop";
    modeRequested: RouteMode | null;
    selectionEngine: CompileSelectionMode | null;
    qualityScore: number;
    compileOkCount: number;
    phraseHitCount: number;
    phraseCount: number;
    usedLearnedRouteTurnCount: number;
    promotionCount: number;
    packIds: string[];
    trainTurnCount: number;
    evalTurnCount: number;
    frozenEvalPackId: string | null;
    frozenEvalRouterIdentity: string | null;
    scannerEvidence: RecordedSessionReplayScannerEvidenceV1;
    scoreHash: string;
}
export interface RecordedSessionReplayModeReportV1 {
    mode: RecordedSessionReplayMode;
    summary: RecordedSessionReplayModeSummaryV1;
    turns: RecordedSessionReplayTurnReportV1[];
}
export interface RecordedSessionReplayBundleV1 {
    contract: typeof RECORDED_SESSION_BUNDLE_CONTRACT;
    traceId: string;
    source: RecordedSessionReplayFixtureV1["source"];
    recordedAt: string;
    generatedAt: string;
    traceHash: string;
    fixtureHash: string;
    scoreHash: string;
    bundleHash: string;
    privacy: RecordedSessionReplayFixtureV1["privacy"];
    modes: RecordedSessionReplayModeReportV1[];
    summary: {
        winnerMode: RecordedSessionReplayMode | null;
        ranking: Array<{
            mode: RecordedSessionReplayMode;
            qualityScore: number;
        }>;
    };
}
export interface RecordedSessionReplayRescoreReportV1 {
    scoreHash: string;
    modes: Array<{
        mode: RecordedSessionReplayMode;
        qualityScore: number;
        scoreHash: string;
    }>;
}
export interface RecordedSessionReplayBundleHashVerificationV1 {
    bundleHashMatches: boolean;
    scoreHashMatches: boolean;
}
export interface RecordedSessionReplayProofManifestV1 {
    contract: "recorded_session_replay_proof_manifest.v1";
    traceId: string;
    source: RecordedSessionReplayBundleV1["source"];
    recordedAt: string;
    generatedAt: string;
    hashAlgorithm: "sha256";
    modeOrder: RecordedSessionReplayMode[];
    contracts: {
        trace: typeof RECORDED_SESSION_TRACE_CONTRACT;
        fixture: typeof RECORDED_SESSION_FIXTURE_CONTRACT;
        bundle: typeof RECORDED_SESSION_BUNDLE_CONTRACT;
        environment: "recorded_session_replay_environment.v1";
        summaryTables: "recorded_session_replay_summary_tables.v1";
        coverageSnapshot: "recorded_session_replay_coverage_snapshot.v1";
        hardeningSnapshot: "recorded_session_replay_hardening_snapshot.v1";
        hashes: "recorded_session_replay_hashes.v1";
    };
    hashes: {
        traceHash: string;
        fixtureHash: string;
        scoreHash: string;
        bundleHash: string;
    };
    files: {
        trace: string;
        fixture: string;
        bundle: string;
        environment: string;
        summary: string;
        summaryTables: string;
        coverageSnapshot: string;
        hardeningSnapshot: string;
        hashes: string;
        modes: Array<{
            mode: RecordedSessionReplayMode;
            path: string;
        }>;
    };
}
export interface RecordedSessionReplayProofEnvironmentV1 {
    contract: "recorded_session_replay_environment.v1";
    runtimeOwner: "openclaw";
    generator: {
        packageName: "@openclawbrain/cli";
        entrypoint: "writeRecordedSessionReplayProofBundle";
        nodeVersion: string;
        platform: string;
        arch: string;
    };
    determinism: {
        hashAlgorithm: "sha256";
        canonicalJson: true;
        modeOrder: RecordedSessionReplayMode[];
        scratchReplayRoot: "temporary_directory";
    };
}
export interface RecordedSessionReplayProofModeTableRowV1 {
    mode: RecordedSessionReplayMode;
    turnCount: number;
    qualityScore: number;
    compileOkCount: number;
    phraseHitCount: number;
    phraseCount: number;
    usedLearnedRouteTurnCount: number;
    promotionCount: number;
    exportTurnCount: number;
    humanLabelCount: number;
    attributedTurnCount: number;
    activePackChangeCount: number;
    warningCount: number;
    scoreHash: string;
}
export interface RecordedSessionReplayProofTurnTableRowV1 {
    mode: RecordedSessionReplayMode;
    turnId: string;
    qualityScore: number;
    compileOk: boolean;
    phraseHitCount: number;
    phraseCount: number;
    usedLearnedRouteFn: boolean;
    promoted: boolean;
    activePackId: string | null;
    selectionDigest: string | null;
    eventExportDigest: string | null;
    warningCount: number;
}
export interface RecordedSessionReplayProofSummaryTablesV1 {
    contract: "recorded_session_replay_summary_tables.v1";
    traceId: string;
    winnerMode: RecordedSessionReplayMode | null;
    ranking: RecordedSessionReplayBundleV1["summary"]["ranking"];
    modes: RecordedSessionReplayProofModeTableRowV1[];
    turns: RecordedSessionReplayProofTurnTableRowV1[];
}
export interface RecordedSessionReplayProofCoverageSnapshotModeRowV1 {
    mode: RecordedSessionReplayMode;
    turnCount: number;
    compileOkRate: number | null;
    phraseHitRate: number | null;
    learnedRouteTurnRate: number | null;
    attributedTurnRate: number | null;
}
export interface RecordedSessionReplayProofCoverageSnapshotV1 {
    contract: "recorded_session_replay_coverage_snapshot.v1";
    traceId: string;
    winnerMode: RecordedSessionReplayMode | null;
    totalTurns: number;
    compileOkTurnCount: number;
    compileOkRate: number | null;
    phraseHitCount: number;
    phraseCount: number;
    phraseHitRate: number | null;
    modes: RecordedSessionReplayProofCoverageSnapshotModeRowV1[];
}
export interface RecordedSessionReplayProofHardeningSnapshotModeRowV1 {
    mode: RecordedSessionReplayMode;
    warningCount: number;
    compileFailureCount: number;
    promotionCount: number;
    exportTurnCount: number;
    attributedTurnCount: number;
}
export interface RecordedSessionReplayProofHardeningSnapshotV1 {
    contract: "recorded_session_replay_hardening_snapshot.v1";
    traceId: string;
    totalTurns: number;
    compileFailureCount: number;
    compileFailureRate: number | null;
    warningCount: number;
    promotionCount: number;
    exportTurnCount: number;
    attributedTurnCount: number;
    modes: RecordedSessionReplayProofHardeningSnapshotModeRowV1[];
}
export interface RecordedSessionReplayProofFileHashEntryV1 {
    path: string;
    digest: string;
}
export interface RecordedSessionReplayProofHashesV1 {
    contract: "recorded_session_replay_hashes.v1";
    algorithm: "sha256";
    semantic: {
        traceHash: string;
        fixtureHash: string;
        scoreHash: string;
        bundleHash: string;
    };
    files: RecordedSessionReplayProofFileHashEntryV1[];
}
export interface RecordedSessionReplayProofModeOutputV1 {
    mode: RecordedSessionReplayMode;
    path: string;
    report: RecordedSessionReplayModeReportV1;
}
export interface RecordedSessionReplayProofBundleDescriptorV1 {
    rootDir: string;
    manifestPath: string;
    tracePath: string;
    fixturePath: string;
    bundlePath: string;
    environmentPath: string;
    summaryPath: string;
    summaryTablesPath: string;
    coverageSnapshotPath: string;
    hardeningSnapshotPath: string;
    hashesPath: string;
    manifest: RecordedSessionReplayProofManifestV1;
    trace: RecordedSessionTraceV1;
    fixture: RecordedSessionReplayFixtureV1;
    bundle: RecordedSessionReplayBundleV1;
    environment: RecordedSessionReplayProofEnvironmentV1;
    summaryText: string;
    summaryTables: RecordedSessionReplayProofSummaryTablesV1;
    coverageSnapshot: RecordedSessionReplayProofCoverageSnapshotV1;
    hardeningSnapshot: RecordedSessionReplayProofHardeningSnapshotV1;
    hashes: RecordedSessionReplayProofHashesV1;
    modeOutputs: RecordedSessionReplayProofModeOutputV1[];
}
export interface RecordedSessionReplayProofBundleValidationV1 {
    contract: "recorded_session_replay_proof_validation.v1";
    ok: boolean;
    rootDir: string;
    expectedFileCount: number;
    verifiedFileCount: number;
    fileHashesMatch: boolean;
    bundleHashMatches: boolean;
    scoreHashMatches: boolean;
    errors: string[];
}
export interface RecordedSessionReplayServedPackAdapterPrepareInputV1 {
    activationRoot: string;
    seedPackRoot: string;
    fixture: RecordedSessionReplayFixtureV1;
}
export interface RecordedSessionReplayServedPackAdapterV1 {
    /** @internal Replay-only served-pack adapter hook for authoritative candidate evaluation. */
    prepare(input: RecordedSessionReplayServedPackAdapterPrepareInputV1): void;
}
export interface WriteRecordedSessionReplayProofBundleInputV1 {
    rootDir: string;
    trace: RecordedSessionTraceV1;
    scratchRootDir?: string | null;
    learnedRouteSelectionOverride?: LearnedRouteSelectionOverride | null;
    learnedRouteServedPackAdapter?: RecordedSessionReplayServedPackAdapterV1 | null;
}
export interface RecordedSessionReplayOptionsV1 {
    /** @internal Replay-only learned-route selection override. */
    learnedRouteSelectionOverride?: LearnedRouteSelectionOverride | null;
    /** @internal Replay-only served-pack adapter hook for authoritative candidate evaluation. */
    learnedRouteServedPackAdapter?: RecordedSessionReplayServedPackAdapterV1 | null;
}
export declare function buildRecordedSessionReplayFixture(trace: RecordedSessionTraceV1): RecordedSessionReplayFixtureV1;
export declare function runRecordedSessionReplay(rootDir: string, fixture: RecordedSessionReplayFixtureV1, options?: RecordedSessionReplayOptionsV1): RecordedSessionReplayBundleV1;
export declare function rescoreRecordedSessionReplayBundle(bundle: RecordedSessionReplayBundleV1): RecordedSessionReplayRescoreReportV1;
export declare function verifyRecordedSessionReplayBundleHashes(bundle: RecordedSessionReplayBundleV1): RecordedSessionReplayBundleHashVerificationV1;
export declare function loadRecordedSessionReplayProofBundle(rootDir: string): RecordedSessionReplayProofBundleDescriptorV1;
export declare function validateRecordedSessionReplayProofBundle(rootDir: string): RecordedSessionReplayProofBundleValidationV1;
export declare function writeRecordedSessionReplayProofBundle(input: WriteRecordedSessionReplayProofBundleInputV1): RecordedSessionReplayProofBundleDescriptorV1;
export type OperatorSurfaceStatus = "ok" | "warn" | "fail";
export type OperatorFindingSeverity = "pass" | "warn" | "fail";
export type OperatorLastPromotionConfidence = "proven_from_previous_pointer" | "unknown_from_local_pointers" | "no_active_pack";
export declare const OPERATOR_API_CONTRACT_ID: "openclaw_operator_api.v1";
export declare const SUPPORTED_OPERATOR_API_FAMILIES: readonly ["bootstrap_attach", "status", "export", "refresh", "promote", "rollback", "proof_observability"];
export type SupportedOperatorApiFamily = (typeof SUPPORTED_OPERATOR_API_FAMILIES)[number];
export type OperatorApiRouteScope = "cli" | "programmatic" | "proof_lane";
export interface OperatorApiRouteDescriptor {
    family: SupportedOperatorApiFamily;
    scope: OperatorApiRouteScope;
    packageName: string;
    entrypoints: readonly string[];
    summary: string;
    notes: readonly string[];
}
export interface OperatorApiContractV1 {
    contract: typeof OPERATOR_API_CONTRACT_ID;
    runtimeOwner: "openclaw";
    scope: "narrow_supported_operator_surface";
    families: readonly SupportedOperatorApiFamily[];
    routes: readonly OperatorApiRouteDescriptor[];
    quarantinedSurface: readonly string[];
}
export declare const OPERATOR_API_CONTRACT_V1: {
    readonly contract: "openclaw_operator_api.v1";
    readonly runtimeOwner: "openclaw";
    readonly scope: "narrow_supported_operator_surface";
    readonly families: readonly ["bootstrap_attach", "status", "export", "refresh", "promote", "rollback", "proof_observability"];
    readonly routes: readonly [{
        readonly family: "bootstrap_attach";
        readonly scope: "programmatic";
        readonly packageName: "@openclawbrain/openclaw";
        readonly entrypoints: readonly ["bootstrapRuntimeAttach", "formatBootstrapRuntimeAttachReport", "describeAttachStatus"];
        readonly summary: "Bootstrap the first current-profile attach, print the next operator step cleanly, and prove the initial handoff state without pretending live learning has already run.";
        readonly notes: readonly ["Zero-event bootstrap is supported and stays explicit through awaiting_first_export.", "Attach serves only from activation's active slot after bootstrap completes.", "bootstrapRuntimeAttach() returns the canonical current-profile answer plus copy-paste-ready next-step output for the resolved activation root."];
    }, {
        readonly family: "status";
        readonly scope: "cli";
        readonly packageName: "@openclawbrain/openclaw";
        readonly entrypoints: readonly ["openclawbrain status", "describeCurrentProfileBrainStatus"];
        readonly summary: "Read the canonical current-profile brain-status object for the active Host/Profile/Brain/Attachment boundary.";
        readonly notes: readonly ["Status is the first operator read path.", "describeCurrentProfileBrainStatus() freezes the supported Host/Profile/Brain/Attachment answer shape for the current profile.", "Use activation and export observability proof helpers when you need candidate/previous or export-freshness detail."];
    }, {
        readonly family: "export";
        readonly scope: "programmatic";
        readonly packageName: "@openclawbrain/openclaw";
        readonly entrypoints: readonly ["buildNormalizedRuntimeEventExport", "writeRuntimeEventExportBundle", "loadRuntimeEventExportBundle"];
        readonly summary: "Emit the deterministic learner handoff artifact explicitly instead of folding export into a larger implicit runtime loop.";
        readonly notes: readonly ["Export is an off-hot-path operator handoff artifact, not proof of immediate active-pack mutation.", "Bundle roots and normalized payloads are both accepted downstream by observability surfaces."];
    }, {
        readonly family: "refresh";
        readonly scope: "programmatic";
        readonly packageName: "@openclawbrain/learner";
        readonly entrypoints: readonly ["createAlwaysOnLearningRuntimeState", "advanceAlwaysOnLearningRuntime", "materializeAlwaysOnLearningCandidatePack"];
        readonly summary: "Refresh candidate learning state explicitly through the learner boundary before any activation-pointer move happens.";
        readonly notes: readonly ["Refresh is PG-only candidate-pack materialization in this repo.", "Refresh does not mutate the currently served active pack in place."];
    }, {
        readonly family: "promote";
        readonly scope: "programmatic";
        readonly packageName: "@openclawbrain/pack-format";
        readonly entrypoints: readonly ["stageCandidatePack", "promoteCandidatePack"];
        readonly summary: "Stage and promote activation-ready candidate packs through explicit pointer changes.";
        readonly notes: readonly ["Promotion is the only path that changes which pack is served.", "Candidate and previous remain inspectable around the pointer move."];
    }, {
        readonly family: "rollback";
        readonly scope: "cli";
        readonly packageName: "@openclawbrain/openclaw";
        readonly entrypoints: readonly ["openclawbrain rollback", "rollbackRuntimeAttach", "formatOperatorRollbackReport"];
        readonly summary: "Preview and apply the explicit active<-previous / active->candidate rollback move.";
        readonly notes: readonly ["Rollback is blocked when the previous pointer is unavailable.", "Dry-run is the required first read path for safe operator rollback."];
    }, {
        readonly family: "proof_observability";
        readonly scope: "programmatic";
        readonly packageName: "@openclawbrain/openclaw";
        readonly entrypoints: readonly ["describeAttachStatus", "describeKernelBrainBoundary"];
        readonly summary: "Prove the local attach and kernel-vs-brain boundary from the shipped bridge surface.";
        readonly notes: readonly ["Use these for repo-local or installed-package operator proof reads.", "These surfaces report the promoted artifact boundary, not full live runtime plasticity."];
    }, {
        readonly family: "proof_observability";
        readonly scope: "programmatic";
        readonly packageName: "@openclawbrain/pack-format";
        readonly entrypoints: readonly ["describeActivationObservability"];
        readonly summary: "Inspect activation health, freshness, route artifacts, rollback lineage, and slot readiness.";
        readonly notes: readonly ["Activation observability is the ground truth for active/candidate/previous slot inspection."];
    }, {
        readonly family: "proof_observability";
        readonly scope: "programmatic";
        readonly packageName: "@openclawbrain/event-export";
        readonly entrypoints: readonly ["describeNormalizedEventExportObservability"];
        readonly summary: "Inspect supervision freshness and teacher freshness from the exported learner handoff artifact.";
        readonly notes: readonly ["Export observability is local-to-export proof only."];
    }, {
        readonly family: "proof_observability";
        readonly scope: "proof_lane";
        readonly packageName: "workspace";
        readonly entrypoints: readonly ["pnpm current-profile-lifecycle:smoke", "pnpm observability:smoke"];
        readonly summary: "Run the repo-local proof lanes that derive operator truth from the canonical current-profile status object plus activation observability.";
        readonly notes: readonly ["These lanes are proof machinery, not a second semver-stable API."];
    }];
    readonly quarantinedSurface: readonly ["openclawbrain-ops doctor was deleted; use the canonical current-profile status object plus proof helpers instead of a parallel troubleshooting surface.", "buildOperatorSurfaceReport / formatOperatorStatusReport / formatOperatorDoctorReport were historical parallel status surfaces and are not the supported operator API.", "runContinuousProductLoopTurn collapses export/refresh/promote into one proof helper and is not the supported operator API.", "runRecordedSessionReplay and recorded-session fixtures are proof helpers, not operator API.", "release scripts, root smoke plumbing, and workspace layout are proof-and-build machinery, not operator API.", "runRuntimeTurn is a runtime convenience wrapper and not the narrow operator export contract.", "createAsyncTeacherLiveLoop is supporting internals for refresh/teacher snapshots, not the narrow operator contract."];
};
export declare const OPENCLAW_OPERATOR_NOUNS_V1: readonly ["Host", "Profile", "Brain", "Attachment"];
export declare const CURRENT_PROFILE_BRAIN_STATUS_CONTRACT: "current_profile_brain_status.v1";
export declare const BRAIN_ATTACHMENT_POLICY_SEMANTICS_V1: {
    readonly undeclared: "The Host has not declared whether the current Profile's Brain attachment policy is shared or dedicated; do not infer profile exclusivity from activation state alone.";
    readonly dedicated: "The Host declares a dedicated Brain attachment policy: one Profile is intentionally attached to one Brain activation root, and operators may treat the served Brain state as profile-specific until the attachment changes.";
    readonly shared: "The Host declares a shared Brain attachment policy: multiple Profiles may intentionally attach to the same Brain activation root, attribution must stay current-profile explicit, and operators must not treat later served context as profile-exclusive.";
};
export type OperatorBrainState = InitHandoffState | "no_active_pack";
export type OperatorServePathState = "serving_active_pack" | "fail_open_static_context" | "hard_fail" | "unprobed";
export interface OperatorSurfaceInput {
    activationRoot: string;
    openclawHome?: string | null;
    updatedAt?: string | null;
    eventExportPath?: string | null;
    teacherSnapshotPath?: string | null;
    brainAttachmentPolicy?: RuntimeTurnBrainAttachmentPolicyV1 | null;
}
export interface OperatorSurfaceSlotSummary {
    slot: ActivationPointerSlot;
    packId: string;
    activationReady: boolean;
    routePolicy: RuntimeCompileTargetV1["routePolicy"];
    routerIdentity: string | null;
    workspaceSnapshot: string;
    workspaceRevision: string | null;
    eventRange: RuntimeCompileTargetV1["eventRange"];
    eventExportDigest: string | null;
    builtAt: string;
    updatedAt: string | null;
    findings: string[];
}
export interface OperatorLastPromotionSummary {
    known: boolean;
    at: string | null;
    confidence: OperatorLastPromotionConfidence;
    note: string;
}
export interface OperatorBrainStateSummary {
    state: OperatorBrainState;
    initMode: LearningBootProfile | null;
    runtimePlasticitySource: RuntimePlasticitySourceV1 | null;
    seedStateVisible: boolean;
    seedBlockCount: number;
    activePackId: string | null;
    activeWorkspaceSnapshot: string | null;
    activeEventExportDigest: string | null;
    detail: string;
}
export interface OperatorActivationStateSummary {
    state: CurrentProfileActivationStateV1;
    detail: string;
    inspectionError: string | null;
}
export interface OperatorGraphSummary {
    available: boolean;
    runtimePlasticitySource: RuntimePlasticitySourceV1 | null;
    structuralOps: GraphEvolutionLogV1["structuralOps"] | null;
    connectDiagnostics: PackGraphConnectDiagnosticsV1 | null;
    changed: boolean | null;
    blockCount: number | null;
    operationsApplied: GraphEvolutionLogV1["structuralEvolutionSummary"]["operationsApplied"];
    liveBlockCount: number | null;
    prunedBlockCount: number | null;
    prePruneBlockCount: number | null;
    strongestBlockId: string | null;
    operatorSummary: string | null;
    latestMaterialization: {
        known: boolean;
        packId: string | null;
        changed: boolean | null;
        connectDiagnostics: PackGraphConnectDiagnosticsV1 | null;
        operatorSummary: string | null;
        detail: string;
    };
    detail: string;
}
export type OperatorObservabilitySource = "active_pack" | "event_export" | "materialized_candidate" | "missing";
export type OperatorPolicyGradientVersion = "v1" | "v2" | "unavailable";
export interface OperatorLabelFlowSummary {
    source: OperatorObservabilitySource;
    humanLabelCount: number | null;
    selfLabelCount: number | null;
    asyncTeacherArtifactCount: number | null;
    implicitPositiveCount: number | null;
    detail: string;
}
export interface OperatorLearningPathSummary {
    available: boolean;
    source: OperatorObservabilitySource;
    policyGradientVersion: OperatorPolicyGradientVersion;
    policyGradientMethod: string | null;
    objective: string | null;
    targetConstruction: string | null;
    connectOpsFired: number | null;
    reconstructedTrajectoryCount: number | null;
    detail: string;
}
export interface OperatorServePathSummary {
    state: OperatorServePathState;
    fallbackToStaticContext: boolean;
    hardRequirementViolated: boolean;
    activePackId: string | null;
    usedLearnedRouteFn: boolean | null;
    routerIdentity: string | null;
    selectionMode: string | null;
    selectionDigest: string | null;
    refreshStatus: string | null;
    freshnessChecksum: string | null;
    requestedBudgetStrategy: string | null;
    resolvedBudgetStrategy: string | null;
    resolvedMaxContextBlocks: number | null;
    structuralBudgetSource: string | null;
    structuralBudgetEvidence: string | null;
    structuralBudgetPressures: string | null;
    structuralDecision: CurrentProfileStructuralDecisionV1;
    contextAttribution: ContextAttributionSummaryV1;
    timing: BrainServeHotPathTimingV1;
    error: string | null;
}
export interface OperatorDoctorFinding {
    severity: OperatorFindingSeverity;
    code: string;
    summary: string;
    detail: string;
}
export interface OperatorSupervisionSummary {
    available: boolean;
    sourcePath: string | null;
    sourceKind: "bundle_root" | "payload" | "missing";
    exportDigest: string | null;
    exportedAt: string | null;
    flowing: boolean | null;
    scanPolicy: LearningScanPolicy | null;
    scanSurfaceCount: number;
    scanSurfaces: string[];
    sourceCount: number;
    freshestSourceStream: string | null;
    freshestCreatedAt: string | null;
    freshestKind: string | null;
    humanLabelCount: number | null;
    selfLabelCount: number | null;
    attributedEventCount: number | null;
    totalEventCount: number | null;
    selectionDigestCount: number | null;
    sources: string[];
    detail: string;
}
export interface OperatorTeacherLoopSummary {
    available: boolean;
    sourcePath: string | null;
    sourceKind: "watch_snapshot" | "async_snapshot" | "missing";
    snapshotUpdatedAt: string | null;
    lastRunAt: string | null;
    lastNoOpReason: TeacherLoopNoOpReason | "unavailable";
    latestFreshness: AsyncTeacherLiveLoopDiagnosticsV1["latestFreshness"] | "unavailable";
    startedAt: string | null;
    lastHeartbeatAt: string | null;
    lastScanAt: string | null;
    pollIntervalSeconds: number | null;
    watchState: OperatorPassiveLearningWatchState;
    watch: OperatorPassiveLearningWatchSummary;
    lastProcessedAt: string | null;
    artifactCount: number | null;
    queueDepth: number | null;
    queueCapacity: number | null;
    running: boolean | null;
    replayedBundleCount: number | null;
    replayedEventCount: number | null;
    exportedBundleCount: number | null;
    exportedEventCount: number | null;
    sessionTailSessionsTracked: number | null;
    sessionTailBridgedEventCount: number | null;
    localSessionTailNoopReason: string | null;
    learningCadence: LearningCadence | "unavailable";
    scanPolicy: LearningScanPolicy | "unavailable";
    liveSlicesPerCycle: number | null;
    backfillSlicesPerCycle: number | null;
    failureMode: WatchTeacherSnapshotFailureV1["mode"] | "none" | "unavailable";
    failureDetail: string | null;
    lastAppliedMaterializationJobId: string | null;
    lastMaterializedPackId: string | null;
    lastObservedDelta: CurrentProfilePassiveLearningDeltaSummaryV1;
    notes: string[];
    detail: string;
}
export type OperatorLearningBacklogState = "unavailable" | "awaiting_first_export" | "principal_live_priority" | "principal_backfill_priority" | "live_priority" | "backfill_only" | "caught_up";
export type OperatorLearningWarningState = "teacher_snapshot_unavailable" | "awaiting_first_export" | "principal_live_backlog" | "principal_backfill_pending" | "active_pack_behind_latest_principal" | "passive_backfill_pending" | "teacher_queue_full" | "teacher_labels_stale" | "teacher_no_artifacts";
export interface OperatorLearningSummary {
    available: boolean;
    sourcePath: string | null;
    bootstrapped: boolean | null;
    mode: AlwaysOnLearningRuntimePlanV1["mode"] | "unavailable";
    nextPriorityLane: AlwaysOnLearningRuntimePlanV1["nextPriorityLane"] | "unavailable";
    nextPriorityBucket: AlwaysOnLearningRuntimePlanV1["nextPriorityBucket"] | "unavailable";
    backlogState: OperatorLearningBacklogState;
    pendingLive: number | null;
    pendingBackfill: number | null;
    pendingTotal: number | null;
    pendingByBucket: AlwaysOnLearningRuntimePlanV1["pending"]["byBucket"] | null;
    freshLivePriority: boolean | null;
    principalCheckpointCount: number | null;
    pendingPrincipalCount: number | null;
    oldestUnlearnedPrincipalEvent: PendingPrincipalEventV1 | null;
    newestPendingPrincipalEvent: PendingPrincipalEventV1 | null;
    leadingPrincipalCheckpoint: PrincipalLearningCheckpointV1 | null;
    principalCheckpoints: PrincipalLearningCheckpointV1[];
    principalLagToPromotion: {
        activeEventRangeEnd: number | null;
        latestPrincipalSequence: number | null;
        sequenceLag: number | null;
        status: "unavailable" | "caught_up" | "pending_promotion";
    };
    warningStates: OperatorLearningWarningState[];
    learnedRange: RuntimeCompileTargetV1["eventRange"] | null;
    materializationCount: number | null;
    lastMaterializedAt: string | null;
    lastMaterializationReason: AlwaysOnLearningMaterializationJobV1["reason"] | null;
    lastMaterializationLane: AlwaysOnLearningMaterializationJobV1["lane"] | null;
    lastMaterializationPriority: AlwaysOnLearningMaterializationJobV1["priority"] | null;
    lastMaterializedPackId: string | null;
    detail: string;
}
export interface OperatorPrincipalItemSummary {
    eventId: string;
    contract: NormalizedEventV1["contract"];
    kind: NormalizedEventV1["kind"];
    sequence: number;
    createdAt: string;
    teacherIdentity: string;
    teacherAuthority: NonNullable<NormalizedEventV1["principal"]>["teacherAuthority"];
    priorityClass: NonNullable<NormalizedEventV1["principal"]>["priorityClass"];
    scopeKind: NonNullable<NormalizedEventV1["principal"]>["principalScope"]["kind"];
    supersedes: string[];
}
export interface OperatorPrincipalFeedbackSummary extends OperatorPrincipalItemSummary {
    content: string;
    relatedInteractionId: string | null;
}
export interface OperatorPrincipalPromotionSummary {
    known: boolean;
    at: string | null;
    activePackId: string | null;
    activeEventRangeEnd: number | null;
    includesLatestFeedback: boolean | null;
    includesLatestCorrection: boolean | null;
    note: string;
}
export interface OperatorPrincipalObservabilitySummary {
    available: boolean;
    sourcePath: string | null;
    sourceKind: "bundle_root" | "payload" | "missing";
    latestFeedback: OperatorPrincipalFeedbackSummary | null;
    latestCorrection: OperatorPrincipalFeedbackSummary | null;
    pendingCount: number | null;
    pendingItems: OperatorPrincipalItemSummary[];
    latestPromotion: OperatorPrincipalPromotionSummary;
    servingDownstreamOfLatestCorrection: boolean | null;
    detail: string;
}
export interface OperatorRouteFnFreshnessSummary {
    available: boolean;
    activePackId: string | null;
    routerIdentity: string | null;
    routerChecksum: string | null;
    trainedAt: string | null;
    updatedAt: string | null;
    usedAt: string | null;
    lastDecisionAt: string | null;
    lastDecisionUsedLearnedRouteFn: boolean | null;
    detail: string;
}
interface OperatorHookSummary {
    scope: "exact_openclaw_home" | "activation_root_only";
    openclawHome: string | null;
    extensionDir: string | null;
    hookPath: string | null;
    runtimeGuardPath: string | null;
    manifestPath: string | null;
    packageJsonPath: string | null;
    manifestId: string | null;
    installId: string | null;
    packageName: string | null;
    packageVersion: string | null;
    installLayout: import("./openclaw-plugin-install.js").OpenClawBrainInstallLayout | null;
    additionalInstallCount: number;
    installState: CurrentProfileHookInstallStateV1;
    loadability: CurrentProfileHookLoadabilityV1;
    loadProof: CurrentProfileHookLoadProofV1;
    guardSeverity: import("./openclaw-hook-truth.js").OpenClawBrainHookGuardSeverity;
    guardActionability: import("./openclaw-hook-truth.js").OpenClawBrainHookGuardActionability;
    guardSummary: string;
    guardAction: string;
    desynced: boolean;
    detail: string;
}
interface OperatorAttachmentTruthSummary {
    state: CurrentProfileAttachmentStateV1;
    proofState: CurrentProfileAttachmentProofStateV1;
    watchOnly: boolean;
    activationRoot: string | null;
    servingSlot: "active" | "none";
    detail: string;
}
interface OperatorManyProfileSupportSummary {
    operatorSurface: "current_profile_only";
    declaredAttachmentPolicy: RuntimeTurnBrainAttachmentPolicyV1;
    sameGatewayIntent: "undeclared" | "dedicated_current_profile_boundary" | "shared_attachment_declared";
    checkedInProofTopology: "two_local_gateways_dedicated_only";
    sameGatewayProof: false;
    sharedWriteSafetyProof: false;
    detail: string;
}
interface OperatorSurfaceReport {
    generatedAt: string;
    activationRoot: string;
    status: OperatorSurfaceStatus;
    activation: OperatorActivationStateSummary;
    active: OperatorSurfaceSlotSummary | null;
    candidate: OperatorSurfaceSlotSummary | null;
    previous: OperatorSurfaceSlotSummary | null;
    freshness: {
        activeBehindPromotionReadyCandidate: boolean;
        candidateAheadBy: string[];
    };
    brain: OperatorBrainStateSummary;
    graph: OperatorGraphSummary;
    labelFlow: OperatorLabelFlowSummary;
    learningPath: OperatorLearningPathSummary;
    learnedRouting: {
        required: boolean;
        available: boolean;
        routerIdentity: string | null;
        routeFnVersion: string | null;
        trainingMethod: string | null;
        routerTrainedAt: string | null;
        objective: string | null;
        pgProfile: ActivationObservabilityReport["learnedRouteFn"]["pgProfile"];
        routerChecksum: string | null;
        objectiveChecksum: string | null;
        updateMechanism: string | null;
        updateVersion: string | null;
        updateCount: number | null;
        supervisionCount: number | null;
        collectedLabelsTotal: number | null;
        freshnessChecksum: string | null;
        handoffState: InitHandoffState;
        initMode: LearningBootProfile | null;
        seedStateVisible: boolean;
    };
    servePath: OperatorServePathSummary;
    promotion: {
        allowed: boolean;
        findings: string[];
        lastPromotion: OperatorLastPromotionSummary;
        activeUpdatedAt: string | null;
        candidateUpdatedAt: string | null;
        previousUpdatedAt: string | null;
    };
    rollback: {
        allowed: boolean;
        findings: string[];
        previousPackId: string | null;
        state: "ready" | "blocked" | "unknown";
    };
    supervision: OperatorSupervisionSummary;
    learning: OperatorLearningSummary;
    teacherLoop: OperatorTeacherLoopSummary;
    routeFn: OperatorRouteFnFreshnessSummary;
    hook: OperatorHookSummary;
    attachmentTruth: OperatorAttachmentTruthSummary;
    principal: OperatorPrincipalObservabilitySummary;
    manyProfile: OperatorManyProfileSupportSummary;
    findings: OperatorDoctorFinding[];
}
export interface CurrentProfileBrainStatusInput extends OperatorSurfaceInput {
    brainAttachmentPolicy?: RuntimeTurnBrainAttachmentPolicyV1 | null;
    profileId?: string | null;
}
export type CurrentProfileBrainStatusV1 = CurrentProfileBrainStatusAnswerV1;
export declare function summarizeNormalizedEventExportLabelFlow(normalizedEventExport: NormalizedEventExportV1, asyncTeacherArtifactCount?: number): OperatorLabelFlowSummary;
export declare function summarizeLearningPathFromMaterialization(materialization: AlwaysOnLearningMaterializationJobV1 | null): OperatorLearningPathSummary;
export declare function buildOperatorSurfaceReport(input: OperatorSurfaceInput): OperatorSurfaceReport;
export declare function describeCurrentProfileBrainStatus(input: CurrentProfileBrainStatusInput): CurrentProfileBrainStatusV1;
export declare function formatOperatorRollbackReport(result: RollbackRuntimeAttachResult): string;
/**
 * A per-turn summary of what was kernel-injected vs brain-compiled in a given
 * compile response. Useful for operator observability and debugging.
 *
 * "Kernel" content is whatever the operator injected directly in the system
 * prompt — this function cannot observe it, only describe the brain side.
 * The `kernelSurface` field is present only when the caller supplies a surface
 * descriptor (from `WorkspaceInjectionSurfaceV1`) alongside the compile
 * response; otherwise it is null.
 */
export interface KernelBrainBoundaryDescriptionV1 {
    /**
     * Summary of what the brain compiler contributed to this turn.
     */
    brain: {
        packId: string;
        mode: "learned" | "heuristic";
        selectedBlockCount: number;
        /** Roles of the selected blocks, deduplicated. */
        selectedRoles: string[];
        /** Whether the learned route_fn ran (true means learned routing was used). */
        usedLearnedRouteFn: boolean;
        /** Whether context was compacted due to char budget pressure. */
        compactionApplied: boolean;
    };
    /**
     * Validation result for the operator-supplied kernel surface descriptor.
     * Null when no surface descriptor was supplied.
     */
    kernelValidation: KernelSurfaceValidationResultV1 | null;
    /**
     * Advisory: based on the compile diagnostics, is the brain context likely
     * covering the query well, or should the operator investigate pack freshness?
     *
     * - `"likely_covered"` — learned routing ran and found token-matched blocks.
     * - `"partial"` — some token matches, but mode was heuristic or routing gap.
     * - `"likely_gap"` — no token matches or heuristic-only; pack may need refresh.
     */
    brainCoverageAdvisory: "likely_covered" | "partial" | "likely_gap";
}
/**
 * Describes the kernel/brain boundary for a single compile response.
 *
 * Combines:
 * - Brain context summary (from the compile response diagnostics)
 * - Kernel surface validation (if a surface descriptor is supplied)
 * - A coverage advisory based on routing signals
 *
 * See `docs/kernel-brain-boundary.md` for the full decision framework.
 */
export declare function describeKernelBrainBoundary(compileResponse: RuntimeCompileResponseV1, surface?: WorkspaceInjectionSurfaceV1): KernelBrainBoundaryDescriptionV1;
export type { BrainEligibleContentKind, KernelContentKind, KernelSurfaceValidationResultV1, WorkspaceInjectionSurfaceV1 } from "@openclawbrain/contracts";
export { CONTRACT_IDS, buildNormalizedEventExport, createFeedbackEvent, createInteractionEvent, validateRuntimeCompileRequest } from "@openclawbrain/contracts";
export { describeNormalizedEventExportObservability } from "@openclawbrain/event-export";
export { describeCompileFallbackUsage } from "@openclawbrain/compiler";
export { describeActivationObservability, inspectActivationState, rollbackActivePack } from "@openclawbrain/pack-format";
export { createOpenClawLocalSessionTail, OpenClawLocalSessionTail, type OpenClawLocalSessionTailChangeKindV1, type OpenClawLocalSessionTailChangeV1, type OpenClawLocalSessionTailCursorV1, type OpenClawLocalSessionTailInput, type OpenClawLocalSessionTailLoopOptionsV1, type OpenClawLocalSessionTailLoopResultV1, type OpenClawLocalSessionTailNoopReasonV1, type OpenClawLocalSessionTailPollResultV1 } from "./session-tail.js";
export { discoverOpenClawMainSessionStores, discoverOpenClawSessionStores, loadOpenClawSessionIndex, readOpenClawAcpStreamFile, readOpenClawSessionFile, type OpenClawMainSessionStoreV1, type OpenClawSessionStoreV1, type OpenClawAcpStreamRecord, type OpenClawInjectedWorkspaceFile, type OpenClawSessionContentPart, type OpenClawSessionCustomRecord, type OpenClawSessionHeaderRecord, type OpenClawSessionIndex, type OpenClawSessionIndexEntry, type OpenClawSessionMessagePayload, type OpenClawSessionMessageRecord, type OpenClawSessionModelChangeRecord, type OpenClawSessionRecord, type OpenClawSessionTextPart, type OpenClawSessionThinkingLevelChangeRecord, type OpenClawSessionThinkingPart, type OpenClawSessionToolCallPart, type OpenClawSystemPromptReport, type OpenClawToolSurfaceEntry } from "./session-store.js";
export { buildPassiveLearningSessionExportFromOpenClawSessionStore, buildPassiveLearningStoreExportFromOpenClawSessionIndex, type OpenClawPassiveLearningPrivacySummaryV1, type OpenClawPassiveLearningSessionEvidenceV1, type OpenClawPassiveLearningSessionExportV1, type OpenClawPassiveLearningStoreExportV1 } from "./local-session-passive-learning.js";
export { DEFAULT_OLLAMA_BASE_URL, DEFAULT_OLLAMA_TIMEOUT_MS, OllamaClient, OllamaClientError, createOllamaClient, type OllamaChatMessage, type OllamaClientOptions, type OllamaFetch, type OllamaFetchRequestInit, type OllamaFetchResponse, type OllamaRequestOptions } from "./ollama-client.js";
export { resolveActivationRoot, type ResolveActivationRootOptions } from "./resolve-activation-root.js";
export { runDaemonCommand, type DaemonCliArgs, type DaemonSubcommand, parseDaemonArgs } from "./daemon.js";
