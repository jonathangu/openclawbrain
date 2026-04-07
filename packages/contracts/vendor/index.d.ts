export declare const CONTRACT_IDS: {
    readonly activationPointers: "activation_pointers.v1";
    readonly artifactManifest: "artifact_manifest.v1";
    readonly brainAttachmentPolicy: "brain_attachment_policy.v1";
    readonly currentProfileBrainStatus: "current_profile_brain_status.v1";
    readonly feedbackEvents: "feedback_events.v1";
    readonly interactionEvents: "interaction_events.v1";
    readonly profileTurnAttribution: "profile_turn_attribution.v1";
    readonly runtimeCompile: "runtime_compile.v1";
    readonly teacherSupervisionArtifact: "teacher_supervision_artifact.v1";
};
export type ContractId = (typeof CONTRACT_IDS)[keyof typeof CONTRACT_IDS];
export type EventContractId = typeof CONTRACT_IDS.interactionEvents | typeof CONTRACT_IDS.feedbackEvents;
export type RouteMode = "heuristic" | "learned";
export type RoutePolicy = "heuristic_allowed" | "requires_learned_routing";
export type RouterRefreshStatus = "updated" | "no_supervision";
export type RouterTrainingMethodV1 = "policy_gradient_v1" | "policy_gradient_v2";
export type RouterAssetKind = "none" | "stub" | "artifact";
export type ActivationPointerSlot = "active" | "candidate" | "previous";
export type InteractionEventKind = "memory_compiled" | "message_delivered" | "operator_override";
export type FeedbackEventKind = "correction" | "teaching" | "approval" | "suppression";
/**
 * Event semantic types capture source-facing event lineage for exports and pack materialization.
 * They do not, by themselves, prove whether later retrieval should treat the block as
 * answer-bearing content or as instructional/meta-teaching scaffolding.
 */
export type EventSemanticTypeV1 = "memory_candidate" | "teacher_signal" | "instructional_scaffolding" | "control_signal" | "delivery_residue" | "observability_residue";
export type RetrievalSemanticClassV1 = "answer_bearing" | "instructional_scaffolding" | "observability_support" | "transport_residue";
export type RetrievalAnswerRoleV1 = "answer_bearing" | "non_answer_bearing";
export type RetrievalInstructionRoleV1 = "non_scaffolding" | "instructional_scaffolding";
export type EventSourceKindV1 = "runtime_turn" | "session_store" | "scanner_export" | "recorded_session_seed";
export type EventDiagnosticIntentV1 = "compile_observability" | "delivery_observability";
export type RouterSupervisionKind = "route_trace" | "human_feedback" | "operator_override" | "self_memory" | "teacher_supervision";
export type TeacherSupervisionKind = FeedbackEventKind | "operator_override";
export type TeacherSupervisionFreshnessStatus = "fresh" | "stale";
export type PrincipalRoleV1 = "principal" | "admin" | "operator" | "user" | "assistant" | "system";
export type TeacherAuthorityV1 = "binding" | "primary_human" | "high" | "normal" | "background";
export type PrincipalPriorityClassV1 = "critical" | "high" | "normal" | "low";
export type PrincipalScopeKindV1 = "global" | "profile" | "session" | "interaction" | "message";
export type LearningBootProfile = "fast_boot_defaults";
export type LearningCadence = "passive_background";
export type LearningScanPolicy = "always_on";
export type RuntimePlasticitySourceV1 = "candidate_build" | "live_loop";
export type LearningBlockRole = "boot_default" | "background_expectation" | "label_surface" | "workspace" | "structural" | "interaction" | "feedback" | "teacher_supervision";
export declare const PACK_GRAPH_SCHEMAS: {
    readonly openclawInit: "openclaw_init_graph.v1";
};
export type PackGraphSchemaV1 = (typeof PACK_GRAPH_SCHEMAS)[keyof typeof PACK_GRAPH_SCHEMAS];
export type OpenClawMarkdownFileRoleV1 = "repo_boundary" | "claims_boundary" | "contracts_reference" | "glossary" | "learning_policy" | "agent_sop" | "attach_quickstart" | "integration_guide" | "operator_guide" | "operator_observability" | "ops_recipe" | "session_replay_proof" | "release_guide" | "evaluation_reproduction" | "setup_guide" | "worked_example";
export type OpenClawMarkdownAudienceV1 = "runtime" | "integrator" | "operator" | "proof";
export type OpenClawMarkdownTierV1 = "core" | "supporting";
export type OpenClawInitNodeKindV1 = "markdown_ontology" | "product_invariant" | "workspace_snapshot" | "event_export" | "event" | "teacher_supervision" | "synthetic_topology";
export type OpenClawInitSourceKindV1 = "markdown" | "product" | "workspace" | "event_export" | "event" | "teacher_supervision" | "synthetic";
export type OpenClawInitHeuristicScopeV1 = "init_priors_and_topology_only";
export type OpenClawLearnedLabelPolicyV1 = "explicit_collected_labels_only";
export interface OpenClawMarkdownFileRoleBindingV1 {
    path: string;
    role: OpenClawMarkdownFileRoleV1;
    audience: OpenClawMarkdownAudienceV1;
    tier: OpenClawMarkdownTierV1;
}
export interface OpenClawInitGraphOntologyV1 {
    schema: typeof PACK_GRAPH_SCHEMAS.openclawInit;
    typedMarkdownSurface: true;
    fileRoles: OpenClawMarkdownFileRoleBindingV1[];
    fastBootRequired: true;
    passiveBackgroundLearningRequired: true;
    heuristicScope: OpenClawInitHeuristicScopeV1;
    learnedLabelPolicy: OpenClawLearnedLabelPolicyV1;
}
export interface OpenClawInitBlockMetadataV1 {
    nodeKind: OpenClawInitNodeKindV1;
    sourceKind: OpenClawInitSourceKindV1;
    fastBootRequired: true;
    passiveBackgroundLearningRequired: true;
    heuristicScope: OpenClawInitHeuristicScopeV1;
    learnedLabelPolicy: OpenClawLearnedLabelPolicyV1;
    fileRole?: OpenClawMarkdownFileRoleBindingV1;
}
export interface LearningLabelSourcesV1 {
    human: string[];
    self: string[];
}
export interface LearningLabelHarvestV1 {
    humanLabels: number;
    selfLabels: number;
    corrections: number;
    teachings: number;
    approvals: number;
    suppressions: number;
    operatorOverrideLabels: number;
    memoryCompileLabels: number;
}
export interface PrincipalScopeV1 {
    kind: PrincipalScopeKindV1;
    profileSelector?: RuntimeTurnProfileSelectorV1;
    sessionId?: string;
    interactionId?: string;
    messageId?: string;
    scopeKey?: string;
}
export interface PrincipalMetadataV1 {
    teacherIdentity: string;
    teacherRole: PrincipalRoleV1;
    teacherAuthority: TeacherAuthorityV1;
    principalScope: PrincipalScopeV1;
    priorityClass: PrincipalPriorityClassV1;
    supersedes?: string[];
}
export interface LearningPrincipalSummaryV1 {
    teacherIdentities: string[];
    teacherRoles: PrincipalRoleV1[];
    teacherAuthorities: TeacherAuthorityV1[];
    priorityClasses: PrincipalPriorityClassV1[];
    scopedEventCount: number;
    supersedingEventCount: number;
}
export interface LearningSurfaceV1 {
    bootProfile: LearningBootProfile;
    learningCadence: LearningCadence;
    scanPolicy: LearningScanPolicy;
    scanSurfaces: string[];
    labelSources: LearningLabelSourcesV1;
    labelHarvest: LearningLabelHarvestV1;
    principalSummary: LearningPrincipalSummaryV1;
}
export interface PackBlockLearningSignalsV1 {
    role: LearningBlockRole;
    humanLabels: number;
    selfLabels: number;
    decayHalfLifeDays: number | null;
    hebbianPulse: number;
}
export type RoutingChannelV1 = "graph" | "short_term" | "vector";
export type InitRouteSeedModeV1 = "heuristic_seed_v1";
export type InitFileRoleV1 = "anchor" | "working_set" | "pointer_index" | "recent_memory" | "archived_memory" | "correction_log" | "reference" | "workspace" | "event_stream" | "synthetic";
export type InitNodeTypeV1 = "file" | "section" | "task" | "rule" | "person" | "project" | "pointer" | "event" | "entity";
export interface PackBlockInitScoreBreakdownV1 {
    role: number;
    authority: number;
    recency: number;
    activeTaskOverlap: number;
    pointerCentrality: number;
    correctionDensity: number;
    entityOverlap: number;
    staleness: number;
    total: number;
}
export interface PackBlockInitSignalsV1 {
    mode: InitRouteSeedModeV1;
    nodeType: InitNodeTypeV1;
    fileRole: InitFileRoleV1;
    seededChannels: RoutingChannelV1[];
    score: number;
    scoreBreakdown: PackBlockInitScoreBreakdownV1;
}
export interface PackBlockRoutingHintsV1 {
    channels: RoutingChannelV1[];
    graphBias?: number;
    shortTermBias?: number;
    vectorBias?: number;
    backgroundLabelAmplification?: number;
}
export interface RoutingChannelScoreSummaryV1 {
    graph: number;
    shortTerm: number;
    vector: number;
}
export interface SparseFeedbackMaskV1 {
    correction: boolean;
    teaching: boolean;
    approval: boolean;
    suppression: boolean;
}
export interface SparseFeedbackPolicyV1 {
    teacherBudget: number;
    teacherDelayMs: number;
    feedbackMask: SparseFeedbackMaskV1;
    backgroundLabelAmplification: number;
}
export type PackGraphEdgeKindV1 = "split" | "merge" | "connect" | "feedback";
export interface PackGraphEdgeV1 {
    targetBlockId: string;
    kind: PackGraphEdgeKindV1;
    weight: number;
}
export interface PackBlockStateV1 {
    strength: number;
    freshness: number;
    traversalBias: number;
    evidenceCount: number;
    splitDepth: number;
    mergedFromCount: number;
    pruned: boolean;
}
export interface PackGraphConnectDiagnosticsV1 {
    requestedBudget: number;
    scoreThreshold: number;
    candidatePairCount: number;
    appliedPairCount: number;
    createdEdgeCount: number;
}
export interface PackGraphEvolutionV1 {
    builtAt: string;
    hebbianApplied: boolean;
    decayApplied: boolean;
    structuralOps: {
        split: number;
        merge: number;
        prune: number;
        connect: number;
    };
    connectDiagnostics?: PackGraphConnectDiagnosticsV1;
    prunedBlockIds: string[];
    strongestBlockId: string | null;
}
export interface RuntimeGraphPlasticityStateV1 {
    source: RuntimePlasticitySourceV1;
    graphChecksum: string;
    builtAt: string;
    sourcePackId: string | null;
    blockCount: number;
    strongestBlockId: string | null;
    eventRange: Pick<NormalizedEventExportV1["range"], "start" | "end" | "count"> | null;
    eventExportDigest: string | null;
    evolution: PackGraphEvolutionV1 | null;
}
export type ContextCompactionMode = "none" | "native";
export interface RuntimeCompileRequestV1 {
    contract: typeof CONTRACT_IDS.runtimeCompile;
    agentId: string;
    userMessage: string;
    maxContextBlocks: number;
    maxContextChars?: number;
    modeRequested: RouteMode;
    activePackId?: string;
    runtimeHints?: string[];
    compactionMode?: ContextCompactionMode;
}
export interface RuntimeContextBlockV1 {
    id: string;
    source: string;
    text: string;
    tokenCount?: number;
    compactedFrom?: string[];
}
export interface PackContextBlockRecordV1 extends RuntimeContextBlockV1 {
    keywords: string[];
    priority: number;
    learning: PackBlockLearningSignalsV1;
    semantic?: EventSemanticMetadataV1;
    init?: OpenClawInitBlockMetadataV1;
    initSeed?: PackBlockInitSignalsV1;
    routing?: PackBlockRoutingHintsV1;
    state?: PackBlockStateV1;
    edges?: PackGraphEdgeV1[];
}
export interface RouteArtifactReferenceV1 {
    assetKind: RouterAssetKind;
    routerIdentity: string | null;
    routerChecksum: string | null;
    routeFnVersion: RouterArtifactV1["strategy"] | null;
    trainingMethod: RouterArtifactV1["training"]["method"] | null;
    trainedAt: string | null;
    eventExportDigest: string | null;
    updateCount: number | null;
    objective: RouterArtifactV1["training"]["objective"]["objective"] | null;
    objectiveChecksum: string | null;
    freshnessChecksum: string | null;
}
export interface ServedArtifactProofV1 {
    packId: string;
    routePolicy: RoutePolicy;
    workspaceSnapshot: string;
    workspaceRevision: string | null;
    eventRange: Pick<NormalizedEventRangeV1, "start" | "end" | "count">;
    eventExportDigest: string | null;
    builtAt: string;
    routeArtifact: RouteArtifactReferenceV1;
}
export interface RuntimeCompileStructuralCandidateSignalV1 {
    blockId: string;
    rank: number;
    score: number;
    selected: boolean;
    selectedBy: "token_match" | "priority_fallback" | null;
    matchedTokens: string[];
    directMatchedTokens: string[];
    traversalActivated: boolean;
    traversalScore: number;
    overlapPruned: boolean;
    compactedFrom: string[];
}
export interface RuntimeCompileStructuralSignalsV1 {
    matchedCandidateCount: number;
    selectedMatchedCount: number;
    selectedPriorityFallbackCount: number;
    overlapPrunedCount: number;
    traversalActivatedCount: number;
    selectedBlockIds: string[];
    overlapPrunedBlockIds: string[];
    traversalActivatedBlockIds: string[];
    candidates: RuntimeCompileStructuralCandidateSignalV1[];
}
export interface RuntimeCompileDiagnosticsV1 {
    modeRequested: RouteMode;
    modeEffective: RouteMode;
    usedLearnedRouteFn: boolean;
    routerIdentity: string | null;
    servedArtifact: ServedArtifactProofV1;
    candidateCount: number;
    selectedCount: number;
    selectedCharCount: number;
    selectedTokenCount: number;
    selectionStrategy: "pack_route_fn_selection_v1";
    selectionDigest: string;
    structuralSignals: RuntimeCompileStructuralSignalsV1;
    compactionMode: ContextCompactionMode;
    compactionApplied: boolean;
    routingChannels: {
        candidates: RoutingChannelScoreSummaryV1;
        selected: RoutingChannelScoreSummaryV1;
    };
    notes: string[];
}
export interface RuntimeCompileResponseV1 {
    contract: typeof CONTRACT_IDS.runtimeCompile;
    packId: string;
    selectedContext: RuntimeContextBlockV1[];
    diagnostics: RuntimeCompileDiagnosticsV1;
}
export type BrainAttachmentModeV1 = "dedicated" | "shared";
export type BrainAttachmentReadScopeV1 = "current_profile_only" | "attached_profiles";
export type BrainAttachmentWriteScopeV1 = "current_profile_only" | "attached_profiles";
export type CurrentProfileBrainStateV1 = "missing" | "seed_state_authoritative" | "pg_promoted_pack_authoritative" | "no_active_pack";
export type CurrentProfileServeStateV1 = "serving_active_pack" | "fail_open_static_context" | "hard_fail" | "unprobed";
export type CurrentProfileAttachmentStateV1 = "attached" | "not_attached" | "unknown";
export type CurrentProfileAttachmentProofStateV1 = "self_proving" | "activation_root_only";
export type CurrentProfileBrainStatusLevelV1 = "ok" | "warn" | "fail";
export type CurrentProfileActivationStateV1 = "healthy_seed" | "awaiting_first_export" | "active_promoted" | "stale_incomplete" | "broken_install" | "detached";
export type CurrentProfileHookScopeV1 = "exact_openclaw_home" | "activation_root_only";
export type CurrentProfileHookInstallStateV1 = "installed" | "not_installed" | "blocked_by_allowlist" | "unverified";
export type CurrentProfileHookLoadabilityV1 = "loadable" | "blocked" | "not_installed" | "unverified";
export type CurrentProfileHookLoadProofV1 = "status_probe_ready" | "not_ready";
export type TurnAttributionEvidenceV1 = "route_fn_and_brain_context" | "brain_context_only" | "route_fn_only" | "stable_kernel_only" | "fail_open_static_context" | "hard_fail" | "unprobed";
export interface DedicatedBrainAttachmentPolicySemanticsV1 {
    mode: "dedicated";
    readScope: "current_profile_only";
    writeScope: "current_profile_only";
    currentProfileExclusive: true;
    requiresProfileAttribution: true;
    detail: string;
}
export interface SharedBrainAttachmentPolicySemanticsV1 {
    mode: "shared";
    readScope: "attached_profiles";
    writeScope: "attached_profiles";
    currentProfileExclusive: false;
    requiresProfileAttribution: true;
    detail: string;
}
export type BrainAttachmentPolicySemanticsV1 = DedicatedBrainAttachmentPolicySemanticsV1 | SharedBrainAttachmentPolicySemanticsV1;
export interface BrainAttachmentPolicyV1 {
    contract: typeof CONTRACT_IDS.brainAttachmentPolicy;
    policy: BrainAttachmentPolicySemanticsV1;
}
export interface ProfileTurnAttributionV1 {
    contract: typeof CONTRACT_IDS.profileTurnAttribution;
    hostRuntimeOwner: "openclaw";
    profileSelector: RuntimeTurnProfileSelectorV1;
    profileId: string | null;
    brainAttachmentPolicy: RuntimeTurnBrainAttachmentPolicyV1;
    brainStatus: RuntimeTurnBrainStatusV1;
    sessionId: string;
    channel: string;
    interactionEventId: string;
    createdAt: string;
    packId: string | null;
    routerIdentity: string | null;
    usedLearnedRouteFn: boolean | null;
    selectionMode: string | null;
    selectionTiers: string | null;
    selectionDigest: string | null;
    contextFingerprint: RuntimeContextFingerprintV1;
    selectedContextCount: number;
    stableKernelBlockCount: number;
    brainCompiledBlockCount: number;
    stableKernelSources: string[];
    brainCompiledSources: string[];
    contextEvidence: TurnAttributionEvidenceV1;
    detail: string;
}
export interface CurrentProfileHostV1 {
    noun: "Host";
    runtimeOwner: "openclaw";
    activationRoot: string;
}
export interface CurrentProfileProfileV1 {
    noun: "Profile";
    selector: "current_profile";
    profileId: string | null;
    detail: string;
}
export interface CurrentProfileBrainV1 {
    noun: "Brain";
    activationRoot: string | null;
    logRoot: string | null;
    activePackId: string | null;
    initMode: LearningBootProfile | null;
    state: CurrentProfileBrainStateV1;
    routeFreshness: RouterRefreshStatus | "unknown";
    routerIdentity: string | null;
    routerChecksum: string | null;
    lastExportAt: string | null;
    lastLearningUpdateAt: string | null;
    lastPromotionAt: string | null;
    summary: string;
    detail: string;
}
export interface CurrentProfileHookV1 {
    noun: "Hook";
    scope: CurrentProfileHookScopeV1;
    openclawHome: string | null;
    hookPath: string | null;
    runtimeGuardPath: string | null;
    manifestPath: string | null;
    installState: CurrentProfileHookInstallStateV1;
    loadability: CurrentProfileHookLoadabilityV1;
    loadProof: CurrentProfileHookLoadProofV1;
    desynced: boolean;
    detail: string;
}
export interface CurrentProfileAttachmentV1 {
    noun: "Attachment";
    state: CurrentProfileAttachmentStateV1;
    activationRoot: string | null;
    servingSlot: "active" | "none";
    policyMode: RuntimeTurnBrainAttachmentPolicyV1;
    policy: BrainAttachmentPolicySemanticsV1 | null;
    proofState: CurrentProfileAttachmentProofStateV1;
    watchOnly: boolean;
    detail: string;
}
export type StructuralBudgetStrategyV1 = "fixed_v1" | "empirical_v1";
export type StructuralDecisionOriginV1 = "manual_caller_shape" | "empirical_control" | "default_path_control" | "unknown";
export type StructuralDecisionBasisV1 = "caller_override" | "compile_structural_signals" | "graph_evolution" | "fixed_default" | "fixed_fallback" | "no_evidence_fallback" | "no_compile_signal_evidence_fallback" | "unknown";
export interface CurrentProfileStructuralDecisionV1 {
    origin: StructuralDecisionOriginV1;
    basis: StructuralDecisionBasisV1;
    requestedBudgetStrategy: StructuralBudgetStrategyV1 | null;
    resolvedBudgetStrategy: StructuralBudgetStrategyV1 | null;
    resolvedMaxContextBlocks: number | null;
    detail: string;
}
export interface BrainServeHotPathTimingV1 {
    scope: "brain_serve_hot_path_only";
    totalMs: number | null;
    routeSelectionMs: number | null;
    promptAssemblyMs: number | null;
    otherMs: number | null;
    backgroundWorkIncluded: false;
    detail: string;
}
export interface CurrentProfileBrainStatusSummaryV1 {
    status: CurrentProfileBrainStatusLevelV1;
    brainState: CurrentProfileBrainStateV1;
    serveState: CurrentProfileServeStateV1;
    activationState: CurrentProfileActivationStateV1;
    usedLearnedRouteFn: boolean | null;
    failOpen: boolean;
    awaitingFirstExport: boolean;
    structuralDecision: CurrentProfileStructuralDecisionV1;
    timing: BrainServeHotPathTimingV1;
    detail: string;
}
export type CurrentProfilePassiveLearningWatchStateV1 = "watching" | "snapshot_only" | "stale_snapshot" | "not_visible";
export type CurrentProfilePassiveLearningExportStateV1 = "awaiting_first_export" | "latest_export_visible" | "history_only";
export type CurrentProfilePassiveLearningBacklogStateV1 = "unknown" | "awaiting_first_export" | "principal_live_priority" | "principal_backfill_priority" | "live_priority" | "backfill_only" | "caught_up";
export type CurrentProfilePassiveLearningDeltaTransitionKindV1 = "staged_candidate" | "promoted_active";
export interface CurrentProfilePassiveLearningDeltaPackTransitionV1 {
    kind: CurrentProfilePassiveLearningDeltaTransitionKindV1;
    fromPackId: string | null;
    toPackId: string;
}
export interface CurrentProfilePassiveLearningDeltaSummaryV1 {
    available: boolean;
    observedAt: string | null;
    exported: boolean | null;
    labeled: boolean | null;
    promoted: boolean | null;
    served: boolean | null;
    latestPackTransition: CurrentProfilePassiveLearningDeltaPackTransitionV1 | null;
    explanation: string;
}
export interface CurrentProfilePassiveLearningSummaryV1 {
    learnerRunning: boolean;
    firstExportOccurred: boolean;
    watchState: CurrentProfilePassiveLearningWatchStateV1;
    exportState: CurrentProfilePassiveLearningExportStateV1;
    backlogState: CurrentProfilePassiveLearningBacklogStateV1;
    pendingLive: number | null;
    pendingBackfill: number | null;
    lastWatchHeartbeatAt: string | null;
    watchIntervalSeconds: number | null;
    lastExportAt: string | null;
    lastPromotionAt: string | null;
    currentServingPackId: string | null;
    lastMaterializedPackId: string | null;
    lastObservedDelta: CurrentProfilePassiveLearningDeltaSummaryV1;
    detail: string;
}
export interface CurrentProfileBrainStatusAnswerV1 {
    contract: typeof CONTRACT_IDS.currentProfileBrainStatus;
    generatedAt: string;
    host: CurrentProfileHostV1;
    profile: CurrentProfileProfileV1;
    brain: CurrentProfileBrainV1;
    hook: CurrentProfileHookV1;
    attachment: CurrentProfileAttachmentV1;
    brainStatus: CurrentProfileBrainStatusSummaryV1;
    passiveLearning: CurrentProfilePassiveLearningSummaryV1;
    currentTurnAttribution: ProfileTurnAttributionV1 | null;
}
export interface RuntimeCompileTargetV1 {
    packId: string;
    routePolicy: RoutePolicy;
    routerIdentity: string | null;
    workspaceSnapshot: string;
    workspaceRevision: string | null;
    eventRange: Pick<NormalizedEventRangeV1, "start" | "end" | "count">;
    eventExportDigest: string | null;
    builtAt: string;
}
export interface RuntimeCompileExpectationV1 {
    packId?: string;
    routePolicy?: RoutePolicy;
    routerIdentity?: string | null;
    workspaceSnapshot?: string;
    workspaceRevision?: string | null;
    eventRange?: Pick<NormalizedEventRangeV1, "start" | "end" | "count">;
    eventExportDigest?: string | null;
    builtAt?: string;
}
export interface NormalizedEventSourceV1 {
    runtimeOwner: "openclaw";
    stream: string;
}
export interface EventSemanticMetadataV1 {
    semanticType: EventSemanticTypeV1;
    sourceKind: EventSourceKindV1;
    diagnosticIntent?: EventDiagnosticIntentV1;
}
export interface RetrievalSemanticClassSemanticsV1 {
    semanticClass: RetrievalSemanticClassV1;
    answerRole: RetrievalAnswerRoleV1;
    instructionRole: RetrievalInstructionRoleV1;
    detail: string;
}
export interface EventSemanticSurfaceV1 {
    semanticTypes: EventSemanticTypeV1[];
    sourceKinds: EventSourceKindV1[];
    diagnosticIntents: EventDiagnosticIntentV1[];
}
export type RuntimeTurnBrainAttachmentPolicyV1 = "undeclared" | "dedicated" | "shared";
export type RuntimeTurnProfileSelectorV1 = string;
export type RuntimeTurnBrainStatusV1 = "serving_active_pack" | "fail_open_static_context" | "hard_fail";
export type ContextContributionEvidenceStateV1 = "route_fn_and_brain_context" | "brain_context_only" | "route_fn_only" | "stable_kernel_only" | "fail_open_static_context" | "hard_fail" | "unprobed";
export interface RuntimeContextFingerprintV1 {
    digest: string;
    selectionDigest: string | null;
    promptContextDigest: string | null;
    promptContextFingerprints: string[];
    workspaceInjectionSurfaceDigest: string | null;
    runtimeHintsDigest: string | null;
    runtimeHints: string[];
    profileLineageDigest: string;
    profileLineage: string[];
    sessionLineageDigest: string;
    sessionLineage: string[];
    brainLineageDigest: string;
    brainLineage: string[];
}
export interface RuntimeTurnAttributionV1 {
    hostRuntimeOwner: "openclaw";
    profileSelector: RuntimeTurnProfileSelectorV1;
    profileId: string | null;
    brainAttachmentPolicy: RuntimeTurnBrainAttachmentPolicyV1;
    brainStatus: RuntimeTurnBrainStatusV1;
    activePackId: string | null;
    usedLearnedRouteFn: boolean | null;
    routerIdentity: string | null;
    selectionDigest: string | null;
    selectionTiers: string | null;
    contextFingerprint: RuntimeContextFingerprintV1;
    contextEvidence: Exclude<ContextContributionEvidenceStateV1, "unprobed"> | null;
}
export interface InteractionEventV1 {
    contract: typeof CONTRACT_IDS.interactionEvents;
    eventId: string;
    agentId: string;
    sessionId: string;
    channel: string;
    sequence: number;
    kind: InteractionEventKind;
    createdAt: string;
    source: NormalizedEventSourceV1;
    packId?: string;
    messageId?: string;
    semantic?: EventSemanticMetadataV1;
    principal?: PrincipalMetadataV1;
    attribution?: RuntimeTurnAttributionV1;
}
export interface FeedbackEventV1 {
    contract: typeof CONTRACT_IDS.feedbackEvents;
    eventId: string;
    agentId: string;
    sessionId: string;
    channel: string;
    sequence: number;
    kind: FeedbackEventKind;
    createdAt: string;
    source: NormalizedEventSourceV1;
    content: string;
    messageId?: string;
    semantic?: EventSemanticMetadataV1;
    principal?: PrincipalMetadataV1;
    attribution?: RuntimeTurnAttributionV1;
    relatedInteractionId?: string;
}
export type NormalizedEventV1 = InteractionEventV1 | FeedbackEventV1;
export interface NormalizedEventRangeV1 {
    start: number;
    end: number;
    count: number;
    firstEventId: string | null;
    lastEventId: string | null;
    firstCreatedAt: string | null;
    lastCreatedAt: string | null;
}
export interface EventExportProvenanceV1 {
    runtimeOwner: "openclaw";
    sessionId: string | null;
    channel: string | null;
    interactionCount: number;
    feedbackCount: number;
    sourceStreams: string[];
    contracts: EventContractId[];
    exportDigest: string;
    semanticSurface?: EventSemanticSurfaceV1;
    learningSurface: LearningSurfaceV1;
}
export interface NormalizedEventExportV1 {
    interactionEvents: InteractionEventV1[];
    feedbackEvents: FeedbackEventV1[];
    range: NormalizedEventRangeV1;
    provenance: EventExportProvenanceV1;
}
export interface TeacherSupervisionArtifactSourceV1 {
    runtimeOwner: "openclaw";
    sessionId: string;
    channel: string;
    sourceStreams: string[];
    eventRange: Pick<NormalizedEventRangeV1, "start" | "end" | "count">;
    eventExportDigest: string;
}
export interface TeacherSupervisionFreshnessV1 {
    status: TeacherSupervisionFreshnessStatus;
    observedAt: string;
    newestSourceCreatedAt: string;
    ageMs: number;
    staleAfterMs: number;
}
export interface TeacherSupervisionArtifactV1 {
    contract: typeof CONTRACT_IDS.teacherSupervisionArtifact;
    artifactId: string;
    dedupId: string;
    kind: TeacherSupervisionKind;
    createdAt: string;
    source: TeacherSupervisionArtifactSourceV1;
    sourceEventIds: string[];
    relatedInteractionId: string | null;
    principal?: PrincipalMetadataV1;
    content: string;
    freshness: TeacherSupervisionFreshnessV1;
}
export interface WorkspaceMetadataV1 {
    workspaceId: string;
    snapshotId: string;
    capturedAt: string;
    rootDir: string;
    branch: string | null;
    revision: string | null;
    dirty: boolean;
    manifestDigest: string | null;
    labels: string[];
    files: string[];
}
export interface ArtifactManifestV1 {
    contract: typeof CONTRACT_IDS.artifactManifest;
    packId: string;
    immutable: true;
    routePolicy: RoutePolicy;
    runtimeAssets: {
        graphPath: string;
        vectorPath: string;
        router: {
            kind: RouterAssetKind;
            identity: string | null;
            artifactPath: string | null;
        };
    };
    payloadChecksums: {
        graph: string;
        vector: string;
        router: string | null;
    };
    routeArtifact: RouteArtifactReferenceV1;
    modelFingerprints: string[];
    provenance: {
        workspace: WorkspaceMetadataV1;
        workspaceSnapshot: string;
        eventRange: NormalizedEventRangeV1;
        eventExports: EventExportProvenanceV1 | null;
        learningSurface: LearningSurfaceV1;
        builtAt: string;
        offlineArtifacts: string[];
    };
    graphDynamics: {
        bootstrapping: {
            fastBootDefaults: boolean;
            passiveBackgroundLearning: boolean;
        };
        runtimePlasticitySource: RuntimePlasticitySourceV1;
        hebbian: {
            enabled: boolean;
            learningRate: number;
        };
        decay: {
            enabled: boolean;
            halfLifeDays: number;
        };
        structuralOps: {
            split: number;
            merge: number;
            prune: number;
            connect: number;
        };
    };
}
export type ArtifactProvenanceV1 = ArtifactManifestV1["provenance"];
export interface ActivationPointerRecordV1 {
    slot: ActivationPointerSlot;
    packId: string;
    packRootDir: string;
    manifestPath: string;
    manifestDigest: string;
    routePolicy: RoutePolicy;
    routerIdentity: string | null;
    workspaceSnapshot: string;
    workspaceRevision: string | null;
    eventRange: Pick<NormalizedEventRangeV1, "start" | "end" | "count">;
    eventExportDigest: string | null;
    builtAt: string;
    updatedAt: string;
}
export interface ActivationPointersV1 {
    contract: typeof CONTRACT_IDS.activationPointers;
    active: ActivationPointerRecordV1 | null;
    candidate: ActivationPointerRecordV1 | null;
    previous: ActivationPointerRecordV1 | null;
}
export interface PackVectorEmbeddingV1 {
    model: string;
    values: number[];
}
export interface PackVectorEntryV1 {
    blockId: string;
    keywords: string[];
    boost: number;
    weights?: Record<string, number>;
    embedding?: PackVectorEmbeddingV1;
}
export interface PackVectorsPayloadV1 {
    packId: string;
    entries: PackVectorEntryV1[];
}
export interface PackGraphPayloadV1 {
    packId: string;
    schema?: PackGraphSchemaV1;
    ontology?: OpenClawInitGraphOntologyV1;
    blocks: PackContextBlockRecordV1[];
    evolution?: PackGraphEvolutionV1;
}
export interface RouterPolicyUpdateV1 {
    blockId: string;
    delta: number;
    evidenceCount: number;
    rewardSum: number;
    tokenWeights: Record<string, number>;
    traceIds: string[];
}
export interface RouterTraceV1 {
    traceId: string;
    sourceEventId: string;
    sourceContract: ContractId;
    sourceKind: InteractionEventKind | FeedbackEventKind;
    supervisionKind: RouterSupervisionKind;
    targetBlockIds: string[];
    reward: number;
    queryTokens: string[];
    queryVector: Record<string, number>;
}
export interface RouterRefreshDiagnosticsV1 {
    method: RouterTrainingMethodV1;
    status: RouterRefreshStatus;
    eventExportDigest: string | null;
    routeTraceCount: number;
    supervisionCount: number;
    updateCount: number;
    collectedLabels: RouterCollectedLabelCountsV1;
    objective: RouterPgObjectiveV1;
    queryChecksum: string;
    weightsChecksum: string;
    freshnessChecksum: string;
    noOpReason: string | null;
}
export interface RouterCollectedLabelCountsV1 {
    total: number;
    humanFeedback: number;
    operatorOverride: number;
    selfMemory: number;
}
export interface RouterPgProfileV1 {
    traceSource: "event_reconstruction" | "serve_time_decision_log";
    actionSpace: "pack_block_softmax" | "graph_local_neighbor_softmax";
    targetConstruction: "event_block_plus_related_interaction" | "trajectory_reconstruction";
    rewardSignal: "explicit_label_reward_table_v1";
    baseline: "none" | "exponential_moving_average";
    offPolicyCorrection: "none";
    updateCadence: "candidate_pack_refresh";
    trajectorySource?: "serve_decision_reconstruction";
}
export declare const ROUTER_PG_PROFILE_V1: RouterPgProfileV1;
export interface RouterPgProfileV2 {
    traceSource: "serve_time_decision_log";
    actionSpace: "graph_local_neighbor_softmax";
    targetConstruction: "trajectory_reconstruction";
    rewardSignal: "explicit_label_reward_table_v1";
    baseline: "exponential_moving_average";
    offPolicyCorrection: "none";
    updateCadence: "candidate_pack_refresh";
    trajectorySource: "serve_decision_reconstruction";
}
export declare const ROUTER_PG_PROFILE_V2: RouterPgProfileV2;
export interface RouterPgObjectiveV1 {
    updateMechanism: "policy_gradient";
    updateVersion: "route_pg_update_v1" | "route_pg_update_v2";
    objective: "supervised_route_pg_v1" | "supervised_route_pg_v2";
    profile: RouterPgProfileV1 | RouterPgProfileV2;
    objectiveChecksum: string;
}
export interface RouterArtifactV1 {
    routerIdentity: string;
    strategy: "learned_route_fn_v1";
    trainedAt: string;
    requiresLearnedRouting: boolean;
    training: RouterRefreshDiagnosticsV1;
    traces: RouterTraceV1[];
    policyUpdates: RouterPolicyUpdateV1[];
}
export declare function describeRetrievalSemanticClass(value: RetrievalSemanticClassV1): RetrievalSemanticClassSemanticsV1;
export declare function buildEventSemanticSurface(events: readonly NormalizedEventV1[]): EventSemanticSurfaceV1;
export declare function createDefaultLearningSurface(scanSurfaces?: readonly string[]): LearningSurfaceV1;
export declare function buildLearningSurface(events: readonly NormalizedEventV1[]): LearningSurfaceV1;
export declare function canonicalJson(value: unknown): string;
export declare function checksumJsonPayload(value: unknown): string;
export declare function computeRouterQueryChecksum(traces: readonly RouterTraceV1[]): string;
export declare function computeRouterWeightsChecksum(policyUpdates: readonly RouterPolicyUpdateV1[]): string;
export declare function computeRouterCollectedLabelCounts(traces: readonly RouterTraceV1[]): RouterCollectedLabelCountsV1;
export declare function computeRouterObjectiveChecksum(input: {
    updateMechanism: RouterPgObjectiveV1["updateMechanism"];
    updateVersion: RouterPgObjectiveV1["updateVersion"];
    objective: RouterPgObjectiveV1["objective"];
    profile: RouterPgObjectiveV1["profile"];
    eventExportDigest: string | null;
    routeTraceCount: number;
    supervisionCount: number;
    collectedLabels: RouterCollectedLabelCountsV1;
    queryChecksum: string;
}): string;
export declare function computeRouterFreshnessChecksum(input: {
    method: RouterTrainingMethodV1;
    trainedAt: string;
    status: RouterRefreshStatus;
    eventExportDigest: string | null;
    routeTraceCount: number;
    supervisionCount: number;
    updateCount: number;
}): string;
export declare function buildRouteArtifactReference(input: {
    routerAssetKind: RouterAssetKind;
    routerIdentity: string | null;
    routerChecksum: string | null;
    router: RouterArtifactV1 | null;
    eventExportDigest?: string | null;
}): RouteArtifactReferenceV1;
export declare function buildServedArtifactProof(target: RuntimeCompileTargetV1, routeArtifact: RouteArtifactReferenceV1): ServedArtifactProofV1;
export declare function createInteractionEvent(value: Omit<InteractionEventV1, "contract">): InteractionEventV1;
export declare function createFeedbackEvent(value: Omit<FeedbackEventV1, "contract">): FeedbackEventV1;
export declare function sortNormalizedEvents(events: readonly NormalizedEventV1[]): NormalizedEventV1[];
export declare function createExplicitEventRange(value: {
    start: number;
    end: number;
}): NormalizedEventRangeV1;
export declare function buildNormalizedEventRange(events: readonly NormalizedEventV1[]): NormalizedEventRangeV1;
export declare function buildNormalizedEventExport(value: {
    interactionEvents: readonly InteractionEventV1[];
    feedbackEvents: readonly FeedbackEventV1[];
}): NormalizedEventExportV1;
export declare function validateRuntimeCompileRequest(value: RuntimeCompileRequestV1): string[];
export declare function validateRuntimeCompileExpectation(value: RuntimeCompileExpectationV1): string[];
export declare function validateRuntimeCompileTargetExpectation(target: RuntimeCompileTargetV1, expectation: RuntimeCompileExpectationV1): string[];
export declare function validateRouteArtifactReference(value: RouteArtifactReferenceV1, options?: {
    routePolicy?: RoutePolicy;
    label?: string;
}): string[];
export declare function validateServedArtifactProof(value: ServedArtifactProofV1, label?: string): string[];
export declare function validateRuntimeCompileResponse(value: RuntimeCompileResponseV1): string[];
export declare function validateBrainServeHotPathTiming(value: BrainServeHotPathTimingV1, label?: string): string[];
export declare function validateBrainAttachmentPolicySemantics(value: BrainAttachmentPolicySemanticsV1, label?: string): string[];
export declare function validateBrainAttachmentPolicy(value: BrainAttachmentPolicyV1): string[];
export declare function validateRuntimeContextFingerprint(value: RuntimeContextFingerprintV1): string[];
export declare function validateProfileTurnAttribution(value: ProfileTurnAttributionV1): string[];
export declare function validateCurrentProfileBrainStatus(value: CurrentProfileBrainStatusAnswerV1): string[];
export declare function validateNormalizedEventSource(value: NormalizedEventSourceV1): string[];
export declare function validateEventSemanticMetadata(value: EventSemanticMetadataV1): string[];
export declare function validateEventSemanticSurface(value: EventSemanticSurfaceV1): string[];
export declare function validatePrincipalScope(value: PrincipalScopeV1): string[];
export declare function validatePrincipalMetadata(value: PrincipalMetadataV1): string[];
export declare function validateRuntimeTurnAttribution(value: RuntimeTurnAttributionV1): string[];
export declare function validateInteractionEvent(value: InteractionEventV1): string[];
export declare function validateFeedbackEvent(value: FeedbackEventV1): string[];
export declare function validateNormalizedEventRange(value: NormalizedEventRangeV1): string[];
export declare function validateTeacherSupervisionArtifact(value: TeacherSupervisionArtifactV1): string[];
export declare function validateWorkspaceMetadata(value: WorkspaceMetadataV1): string[];
export declare function validateLearningSurface(value: LearningSurfaceV1): string[];
export declare function validatePackBlockLearningSignals(value: PackBlockLearningSignalsV1, blockId?: string): string[];
export declare function validateEventExportProvenance(value: EventExportProvenanceV1, eventRange?: NormalizedEventRangeV1): string[];
export declare function validateNormalizedEventExport(value: NormalizedEventExportV1): string[];
export declare function validateArtifactManifest(value: ArtifactManifestV1): string[];
export declare function validateActivationPointerRecord(value: ActivationPointerRecordV1, expectedSlot?: ActivationPointerSlot): string[];
export declare function validateActivationPointers(value: ActivationPointersV1): string[];
export declare function validateOpenClawInitBlockMetadata(value: OpenClawInitBlockMetadataV1, label?: string): string[];
export declare function validateOpenClawInitGraphOntology(value: OpenClawInitGraphOntologyV1): string[];
export declare function validatePackGraphPayload(value: PackGraphPayloadV1, expectedPackId?: string): string[];
export declare function validatePackVectorsPayload(value: PackVectorsPayloadV1, graph?: PackGraphPayloadV1): string[];
export declare function validateRouterArtifact(value: RouterArtifactV1, manifest?: ArtifactManifestV1): string[];
export declare const FIXTURE_PACK_GRAPH: PackGraphPayloadV1;
export declare const FIXTURE_PACK_VECTORS: PackVectorsPayloadV1;
export declare const FIXTURE_RUNTIME_TURN_ATTRIBUTION: RuntimeTurnAttributionV1;
export declare const FIXTURE_INTERACTION_EVENTS: InteractionEventV1[];
export declare const FIXTURE_FEEDBACK_EVENTS: FeedbackEventV1[];
export declare const FIXTURE_NORMALIZED_EVENT_EXPORT: NormalizedEventExportV1;
export declare const FIXTURE_ROUTER_ARTIFACT: RouterArtifactV1;
export declare const FIXTURE_TEACHER_SUPERVISION_ARTIFACT: TeacherSupervisionArtifactV1;
export declare const FIXTURE_WORKSPACE_METADATA: WorkspaceMetadataV1;
export declare const FIXTURE_ARTIFACT_MANIFEST: ArtifactManifestV1;
export declare const FIXTURE_ACTIVATION_POINTERS: ActivationPointersV1;
export declare const FIXTURE_RUNTIME_COMPILE_REQUEST: RuntimeCompileRequestV1;
export declare const FIXTURE_RUNTIME_COMPILE_RESPONSE: RuntimeCompileResponseV1;
export declare const FIXTURE_DEDICATED_BRAIN_ATTACHMENT_POLICY: BrainAttachmentPolicyV1;
export declare const FIXTURE_SHARED_BRAIN_ATTACHMENT_POLICY: BrainAttachmentPolicyV1;
export declare const FIXTURE_PROFILE_TURN_ATTRIBUTION: ProfileTurnAttributionV1;
export declare const FIXTURE_CURRENT_PROFILE_BRAIN_STATUS: CurrentProfileBrainStatusAnswerV1;
export declare const FIXTURE_INTERACTION_EVENT: InteractionEventV1;
/**
 * Coarse category for a piece of workspace injection content.
 *
 * - `"kernel"` — must stay in the system prompt, injected directly by OpenClaw
 *   on every turn, immune to brain routing and budget pressure.
 * - `"brain_eligible"` — can migrate into a pack as source material and be
 *   compiled into context by the brain compiler.
 */
export type WorkspaceInjectionCategory = "kernel" | "brain_eligible";
/**
 * Kinds of kernel content. Kernel content is injected directly in the system
 * prompt and never mediated by brain compilation.
 *
 * - `"identity_anchor"` — core role / persona definition ("You are the OpenClaw
 *   assistant for Acme Corp.").
 * - `"safety_constraint"` — hard behavioral prohibitions ("Never output PII.").
 * - `"channel_policy"` — required response format or channel behavior ("Always
 *   respond in JSON.").
 * - `"capability_restriction"` — allowed/prohibited tools or operations.
 * - `"session_metadata"` — per-session operator context (user ID, tier,
 *   timestamp) that changes each turn.
 */
export type KernelContentKind = "identity_anchor" | "safety_constraint" | "channel_policy" | "capability_restriction" | "session_metadata";
/**
 * Kinds of brain-eligible content. These can be moved out of the system prompt
 * into pack source material and compiled selectively by the brain.
 *
 * - `"domain_knowledge"` — product facts, API reference, technical docs.
 * - `"workspace_state"` — current repo/file state, recent changes, open tasks.
 * - `"soft_behavioral_pref"` — preferred tone, verbosity, language style.
 * - `"project_context"` — past architectural decisions, team conventions.
 * - `"teaching_example"` — examples of good and corrected responses.
 */
export type BrainEligibleContentKind = "domain_knowledge" | "workspace_state" | "soft_behavioral_pref" | "project_context" | "teaching_example";
/** A single kernel section in a proposed workspace injection surface. */
export interface KernelSectionDescriptorV1 {
    kind: KernelContentKind;
    /** Short human-readable description of this section's content. */
    description: string;
    /** Estimated char count for kernel budget tracking. Optional. */
    estimatedChars?: number;
}
/** A single brain-eligible section in a proposed workspace injection surface. */
export interface BrainEligibleSectionDescriptorV1 {
    kind: BrainEligibleContentKind;
    /** Short human-readable description of this section's content. */
    description: string;
    /**
     * The `LearningBlockRole` this content should map to when written as pack
     * source material. This guides how the learner seeds and routes the block.
     */
    mappedRole: LearningBlockRole;
    /** Estimated char count (helps operators size pack budgets). Optional. */
    estimatedChars?: number;
}
/**
 * A human-authored descriptor of a proposed kernel/brain split for an
 * operator's workspace injection surface.
 *
 * Use `validateKernelSurface()` to check for obvious misclassifications.
 */
export interface WorkspaceInjectionSurfaceV1 {
    /** Sections that will remain in the system prompt permanently. */
    kernelSections: KernelSectionDescriptorV1[];
    /** Sections that will migrate to pack source material. */
    brainEligibleSections: BrainEligibleSectionDescriptorV1[];
    /**
     * Sections the operator could not clearly classify. These should be resolved
     * before using the surface descriptor in production.
     */
    ambiguous?: Array<{
        description: string;
        tentativeCategory: WorkspaceInjectionCategory;
    }>;
}
/** Severity of a finding from `validateKernelSurface()`. */
export type KernelSurfaceFindingSeverity = "PASS" | "WARN" | "FAIL";
/** A single finding from `validateKernelSurface()`. */
export interface KernelSurfaceFindingV1 {
    severity: KernelSurfaceFindingSeverity;
    code: string;
    message: string;
}
/** Result of `validateKernelSurface()`. */
export interface KernelSurfaceValidationResultV1 {
    /** Overall severity: worst finding severity, or "PASS" if no findings. */
    severity: KernelSurfaceFindingSeverity;
    findings: KernelSurfaceFindingV1[];
}
/**
 * Validates a proposed `WorkspaceInjectionSurfaceV1` for obvious
 * misclassifications and missing required kernel kinds.
 *
 * Checks performed:
 * - Brain-eligible section roles must be compatible with their kind.
 * - Safety constraint and identity anchor are recommended as kernel.
 * - Ambiguous sections produce WARN findings.
 * - No brain-eligible content should be classified as kernel (heuristic check
 *   based on kind: `teaching_example` and `workspace_state` in kernel → FAIL).
 *
 * Returns a `KernelSurfaceValidationResultV1` with all findings. The caller
 * decides whether to act on WARN vs FAIL findings.
 *
 * Note: this validates the *descriptor* — it does not parse system prompt text.
 * Automatic text extraction is not yet supported.
 */
export declare function validateKernelSurface(surface: WorkspaceInjectionSurfaceV1): KernelSurfaceValidationResultV1;
export declare const FIXTURE_FEEDBACK_EVENT: FeedbackEventV1;
