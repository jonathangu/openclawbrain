import { type ActivationPointerRecordV1, type ActivationPointerSlot, type ActivationPointersV1, type ArtifactManifestV1, type LearningBlockRole, type LearningBootProfile, type PackGraphConnectDiagnosticsV1, type PackGraphPayloadV1, type PackVectorsPayloadV1, type ServedArtifactProofV1, type RuntimeCompileTargetV1, type RuntimePlasticitySourceV1, type RouterArtifactV1, type RouterPgProfileV1, type RouterPgProfileV2 } from "@openclawbrain/contracts";
export declare const PACK_LAYOUT: {
    readonly graph: "graph.json";
    readonly manifest: "manifest.json";
    readonly router: "router/model.json";
    readonly vectors: "vectors.json";
};
export declare const ACTIVATION_LAYOUT: {
    readonly pointers: "activation-pointers.json";
};
export interface PackDescriptor {
    rootDir: string;
    manifestPath: string;
    graphPath: string;
    vectorPath: string;
    routerPath: string | null;
    manifest: ArtifactManifestV1;
    graph: PackGraphPayloadV1;
    vectors: PackVectorsPayloadV1;
    router: RouterArtifactV1 | null;
}
export interface ActivationStateDescriptor {
    rootDir: string;
    pointerPath: string;
    pointers: ActivationPointersV1;
}
export interface ActivationSlotInspection {
    slot: ActivationPointerSlot;
    packId: string;
    routePolicy: ArtifactManifestV1["routePolicy"];
    routerIdentity: string | null;
    workspaceSnapshot: string;
    workspaceRevision: string | null;
    eventRange: ActivationPointerRecordV1["eventRange"];
    eventExportDigest: string | null;
    builtAt: string;
    activationReady: boolean;
    findings: string[];
}
export interface ActivationOperationPreview {
    allowed: boolean;
    findings: string[];
    nextPointers: ActivationPointersV1 | null;
}
export interface ActivationMutationOptions {
    updatedAt?: string | null;
    reason?: string | null;
}
export interface ActivationInspection {
    rootDir: string;
    pointerPath: string;
    pointers: ActivationPointersV1;
    active: ActivationSlotInspection | null;
    candidate: ActivationSlotInspection | null;
    previous: ActivationSlotInspection | null;
    promotion: ActivationOperationPreview;
    rollback: ActivationOperationPreview;
}
export interface LearnedRouteFnFreshnessReport {
    packId: string | null;
    required: boolean;
    available: boolean;
    routerAssetKind: ArtifactManifestV1["runtimeAssets"]["router"]["kind"] | null;
    routerIdentity: string | null;
    routeFnVersion: RouterArtifactV1["strategy"] | null;
    trainingMethod: RouterArtifactV1["training"]["method"] | null;
    routerChecksum: string | null;
    routerTrainedAt: string | null;
    packBuiltAt: string | null;
    workspaceSnapshot: string | null;
    eventExportDigest: string | null;
    updateMechanism: RouterArtifactV1["training"]["objective"]["updateMechanism"] | null;
    updateVersion: RouterArtifactV1["training"]["objective"]["updateVersion"] | null;
    objective: RouterArtifactV1["training"]["objective"]["objective"] | null;
    pgProfile: RouterPgProfileV1 | RouterPgProfileV2 | null;
    objectiveChecksum: string | null;
    collectedLabels: RouterArtifactV1["training"]["collectedLabels"] | null;
    supervisionCount: number | null;
    updateCount: number | null;
    weightsChecksum: string | null;
    freshnessChecksum: string | null;
}
export interface RouteArtifactDiffReport {
    activePackId: string | null;
    candidatePackId: string | null;
    comparable: boolean;
    routerChanged: boolean;
    objectiveChanged: boolean;
    weightsChanged: boolean;
    freshnessChanged: boolean;
    labelCountsChanged: boolean;
    updateCountChanged: boolean;
    activeRouterChecksum: string | null;
    candidateRouterChecksum: string | null;
    activeObjectiveChecksum: string | null;
    candidateObjectiveChecksum: string | null;
    activeWeightsChecksum: string | null;
    candidateWeightsChecksum: string | null;
    activeFreshnessChecksum: string | null;
    candidateFreshnessChecksum: string | null;
    activeCollectedLabels: RouterArtifactV1["training"]["collectedLabels"] | null;
    candidateCollectedLabels: RouterArtifactV1["training"]["collectedLabels"] | null;
    activeUpdateCount: number | null;
    candidateUpdateCount: number | null;
    activeTopUpdatedBlocks: string[];
    candidateTopUpdatedBlocks: string[];
}
export interface GraphDynamicsFreshnessReport {
    packId: string | null;
    graphChecksum: string | null;
    builtAt: string | null;
    workspaceSnapshot: string | null;
    eventRange: RuntimeCompileTargetV1["eventRange"] | null;
    eventExportDigest: string | null;
    runtimePlasticitySource: ArtifactManifestV1["graphDynamics"]["runtimePlasticitySource"] | null;
    bootstrapping: ArtifactManifestV1["graphDynamics"]["bootstrapping"] | null;
    hebbian: ArtifactManifestV1["graphDynamics"]["hebbian"] | null;
    decay: ArtifactManifestV1["graphDynamics"]["decay"] | null;
    structuralOps: ArtifactManifestV1["graphDynamics"]["structuralOps"] | null;
}
export interface GraphSplitSummaryV1 {
    count: number;
    blockIds: string[];
    sources: string[];
    strongestBlockId: string | null;
}
export interface GraphEvolutionLogV1 {
    packId: string;
    provenance: RuntimePlasticitySourceV1;
    builtAt: string;
    graphChecksum: string;
    blockCount: number;
    structuralOps: {
        split: number;
        merge: number;
        prune: number;
        connect: number;
    };
    connectDiagnostics: PackGraphConnectDiagnosticsV1 | null;
    structuralEvolutionSummary: {
        changed: boolean;
        operationsApplied: Array<"split" | "merge" | "prune" | "connect">;
        liveBlockCount: number;
        prunedBlockCount: number;
        prePruneBlockCount: number;
        operatorSummary: string;
    };
    prunedBlockIds: string[];
    hebbianSummary: {
        applied: boolean;
        learningRate: number;
    };
    decaySummary: {
        applied: boolean;
        halfLifeDays: number;
    };
    strongestBlockId: string | null;
    eventExportDigest: string | null;
}
export interface PromotionFreshnessDeltaReport {
    builtAt: boolean;
    eventRangeEnd: boolean;
    eventRangeCount: boolean;
    workspaceSnapshot: boolean;
    workspaceRevision: boolean;
    eventExportDigest: boolean;
}
export interface PromotionFreshnessReport {
    activePackId: string | null;
    candidatePackId: string | null;
    previousPackId: string | null;
    activeUpdatedAt: string | null;
    candidateUpdatedAt: string | null;
    previousUpdatedAt: string | null;
    promotionAllowed: boolean;
    promotionFindings: string[];
    rollbackAllowed: boolean;
    rollbackFindings: string[];
    activeBehindPromotionReadyCandidate: boolean;
    candidateAheadBy: PromotionFreshnessDeltaReport | null;
}
export type InitHandoffState = "missing" | "seed_state_authoritative" | "pg_promoted_pack_authoritative";
export interface InitHandoffReport {
    packId: string | null;
    initMode: LearningBootProfile | null;
    seedStateVisible: boolean;
    seedBlockCount: number;
    seedSources: string[];
    seedRoles: LearningBlockRole[];
    handoffState: InitHandoffState;
    pgRouteAuthoritative: boolean;
    learnedRouteUpdateCount: number | null;
}
export interface ActivationObservabilityReport {
    slot: ActivationPointerSlot;
    target: RuntimeCompileTargetV1 | null;
    servedArtifact: ServedArtifactProofV1 | null;
    learnedRouteFn: LearnedRouteFnFreshnessReport;
    routeArtifactDiff: RouteArtifactDiffReport;
    graphDynamics: GraphDynamicsFreshnessReport;
    graphEvolutionLog: GraphEvolutionLogV1 | null;
    promotionFreshness: PromotionFreshnessReport;
    initHandoff: InitHandoffReport;
}
export interface ActivationObservabilityOptions {
    requireActivationReady?: boolean;
    updatedAt?: string;
}
export declare function describePackInitHandoff(packOrRootDir: PackDescriptor | string): InitHandoffReport;
export declare function summarizeGraphSplitBlocks(pack: Pick<PackDescriptor, "graph">): GraphSplitSummaryV1;
export declare function summarizeStructuralGraphEvolution(input: {
    blockCount: number;
    strongestBlockId: string | null;
    structuralOps: GraphEvolutionLogV1["structuralOps"];
    prunedBlockCount: number;
    connectDiagnostics: PackGraphConnectDiagnosticsV1 | null;
}): GraphEvolutionLogV1["structuralEvolutionSummary"];
export declare function describeGraphEvolutionLog(pack: PackDescriptor): GraphEvolutionLogV1;
export declare function validatePackDescriptor(manifest: ArtifactManifestV1): string[];
export declare function validatePackActivationReadiness(packOrRootDir: PackDescriptor | string): string[];
export declare function computePayloadChecksum(value: unknown): string;
export declare function writePackFile(rootDir: string, relativePath: string, payload: unknown): string;
export declare function describePackCompileTarget(packOrRootDir: PackDescriptor | string): RuntimeCompileTargetV1;
export declare function loadPackFromActivation(rootDir: string, slot?: ActivationPointerSlot, options?: {
    requireActivationReady?: boolean;
}): PackDescriptor | null;
export declare function describeActivationTarget(rootDir: string, slot?: ActivationPointerSlot, options?: {
    requireActivationReady?: boolean;
}): RuntimeCompileTargetV1 | null;
export declare function loadActivationPointers(rootDir: string): ActivationStateDescriptor;
export declare function inspectActivationState(rootDir: string, updatedAt?: string): ActivationInspection;
export declare function describeActivationObservability(rootDir: string, slot?: ActivationPointerSlot, options?: ActivationObservabilityOptions): ActivationObservabilityReport;
export declare function activatePack(rootDir: string, packRootDir: string, updatedAtOrOptions?: string | ActivationMutationOptions): ActivationStateDescriptor;
export declare function stageCandidatePack(rootDir: string, packRootDir: string, updatedAtOrOptions?: string | ActivationMutationOptions): ActivationStateDescriptor;
export declare function promoteCandidatePack(rootDir: string, updatedAtOrOptions?: string | ActivationMutationOptions): ActivationStateDescriptor;
export declare function rollbackActivePack(rootDir: string, updatedAtOrOptions?: string | ActivationMutationOptions): ActivationStateDescriptor;
export { LEARNING_SPINE_LOG_LAYOUT, appendLearningSpineLogEntry, buildLearningSpineLogId, readLearningSpineLogEntries, resolveLearningSpineLogPath, type LearningSpineActivationPackSnapshotV1, type LearningSpineActivationPointersSnapshotV1, type LearningSpineLaterServedTurnProofV1, type LearningSpineLogStream, type LearningSpinePgRouteUpdateLogEntryV1, type LearningSpinePromotionActivationLogEntryV1, type LearningSpinePromotionLinkV1, type LearningSpineServeRouteBreadcrumbsV1, type LearningSpineServeRouteCandidateScoreV1, type LearningSpineServeRouteDecisionLogEntryV1, type LearningSpineSupervisionLabelBindingLogEntryV1 } from "./learning-spine-logs.js";
export declare function loadPack(rootDir: string): PackDescriptor;
