import { type DatabaseLike } from './sqlite-driver.js';
import { type ScopeContext } from './scope.js';
import type { MemoryNode, MemoryType, MemoryEdge, EdgeRelation, RouteDecision, RouteKind, RouteFrameV2, InjectionEvent, InjectionOutcome, BackgroundJob, JobKind, ProofEvent, DistillationRun, RouteExample, RoutePolicySnapshot, RouteGraphSnapshot, RouteTeacherRun, RouteCounterfactual, RouteTrainingExampleV2, RoutePolicySnapshotV2, RouteFrameV3, RouteActionPrototypeV3, RoutePairExampleV3, RouteBanditFeedbackV3, RouteBanditStateV3, RoutePolicySnapshotV3, RouteShadowDecisionV3, RouteCalibrationExampleV3, RouteActionFamilyStatsV3, RoutePolicyCandidateReportV3, RouteEvalCaseV3, RouteEvalCaseLabelV3, CaptureAuditRow } from './memory-types.js';
export declare const uuid: () => `${string}-${string}-${string}-${string}-${string}`;
export declare const now: () => string;
export interface MemoryStoreOptions {
    activationRoot: string;
    agentId: string;
}
export declare class MemoryStore {
    private db;
    private dbPath;
    private ownerAgentId;
    constructor(options: MemoryStoreOptions);
    private migrate;
    close(): void;
    insertMemory(node: Omit<MemoryNode, 'id' | 'createdAt' | 'updatedAt' | 'lastSeenAt'> & {
        id?: string;
    }): MemoryNode;
    getMemory(id: string): MemoryNode | null;
    findMemoryByNormalizedKey(agentId: string, normalizedKey: string, scopeKind: string, scopeKey?: string): MemoryNode | null;
    updateMemory(id: string, updates: Partial<MemoryNode>): MemoryNode | null;
    supersedeMemory(existingId: string, supersededById: string): void;
    softDeleteMemory(id: string): void;
    searchMemories(query: string, agentId: string, opts?: {
        limit?: number;
        offset?: number;
        scopeContext?: ScopeContext;
    }): MemoryNode[];
    listMemories(agentId: string, opts?: {
        type?: MemoryType;
        limit?: number;
        scopeContext?: ScopeContext;
    }): MemoryNode[];
    countMemories(agentId: string, type?: MemoryType): number;
    insertEdge(edge: Omit<MemoryEdge, 'id' | 'createdAt' | 'updatedAt'> & {
        id?: string;
    }): MemoryEdge;
    upsertEdge(agentId: string, fromId: string, toId: string, relation: EdgeRelation): MemoryEdge;
    getEdges(memoryId: string, relation?: EdgeRelation): MemoryEdge[];
    insertInjection(inj: Omit<InjectionEvent, 'id' | 'injectedAt' | 'outcome'> & {
        id?: string;
        injectedAt?: string;
        outcome?: InjectionOutcome;
    }): InjectionEvent;
    resolveInjectionOutcome(injectionId: string, outcome: InjectionOutcome, correctionSignal?: string, scope?: {
        agentId?: string;
        runId?: string;
        turnId?: string;
        sessionId?: string;
    }): number;
    getPendingInjections(agentId: string): InjectionEvent[];
    getInjectionsForRouteDecision(routeDecisionId: string): InjectionEvent[];
    insertRouteFrame(frame: Omit<RouteFrameV2, 'id' | 'createdAt'> & {
        id?: string;
        createdAt?: string;
    }): RouteFrameV2;
    getRouteFrame(id: string): RouteFrameV2 | null;
    insertRouteFrameV3(frame: Omit<RouteFrameV3, 'id' | 'createdAt'> & {
        id?: string;
        createdAt?: string;
    }): RouteFrameV3;
    listRouteFramesV3(agentId: string, limit?: number): RouteFrameV3[];
    upsertRouteActionPrototypeV3(prototype: Omit<RouteActionPrototypeV3, 'createdAt' | 'updatedAt'> & {
        createdAt?: string;
        updatedAt?: string;
    }): RouteActionPrototypeV3;
    listRouteActionPrototypesV3(agentId: string, limit?: number): RouteActionPrototypeV3[];
    getRouteActionPrototypeV3(id: string): RouteActionPrototypeV3 | null;
    setRouteActionPrototypeStatusV3(id: string, status: RouteActionPrototypeV3['status']): RouteActionPrototypeV3 | null;
    insertRoutePairExampleV3(example: Omit<RoutePairExampleV3, 'id' | 'createdAt'> & {
        id?: string;
        createdAt?: string;
    }): RoutePairExampleV3;
    listRoutePairExamplesV3(agentId: string, limit?: number): RoutePairExampleV3[];
    insertRouteBanditFeedbackV3(feedback: Omit<RouteBanditFeedbackV3, 'id' | 'createdAt'> & {
        id?: string;
        createdAt?: string;
    }): RouteBanditFeedbackV3;
    getRouteBanditStateV3(agentId: string): RouteBanditStateV3 | null;
    upsertRouteBanditStateV3(state: RouteBanditStateV3): RouteBanditStateV3;
    insertRouteShadowDecisionV3(decision: Omit<RouteShadowDecisionV3, 'id' | 'createdAt'> & {
        id?: string;
        createdAt?: string;
    }): RouteShadowDecisionV3;
    listRouteShadowDecisionsV3(agentId: string, limit?: number, routeDecisionId?: string): RouteShadowDecisionV3[];
    replaceRouteCalibrationExamplesV3(agentId: string, snapshotId: string, examples: Array<Omit<RouteCalibrationExampleV3, 'id' | 'createdAt'> & {
        id?: string;
        createdAt?: string;
    }>): RouteCalibrationExampleV3[];
    listRouteCalibrationExamplesV3(agentId: string, limit?: number, snapshotId?: string): RouteCalibrationExampleV3[];
    replaceRouteEvalCasesV3(agentId: string, snapshotId: string, cases: Array<Omit<RouteEvalCaseV3, 'id' | 'agentId' | 'snapshotId' | 'createdAt'> & {
        id?: string;
        createdAt?: string;
        labels?: Array<Omit<RouteEvalCaseLabelV3, 'id' | 'agentId' | 'caseId' | 'createdAt'> & {
            id?: string;
            createdAt?: string;
        }>;
    }>): RouteEvalCaseV3[];
    listRouteEvalCasesV3(agentId: string, limit?: number, snapshotId?: string): RouteEvalCaseV3[];
    listRouteEvalCaseLabelsV3(agentId: string, limit?: number, caseId?: string): RouteEvalCaseLabelV3[];
    upsertRouteActionFamilyStatsV3(stats: RouteActionFamilyStatsV3): RouteActionFamilyStatsV3;
    listRouteActionFamilyStatsV3(agentId: string, limit?: number): RouteActionFamilyStatsV3[];
    insertRoutePolicyCandidateReportV3(report: Omit<RoutePolicyCandidateReportV3, 'id' | 'createdAt'> & {
        id?: string;
        createdAt?: string;
    }): RoutePolicyCandidateReportV3;
    listRoutePolicyCandidateReportsV3(agentId: string, limit?: number): RoutePolicyCandidateReportV3[];
    insertPolicySnapshotV3(snapshot: Omit<RoutePolicySnapshotV3, 'id' | 'createdAt'> & {
        id?: string;
        createdAt?: string;
    }): RoutePolicySnapshotV3;
    getActivePolicySnapshotV3(agentId: string): RoutePolicySnapshotV3 | null;
    listPolicySnapshotsV3(agentId: string, limit?: number): RoutePolicySnapshotV3[];
    insertRouteDecision(decision: Omit<RouteDecision, 'id' | 'createdAt'> & {
        id?: string;
    }): RouteDecision;
    getRouteDecision(id: string): RouteDecision | null;
    resolveRouteDecision(id: string, outcome: string, reward: number): void;
    finalizeRouteShadowDecisionsV3(routeDecisionId: string, actualRoute: RouteKind, reward: number): void;
    getRecentRouteDecisions(agentId: string, limit?: number): RouteDecision[];
    getUnresolvedRouteDecisions(agentId: string): RouteDecision[];
    getResolvedRouteDecisions(agentId: string, limit?: number): RouteDecision[];
    countRouteDecisions(agentId: string): number;
    countRouteDecisionsByLatencyTier(agentId: string, latencyTier: string): number;
    countSyncPlannerCalls(agentId: string): number;
    averageSyncPlannerLatency(agentId: string): number;
    countSyncPlannerFallbacks(agentId: string): number;
    countRouteExamples(agentId: string, polarity?: 'all' | 'positive' | 'negative'): number;
    insertRouteExample(example: Omit<RouteExample, 'id' | 'createdAt'> & {
        id?: string;
    }): RouteExample;
    getRouteExamples(agentId: string, limit?: number): RouteExample[];
    hasRouteExampleForDecision(agentId: string, routeDecisionId: string): boolean;
    getActivePolicySnapshot(agentId: string): RoutePolicySnapshot | null;
    insertPolicySnapshot(snapshot: Omit<RoutePolicySnapshot, 'id' | 'createdAt'> & {
        id?: string;
    }): RoutePolicySnapshot;
    listPolicySnapshots(agentId: string, limit?: number): RoutePolicySnapshot[];
    insertDistillationRun(run: Omit<DistillationRun, 'id' | 'createdAt'> & {
        id?: string;
    }): DistillationRun;
    insertCaptureAudit(row: Omit<CaptureAuditRow, 'id' | 'createdAt'> & {
        id?: string;
        createdAt?: string;
    }): CaptureAuditRow;
    listCaptureAudit(agentId: string, limit?: number): CaptureAuditRow[];
    countCaptureAudit(agentId: string): number;
    enqueueJob(job: Omit<BackgroundJob, 'id' | 'createdAt' | 'updatedAt' | 'status' | 'attempts' | 'startedAt' | 'finishedAt' | 'error'> & {
        id?: string;
    }): BackgroundJob;
    claimNextJob(kind?: JobKind, agentId?: string): BackgroundJob | null;
    completeJob(id: string): void;
    failJob(id: string, error: string, retryAfterMs?: number): void;
    getJobQueueDepth(agentId?: string): number;
    adjustMemoryScore(memoryId: string, patch: {
        importanceDelta?: number;
        confidenceDelta?: number;
        freshnessDelta?: number;
        useCountDelta?: number;
        usefulCountDelta?: number;
        captureCountDelta?: number;
    }): MemoryNode | null;
    pruneMemories(agentId: string, maxNodes: number): number;
    consolidateMemories(agentId: string, limit?: number): number;
    decayFreshness(agentId: string, decayPerDay?: number): number;
    getRouteExamplesByPolarity(agentId: string, polarity: 'positive' | 'negative', limit?: number): any[];
    getConnectedMemories(memoryId: string, maxDepth?: number, agentId?: string, scopeContext?: ScopeContext): MemoryNode[];
    countEdgesForAgent(agentId: string): number;
    insertRouteGraphSnapshot(snapshot: Omit<RouteGraphSnapshot, 'id' | 'createdAt'> & {
        id?: string;
        createdAt?: string;
    }): RouteGraphSnapshot;
    getRouteGraphSnapshot(routeDecisionId: string): RouteGraphSnapshot | null;
    insertRouteTeacherRun(run: Omit<RouteTeacherRun, 'id' | 'createdAt'> & {
        id?: string;
        createdAt?: string;
    }): RouteTeacherRun;
    hasRouteTeacherRunForDecision(routeDecisionId: string): boolean;
    listRouteTeacherRuns(agentId: string, limit?: number): RouteTeacherRun[];
    insertRouteCounterfactual(counterfactual: Omit<RouteCounterfactual, 'id' | 'createdAt'> & {
        id?: string;
        createdAt?: string;
    }): RouteCounterfactual;
    listRouteCounterfactuals(agentId: string, routeDecisionId?: string, limit?: number): RouteCounterfactual[];
    insertRouteTrainingExampleV2(example: Omit<RouteTrainingExampleV2, 'id' | 'createdAt'> & {
        id?: string;
        createdAt?: string;
    }): RouteTrainingExampleV2;
    listRouteTrainingExamplesV2(agentId: string, limit?: number): RouteTrainingExampleV2[];
    countRouteTrainingExamplesV2(agentId: string): number;
    insertPolicySnapshotV2(snapshot: Omit<RoutePolicySnapshotV2, 'id' | 'createdAt'> & {
        id?: string;
        createdAt?: string;
    }): RoutePolicySnapshotV2;
    getActivePolicySnapshotV2(agentId: string): RoutePolicySnapshotV2 | null;
    listPolicySnapshotsV2(agentId: string, limit?: number): RoutePolicySnapshotV2[];
    insertProofEvent(event: Omit<ProofEvent, 'id' | 'createdAt'> & {
        id?: string;
    }): ProofEvent;
    getProofEvents(agentId: string, limit?: number): ProofEvent[];
    pruneProofEvents(agentId: string, retain: number): void;
    writeStatusSnapshot(agentId: string, status: Record<string, unknown>): Record<string, unknown>;
    readStatusSnapshot(agentId: string): Record<string, unknown> | null;
    transaction<T>(fn: () => T): T;
}
export declare function dbPathForAgent(activationRoot: string, agentId: string): string;
export declare function openDb(dbPath: string): DatabaseLike;
