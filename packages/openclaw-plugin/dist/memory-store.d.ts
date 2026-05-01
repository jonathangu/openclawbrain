import BetterSqlite3 from 'better-sqlite3';
import type { MemoryNode, MemoryType, MemoryEdge, EdgeRelation, RouteDecision, InjectionEvent, InjectionOutcome, BackgroundJob, JobKind, ProofEvent, DistillationRun, RouteExample, RoutePolicySnapshot } from './memory-types.js';
export declare const uuid: () => `${string}-${string}-${string}-${string}-${string}`;
export declare const now: () => string;
export interface MemoryStoreOptions {
    activationRoot: string;
    agentId: string;
}
export declare class MemoryStore {
    private db;
    private dbPath;
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
    }): MemoryNode[];
    listMemories(agentId: string, opts?: {
        type?: MemoryType;
        limit?: number;
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
    resolveInjectionOutcome(injectionId: string, outcome: InjectionOutcome, correctionSignal?: string): void;
    getPendingInjections(agentId: string): InjectionEvent[];
    getInjectionsForRouteDecision(routeDecisionId: string): InjectionEvent[];
    insertRouteDecision(decision: Omit<RouteDecision, 'id' | 'createdAt'> & {
        id?: string;
    }): RouteDecision;
    getRouteDecision(id: string): RouteDecision | null;
    resolveRouteDecision(id: string, outcome: string, reward: number): void;
    getRecentRouteDecisions(agentId: string, limit?: number): RouteDecision[];
    getUnresolvedRouteDecisions(agentId: string): RouteDecision[];
    getResolvedRouteDecisions(agentId: string, limit?: number): RouteDecision[];
    countRouteDecisions(agentId: string): number;
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
    enqueueJob(job: Omit<BackgroundJob, 'id' | 'createdAt' | 'updatedAt' | 'status' | 'attempts' | 'startedAt' | 'finishedAt' | 'error'> & {
        id?: string;
    }): BackgroundJob;
    claimNextJob(kind?: JobKind): BackgroundJob | null;
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
export declare function openDb(dbPath: string): BetterSqlite3.Database;
