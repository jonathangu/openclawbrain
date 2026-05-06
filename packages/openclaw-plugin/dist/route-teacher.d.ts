import type { LlmClient } from './llm-client.js';
import type { MemoryStore } from './memory-store.js';
import type { MemoryNode, RouteDecision, RouteGraphSnapshot } from './memory-types.js';
export interface RouteTeacherReport {
    teacherRuns: number;
    counterfactuals: number;
    examples: number;
    policySnapshotId?: string;
    policySnapshotV3Id?: string;
    routeFramesV3?: number;
    pairExamplesV3?: number;
}
export declare class RouteTeacher {
    private store;
    private config;
    private client?;
    constructor(options: {
        store: MemoryStore;
        config: any;
        client?: LlmClient | null;
    });
    run(agentId: string): Promise<RouteTeacherReport>;
    teachDecision(agentId: string, decision: RouteDecision): Promise<RouteTeacherReport>;
}
export declare function buildRouteGraphSnapshot(store: MemoryStore, agentId: string, routeDecisionId: string, queries: string[], candidates: MemoryNode[], graphDepth: 0 | 1 | 2): RouteGraphSnapshot;
