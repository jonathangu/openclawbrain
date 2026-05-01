import type { MemoryStore } from './memory-store.js';
export interface RouteLearningRunReport {
    resolvedDecisions: number;
    examplesCreated: number;
    memoryUpdates: number;
    snapshotId?: string;
}
export declare class RouteLearning {
    private store;
    private config;
    constructor(options: {
        store: MemoryStore;
        config: any;
    });
    run(agentId: string): RouteLearningRunReport;
}
