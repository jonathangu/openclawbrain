import type { TurnEventPacket } from './capture.js';
import type { MemoryStore } from './memory-store.js';
export interface BackgroundLearningReport {
    outcomeResolutions: number;
    routeDecisionsResolved: number;
    routeExamplesCreated: number;
    memoryUpdates: number;
    snapshotId?: string;
    consolidatedMemories?: number;
    prunedMemories: number;
    lastRunAt: string;
}
export declare class BackgroundLearner {
    private store;
    private config;
    constructor(options: {
        store: MemoryStore;
        config: any;
    });
    processOutcomeClassification(agentId: string, packet: TurnEventPacket): BackgroundLearningReport;
    processAgentEnd(agentId: string, packet: TurnEventPacket): BackgroundLearningReport;
    runMaintenance(agentId: string): BackgroundLearningReport;
}
