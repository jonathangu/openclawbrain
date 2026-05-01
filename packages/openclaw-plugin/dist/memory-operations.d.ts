import type { FeedbackDistillation } from './memory-types.js';
import type { TurnEventPacket } from './capture.js';
import { MemoryStore } from './memory-store.js';
export interface ApplyFeedbackResult {
    memoryIds: string[];
    storedCandidates: number;
    resolvedInjections: number;
}
export declare class MemoryOperationApplier {
    private store;
    private config;
    constructor(options: {
        store: MemoryStore;
        config: any;
    });
    applyDistillation(distillation: FeedbackDistillation, packet: TurnEventPacket): ApplyFeedbackResult;
    private upsertCandidate;
}
