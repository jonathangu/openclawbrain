import type { FeedbackDistillation } from './memory-types.js';
import type { TurnEventPacket } from './capture.js';
import { MemoryStore } from './memory-store.js';
import { type CaptureIntentResult } from './capture-intent.js';
export interface ApplyFeedbackResult {
    memoryIds: string[];
    storedCandidates: number;
    rejectedCandidates: number;
    rejectionReasons: string[];
    resolvedInjections: number;
    deletedOrSuppressed: number;
}
export declare class MemoryOperationApplier {
    private store;
    private config;
    constructor(options: {
        store: MemoryStore;
        config: any;
    });
    applyDistillation(distillation: FeedbackDistillation, packet: TurnEventPacket, context?: {
        captureIntent?: CaptureIntentResult;
    }): ApplyFeedbackResult;
    private applyDeleteOrSuppress;
    private isSafeToStore;
    private upsertCandidate;
}
