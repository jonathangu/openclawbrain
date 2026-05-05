import type { BackgroundJob, JobKind } from './memory-types.js';
import { MemoryStore } from './memory-store.js';
export interface JobQueueOptions {
    store: MemoryStore;
}
export declare class JobQueue {
    private store;
    constructor(options: JobQueueOptions);
    enqueue(agentId: string, kind: JobKind, payload: Record<string, unknown>, options?: {
        priority?: number;
        maxAttempts?: number;
        delayMs?: number;
    }): BackgroundJob;
    enqueueFeedbackDistillation(agentId: string, payload: Record<string, unknown>, options?: {
        priority?: number;
        delayMs?: number;
    }): BackgroundJob;
    enqueueRouteLearning(agentId: string, payload: Record<string, unknown>, options?: {
        priority?: number;
        delayMs?: number;
    }): BackgroundJob;
    enqueueRouteTeacher(agentId: string, payload: Record<string, unknown>, options?: {
        priority?: number;
        delayMs?: number;
    }): BackgroundJob;
    enqueueOutcomeClassification(agentId: string, payload: Record<string, unknown>, options?: {
        priority?: number;
        delayMs?: number;
    }): BackgroundJob;
    enqueueConsolidation(agentId: string, payload: Record<string, unknown>, options?: {
        priority?: number;
        delayMs?: number;
    }): BackgroundJob;
    claimNext(kind?: JobKind, agentId?: string): BackgroundJob | null;
    complete(jobId: string): void;
    fail(jobId: string, error: string, retryAfterMs?: number): void;
    depth(agentId?: string): number;
}
