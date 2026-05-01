import { now } from './memory-store.js';
export class JobQueue {
    store;
    constructor(options) {
        this.store = options.store;
    }
    enqueue(agentId, kind, payload, options = {}) {
        const availableAt = options.delayMs
            ? new Date(Date.now() + options.delayMs).toISOString()
            : now();
        return this.store.enqueueJob({
            agentId,
            kind,
            priority: options.priority ?? 0,
            payload,
            maxAttempts: options.maxAttempts ?? 3,
            availableAt,
        });
    }
    enqueueFeedbackDistillation(agentId, payload, options = {}) {
        return this.enqueue(agentId, 'feedback_distillation', payload, { priority: options.priority ?? 10, delayMs: options.delayMs });
    }
    enqueueRouteLearning(agentId, payload, options = {}) {
        return this.enqueue(agentId, 'route_learning', payload, { priority: options.priority ?? 5, delayMs: options.delayMs });
    }
    enqueueOutcomeClassification(agentId, payload, options = {}) {
        return this.enqueue(agentId, 'outcome_classification', payload, { priority: options.priority ?? 5, delayMs: options.delayMs });
    }
    enqueueConsolidation(agentId, payload, options = {}) {
        return this.enqueue(agentId, 'consolidation', payload, { priority: options.priority ?? 1, delayMs: options.delayMs });
    }
    claimNext(kind) {
        return this.store.claimNextJob(kind);
    }
    complete(jobId) {
        this.store.completeJob(jobId);
    }
    fail(jobId, error, retryAfterMs) {
        this.store.failJob(jobId, error, retryAfterMs);
    }
    depth(agentId) {
        return this.store.getJobQueueDepth(agentId);
    }
}
