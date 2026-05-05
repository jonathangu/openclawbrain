import type { BackgroundJob, JobKind } from './memory-types.js';
import { MemoryStore, now } from './memory-store.js';

export interface JobQueueOptions {
  store: MemoryStore;
}

export class JobQueue {
  private store: MemoryStore;

  constructor(options: JobQueueOptions) {
    this.store = options.store;
  }

  enqueue(agentId: string, kind: JobKind, payload: Record<string, unknown>, options: { priority?: number; maxAttempts?: number; delayMs?: number } = {}): BackgroundJob {
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

  enqueueFeedbackDistillation(agentId: string, payload: Record<string, unknown>, options: { priority?: number; delayMs?: number } = {}) {
    return this.enqueue(agentId, 'feedback_distillation', payload, { priority: options.priority ?? 10, delayMs: options.delayMs });
  }

  enqueueRouteLearning(agentId: string, payload: Record<string, unknown>, options: { priority?: number; delayMs?: number } = {}) {
    return this.enqueue(agentId, 'route_learning', payload, { priority: options.priority ?? 5, delayMs: options.delayMs });
  }

  enqueueRouteTeacher(agentId: string, payload: Record<string, unknown>, options: { priority?: number; delayMs?: number } = {}) {
    return this.enqueue(agentId, 'route_teacher', payload, { priority: options.priority ?? 4, delayMs: options.delayMs, maxAttempts: 2 });
  }

  enqueueOutcomeClassification(agentId: string, payload: Record<string, unknown>, options: { priority?: number; delayMs?: number } = {}) {
    return this.enqueue(agentId, 'outcome_classification', payload, { priority: options.priority ?? 5, delayMs: options.delayMs });
  }

  enqueueConsolidation(agentId: string, payload: Record<string, unknown>, options: { priority?: number; delayMs?: number } = {}) {
    return this.enqueue(agentId, 'consolidation', payload, { priority: options.priority ?? 1, delayMs: options.delayMs });
  }

  claimNext(kind?: JobKind, agentId?: string) {
    return this.store.claimNextJob(kind, agentId);
  }

  complete(jobId: string) {
    this.store.completeJob(jobId);
  }

  fail(jobId: string, error: string, retryAfterMs?: number) {
    this.store.failJob(jobId, error, retryAfterMs);
  }

  depth(agentId?: string) {
    return this.store.getJobQueueDepth(agentId);
  }
}
