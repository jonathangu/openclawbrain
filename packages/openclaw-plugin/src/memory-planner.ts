import type { ContextSelection, FeedbackDistillation } from './memory-types.js';
import type { TurnEventPacket } from './capture.js';
import { ContextSelector } from './context-selector.js';
import { FeedbackDistiller } from './feedback-distiller.js';
import type { LlmClient } from './llm-client.js';
import { MemoryOperationApplier } from './memory-operations.js';
import type { MemoryStore } from './memory-store.js';
import type { RoutePlan } from './route-fn.js';
import { RouteFn } from './route-fn.js';

export interface MemoryPlannerResult {
  routePlan: RoutePlan;
  feedbackDistillation?: FeedbackDistillation;
  contextSelection?: ContextSelection;
}

export class MemoryPlanner {
  private config: any;
  private routeFn: RouteFn;
  private contextSelector: ContextSelector;
  private distiller?: FeedbackDistiller;
  private store: MemoryStore;

  constructor(options: { config: any; routeFn: RouteFn; store: MemoryStore; client?: LlmClient }) {
    this.config = options.config;
    this.routeFn = options.routeFn;
    this.contextSelector = new ContextSelector(options.config);
    this.distiller = options.client ? new FeedbackDistiller({ client: options.client, config: options.config }) : undefined;
    this.store = options.store;
  }

  async run(packet: TurnEventPacket): Promise<MemoryPlannerResult> {
    const routePlan = this.routeFn.plan(packet);
    let feedbackDistillation: FeedbackDistillation | undefined;
    if (routePlan.enqueueCapture && this.distiller) {
      const result = await this.distiller.distill(packet);
      feedbackDistillation = result.output;
      if (feedbackDistillation.shouldStore || feedbackDistillation.injectionFeedback.length > 0) {
        new MemoryOperationApplier({ store: this.store, config: this.config }).applyDistillation(feedbackDistillation, packet);
      }
    }
    if (!routePlan.shouldRetrieve) return { routePlan, feedbackDistillation };
    const candidates = retrieveCandidates(this.store, packet.agentId, routePlan.retrievalPlan.queries, routePlan.retrievalPlan.memoryTypes, routePlan.retrievalPlan.maxCandidates);
    const contextSelection = this.contextSelector.select({ packet, plan: routePlan, candidates });
    return { routePlan, feedbackDistillation, contextSelection };
  }
}

function retrieveCandidates(store: MemoryStore, agentId: string, queries: string[], memoryTypes: string[], maxCandidates: number) {
  const seen = new Set<string>();
  const results: any[] = [];
  for (const query of queries) {
    for (const candidate of store.searchMemories(query, agentId, { limit: maxCandidates })) {
      if (seen.has(candidate.id)) continue;
      seen.add(candidate.id);
      results.push(candidate);
      if (results.length >= maxCandidates) return results;
    }
  }
  for (const memoryType of memoryTypes) {
    for (const candidate of store.listMemories(agentId, { type: memoryType as any, limit: maxCandidates })) {
      if (seen.has(candidate.id)) continue;
      seen.add(candidate.id);
      results.push(candidate);
      if (results.length >= maxCandidates) return results;
    }
  }
  return results;
}
