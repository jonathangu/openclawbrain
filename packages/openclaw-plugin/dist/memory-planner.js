import { ContextSelector } from './context-selector.js';
import { FeedbackDistiller } from './feedback-distiller.js';
import { MemoryOperationApplier } from './memory-operations.js';
export class MemoryPlanner {
    config;
    routeFn;
    contextSelector;
    distiller;
    store;
    constructor(options) {
        this.config = options.config;
        this.routeFn = options.routeFn;
        this.contextSelector = new ContextSelector(options.config);
        this.distiller = options.client ? new FeedbackDistiller({ client: options.client, config: options.config }) : undefined;
        this.store = options.store;
    }
    async run(packet) {
        const routePlan = this.routeFn.plan(packet);
        let feedbackDistillation;
        if (routePlan.enqueueCapture && this.distiller) {
            const result = await this.distiller.distill(packet);
            feedbackDistillation = result.output;
            if (feedbackDistillation.shouldStore || feedbackDistillation.injectionFeedback.length > 0) {
                new MemoryOperationApplier({ store: this.store, config: this.config }).applyDistillation(feedbackDistillation, packet);
            }
        }
        if (!routePlan.shouldRetrieve)
            return { routePlan, feedbackDistillation };
        const candidates = retrieveCandidates(this.store, packet.agentId, routePlan.retrievalPlan.queries, routePlan.retrievalPlan.memoryTypes, routePlan.retrievalPlan.maxCandidates);
        const contextSelection = this.contextSelector.select({ packet, plan: routePlan, candidates });
        return { routePlan, feedbackDistillation, contextSelection };
    }
}
function retrieveCandidates(store, agentId, queries, memoryTypes, maxCandidates) {
    const seen = new Set();
    const results = [];
    for (const query of queries) {
        for (const candidate of store.searchMemories(query, agentId, { limit: maxCandidates })) {
            if (seen.has(candidate.id))
                continue;
            seen.add(candidate.id);
            results.push(candidate);
            if (results.length >= maxCandidates)
                return results;
        }
    }
    for (const memoryType of memoryTypes) {
        for (const candidate of store.listMemories(agentId, { type: memoryType, limit: maxCandidates })) {
            if (seen.has(candidate.id))
                continue;
            seen.add(candidate.id);
            results.push(candidate);
            if (results.length >= maxCandidates)
                return results;
        }
    }
    return results;
}
