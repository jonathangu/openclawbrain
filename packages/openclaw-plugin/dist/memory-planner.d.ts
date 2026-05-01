import type { ContextSelection, FeedbackDistillation } from './memory-types.js';
import type { TurnEventPacket } from './capture.js';
import type { LlmClient } from './llm-client.js';
import type { MemoryStore } from './memory-store.js';
import type { RoutePlan } from './route-fn.js';
import { RouteFn } from './route-fn.js';
export interface MemoryPlannerResult {
    routePlan: RoutePlan;
    feedbackDistillation?: FeedbackDistillation;
    contextSelection?: ContextSelection;
}
export interface MemoryPlannerRunOptions {
    timeoutMs?: number;
    fallback?: () => RoutePlan;
}
export declare class MemoryPlanner {
    private config;
    private routeFn;
    private contextSelector;
    private distiller?;
    private store;
    private client?;
    constructor(options: {
        config: any;
        routeFn: RouteFn;
        store: MemoryStore;
        client?: LlmClient;
    });
    run(packet: TurnEventPacket, options?: MemoryPlannerRunOptions): Promise<MemoryPlannerResult>;
    private runInner;
    private runWithFallback;
    private planWithLlm;
}
