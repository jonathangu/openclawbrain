import type { MemoryAuthorityDecisionKind, MemoryAuthorityEventType, MemoryAuthorityResolution, MemoryNode, MemoryValidity } from './memory-types.js';
import type { TurnEventPacket } from './capture.js';
import type { RoutePlan } from './route-fn.js';
import type { MemoryStore } from './memory-store.js';
export interface MemoryAuthorityResolverOptions {
    config?: any;
    store?: MemoryStore;
}
export interface MemoryAuthorityResolverInput {
    packet: TurnEventPacket;
    plan: RoutePlan;
    candidates: MemoryNode[];
}
export declare class MemoryAuthorityResolver {
    private config;
    private store?;
    constructor(options?: MemoryAuthorityResolverOptions);
    resolve(input: MemoryAuthorityResolverInput): MemoryAuthorityResolution[];
    resolveOne(memory: MemoryNode, packet: TurnEventPacket, plan: RoutePlan): MemoryAuthorityResolution;
}
export declare function defaultValidityForMemory(memory: MemoryNode, overrides?: Partial<MemoryValidity>): MemoryValidity;
export declare function authorityEventTypeForDecision(decisionKind: MemoryAuthorityDecisionKind): MemoryAuthorityEventType;
