import type { ContextSelection, MemoryNode } from './memory-types.js';
import type { TurnEventPacket } from './capture.js';
import type { RoutePlan } from './route-fn.js';
import type { MemoryStore } from './memory-store.js';
export declare class ContextSelector {
    private config;
    constructor(config: any);
    select(input: {
        packet: TurnEventPacket;
        plan: RoutePlan;
        candidates: MemoryNode[];
        store?: MemoryStore;
    }): ContextSelection;
}
