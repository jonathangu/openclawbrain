import type { ContextSelection, MemoryNode } from './memory-types.js';
import type { TurnEventPacket } from './capture.js';
import type { RoutePlan } from './route-fn.js';
export declare class ContextSelector {
    private config;
    constructor(config: any);
    select(input: {
        packet: TurnEventPacket;
        plan: RoutePlan;
        candidates: MemoryNode[];
    }): ContextSelection;
}
