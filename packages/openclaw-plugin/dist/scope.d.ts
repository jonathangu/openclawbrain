import type { MemoryNode } from './memory-types.js';
import type { TurnEventPacket } from './capture.js';
export interface ScopeContext {
    agentId: string;
    sessionId?: string;
    sessionKey?: string;
    channelId?: string;
    messageProvider?: string;
    repo?: string;
    project?: string;
    app?: string;
    task?: string;
    tool?: string;
    person?: string;
}
export declare function scopeContextFromPacket(packet?: Partial<TurnEventPacket>): ScopeContext;
export declare function defaultScopeContext(agentId?: string): ScopeContext;
export declare function memoryInScope(memory: MemoryNode, context: ScopeContext): boolean;
export declare function filterMemoriesForScope(memories: MemoryNode[], context?: ScopeContext): MemoryNode[];
