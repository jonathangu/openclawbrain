export { normalizePluginConfig, resolveOpenClawBrainConfig } from './config.js';
export { redactText, hashText } from './redact.js';
export { decidePolicy, classifyTurn } from './policy.js';
export { readActivationContext } from './context-files.js';
export { appendProofEvent, readProofEvents, readStatus, writeStatus } from './proof-store.js';
export { buildStatus } from './status.js';
export declare const openClawBrainPluginEntry: {
    id: string;
    name: string;
    version: string;
    register(api?: any): void;
};
export default openClawBrainPluginEntry;
export declare function redactedTurnFromPromptEvent(event?: any, config?: any): {
    turnId: string;
    agentId: string;
    promptHash: string;
    summary: string;
    sessionKeyHash: string;
    sessionIdHash: string;
    runIdHash: string;
    openclawProfile: string;
    turnType: string;
};
export declare function handleTurnHook(event?: any, config?: any, api?: any, phase?: string): Promise<{
    prependContext?: undefined;
} | {
    prependContext: string;
}>;
