import type { MemoryType } from './memory-types.js';
export type MemoryScopeKind = 'global_user' | 'agent' | 'repo' | 'project' | 'app' | 'person' | 'channel' | 'session' | 'task' | 'tool';
export interface MemoryScopeHint {
    kind: MemoryScopeKind;
    key?: string;
}
export type CaptureIntent = 'explicit_store' | 'explicit_update' | 'standing_preference' | 'standing_workflow' | 'project_fact' | 'tool_convention' | 'routing_rule' | 'agent_assignment' | 'recall_rule' | 'sensitive_secret' | 'delete_or_suppress' | 'retrieval_question' | 'one_off' | 'ambiguous';
export type RetrievalIntent = 'needs_memory' | 'may_need_memory' | 'no_retrieval' | 'memory_management' | 'recall_value_request';
export type RiskHint = 'ordinary' | 'private' | 'codeword_like_value' | 'credential_like_secret' | 'ambiguous_sensitive_recall' | 'benign_recall' | 'delete_requested';
export type SensitiveValueRisk = {
    kind: 'ordinary';
    plaintextAllowed: true;
    proactiveInjectionAllowed: true;
    reason: string;
} | {
    kind: 'user_authorized_recall';
    plaintextAllowed: boolean;
    proactiveInjectionAllowed: false;
    reason: string;
} | {
    kind: 'ambiguous_codeword';
    plaintextAllowed: false;
    proactiveInjectionAllowed: false;
    reason: string;
} | {
    kind: 'credential_secret';
    plaintextAllowed: false;
    proactiveInjectionAllowed: false;
    reason: string;
};
export interface CaptureIntentResult {
    shouldConsiderCapture: boolean;
    intent: CaptureIntent;
    confidence: number;
    reason: string;
    matchedSignals: string[];
    riskHints: RiskHint[];
    proposedScope?: MemoryScopeHint;
}
export interface RetrievalIntentResult {
    shouldRetrieve: boolean;
    intent: RetrievalIntent;
    confidence: number;
    query: string;
    scopeHints: MemoryScopeHint[];
    includeRecallRules: boolean;
    memoryTypes: MemoryType[];
}
export declare function detectCaptureIntent(input: {
    latestUserMessageRedacted: string;
    agentId?: string;
    sessionId?: string;
    sessionKey?: string;
}): CaptureIntentResult;
export declare function detectRetrievalIntent(input: {
    latestUserMessageRedacted: string;
    agentId?: string;
    sessionId?: string;
    sessionKey?: string;
}): RetrievalIntentResult;
export declare function classifySensitiveValue(text: string, captureIntent?: CaptureIntent): SensitiveValueRisk;
export declare function captureStoreThreshold(intent: CaptureIntent): number;
