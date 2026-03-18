export interface OpenClawInjectedWorkspaceFile {
    name: string;
    path: string;
    missing: boolean;
    rawChars: number;
    injectedChars: number;
    truncated: boolean;
}
export interface OpenClawToolSurfaceEntry {
    name: string;
    summaryChars: number;
    schemaChars: number;
    propertiesCount: number;
}
export interface OpenClawSystemPromptReport {
    source?: string;
    generatedAt?: number;
    sessionId?: string;
    sessionKey?: string;
    provider?: string;
    model?: string;
    workspaceDir?: string;
    injectedWorkspaceFiles?: readonly OpenClawInjectedWorkspaceFile[];
    tools?: {
        listChars?: number;
        schemaChars?: number;
        entries?: readonly OpenClawToolSurfaceEntry[];
        [key: string]: unknown;
    };
    [key: string]: unknown;
}
export interface OpenClawSessionIndexEntry {
    sessionId: string;
    updatedAt: number;
    sessionFile?: string;
    model?: string;
    modelProvider?: string;
    chatType?: string;
    origin?: Record<string, unknown>;
    deliveryContext?: Record<string, unknown>;
    systemPromptReport?: OpenClawSystemPromptReport;
    [key: string]: unknown;
}
export type OpenClawSessionIndex = Record<string, OpenClawSessionIndexEntry>;
export interface OpenClawSessionTextPart {
    type: "text";
    text: string;
    textSignature?: string;
    [key: string]: unknown;
}
export interface OpenClawSessionThinkingPart {
    type: "thinking";
    thinking: string;
    thinkingSignature?: string;
    [key: string]: unknown;
}
export interface OpenClawSessionToolCallPart {
    type: "toolCall";
    id: string;
    name: string;
    arguments: Record<string, unknown>;
    partialJson?: string;
    [key: string]: unknown;
}
export interface OpenClawSessionUnknownContentPart {
    type: string;
    [key: string]: unknown;
}
export type OpenClawSessionContentPart = OpenClawSessionTextPart | OpenClawSessionThinkingPart | OpenClawSessionToolCallPart | OpenClawSessionUnknownContentPart;
export interface OpenClawSessionMessagePayload {
    role: string;
    content: readonly OpenClawSessionContentPart[];
    timestamp: number;
    toolCallId?: string;
    toolName?: string;
    details?: Record<string, unknown>;
    isError?: boolean;
    api?: string;
    provider?: string;
    model?: string;
    usage?: Record<string, unknown>;
    stopReason?: string;
    [key: string]: unknown;
}
export interface OpenClawSessionHeaderRecord {
    type: "session";
    version: number;
    id: string;
    timestamp: string;
    cwd: string;
}
export interface OpenClawSessionModelChangeRecord {
    type: "model_change";
    id: string;
    parentId: string | null;
    timestamp: string;
    provider: string;
    modelId: string;
}
export interface OpenClawSessionThinkingLevelChangeRecord {
    type: "thinking_level_change";
    id: string;
    parentId: string | null;
    timestamp: string;
    thinkingLevel: string;
}
export interface OpenClawSessionCustomRecord {
    type: "custom";
    customType: string;
    data: Record<string, unknown>;
    id: string;
    parentId: string | null;
    timestamp: string;
}
export interface OpenClawSessionMessageRecord {
    type: "message";
    id: string;
    parentId: string | null;
    timestamp: string;
    message: OpenClawSessionMessagePayload;
}
export type OpenClawSessionRecord = OpenClawSessionHeaderRecord | OpenClawSessionModelChangeRecord | OpenClawSessionThinkingLevelChangeRecord | OpenClawSessionCustomRecord | OpenClawSessionMessageRecord;
export interface OpenClawAcpStreamRecord {
    ts: string;
    epochMs: number;
    runId: string;
    parentSessionKey: string;
    childSessionKey: string;
    agentId: string;
    kind: string;
    contextKey?: string;
    text?: string;
    delta?: string;
    phase?: string;
    data?: Record<string, unknown>;
    [key: string]: unknown;
}
export interface OpenClawMainSessionStoreV1 {
    profileRoot: string;
    agentId: "main";
    sessionsDir: string;
    indexPath: string;
}
/**
 * Generic session store that can represent any agent (main, subagent, ACP, etc.).
 */
export interface OpenClawSessionStoreV1 {
    profileRoot: string;
    agentId: string;
    sessionsDir: string;
    indexPath: string;
}
export declare function loadOpenClawSessionIndex(indexFilePath: string): OpenClawSessionIndex;
export declare function readOpenClawSessionFile(sessionFilePath: string): OpenClawSessionRecord[];
export declare function readOpenClawAcpStreamFile(streamFilePath: string): OpenClawAcpStreamRecord[];
export declare function discoverOpenClawMainSessionStores(options?: {
    homeDir?: string;
    profileRoots?: readonly string[];
}): OpenClawMainSessionStoreV1[];
/**
 * Discover session stores for ALL agents under each profile root.
 * Scans every directory under `agents/` (not just `agents/main/`),
 * finding subagent, ACP, and any other agent session stores.
 */
export declare function discoverOpenClawSessionStores(options?: {
    homeDir?: string;
    profileRoots?: readonly string[];
}): OpenClawSessionStoreV1[];
