export type CodexBridgeSource = 'app_server' | 'sqlite_fallback' | 'none' | 'mock';
export type CodexBridgeEventClass = 'completion' | 'failure' | 'blocker' | 'approval_required' | 'auth_failure' | 'assistant_message' | 'user_message' | 'turn_started' | 'turn_completed' | 'status_snapshot' | 'watch_created' | 'binding_created' | 'binding_removed' | 'outbound_write' | 'delivery_failed' | 'handoff' | 'quiet';
export interface CodexBridgeConfig {
    enabled: boolean;
    statePaths: string[];
    bridgeStatePath: string;
    preferAppServer: boolean;
    appServerCommand: string;
    appServerArgs: string[];
    appServerUrl: string;
    appServerTimeoutMs: number;
    staleAfterMs: number;
    maxThreads: number;
    watchPollIntervalMs: number;
    messageWatchesEnabled: boolean;
    directMessageCopyEnabled: boolean;
    telegramForwardingMode: 'redacted' | 'raw_trusted' | 'metadata_only';
    enableTelegramWrites: boolean;
    enableTelegramSteer: boolean;
    trustOpenClawAuth: boolean;
    allowLatestTargetForWrites: boolean;
    highRiskTelegramWrites: boolean;
    trustedTelegramSenders: string[];
    repoAllowlist: string[];
    readAllowlist: string[];
    writeAllowlist: string[];
    destructiveWriteAllowlist: string[];
    notifyChannel: string;
    notifyTarget: string;
}
export interface CodexThreadSummary {
    id: string;
    title: string;
    cwd: string;
    branch?: string;
    sha?: string;
    model?: string;
    reasoningEffort?: string;
    rolloutPath?: string;
    firstUserMessage?: string;
    updatedAtMs: number;
    archived: boolean;
    goal?: CodexGoalSummary;
    source: CodexBridgeSource;
}
export interface CodexGoalSummary {
    goalId?: string;
    objective: string;
    status: string;
    tokenBudget?: number;
    tokensUsed: number;
    timeUsedSeconds: number;
    updatedAtMs: number;
}
export interface CodexBridgeStatus {
    ok: boolean;
    bridge: 'openclawbrain-codex-continuity';
    source: CodexBridgeSource;
    stale: boolean;
    staleReason?: string;
    generatedAt: string;
    capabilities: {
        canReadThreads: boolean;
        canReadGoals: boolean;
        canReadMessages: boolean;
        canSubscribe: boolean;
        canStartTurn: boolean;
        canSteerTurn: boolean;
        canWrite: boolean;
        appServerAvailable: boolean;
        sqliteFallbackAvailable: boolean;
    };
    counts: {
        threads: number;
        activeGoals: number;
        watched: number;
    };
    latestThreads: CodexThreadSummary[];
    activeGoals: CodexThreadSummary[];
    errors: string[];
    writeControl: {
        enabled: boolean;
        reason: string;
    };
}
export interface CodexHandoffBrief {
    ok: boolean;
    source: CodexBridgeSource;
    threadId?: string;
    title?: string;
    observedFacts: string[];
    codexReportedClaims: string[];
    evidence: string[];
    interpretation: string[];
    nextActions: string[];
    stale: boolean;
    generatedAt: string;
}
export interface CodexBridgeWatch {
    id: string;
    agentId: string;
    scope: 'thread' | 'repo' | 'goal';
    threadId?: string;
    goalKey?: string;
    notifyChannel: string;
    notifyTarget: string;
    accountId?: string;
    messageThreadId?: string;
    allowedClasses: CodexBridgeEventClass[];
    expiresAt?: string;
    status: 'active' | 'expired' | 'completed' | 'paused';
    dedupeKeyLastSeen?: string;
    lastEventAt?: string;
    sensitivity: 'normal' | 'sensitive' | 'no_telegram_details';
    verbosity: 'completion_only' | 'blockers_and_completion' | 'periodic_digest' | 'terminal_only' | 'assistant_messages' | 'messages_and_terminal' | 'explicit_all';
    createdAt: string;
    updatedAt: string;
}
export interface CodexBridgeEvent {
    id: string;
    agentId: string;
    eventType: string;
    eventClass: CodexBridgeEventClass;
    source: CodexBridgeSource;
    threadId?: string;
    goalKey?: string;
    decision: 'notified' | 'suppressed' | 'recorded' | 'rejected';
    notified: boolean;
    reason: string;
    redactedSummary: string;
    dedupeKey: string;
    createdAt: string;
}
export interface CodexAppServerReader {
    listThreads(options: {
        limit: number;
        searchTerm?: string;
        timeoutMs: number;
    }): Promise<unknown>;
}
export interface CodexAppServerWriter {
    sendMessage(input: {
        threadId: string;
        cwd?: string;
        model?: string;
        message: string;
        timeoutMs: number;
    }): Promise<{
        ok: boolean;
        turnId?: string;
        status?: string;
        activeTurnId?: string;
        possiblySent?: boolean;
        error?: string;
    }>;
    steerMessage?(input: {
        threadId: string;
        cwd?: string;
        model?: string;
        message: string;
        expectedTurnId?: string;
        timeoutMs: number;
    }): Promise<{
        ok: boolean;
        turnId?: string;
        status?: string;
        activeTurnId?: string;
        possiblySent?: boolean;
        error?: string;
    }>;
}
export interface CodexBridgeDeps {
    appServerReader?: CodexAppServerReader;
    appServerWriter?: CodexAppServerWriter;
    nowMs?: () => number;
}
export interface CodexTranscriptMessage {
    id: string;
    threadId: string;
    role: 'user' | 'assistant';
    text: string;
    timestamp: string;
    source: 'rollout_jsonl' | 'event_msg';
    lineNumber: number;
    byteOffset: number;
    messageKind: 'final_message' | 'ui_event';
    hash: string;
}
export interface CodexConversationBinding {
    id: string;
    agentId: string;
    chatKeyHash: string;
    senderKeyHash: string;
    threadId: string;
    title?: string;
    cwd?: string;
    createdAt: string;
    updatedAt: string;
}
export declare const DEFAULT_CODEX_BRIDGE_CONFIG: CodexBridgeConfig;
export declare function normalizeCodexBridgeConfig(source?: any): CodexBridgeConfig;
export declare function buildCodexBridgeStatus(config: any, agentId?: string, deps?: CodexBridgeDeps): Promise<CodexBridgeStatus>;
export declare function readCodexThreadsFromSqlite(bridgeConfig: CodexBridgeConfig, options?: {
    limit?: number;
    searchTerm?: string;
    nowMs?: number;
}): {
    threads: CodexThreadSummary[];
    sourcePath?: string;
    errors: string[];
};
export declare function readCodexTranscriptMessages(thread: CodexThreadSummary, options?: {
    limit?: number;
    role?: 'assistant' | 'user' | 'all';
    afterLine?: number;
}): {
    ok: boolean;
    messages: CodexTranscriptMessage[];
    errors: string[];
    truncated: boolean;
    rolloutPath?: string;
};
export declare function formatCodexMessages(thread: CodexThreadSummary, result: {
    messages: CodexTranscriptMessage[];
    errors?: string[];
}, options?: {
    title?: string;
    full?: boolean;
}): string;
export declare function buildCodexHandoff(status: CodexBridgeStatus, threadId?: string): CodexHandoffBrief;
export declare function handleBrainCommand(ctx: any, config: any, api?: any): Promise<{
    text: string;
    continueAgent?: boolean;
}>;
export declare function processCodexBridgeWatches(config: any, api?: any, deps?: CodexBridgeDeps): Promise<{
    ok: boolean;
    processed: number;
    notified: number;
}>;
export declare class CodexBridgeStore {
    private db;
    constructor(options: {
        config: any;
        agentId: string;
    });
    close(): void;
    createWatch(input: Omit<CodexBridgeWatch, 'id' | 'status' | 'dedupeKeyLastSeen' | 'lastEventAt' | 'createdAt' | 'updatedAt'> & {
        id?: string;
        expiresAt?: string;
    }): CodexBridgeWatch;
    getWatch(id: string): CodexBridgeWatch | null;
    listWatches(agentId: string, options?: {
        activeOnly?: boolean;
    }): CodexBridgeWatch[];
    updateWatchEvent(id: string, patch: {
        dedupeKeyLastSeen?: string;
        lastEventAt?: string;
        status?: CodexBridgeWatch['status'];
    }): CodexBridgeWatch | null;
    pauseWatches(agentId: string, target: string): number;
    recordEvent(input: Omit<CodexBridgeEvent, 'id' | 'createdAt'> & {
        id?: string;
        createdAt?: string;
    }): CodexBridgeEvent;
    listEvents(agentId: string, limit?: number): CodexBridgeEvent[];
    bindConversation(input: {
        agentId: string;
        chatKey: string;
        senderKey: string;
        thread: CodexThreadSummary;
    }): CodexConversationBinding;
    getBinding(agentId: string, chatKey: string): CodexConversationBinding | null;
    unbindConversation(agentId: string, chatKey: string): boolean;
    getMessageCursor(watchId: string, threadId: string, rolloutPath: string): any | null;
    upsertMessageCursor(input: {
        watchId: string;
        agentId: string;
        threadId: string;
        rolloutPath: string;
        parseCursorLine: number;
        parseCursorByteOffset: number;
        deliveryCursorLine: number;
        deliveryCursorByteOffset: number;
        lastMessageId?: string;
        lastMessageHash?: string;
        fileIdentity?: string;
    }): void;
    recordPendingDelivery(input: {
        watchId: string;
        threadId: string;
        message: CodexTranscriptMessage;
        chatKey: string;
        status: string;
        error?: string;
    }): void;
    recordOutbound(input: {
        agentId: string;
        sourceChannel: string;
        sourceSender: string;
        sourceMessageId?: string;
        threadId: string;
        repoPath?: string;
        riskClass: string;
        confirmationState: string;
        appServerMethod?: string;
        appServerTurnId?: string;
        status: string;
        redactedPreview: string;
        error?: string;
        idempotencyKey: string;
    }): void;
    private migrate;
}
export declare function formatCodexStatus(status: CodexBridgeStatus): string;
export declare function formatCodexThreads(status: CodexBridgeStatus, filter?: string): string;
export declare function formatHandoffBrief(brief: CodexHandoffBrief): string;
