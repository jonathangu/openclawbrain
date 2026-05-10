export type CodexBridgeSource = 'app_server' | 'sqlite_fallback' | 'none' | 'mock';
export type CodexBridgeEventClass = 'completion' | 'failure' | 'blocker' | 'approval_required' | 'auth_failure' | 'status_snapshot' | 'watch_created' | 'handoff' | 'quiet';
export interface CodexBridgeConfig {
    enabled: boolean;
    statePaths: string[];
    bridgeStatePath: string;
    preferAppServer: boolean;
    appServerCommand: string;
    appServerArgs: string[];
    appServerTimeoutMs: number;
    staleAfterMs: number;
    maxThreads: number;
    watchPollIntervalMs: number;
    enableTelegramWrites: boolean;
    trustedTelegramSenders: string[];
    repoAllowlist: string[];
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
        canSubscribe: boolean;
        canStartTurn: boolean;
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
    verbosity: 'completion_only' | 'blockers_and_completion' | 'periodic_digest';
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
export interface CodexBridgeDeps {
    appServerReader?: CodexAppServerReader;
    nowMs?: () => number;
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
    recordEvent(input: Omit<CodexBridgeEvent, 'id' | 'createdAt'> & {
        id?: string;
        createdAt?: string;
    }): CodexBridgeEvent;
    listEvents(agentId: string, limit?: number): CodexBridgeEvent[];
    private migrate;
}
export declare function formatCodexStatus(status: CodexBridgeStatus): string;
export declare function formatCodexThreads(status: CodexBridgeStatus, filter?: string): string;
export declare function formatHandoffBrief(brief: CodexHandoffBrief): string;
