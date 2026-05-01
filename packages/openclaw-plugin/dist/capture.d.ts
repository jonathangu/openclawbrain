export interface TurnEventPacket {
    agentId: string;
    sessionId?: string;
    sessionKey?: string;
    turnId?: string;
    runId?: string;
    sourceHook: string;
    latestUserMessageRedacted: string;
    recentAssistantMessage?: string;
    toolObservations: Array<{
        toolName: string;
        ok: boolean;
        durationMs?: number;
        argsSummary?: string;
        resultSummary?: string;
        errorClass?: string;
    }>;
    recentInjections: Array<{
        injectionId: string;
        memoryId: string;
        outcome?: string;
    }>;
    metadata: Record<string, unknown>;
}
export declare class CaptureOrchestrator {
    fromBeforePromptBuild(event?: any, config?: any): TurnEventPacket;
    fromAgentEnd(event?: any, config?: any): TurnEventPacket;
    fromAfterToolCall(event?: any, config?: any): TurnEventPacket;
}
export declare function sanitizeToolEvent(event?: any, config?: any): {
    toolName: string;
    ok: boolean;
    durationMs: number | undefined;
    argsSummary: string;
    resultSummary: string;
    errorClass: string;
};
