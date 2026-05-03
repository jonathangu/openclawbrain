import type { MemoryNode } from './memory-types.js';
export declare function buildMemoryPromptSupplement(): () => string[];
export declare function buildMemoryCorpusSupplement(config: any): {
    search: ({ query, maxResults }: {
        query: string;
        maxResults?: number;
        agentSessionKey?: string;
    }) => Promise<{
        corpus: string;
        path: string;
        title: string;
        kind: import("./memory-types.js").MemoryType;
        score: number;
        snippet: string;
        id: string;
        startLine: number;
        endLine: number;
        citation: string;
        source: string;
        provenanceLabel: string;
        updatedAt: string;
    }[]>;
    get: ({ lookup }: {
        lookup: string;
        fromLine?: number;
        lineCount?: number;
        agentSessionKey?: string;
    }) => Promise<{
        corpus: string;
        path: string;
        title: string;
        kind: import("./memory-types.js").MemoryType;
        content: string;
        fromLine: number;
        lineCount: number;
        id: string;
        provenanceLabel: string;
        updatedAt: string;
    } | null>;
};
export declare function searchPayload(config: any, agentId: string, query: string, limit?: number): {
    ok: boolean;
    agentId: string;
    reason: string;
} | {
    ok: boolean;
    agentId: string;
    query: string;
    limit: number;
    results: {
        id: string;
        type: import("./memory-types.js").MemoryType;
        normalizedKey: string;
        score: number;
        content: string;
        tags: string[];
        updatedAt: string;
    }[];
};
export declare function graphPayload(config: any, agentId: string, limit?: number): {
    ok: boolean;
    agentId: string;
    reason: string;
} | {
    ok: boolean;
    agentId: string;
    counts: {
        nodes: number;
        edges: number;
    };
    nodes: {
        id: string;
        type: import("./memory-types.js").MemoryType;
        normalizedKey: string;
        content: string;
        importance: number;
        confidence: number;
        supersededBy: string | null;
    }[];
    edges: any[];
};
export declare function learnPayload(config: any, agentId: string, limit?: number): {
    ok: boolean;
    agentId: string;
    reason: string;
} | {
    ok: boolean;
    agentId: string;
    activePolicySnapshot: import("./memory-types.js").RoutePolicySnapshot | null;
    examples: import("./memory-types.js").RouteExample[];
    policySnapshots: import("./memory-types.js").RoutePolicySnapshot[];
};
export declare function auditPayload(config: any, agentId: string, limit?: number): {
    ok: boolean;
    agentId: string;
    reason: string;
} | {
    ok: boolean;
    agentId: string;
    limit: number;
    captureOpportunityRate: number;
    storageAcceptanceRate: number;
    rejectionDistribution: Record<string, number>;
    rows: {
        id: any;
        turnId: any;
        sessionId: any;
        createdAt: any;
        retrievalIntent: any;
        shouldRetrieve: boolean;
        captureIntent: any;
        shouldConsiderCapture: boolean;
        captureJobCreated: any;
        distillerRan: any;
        fallbackRan: any;
        candidateCount: any;
        storedCount: any;
        rejectedCount: any;
        rejectionReasons: any;
        safeCandidatePreview: any;
    }[];
};
export declare function explainLastPayload(config: any, agentId: string, turnId?: string): {
    ok: boolean;
    agentId: string;
    reason: string;
} | {
    ok: boolean;
    agentId: string;
    turnId: string | null;
    createdAt: string;
    summary: string;
    retrieval: {
        intent: any;
        shouldRetrieve: boolean;
        includeRecallRules: boolean;
    };
    capture: {
        signalFound: boolean;
        intent: any;
        confidence: any;
        reason: any;
        matchedSignals: any;
    };
    distiller: {
        ran: boolean;
        model: string | null;
        latencyMs: number | null;
        fallbackRan: boolean;
    };
    storage: {
        candidateCount: number;
        storedCount: number;
        rejectedCount: number;
        reasons: string[];
        safeCandidatePreview: string | null;
    };
};
export declare function extractMemoryId(lookup: string): string;
export declare function renderMemory(memory: MemoryNode): string;
export declare function memoryPath(memory: MemoryNode): string;
