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
    activePolicySnapshot: import("./memory-types.js").RoutePolicySnapshot | null;
    examples: import("./memory-types.js").RouteExample[];
    policySnapshots: import("./memory-types.js").RoutePolicySnapshot[];
};
