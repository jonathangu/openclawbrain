import { type ActivationPointerSlot, type RetrievalSemanticClassV1, type RoutingChannelV1, type RouteMode, type RuntimeCompileExpectationV1, type RuntimeCompileRequestV1, type RuntimeCompileResponseV1, type RuntimeCompileTargetV1 } from "@openclawbrain/contracts";
import { type PackDescriptor } from "@openclawbrain/pack-format";
export type LoadedPack = PackDescriptor;
export type CompileSelectionMode = "flat_rank_v1" | "graph_walk_v1";
export interface RankedContextBlock {
    blockId: string;
    source: string;
    text: string;
    score: number;
    channelScores: {
        graph: number;
        shortTerm: number;
        vector: number;
    };
    routingChannels: RoutingChannelV1[];
    priority: number;
    matchedTokens: string[];
    directMatchedTokens?: string[];
    semanticSimilarity?: number;
    traversalScore?: number;
    tokenCount: number;
    compactedFrom?: string[];
    packOrder: number;
    candidateSemanticClass: RetrievalSemanticClassV1;
    candidateSemanticEvidence: string[];
}
export interface ActivationCompileOptions {
    slot?: ActivationPointerSlot;
    requireActivationReady?: boolean;
    requirePromotionSafe?: boolean;
    expectedTarget?: RuntimeCompileExpectationV1;
    selectionMode?: CompileSelectionMode;
    expectation?: never;
}
export interface RuntimeCompileOptions {
    selectionMode?: CompileSelectionMode;
}
export interface TextEmbeddingResult {
    model: string;
    values: number[];
}
export interface TextEmbedder {
    model: string;
    embed(input: readonly string[]): Promise<readonly TextEmbeddingResult[]>;
}
export interface OllamaEmbedderOptions {
    baseUrl?: string;
    model?: string;
    fetchImpl?: typeof fetch;
    headers?: Record<string, string>;
}
export interface ActivationCompileResolution {
    slot: ActivationPointerSlot;
    pack: LoadedPack;
    target: RuntimeCompileTargetV1;
}
export interface ActivationCompileResult extends RuntimeCompileResponseV1 {
    slot: ActivationPointerSlot;
    target: RuntimeCompileTargetV1;
    response: RuntimeCompileResponseV1;
    economicsLog: ContextEconomicsLogV1;
}
export interface CompileFallbackUsageReport {
    packId: string;
    modeRequested: RouteMode;
    modeEffective: RouteMode;
    usedLearnedRouteFn: boolean;
    routerIdentity: string | null;
    selectionDigest: string;
    selectionMode: "token_match" | "priority_fallback" | null;
    selectionTiers: "token_match+priority_fallback" | "token_match_only" | "priority_fallback_only" | null;
    priorityFallbackUsed: boolean;
    notes: string[];
}
export interface ContextEconomicsLogV1 {
    packId: string;
    plasticitySource: "candidate_build" | "live_loop";
    requestedBudget: {
        maxBlocks: number;
        maxChars: number | null;
    };
    selectionCounts: {
        candidates: number;
        selected: number;
        overlapDropped: number;
        compactedBlocks: number;
    };
    usedBudget: {
        chars: number;
        tokens: number;
        blocks: number;
    };
    packBlockCounts: {
        total: number;
        seed: number;
        brain: number;
    };
    routingChannels: {
        candidates: {
            graph: number;
            shortTerm: number;
            vector: number;
        };
        selected: {
            graph: number;
            shortTerm: number;
            vector: number;
        };
    };
    compaction: {
        applied: boolean;
        mode: string;
        charsBefore: number | null;
        charsAfter: number | null;
    };
}
export declare const DEFAULT_OLLAMA_EMBEDDING_MODEL = "bge-large";
export declare function createOllamaEmbedder(options?: OllamaEmbedderOptions): TextEmbedder;
export declare function determineRouteMode(pack: LoadedPack, requested: RouteMode): RouteMode;
export declare function loadPackForCompile(rootDir: string): LoadedPack;
export declare function resolveActivationCompileTarget(rootDir: string, options?: ActivationCompileOptions): ActivationCompileResolution;
export declare function loadPackForActivationCompile(rootDir: string, options?: ActivationCompileOptions): LoadedPack;
export declare function rankContextBlocks(pack: LoadedPack, request: RuntimeCompileRequestV1): RankedContextBlock[];
export declare function rankContextBlocksWithEmbedder(pack: LoadedPack, request: RuntimeCompileRequestV1, embedder: TextEmbedder): Promise<RankedContextBlock[]>;
export declare function compileRuntime(packOrRoot: LoadedPack | string, request: RuntimeCompileRequestV1, options?: RuntimeCompileOptions): RuntimeCompileResponseV1;
export declare function compileRuntimeWithEmbedder(packOrRoot: LoadedPack | string, request: RuntimeCompileRequestV1, embedder: TextEmbedder, options?: RuntimeCompileOptions): Promise<RuntimeCompileResponseV1>;
export declare function compileRuntimeFromActivation(rootDir: string, request: RuntimeCompileRequestV1, options?: ActivationCompileOptions): ActivationCompileResult;
export declare function compileRuntimeFromActivationWithEmbedder(rootDir: string, request: RuntimeCompileRequestV1, embedder: TextEmbedder, options?: ActivationCompileOptions): Promise<ActivationCompileResult>;
export declare function describeCompileFallbackUsage(response: RuntimeCompileResponseV1): CompileFallbackUsageReport;
