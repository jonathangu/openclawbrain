export { normalizePluginConfig, resolveOpenClawBrainConfig } from './config.js';
export { redactText, hashText } from './redact.js';
export { decidePolicy, classifyTurn } from './policy.js';
export { readActivationContext } from './context-files.js';
export { buildCodexBridgeStatus, buildCodexHandoff, CodexBridgeStore, formatCodexStatus, formatCodexThreads, formatHandoffBrief, handleBrainCommand, normalizeCodexBridgeConfig, processCodexBridgeWatches, } from './codex-continuity.js';
export { appendProofEvent, readProofEvents, readStatus, writeStatus } from './proof-store.js';
export { buildStatus } from './status.js';
export { FakeLlmClient, OllamaNativeLlmClient, OpenAICompatibleLlmClient, isOllamaLoopbackBaseUrl } from './llm-client.js';
export { JsonParseError, JsonTimeoutError, JsonValidationError, runJsonWithValidation, validateWithGuard, withTimeout } from './llm-json.js';
export { CaptureOrchestrator, sanitizeToolEvent } from './capture.js';
export { FeedbackDistiller, validateFeedbackDistillation } from './feedback-distiller.js';
export { MemoryOperationApplier } from './memory-operations.js';
export { GraphMaintenanceEngine, graphMaintenancePayload, handleGraphBrainCommand } from './graph-maintenance.js';
export { MemoryAuthorityResolver, authorityEventTypeForDecision, defaultValidityForMemory } from './memory-authority.js';
export { JobQueue } from './job-queue.js';
export { LatencyController } from './latency-controller.js';
export { RouteCache, RouteFn } from './route-fn.js';
export { maybeDistillAndStorePolicyV2, scorePolicySnapshotV2, validatePolicySnapshotV2 } from './route-policy-v2.js';
export { maybeDistillAndStorePolicyV3, scorePolicySnapshotV3, validatePolicySnapshotV3, ingestRouteLearningArtifactsV3, rankActionPrototypesV3 } from './route-policy-v3.js';
export { buildCalibrationSummaryV3, applyCalibrationV3 } from './route-policy-v3-calibration.js';
export { summarizeReplayEvaluationV3 } from './route-policy-v3-eval.js';
export { detectRoutingModeV3, hybridWeightsForRoutingModeV3 } from './route-policy-v3-routing-mode.js';
export { RouteTeacher, buildRouteGraphSnapshot } from './route-teacher.js';
export { detectCaptureIntent, detectRetrievalIntent, classifySensitiveValue } from './capture-intent.js';
export { ContextSelector } from './context-selector.js';
export { MemoryPlanner } from './memory-planner.js';
export { nativeSqliteSmokeTest } from './native-sqlite.js';
export { BackgroundLearner } from './learning.js';
export { RouteLearning } from './route-learning.js';
export { auditPayload, buildMemoryCorpusSupplement, buildMemoryPromptSupplement, explainLastPayload, extractMemoryId, graphPayload, learnPayload, memoryPath, renderMemory, searchPayload } from './search.js';
export declare const openClawBrainPluginEntry: {
    id: string;
    name: string;
    version: string;
    kind: string;
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
export declare function buildMemoryCapability(resolve: any): {
    promptBuilder: () => string[];
    runtime: {
        getMemorySearchManager({ agentId }: {
            cfg?: any;
            agentId: string;
            purpose?: string;
        }): Promise<{
            manager: null;
            error: string;
        } | {
            manager: {
                search(query: string, opts?: any): Promise<{
                    path: string;
                    startLine: number;
                    endLine: number;
                    score: number;
                    textScore: number;
                    snippet: string;
                    source: "memory";
                    citation: string;
                }[]>;
                readFile({ relPath, from, lines }: {
                    relPath: string;
                    from?: number;
                    lines?: number;
                }): Promise<{
                    nextFrom?: number | undefined;
                    text: string;
                    path: string;
                    from: number;
                    lines: number;
                    truncated: boolean;
                }>;
                status(): {
                    backend: "builtin";
                    provider: string;
                    files: number;
                    chunks: number;
                    dirty: boolean;
                    sources: "memory"[];
                    sourceCounts: {
                        source: "memory";
                        files: number;
                        chunks: number;
                    }[];
                    custom: {
                        agentId: string;
                        plugin: string;
                        pluginVersion: string;
                        nodes: number;
                        edges: number;
                        captureAuditRows: number;
                        routeDecisions: number;
                    };
                };
                sync(): Promise<undefined>;
                getCachedEmbeddingAvailability(): {
                    ok: boolean;
                    checked: boolean;
                    cached: boolean;
                };
                probeEmbeddingAvailability(): Promise<{
                    ok: boolean;
                    checked: boolean;
                }>;
                probeVectorAvailability(): Promise<boolean>;
                close(): Promise<void>;
            };
            error?: undefined;
        }>;
        resolveMemoryBackendConfig(): {
            backend: "builtin";
        };
        closeAllMemorySearchManagers(): Promise<undefined>;
    };
};
export declare function processAutomaticGraphMaintenance(config?: any, api?: any): Promise<void>;
