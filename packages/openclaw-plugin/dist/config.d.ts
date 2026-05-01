export declare const PLUGIN_ID = "openclawbrain";
export declare const PLUGIN_VERSION = "0.1.1";
export declare const DEFAULT_CONFIG: any;
export declare function resolveOpenClawBrainConfig(api?: any): {
    enabled: boolean;
    mode: any;
    activationRoot: any;
    proofEvents: boolean;
    proofRetentionEvents: number;
    maxContextChars: number;
    includeActivationContext: boolean;
    rawTranscriptUpload: boolean;
    failClosedReason: string;
    scopes: {
        agents: any;
    };
    hooks: {
        allowPromptInjection: boolean;
        allowConversationAccess: boolean;
        allowToolObservation: boolean;
    };
    llm: {
        enabled: boolean;
        provider: any;
        routeModel: string;
        plannerModel: string;
        feedbackModel: string;
        learningModel: string;
        baseUrl: string;
        apiKeyEnv: string;
        allowRemoteModels: boolean;
        allowedModels: any;
        temperature: number;
        maxTokens: number;
    };
    latency: {
        noSynchronousLlmByDefault: boolean;
        syncPlannerEnabled: boolean;
        syncPlannerSoftTimeoutMs: number;
        syncPlannerHardTimeoutMs: number;
        maxSyncPlannerCallsPerSession: number;
        maxSyncPlannerCallsPerHour: number;
        fallbackOnTimeout: any;
    };
    capture: {
        enabled: boolean;
        mode: any;
        minConfidence: number;
        immediateCorrectionCapture: boolean;
        postRunWorkflowCapture: boolean;
        storeCandidates: boolean;
        agentEndMode: any;
    };
    routing: {
        enabled: boolean;
        mode: any;
        minRouteConfidence: number;
        maxCandidateMemories: number;
        maxInjectedMemories: number;
        maxInjectedChars: number;
        learnFromOutcomes: boolean;
    };
    learning: {
        enabled: boolean;
        intervalMs: number;
        minExamplesForPolicyUpdate: number;
        maxPositiveExamples: number;
        maxNegativeExamples: number;
        pruneIntervalMs: number;
        maxMemoryNodesPerAgent: number;
    };
    privacy: {
        storeRawTranscript: boolean;
        redactBeforeStore: boolean;
        redactBeforeLlm: boolean;
        storeDistillationInputs: boolean;
        storeDistillationOutputs: boolean;
    };
};
export declare function livePluginEntry(api?: any): any;
export declare function normalizePluginConfig(input?: any): {
    enabled: boolean;
    mode: any;
    activationRoot: any;
    proofEvents: boolean;
    proofRetentionEvents: number;
    maxContextChars: number;
    includeActivationContext: boolean;
    rawTranscriptUpload: boolean;
    failClosedReason: string;
    scopes: {
        agents: any;
    };
    hooks: {
        allowPromptInjection: boolean;
        allowConversationAccess: boolean;
        allowToolObservation: boolean;
    };
    llm: {
        enabled: boolean;
        provider: any;
        routeModel: string;
        plannerModel: string;
        feedbackModel: string;
        learningModel: string;
        baseUrl: string;
        apiKeyEnv: string;
        allowRemoteModels: boolean;
        allowedModels: any;
        temperature: number;
        maxTokens: number;
    };
    latency: {
        noSynchronousLlmByDefault: boolean;
        syncPlannerEnabled: boolean;
        syncPlannerSoftTimeoutMs: number;
        syncPlannerHardTimeoutMs: number;
        maxSyncPlannerCallsPerSession: number;
        maxSyncPlannerCallsPerHour: number;
        fallbackOnTimeout: any;
    };
    capture: {
        enabled: boolean;
        mode: any;
        minConfidence: number;
        immediateCorrectionCapture: boolean;
        postRunWorkflowCapture: boolean;
        storeCandidates: boolean;
        agentEndMode: any;
    };
    routing: {
        enabled: boolean;
        mode: any;
        minRouteConfidence: number;
        maxCandidateMemories: number;
        maxInjectedMemories: number;
        maxInjectedChars: number;
        learnFromOutcomes: boolean;
    };
    learning: {
        enabled: boolean;
        intervalMs: number;
        minExamplesForPolicyUpdate: number;
        maxPositiveExamples: number;
        maxNegativeExamples: number;
        pruneIntervalMs: number;
        maxMemoryNodesPerAgent: number;
    };
    privacy: {
        storeRawTranscript: boolean;
        redactBeforeStore: boolean;
        redactBeforeLlm: boolean;
        storeDistillationInputs: boolean;
        storeDistillationOutputs: boolean;
    };
};
export declare function normalizeScopes(scopes?: any): {
    agents: any;
};
export declare function activationRootForAgent(config: any, agentId?: any): string;
export declare function isAgentAllowed(config: any, agentId: any): any;
export declare function isRemoteUrl(value: any): boolean;
