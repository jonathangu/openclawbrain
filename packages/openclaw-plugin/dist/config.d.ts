export declare const PLUGIN_ID = "openclawbrain";
export declare const PLUGIN_VERSION = "0.2.33";
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
        allowPromptContext: boolean;
        allowConversationAccess: boolean;
        allowToolObservation: boolean;
    };
    llm: {
        enabled: boolean;
        routeModel: any;
        plannerModel: any;
        feedbackModel: any;
        learningModel: any;
        baseUrl: any;
        allowRemoteLlm: boolean;
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
        feedbackTimeoutMs: number;
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
    graphMaintenance: {
        enabled: boolean;
        mode: any;
        intervalMs: number;
        runOnStartup: boolean;
        startupDelayMs: number;
        maxNodesPerRun: number;
        safeAutoApply: boolean;
        maxSafeAutoApplyPerRun: number;
    };
    routeLearning: {
        enabled: boolean;
        teacher: {
            enabled: boolean;
            mode: any;
            maxRunsPerCycle: number;
            minResolvedRewardMagnitude: number;
        };
        counterfactuals: {
            enabled: boolean;
            topK: number;
            maxGraphDepth: number;
        };
        policyV2: {
            enabled: boolean;
            shadowBeforeActivate: boolean;
            minExamples: number;
            maxSyncPlannerRate: number;
            maxNoisyInjectionRate: number;
        };
        policyV3: {
            enabled: boolean;
            updateMode: any;
            shadowBeforeActivate: boolean;
            activationCooldownMs: number;
            coldStartMinSamples: number;
            minFrames: number;
            maxSyncPlannerRate: number;
            maxHarmRate: number;
            explorationAlpha: number;
            minRuleConfidence: number;
            maxRules: number;
            maxRulesPerRoute: number;
            maxRuleSignals: number;
            holdoutFraction: number;
            minHoldoutFrames: number;
            minRouteThresholdSamples: number;
            minCalibratedConfidence: number;
            abstainMargin: number;
            calibrationBuckets: number;
            storeShadowDecisions: boolean;
            maxShadowSnapshots: number;
            minProjectedImprovement: number;
            compactnessMaxDuplicateRate: number;
            prototypeRetirementHarmRate: number;
            prototypeRetirementMinCount: number;
        };
    };
    privacy: {
        storeRawTranscript: boolean;
        redactBeforeStore: boolean;
        redactBeforeLlm: boolean;
        storeDistillationInputs: boolean;
        storeDistillationOutputs: boolean;
    };
    memory: {
        captureMode: any;
        explicitRememberMode: any;
        futureFacingLanguage: any;
        sensitiveRecall: {
            allowUserAuthorizedRecallRules: boolean;
            neverStoreCredentialPlaintext: boolean;
            requireNarrowScope: boolean;
            preventProactiveDisclosure: boolean;
            preferSecureStoreForSensitiveValues: boolean;
        };
        scope: {
            preferNarrowestScope: boolean;
            allowCurrentRepoInference: boolean;
            allowGlobalPreferenceInference: boolean;
        };
        audit: {
            recordSkippedCapture: boolean;
            recordRejectedCandidates: boolean;
            safePreviewOnly: boolean;
            enableMemoryPostmortem: boolean;
        };
    };
    codexBridge: {
        enabled: boolean;
        statePaths: any;
        bridgeStatePath: any;
        preferAppServer: boolean;
        appServerCommand: any;
        appServerArgs: any;
        appServerUrl: any;
        appServerTimeoutMs: number;
        staleAfterMs: number;
        maxThreads: number;
        watchPollIntervalMs: number;
        messageWatchesEnabled: boolean;
        directMessageCopyEnabled: boolean;
        telegramForwardingMode: any;
        enableTelegramWrites: boolean;
        enableTelegramSteer: boolean;
        trustOpenClawAuth: boolean;
        allowLatestTargetForWrites: boolean;
        highRiskTelegramWrites: boolean;
        trustedTelegramSenders: any;
        repoAllowlist: any;
        readAllowlist: any;
        writeAllowlist: any;
        destructiveWriteAllowlist: any;
        notifyChannel: any;
        notifyTarget: any;
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
        allowPromptContext: boolean;
        allowConversationAccess: boolean;
        allowToolObservation: boolean;
    };
    llm: {
        enabled: boolean;
        routeModel: any;
        plannerModel: any;
        feedbackModel: any;
        learningModel: any;
        baseUrl: any;
        allowRemoteLlm: boolean;
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
        feedbackTimeoutMs: number;
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
    graphMaintenance: {
        enabled: boolean;
        mode: any;
        intervalMs: number;
        runOnStartup: boolean;
        startupDelayMs: number;
        maxNodesPerRun: number;
        safeAutoApply: boolean;
        maxSafeAutoApplyPerRun: number;
    };
    routeLearning: {
        enabled: boolean;
        teacher: {
            enabled: boolean;
            mode: any;
            maxRunsPerCycle: number;
            minResolvedRewardMagnitude: number;
        };
        counterfactuals: {
            enabled: boolean;
            topK: number;
            maxGraphDepth: number;
        };
        policyV2: {
            enabled: boolean;
            shadowBeforeActivate: boolean;
            minExamples: number;
            maxSyncPlannerRate: number;
            maxNoisyInjectionRate: number;
        };
        policyV3: {
            enabled: boolean;
            updateMode: any;
            shadowBeforeActivate: boolean;
            activationCooldownMs: number;
            coldStartMinSamples: number;
            minFrames: number;
            maxSyncPlannerRate: number;
            maxHarmRate: number;
            explorationAlpha: number;
            minRuleConfidence: number;
            maxRules: number;
            maxRulesPerRoute: number;
            maxRuleSignals: number;
            holdoutFraction: number;
            minHoldoutFrames: number;
            minRouteThresholdSamples: number;
            minCalibratedConfidence: number;
            abstainMargin: number;
            calibrationBuckets: number;
            storeShadowDecisions: boolean;
            maxShadowSnapshots: number;
            minProjectedImprovement: number;
            compactnessMaxDuplicateRate: number;
            prototypeRetirementHarmRate: number;
            prototypeRetirementMinCount: number;
        };
    };
    privacy: {
        storeRawTranscript: boolean;
        redactBeforeStore: boolean;
        redactBeforeLlm: boolean;
        storeDistillationInputs: boolean;
        storeDistillationOutputs: boolean;
    };
    memory: {
        captureMode: any;
        explicitRememberMode: any;
        futureFacingLanguage: any;
        sensitiveRecall: {
            allowUserAuthorizedRecallRules: boolean;
            neverStoreCredentialPlaintext: boolean;
            requireNarrowScope: boolean;
            preventProactiveDisclosure: boolean;
            preferSecureStoreForSensitiveValues: boolean;
        };
        scope: {
            preferNarrowestScope: boolean;
            allowCurrentRepoInference: boolean;
            allowGlobalPreferenceInference: boolean;
        };
        audit: {
            recordSkippedCapture: boolean;
            recordRejectedCandidates: boolean;
            safePreviewOnly: boolean;
            enableMemoryPostmortem: boolean;
        };
    };
    codexBridge: {
        enabled: boolean;
        statePaths: any;
        bridgeStatePath: any;
        preferAppServer: boolean;
        appServerCommand: any;
        appServerArgs: any;
        appServerUrl: any;
        appServerTimeoutMs: number;
        staleAfterMs: number;
        maxThreads: number;
        watchPollIntervalMs: number;
        messageWatchesEnabled: boolean;
        directMessageCopyEnabled: boolean;
        telegramForwardingMode: any;
        enableTelegramWrites: boolean;
        enableTelegramSteer: boolean;
        trustOpenClawAuth: boolean;
        allowLatestTargetForWrites: boolean;
        highRiskTelegramWrites: boolean;
        trustedTelegramSenders: any;
        repoAllowlist: any;
        readAllowlist: any;
        writeAllowlist: any;
        destructiveWriteAllowlist: any;
        notifyChannel: any;
        notifyTarget: any;
    };
};
export declare function normalizeScopes(scopes?: any): {
    agents: any;
};
export declare function activationRootForAgent(config: any, agentId?: any): string;
export declare function isAgentAllowed(config: any, agentId: any): any;
export declare function isRemoteUrl(value: any): boolean;
