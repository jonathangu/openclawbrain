import os from 'node:os';
import path from 'node:path';
import { safeString } from './redact.js';
export const PLUGIN_ID = 'openclawbrain';
export const PLUGIN_VERSION = '0.2.14';
export const DEFAULT_CONFIG = Object.freeze({
    enabled: true,
    mode: 'balanced',
    activationRoot: '~/.openclawbrain/activation/${agentId}',
    proofEvents: true,
    proofRetentionEvents: 1000,
    maxContextChars: 3000,
    includeActivationContext: true,
    rawTranscriptUpload: false,
    scopes: Object.freeze({ agents: Object.freeze(['main']) }),
    hooks: Object.freeze({ allowPromptContext: true, allowConversationAccess: true, allowToolObservation: true }),
    llm: Object.freeze({
        enabled: true,
        routeModel: 'qwen2.5:32b-instruct',
        plannerModel: 'qwen2.5:32b-instruct',
        feedbackModel: 'qwen2.5:32b-instruct',
        learningModel: 'qwen2.5:32b-instruct',
        baseUrl: 'http://127.0.0.1:11434/v1',
        allowedModels: Object.freeze(['qwen2.5:32b-instruct', 'qwen3.5:9b', 'qwen3.5:35b-a3b', 'gemma4:31b']),
        temperature: 0,
        maxTokens: 1200,
    }),
    latency: Object.freeze({
        noSynchronousLlmByDefault: true,
        syncPlannerEnabled: true,
        syncPlannerSoftTimeoutMs: 900,
        syncPlannerHardTimeoutMs: 1800,
        maxSyncPlannerCallsPerSession: 5,
        maxSyncPlannerCallsPerHour: 30,
        fallbackOnTimeout: 'cached_route',
    }),
    capture: Object.freeze({
        enabled: true,
        mode: 'aggressive',
        minConfidence: 0.7,
        feedbackTimeoutMs: 60000,
        immediateCorrectionCapture: true,
        postRunWorkflowCapture: true,
        storeCandidates: true,
        agentEndMode: 'enqueue',
    }),
    routing: Object.freeze({
        enabled: true,
        mode: 'hybrid_llm_on_cache_miss',
        minRouteConfidence: 0.7,
        maxCandidateMemories: 40,
        maxInjectedMemories: 8,
        maxInjectedChars: 2500,
        learnFromOutcomes: true,
    }),
    learning: Object.freeze({
        enabled: true,
        intervalMs: 60000,
        minExamplesForPolicyUpdate: 5,
        maxPositiveExamples: 25,
        maxNegativeExamples: 25,
        pruneIntervalMs: 3600000,
        maxMemoryNodesPerAgent: 5000,
    }),
    privacy: Object.freeze({
        storeRawTranscript: false,
        redactBeforeStore: true,
        redactBeforeLlm: true,
        storeDistillationInputs: false,
        storeDistillationOutputs: true,
    }),
    memory: Object.freeze({
        captureMode: 'aggressive',
        explicitRememberMode: 'always_consider',
        futureFacingLanguage: 'enqueue_async_capture',
        sensitiveRecall: Object.freeze({
            allowUserAuthorizedRecallRules: true,
            neverStoreCredentialPlaintext: true,
            requireNarrowScope: true,
            preventProactiveDisclosure: true,
            preferSecureStoreForSensitiveValues: true,
        }),
        scope: Object.freeze({
            preferNarrowestScope: true,
            allowCurrentRepoInference: true,
            allowGlobalPreferenceInference: false,
        }),
        audit: Object.freeze({
            recordSkippedCapture: true,
            recordRejectedCandidates: true,
            safePreviewOnly: true,
            enableMemoryPostmortem: true,
        }),
    }),
});
const MODES = new Set(['off', 'proof-only', 'conservative', 'active', 'balanced', 'aggressive']);
export function resolveOpenClawBrainConfig(api = {}) {
    const entry = livePluginEntry(api);
    const pluginScopedConfig = entry?.config && typeof entry.config === 'object'
        ? entry.config
        : api.pluginConfig && typeof api.pluginConfig === 'object'
            ? api.pluginConfig
            : {};
    const hooks = entry?.hooks && typeof entry.hooks === 'object'
        ? entry.hooks
        : pluginScopedConfig.hooks && typeof pluginScopedConfig.hooks === 'object'
            ? pluginScopedConfig.hooks
            : {};
    return normalizePluginConfig({ ...pluginScopedConfig, hooks });
}
export function livePluginEntry(api = {}) {
    try {
        const config = api.runtime?.config?.current?.();
        const entry = config?.plugins?.entries?.openclawbrain;
        return entry && typeof entry === 'object' && !Array.isArray(entry) ? entry : null;
    }
    catch {
        return null;
    }
}
export function normalizePluginConfig(input = {}) {
    const source = input && typeof input === 'object' ? input : {};
    const mode = MODES.has(source.mode) ? source.mode : DEFAULT_CONFIG.mode;
    const rawTranscriptUpload = source.rawTranscriptUpload === true;
    const proofRetentionEvents = clampInteger(source.proofRetentionEvents, 1000, 50, 50000);
    const maxContextChars = clampInteger(source.maxContextChars, 3000, 500, 20000);
    const activationRoot = nonEmptyString(source.activationRoot) || DEFAULT_CONFIG.activationRoot;
    return {
        enabled: source.enabled !== false && !rawTranscriptUpload && mode !== 'off',
        mode,
        activationRoot,
        proofEvents: source.proofEvents !== false,
        proofRetentionEvents,
        maxContextChars,
        includeActivationContext: source.includeActivationContext !== false,
        rawTranscriptUpload,
        failClosedReason: rawTranscriptUpload ? 'raw_transcript_upload_requested' : '',
        scopes: normalizeScopes(source.scopes),
        hooks: {
            allowPromptContext: source.hooks?.allowPromptContext !== false,
            allowConversationAccess: source.hooks?.allowConversationAccess !== false,
            allowToolObservation: source.hooks?.allowToolObservation !== false,
        },
        llm: {
            enabled: source.llm?.enabled !== false,
            routeModel: nonEmptyString(source.llm?.routeModel) || DEFAULT_CONFIG.llm.routeModel,
            plannerModel: nonEmptyString(source.llm?.plannerModel) || DEFAULT_CONFIG.llm.plannerModel,
            feedbackModel: nonEmptyString(source.llm?.feedbackModel) || DEFAULT_CONFIG.llm.feedbackModel,
            learningModel: nonEmptyString(source.llm?.learningModel) || DEFAULT_CONFIG.llm.learningModel,
            baseUrl: nonEmptyString(source.llm?.baseUrl) || DEFAULT_CONFIG.llm.baseUrl,
            allowedModels: Array.isArray(source.llm?.allowedModels) ? source.llm.allowedModels.map((v) => safeString(v)).filter(Boolean) : [...DEFAULT_CONFIG.llm.allowedModels],
            temperature: clampNumber(source.llm?.temperature, DEFAULT_CONFIG.llm.temperature, 0, 2),
            maxTokens: clampInteger(source.llm?.maxTokens, DEFAULT_CONFIG.llm.maxTokens, 1, 100000),
        },
        latency: {
            noSynchronousLlmByDefault: source.latency?.noSynchronousLlmByDefault !== false,
            syncPlannerEnabled: source.latency?.syncPlannerEnabled !== false,
            syncPlannerSoftTimeoutMs: clampInteger(source.latency?.syncPlannerSoftTimeoutMs, DEFAULT_CONFIG.latency.syncPlannerSoftTimeoutMs, 100, 10000),
            syncPlannerHardTimeoutMs: clampInteger(source.latency?.syncPlannerHardTimeoutMs, DEFAULT_CONFIG.latency.syncPlannerHardTimeoutMs, 100, 30000),
            maxSyncPlannerCallsPerSession: clampInteger(source.latency?.maxSyncPlannerCallsPerSession, DEFAULT_CONFIG.latency.maxSyncPlannerCallsPerSession, 0, 1000),
            maxSyncPlannerCallsPerHour: clampInteger(source.latency?.maxSyncPlannerCallsPerHour, DEFAULT_CONFIG.latency.maxSyncPlannerCallsPerHour, 0, 10000),
            fallbackOnTimeout: nonEmptyString(source.latency?.fallbackOnTimeout) || DEFAULT_CONFIG.latency.fallbackOnTimeout,
        },
        capture: {
            enabled: source.capture?.enabled !== false,
            mode: nonEmptyString(source.memory?.captureMode) || nonEmptyString(source.capture?.mode) || DEFAULT_CONFIG.capture.mode,
            minConfidence: clampNumber(source.capture?.minConfidence, DEFAULT_CONFIG.capture.minConfidence, 0, 1),
            feedbackTimeoutMs: clampInteger(source.capture?.feedbackTimeoutMs, DEFAULT_CONFIG.capture.feedbackTimeoutMs, 1000, 300000),
            immediateCorrectionCapture: source.capture?.immediateCorrectionCapture !== false,
            postRunWorkflowCapture: source.capture?.postRunWorkflowCapture !== false,
            storeCandidates: source.capture?.storeCandidates !== false,
            agentEndMode: nonEmptyString(source.capture?.agentEndMode) || DEFAULT_CONFIG.capture.agentEndMode,
        },
        routing: {
            enabled: source.routing?.enabled !== false,
            mode: nonEmptyString(source.routing?.mode) || DEFAULT_CONFIG.routing.mode,
            minRouteConfidence: clampNumber(source.routing?.minRouteConfidence, DEFAULT_CONFIG.routing.minRouteConfidence, 0, 1),
            maxCandidateMemories: clampInteger(source.routing?.maxCandidateMemories, DEFAULT_CONFIG.routing.maxCandidateMemories, 1, 1000),
            maxInjectedMemories: clampInteger(source.routing?.maxInjectedMemories, DEFAULT_CONFIG.routing.maxInjectedMemories, 0, 100),
            maxInjectedChars: clampInteger(source.routing?.maxInjectedChars, DEFAULT_CONFIG.routing.maxInjectedChars, 0, 20000),
            learnFromOutcomes: source.routing?.learnFromOutcomes !== false,
        },
        learning: {
            enabled: source.learning?.enabled !== false,
            intervalMs: clampInteger(source.learning?.intervalMs, DEFAULT_CONFIG.learning.intervalMs, 1000, 86400000),
            minExamplesForPolicyUpdate: clampInteger(source.learning?.minExamplesForPolicyUpdate, DEFAULT_CONFIG.learning.minExamplesForPolicyUpdate, 1, 1000),
            maxPositiveExamples: clampInteger(source.learning?.maxPositiveExamples, DEFAULT_CONFIG.learning.maxPositiveExamples, 1, 1000),
            maxNegativeExamples: clampInteger(source.learning?.maxNegativeExamples, DEFAULT_CONFIG.learning.maxNegativeExamples, 1, 1000),
            pruneIntervalMs: clampInteger(source.learning?.pruneIntervalMs, DEFAULT_CONFIG.learning.pruneIntervalMs, 1000, 86400000),
            maxMemoryNodesPerAgent: clampInteger(source.learning?.maxMemoryNodesPerAgent, DEFAULT_CONFIG.learning.maxMemoryNodesPerAgent, 1, 1000000),
        },
        privacy: {
            storeRawTranscript: false,
            redactBeforeStore: source.privacy?.redactBeforeStore !== false,
            redactBeforeLlm: source.privacy?.redactBeforeLlm !== false,
            storeDistillationInputs: source.privacy?.storeDistillationInputs === true,
            storeDistillationOutputs: source.privacy?.storeDistillationOutputs !== false,
        },
        memory: normalizeMemoryPolicy(source.memory),
    };
}
function normalizeMemoryPolicy(memory = {}) {
    const source = memory && typeof memory === 'object' ? memory : {};
    return {
        captureMode: nonEmptyString(source.captureMode) || DEFAULT_CONFIG.memory.captureMode,
        explicitRememberMode: nonEmptyString(source.explicitRememberMode) || DEFAULT_CONFIG.memory.explicitRememberMode,
        futureFacingLanguage: nonEmptyString(source.futureFacingLanguage) || DEFAULT_CONFIG.memory.futureFacingLanguage,
        sensitiveRecall: {
            allowUserAuthorizedRecallRules: source.sensitiveRecall?.allowUserAuthorizedRecallRules !== false,
            neverStoreCredentialPlaintext: source.sensitiveRecall?.neverStoreCredentialPlaintext !== false,
            requireNarrowScope: source.sensitiveRecall?.requireNarrowScope !== false,
            preventProactiveDisclosure: source.sensitiveRecall?.preventProactiveDisclosure !== false,
            preferSecureStoreForSensitiveValues: source.sensitiveRecall?.preferSecureStoreForSensitiveValues !== false,
        },
        scope: {
            preferNarrowestScope: source.scope?.preferNarrowestScope !== false,
            allowCurrentRepoInference: source.scope?.allowCurrentRepoInference !== false,
            allowGlobalPreferenceInference: source.scope?.allowGlobalPreferenceInference === true,
        },
        audit: {
            recordSkippedCapture: source.audit?.recordSkippedCapture !== false,
            recordRejectedCandidates: source.audit?.recordRejectedCandidates !== false,
            safePreviewOnly: source.audit?.safePreviewOnly !== false,
            enableMemoryPostmortem: source.audit?.enableMemoryPostmortem !== false,
        },
    };
}
export function normalizeScopes(scopes = {}) {
    const agents = Array.isArray(scopes?.agents)
        ? scopes.agents.map((agent) => safeString(agent)).filter(Boolean)
        : [...DEFAULT_CONFIG.scopes.agents];
    return { agents };
}
export function activationRootForAgent(config, agentId = 'main') {
    const resolvedAgentId = safeString(agentId) || 'main';
    const template = safeString(config?.activationRoot) || DEFAULT_CONFIG.activationRoot;
    if (isRemoteUrl(template))
        throw new Error('activationRoot must be a local filesystem path');
    const substituted = template.replaceAll('${agentId}', resolvedAgentId);
    if (substituted === '~')
        return os.homedir();
    if (substituted.startsWith('~/'))
        return path.join(os.homedir(), substituted.slice(2));
    return path.resolve(substituted);
}
export function isAgentAllowed(config, agentId) {
    const agents = Array.isArray(config?.scopes?.agents) ? config.scopes.agents : ['main'];
    return agents.length === 0 || agents.includes(agentId);
}
export function isRemoteUrl(value) {
    return /^[a-z][a-z0-9+.-]*:\/\//i.test(String(value ?? ''));
}
function nonEmptyString(value) {
    return typeof value === 'string' && value.trim() ? value.trim() : '';
}
function clampInteger(value, fallback, min, max) {
    const number = Number(value);
    if (!Number.isFinite(number))
        return fallback;
    return Math.min(max, Math.max(min, Math.trunc(number)));
}
function clampNumber(value, fallback, min, max) {
    const number = Number(value);
    if (!Number.isFinite(number))
        return fallback;
    return Math.min(max, Math.max(min, number));
}
