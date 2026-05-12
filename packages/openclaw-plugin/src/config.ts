import os from 'node:os';
import path from 'node:path';
import { safeString } from './redact.js';

export const PLUGIN_ID = 'openclawbrain';
export const PLUGIN_VERSION = '0.2.30';
export const DEFAULT_CONFIG: any = Object.freeze({
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
    baseUrl: 'http://localhost:11434/v1',
    allowRemoteLlm: false,
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
  graphMaintenance: Object.freeze({
    enabled: true,
    mode: 'passive',
    intervalMs: 900000,
    runOnStartup: true,
    startupDelayMs: 30000,
    maxNodesPerRun: 1000,
    safeAutoApply: true,
    maxSafeAutoApplyPerRun: 5,
  }),
  routeLearning: Object.freeze({
    enabled: true,
    teacher: Object.freeze({ enabled: true, mode: 'background', maxRunsPerCycle: 5, minResolvedRewardMagnitude: 0 }),
    counterfactuals: Object.freeze({ enabled: true, topK: 5, maxGraphDepth: 2 }),
    policyV2: Object.freeze({ enabled: true, shadowBeforeActivate: false, minExamples: 3, maxSyncPlannerRate: 0.05, maxNoisyInjectionRate: 0.05 }),
    policyV3: Object.freeze({
      enabled: true,
      updateMode: 'gated_active',
      shadowBeforeActivate: false,
      activationCooldownMs: 0,
      coldStartMinSamples: 3,
      minFrames: 3,
      maxSyncPlannerRate: 0.1,
      maxHarmRate: 0.2,
      explorationAlpha: 0.35,
      minRuleConfidence: 0.55,
      maxRules: 32,
      maxRulesPerRoute: 10,
      maxRuleSignals: 8,
      holdoutFraction: 0.3,
      minHoldoutFrames: 2,
      minRouteThresholdSamples: 3,
      minCalibratedConfidence: 0.62,
      abstainMargin: 0.05,
      calibrationBuckets: 5,
      storeShadowDecisions: true,
      maxShadowSnapshots: 3,
      minProjectedImprovement: -0.01,
      compactnessMaxDuplicateRate: 0.35,
      prototypeRetirementHarmRate: 0.7,
      prototypeRetirementMinCount: 3,
    }),
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
  codexBridge: Object.freeze({
    enabled: true,
    statePaths: Object.freeze(['~/.codex/state_5.sqlite']),
    bridgeStatePath: '~/.openclawbrain/activation/${agentId}/codex-continuity.sqlite',
    preferAppServer: false,
    appServerCommand: 'codex',
    appServerArgs: Object.freeze(['app-server', 'proxy']),
    appServerTimeoutMs: 1200,
    staleAfterMs: 600000,
    maxThreads: 10,
    watchPollIntervalMs: 60000,
    messageWatchesEnabled: true,
    directMessageCopyEnabled: true,
    telegramForwardingMode: 'redacted',
    enableTelegramWrites: false,
    trustOpenClawAuth: true,
    allowLatestTargetForWrites: false,
    highRiskTelegramWrites: false,
    trustedTelegramSenders: Object.freeze([]),
    repoAllowlist: Object.freeze([]),
    readAllowlist: Object.freeze([]),
    writeAllowlist: Object.freeze([]),
    destructiveWriteAllowlist: Object.freeze([]),
    notifyChannel: 'telegram',
    notifyTarget: '',
  }),
});

const MODES = new Set(['off', 'proof-only', 'conservative', 'active', 'balanced', 'aggressive']);

export function resolveOpenClawBrainConfig(api: any = {}) {
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

export function livePluginEntry(api: any = {}) {
  try {
    const config = api.runtime?.config?.current?.();
    const entry = config?.plugins?.entries?.openclawbrain;
    return entry && typeof entry === 'object' && !Array.isArray(entry) ? entry : null;
  } catch {
    return null;
  }
}

export function normalizePluginConfig(input: any = {}) {
  const source: any = input && typeof input === 'object' ? input : {};
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
      allowRemoteLlm: source.llm?.allowRemoteLlm === true,
      allowedModels: Array.isArray(source.llm?.allowedModels) ? source.llm.allowedModels.map((v: any) => safeString(v)).filter(Boolean) : [...DEFAULT_CONFIG.llm.allowedModels],
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
    graphMaintenance: normalizeGraphMaintenanceConfig(source.graphMaintenance),
    routeLearning: {
      enabled: source.routeLearning?.enabled !== false,
      teacher: {
        enabled: source.routeLearning?.teacher?.enabled !== false,
        mode: nonEmptyString(source.routeLearning?.teacher?.mode) || DEFAULT_CONFIG.routeLearning.teacher.mode,
        maxRunsPerCycle: clampInteger(source.routeLearning?.teacher?.maxRunsPerCycle, DEFAULT_CONFIG.routeLearning.teacher.maxRunsPerCycle, 0, 100),
        minResolvedRewardMagnitude: clampNumber(source.routeLearning?.teacher?.minResolvedRewardMagnitude, DEFAULT_CONFIG.routeLearning.teacher.minResolvedRewardMagnitude, 0, 1),
      },
      counterfactuals: {
        enabled: source.routeLearning?.counterfactuals?.enabled !== false,
        topK: clampInteger(source.routeLearning?.counterfactuals?.topK, DEFAULT_CONFIG.routeLearning.counterfactuals.topK, 1, 50),
        maxGraphDepth: clampInteger(source.routeLearning?.counterfactuals?.maxGraphDepth, DEFAULT_CONFIG.routeLearning.counterfactuals.maxGraphDepth, 0, 2),
      },
      policyV2: {
        enabled: source.routeLearning?.policyV2?.enabled !== false,
        shadowBeforeActivate: source.routeLearning?.policyV2?.shadowBeforeActivate === true,
        minExamples: clampInteger(source.routeLearning?.policyV2?.minExamples, DEFAULT_CONFIG.routeLearning.policyV2.minExamples, 1, 1000),
        maxSyncPlannerRate: clampNumber(source.routeLearning?.policyV2?.maxSyncPlannerRate, DEFAULT_CONFIG.routeLearning.policyV2.maxSyncPlannerRate, 0, 1),
        maxNoisyInjectionRate: clampNumber(source.routeLearning?.policyV2?.maxNoisyInjectionRate, DEFAULT_CONFIG.routeLearning.policyV2.maxNoisyInjectionRate, 0, 1),
      },
      policyV3: {
        enabled: source.routeLearning?.policyV3?.enabled !== false,
        updateMode: ['collect_only', 'distill_shadow', 'gated_active', 'manual_review_required'].includes(String(source.routeLearning?.policyV3?.updateMode || ''))
          ? String(source.routeLearning?.policyV3?.updateMode)
          : DEFAULT_CONFIG.routeLearning.policyV3.updateMode,
        shadowBeforeActivate: source.routeLearning?.policyV3?.shadowBeforeActivate === true,
        activationCooldownMs: clampInteger(source.routeLearning?.policyV3?.activationCooldownMs, DEFAULT_CONFIG.routeLearning.policyV3.activationCooldownMs, 0, 7 * 24 * 60 * 60 * 1000),
        coldStartMinSamples: clampInteger(source.routeLearning?.policyV3?.coldStartMinSamples, DEFAULT_CONFIG.routeLearning.policyV3.coldStartMinSamples, 1, 1000),
        minFrames: clampInteger(source.routeLearning?.policyV3?.minFrames, DEFAULT_CONFIG.routeLearning.policyV3.minFrames, 1, 5000),
        maxSyncPlannerRate: clampNumber(source.routeLearning?.policyV3?.maxSyncPlannerRate, DEFAULT_CONFIG.routeLearning.policyV3.maxSyncPlannerRate, 0, 1),
        maxHarmRate: clampNumber(source.routeLearning?.policyV3?.maxHarmRate, DEFAULT_CONFIG.routeLearning.policyV3.maxHarmRate, 0, 1),
        explorationAlpha: clampNumber(source.routeLearning?.policyV3?.explorationAlpha, DEFAULT_CONFIG.routeLearning.policyV3.explorationAlpha, 0, 5),
        minRuleConfidence: clampNumber(source.routeLearning?.policyV3?.minRuleConfidence, DEFAULT_CONFIG.routeLearning.policyV3.minRuleConfidence, 0, 1),
        maxRules: clampInteger(source.routeLearning?.policyV3?.maxRules, DEFAULT_CONFIG.routeLearning.policyV3.maxRules, 1, 500),
        maxRulesPerRoute: clampInteger(source.routeLearning?.policyV3?.maxRulesPerRoute, DEFAULT_CONFIG.routeLearning.policyV3.maxRulesPerRoute, 1, 100),
        maxRuleSignals: clampInteger(source.routeLearning?.policyV3?.maxRuleSignals, DEFAULT_CONFIG.routeLearning.policyV3.maxRuleSignals, 1, 50),
        holdoutFraction: clampNumber(source.routeLearning?.policyV3?.holdoutFraction, DEFAULT_CONFIG.routeLearning.policyV3.holdoutFraction, 0.05, 0.8),
        minHoldoutFrames: clampInteger(source.routeLearning?.policyV3?.minHoldoutFrames, DEFAULT_CONFIG.routeLearning.policyV3.minHoldoutFrames, 1, 500),
        minRouteThresholdSamples: clampInteger(source.routeLearning?.policyV3?.minRouteThresholdSamples, DEFAULT_CONFIG.routeLearning.policyV3.minRouteThresholdSamples, 1, 100),
        minCalibratedConfidence: clampNumber(source.routeLearning?.policyV3?.minCalibratedConfidence, DEFAULT_CONFIG.routeLearning.policyV3.minCalibratedConfidence, 0, 1),
        abstainMargin: clampNumber(source.routeLearning?.policyV3?.abstainMargin, DEFAULT_CONFIG.routeLearning.policyV3.abstainMargin, 0, 0.5),
        calibrationBuckets: clampInteger(source.routeLearning?.policyV3?.calibrationBuckets, DEFAULT_CONFIG.routeLearning.policyV3.calibrationBuckets, 3, 10),
        storeShadowDecisions: source.routeLearning?.policyV3?.storeShadowDecisions !== false,
        maxShadowSnapshots: clampInteger(source.routeLearning?.policyV3?.maxShadowSnapshots, DEFAULT_CONFIG.routeLearning.policyV3.maxShadowSnapshots, 0, 20),
        minProjectedImprovement: clampNumber(source.routeLearning?.policyV3?.minProjectedImprovement, DEFAULT_CONFIG.routeLearning.policyV3.minProjectedImprovement, -1, 1),
        compactnessMaxDuplicateRate: clampNumber(source.routeLearning?.policyV3?.compactnessMaxDuplicateRate, DEFAULT_CONFIG.routeLearning.policyV3.compactnessMaxDuplicateRate, 0, 1),
        prototypeRetirementHarmRate: clampNumber(source.routeLearning?.policyV3?.prototypeRetirementHarmRate, DEFAULT_CONFIG.routeLearning.policyV3.prototypeRetirementHarmRate, 0, 1),
        prototypeRetirementMinCount: clampInteger(source.routeLearning?.policyV3?.prototypeRetirementMinCount, DEFAULT_CONFIG.routeLearning.policyV3.prototypeRetirementMinCount, 1, 1000),
      },
    },
    privacy: {
      storeRawTranscript: false,
      redactBeforeStore: source.privacy?.redactBeforeStore !== false,
      redactBeforeLlm: source.privacy?.redactBeforeLlm !== false,
      storeDistillationInputs: source.privacy?.storeDistillationInputs === true,
      storeDistillationOutputs: source.privacy?.storeDistillationOutputs !== false,
    },
    memory: normalizeMemoryPolicy(source.memory),
    codexBridge: normalizeCodexBridgeConfig(source.codexBridge),
  };
}

function normalizeGraphMaintenanceConfig(graphMaintenance: any = {}) {
  const source = graphMaintenance && typeof graphMaintenance === 'object' ? graphMaintenance : {};
  const mode = ['off', 'passive', 'dry_run', 'safe_auto'].includes(String(source.mode || ''))
    ? String(source.mode)
    : DEFAULT_CONFIG.graphMaintenance.mode;
  return {
    enabled: source.enabled !== false && mode !== 'off',
    mode,
    intervalMs: clampInteger(source.intervalMs, DEFAULT_CONFIG.graphMaintenance.intervalMs, 60000, 86400000),
    runOnStartup: source.runOnStartup !== false,
    startupDelayMs: clampInteger(source.startupDelayMs, DEFAULT_CONFIG.graphMaintenance.startupDelayMs, 0, 3600000),
    maxNodesPerRun: clampInteger(source.maxNodesPerRun, DEFAULT_CONFIG.graphMaintenance.maxNodesPerRun, 50, 100000),
    safeAutoApply: source.safeAutoApply !== false,
    maxSafeAutoApplyPerRun: clampInteger(source.maxSafeAutoApplyPerRun, DEFAULT_CONFIG.graphMaintenance.maxSafeAutoApplyPerRun, 0, 100),
  };
}

function normalizeCodexBridgeConfig(codexBridge: any = {}) {
  const source = codexBridge && typeof codexBridge === 'object' ? codexBridge : {};
  const statePaths = Array.isArray(source.statePaths)
    ? source.statePaths.map((item: any) => safeString(item)).filter(Boolean)
    : typeof source.statePath === 'string'
      ? [source.statePath]
      : [...DEFAULT_CONFIG.codexBridge.statePaths];
  const appServerArgs = Array.isArray(source.appServerArgs)
    ? source.appServerArgs.map((item: any) => safeString(item)).filter(Boolean)
    : [...DEFAULT_CONFIG.codexBridge.appServerArgs];
  return {
    enabled: source.enabled !== false,
    statePaths: statePaths.length ? statePaths : [...DEFAULT_CONFIG.codexBridge.statePaths],
    bridgeStatePath: nonEmptyString(source.bridgeStatePath) || DEFAULT_CONFIG.codexBridge.bridgeStatePath,
    preferAppServer: source.preferAppServer === true,
    appServerCommand: nonEmptyString(source.appServerCommand) || DEFAULT_CONFIG.codexBridge.appServerCommand,
    appServerArgs: appServerArgs.length ? appServerArgs : [...DEFAULT_CONFIG.codexBridge.appServerArgs],
    appServerTimeoutMs: clampInteger(source.appServerTimeoutMs, DEFAULT_CONFIG.codexBridge.appServerTimeoutMs, 100, 30000),
    staleAfterMs: clampInteger(source.staleAfterMs, DEFAULT_CONFIG.codexBridge.staleAfterMs, 1000, 86400000),
    maxThreads: clampInteger(source.maxThreads, DEFAULT_CONFIG.codexBridge.maxThreads, 1, 100),
    watchPollIntervalMs: clampInteger(source.watchPollIntervalMs, DEFAULT_CONFIG.codexBridge.watchPollIntervalMs, 5000, 86400000),
    messageWatchesEnabled: source.messageWatchesEnabled !== false,
    directMessageCopyEnabled: source.directMessageCopyEnabled !== false,
    telegramForwardingMode: ['redacted', 'raw_trusted', 'metadata_only'].includes(String(source.telegramForwardingMode))
      ? String(source.telegramForwardingMode)
      : DEFAULT_CONFIG.codexBridge.telegramForwardingMode,
    enableTelegramWrites: source.enableTelegramWrites === true,
    trustOpenClawAuth: source.trustOpenClawAuth !== false,
    allowLatestTargetForWrites: source.allowLatestTargetForWrites === true,
    highRiskTelegramWrites: source.highRiskTelegramWrites === true,
    trustedTelegramSenders: Array.isArray(source.trustedTelegramSenders) ? source.trustedTelegramSenders.map((item: any) => safeString(item)).filter(Boolean) : [],
    repoAllowlist: Array.isArray(source.repoAllowlist) ? source.repoAllowlist.map((item: any) => safeString(item)).filter(Boolean) : [],
    readAllowlist: Array.isArray(source.readAllowlist) ? source.readAllowlist.map((item: any) => safeString(item)).filter(Boolean) : [],
    writeAllowlist: Array.isArray(source.writeAllowlist) ? source.writeAllowlist.map((item: any) => safeString(item)).filter(Boolean) : [],
    destructiveWriteAllowlist: Array.isArray(source.destructiveWriteAllowlist) ? source.destructiveWriteAllowlist.map((item: any) => safeString(item)).filter(Boolean) : [],
    notifyChannel: nonEmptyString(source.notifyChannel) || DEFAULT_CONFIG.codexBridge.notifyChannel,
    notifyTarget: nonEmptyString(source.notifyTarget) || DEFAULT_CONFIG.codexBridge.notifyTarget,
  };
}

function normalizeMemoryPolicy(memory: any = {}) {
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

export function normalizeScopes(scopes: any = {}) {
  const agents = Array.isArray(scopes?.agents)
    ? scopes.agents.map((agent: any) => safeString(agent)).filter(Boolean)
    : [...DEFAULT_CONFIG.scopes.agents];
  return { agents };
}

export function activationRootForAgent(config: any, agentId: any = 'main') {
  const resolvedAgentId = safeString(agentId) || 'main';
  const template = safeString(config?.activationRoot) || DEFAULT_CONFIG.activationRoot;
  if (isRemoteUrl(template)) throw new Error('activationRoot must be a local filesystem path');
  const substituted = template.replaceAll('${agentId}', resolvedAgentId);
  if (substituted === '~') return os.homedir();
  if (substituted.startsWith('~/')) return path.join(os.homedir(), substituted.slice(2));
  return path.resolve(substituted);
}

export function isAgentAllowed(config: any, agentId: any) {
  const agents = Array.isArray(config?.scopes?.agents) ? config.scopes.agents : ['main'];
  return agents.length === 0 || agents.includes(agentId);
}

export function isRemoteUrl(value: any) {
  return /^[a-z][a-z0-9+.-]*:\/\//i.test(String(value ?? ''));
}

function nonEmptyString(value: any) {
  return typeof value === 'string' && value.trim() ? value.trim() : '';
}

function clampInteger(value: any, fallback: number, min: number, max: number) {
  const number = Number(value);
  if (!Number.isFinite(number)) return fallback;
  return Math.min(max, Math.max(min, Math.trunc(number)));
}

function clampNumber(value: any, fallback: number, min: number, max: number) {
  const number = Number(value);
  if (!Number.isFinite(number)) return fallback;
  return Math.min(max, Math.max(min, number));
}
