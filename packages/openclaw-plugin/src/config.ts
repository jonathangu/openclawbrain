import os from 'node:os';
import path from 'node:path';
import { safeString } from './redact.js';

export const PLUGIN_ID = 'openclawbrain';
export const PLUGIN_VERSION = '0.2.9';
export const DEFAULT_CONFIG: any = Object.freeze({
  enabled: false,
  mode: 'balanced',
  activationRoot: '~/.openclawbrain/activation/${agentId}',
  proofEvents: true,
  proofRetentionEvents: 1000,
  maxContextChars: 3000,
  includeActivationContext: true,
  rawTranscriptUpload: false,
  scopes: Object.freeze({ agents: Object.freeze(['main']) }),
  hooks: Object.freeze({ allowPromptContext: false, allowConversationAccess: false, allowToolObservation: false }),
  llm: Object.freeze({
    enabled: false,
    allowedModels: Object.freeze([]),
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
    mode: 'hybrid',
    minConfidence: 0.7,
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
    maxInjectedMemories: 5,
    maxInjectedChars: 1200,
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
    enabled: source.enabled === true && !rawTranscriptUpload && mode !== 'off',
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
      allowPromptContext: source.hooks?.allowPromptContext === true,
      allowConversationAccess: source.hooks?.allowConversationAccess === true,
      allowToolObservation: source.hooks?.allowToolObservation === true,
    },
    llm: {
      enabled: source.llm?.enabled === true,
      routeModel: nonEmptyString(source.llm?.routeModel) || '',
      plannerModel: nonEmptyString(source.llm?.plannerModel) || '',
      feedbackModel: nonEmptyString(source.llm?.feedbackModel) || '',
      learningModel: nonEmptyString(source.llm?.learningModel) || '',
      baseUrl: nonEmptyString(source.llm?.baseUrl) || '',
      allowedModels: Array.isArray(source.llm?.allowedModels) ? source.llm.allowedModels.map((v: any) => safeString(v)).filter(Boolean) : [],
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
      mode: nonEmptyString(source.capture?.mode) || DEFAULT_CONFIG.capture.mode,
      minConfidence: clampNumber(source.capture?.minConfidence, DEFAULT_CONFIG.capture.minConfidence, 0, 1),
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
