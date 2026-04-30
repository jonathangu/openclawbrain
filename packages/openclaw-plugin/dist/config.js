import os from 'node:os';
import path from 'node:path';
import { safeString } from './redact.js';
export const PLUGIN_ID = 'openclawbrain';
export const PLUGIN_VERSION = '0.1.1';
export const DEFAULT_CONFIG = Object.freeze({
    enabled: false,
    mode: 'conservative',
    activationRoot: '~/.openclawbrain/activation/${agentId}',
    proofEvents: true,
    proofRetentionEvents: 1000,
    maxContextChars: 3000,
    includeActivationContext: true,
    rawTranscriptUpload: false,
    scopes: Object.freeze({ agents: Object.freeze(['main']) }),
    hooks: Object.freeze({ allowPromptInjection: false, allowConversationAccess: false })
});
const MODES = new Set(['off', 'proof-only', 'conservative', 'active']);
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
        const config = api.runtime?.config?.loadConfig?.();
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
            allowPromptInjection: source.hooks?.allowPromptInjection === true,
            allowConversationAccess: source.hooks?.allowConversationAccess === true
        }
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
