import { DEFAULT_CONFIG, PLUGIN_ID, PLUGIN_VERSION, isAgentAllowed, normalizePluginConfig, resolveOpenClawBrainConfig } from './config.js';
import { buildInjectionText, ensureActivationRoot, readActivationContext } from './context-files.js';
import { decidePolicy } from './policy.js';
import { appendProofEvent, readProofEvents, readStatus, writeStatus } from './proof-store.js';
import { buildStatus } from './status.js';
import { clipText, eventId, hashText, latestUserTextFromEvent, redactText, safeString } from './redact.js';
export { normalizePluginConfig, resolveOpenClawBrainConfig } from './config.js';
export { redactText, hashText } from './redact.js';
export { decidePolicy, classifyTurn } from './policy.js';
export { readActivationContext } from './context-files.js';
export { appendProofEvent, readProofEvents, readStatus, writeStatus } from './proof-store.js';
export { buildStatus } from './status.js';
export { FakeLlmClient, OpenAICompatibleLlmClient } from './llm-client.js';
export { JsonParseError, JsonTimeoutError, JsonValidationError, runJsonWithValidation, validateWithGuard, withTimeout } from './llm-json.js';
export const openClawBrainPluginEntry = {
    id: PLUGIN_ID,
    name: 'OpenClawBrain',
    version: PLUGIN_VERSION,
    register(api = {}) {
        const resolve = () => resolveOpenClawBrainConfig(api);
        registerFirstClassSurfaces(api, resolve);
        registerPromptHooks(api, resolve);
        registerLifecycleHooks(api, resolve);
    }
};
export default openClawBrainPluginEntry;
export function redactedTurnFromPromptEvent(event = {}, config = DEFAULT_CONFIG) {
    const agentId = agentIdFromEvent(event);
    const rawPrompt = latestUserTextFromEvent(event);
    const redactedPrompt = redactText(rawPrompt, config.maxContextChars || 3000);
    const ctx = event.ctx || {};
    return {
        turnId: safeString(event.turnId ?? event.turn_id ?? event.requestId ?? event.request_id ?? ctx.runId ?? 'turn-redacted'),
        agentId,
        promptHash: hashText(rawPrompt),
        summary: clipText(redactedPrompt, 500),
        sessionKeyHash: hashText(ctx.sessionKey ?? event.sessionKey ?? event.session_key ?? ''),
        sessionIdHash: hashText(ctx.sessionId ?? event.sessionId ?? event.session_id ?? ''),
        runIdHash: hashText(ctx.runId ?? event.runId ?? event.run_id ?? ''),
        openclawProfile: safeString(ctx.openclawProfile ?? ctx.profile ?? event.openclawProfile ?? event.profile ?? agentId),
        turnType: safeString(event.turnType ?? event.turn_type ?? '')
    };
}
export async function handleTurnHook(event = {}, config = normalizePluginConfig(), api = {}, phase = 'before_prompt_build') {
    const agentId = agentIdFromEvent(event);
    if (config.rawTranscriptUpload === true) {
        await writeProofForDecision(event, config, api, { kind: 'stay_silent', slice: 'unknown', reasonCode: 'raw_transcript_upload_requested' }, phase, [], []);
        return {};
    }
    if (!config.enabled || config.mode === 'off')
        return {};
    if (!isAgentAllowed(config, agentId))
        return {};
    const redactedTurn = redactedTurnFromPromptEvent(event, config);
    const decision = decidePolicy({
        mode: config.mode,
        redactedPrompt: redactedTurn.summary,
        redactedTurn,
        turnType: redactedTurn.turnType,
        event,
        tools: event.tools
    });
    if (decision.kind === 'stay_silent' || decision.kind === 'proof_only') {
        await writeProofForDecision(event, config, api, decision, phase, [], []);
        await writeDecisionStatus(config, redactedTurn, decision);
        return {};
    }
    if (config.hooks.allowPromptInjection !== true) {
        const failClosed = { kind: 'stay_silent', slice: decision.slice, reasonCode: 'prompt_injection_disabled' };
        await writeProofForDecision(event, config, api, failClosed, phase, [], []);
        await writeDecisionStatus(config, redactedTurn, failClosed);
        return {};
    }
    const activationContext = await readActivationContext(config, redactedTurn.agentId, decision);
    const injection = buildInjectionText(decision, activationContext, config);
    await writeProofForDecision(event, config, api, decision, phase, activationContext.usedFileIdsRedacted, activationContext.rejectedFiles);
    await writeDecisionStatus(config, redactedTurn, decision);
    if (!injection)
        return {};
    return { prependContext: injection };
}
function registerPromptHooks(api, resolve) {
    safeRegisterHook(api, 'before_prompt_build', async (event = {}) => handleTurnHook(event, resolve(), api, 'before_prompt_build'));
    safeRegisterOptionalHook(api, 'agent_turn_prepare', async (event = {}) => handleTurnHook(event, resolve(), api, 'agent_turn_prepare'));
}
function registerLifecycleHooks(api, resolve) {
    safeRegisterHook(api, 'model_call_started', async (event = {}) => writeTelemetryEvent('model_call_started', event, resolve(), api));
    safeRegisterHook(api, 'model_call_ended', async (event = {}) => writeTelemetryEvent('model_call_ended', event, resolve(), api));
    safeRegisterHook(api, 'gateway_start', async (event = {}) => writeGatewayStatus('gateway_start', event, resolve(), api));
    safeRegisterHook(api, 'gateway_stop', async (event = {}) => writeGatewayStatus('gateway_stop', event, resolve(), api));
    if (resolve().hooks.allowConversationAccess === true) {
        safeRegisterOptionalHook(api, 'agent_end', async (event = {}) => writeTelemetryEvent('agent_end', event, resolve(), api));
    }
}
function registerFirstClassSurfaces(api, resolve) {
    api.registerService?.({
        id: PLUGIN_ID,
        start: async () => {
            await writeGatewayStatus('service_start', {}, resolve(), api);
        },
        stop: async () => {
            await writeGatewayStatus('service_stop', {}, resolve(), api);
        }
    });
    api.registerHttpRoute?.({
        path: '/plugins/openclawbrain/status',
        auth: 'gateway',
        match: 'exact',
        replaceExisting: true,
        handler: async (req, res) => writeJson(res, await statusPayload(resolve(), req))
    });
    api.registerHttpRoute?.({
        path: '/plugins/openclawbrain/proof',
        auth: 'gateway',
        match: 'prefix',
        replaceExisting: true,
        handler: async (req, res) => writeJson(res, await proofPayload(resolve(), limitFromRequest(req)))
    });
}
async function statusPayload(config, req = {}) {
    const agentId = safeString(req.query?.agentId ?? req.query?.agent ?? config.scopes.agents[0] ?? 'main') || 'main';
    let persisted = null;
    try {
        persisted = await readStatus({ activationRoot: config.activationRoot, agentId });
    }
    catch {
        persisted = null;
    }
    return { ...buildStatus(config, { agentId }), persisted };
}
async function proofPayload(config, limit = 20) {
    const agentId = config.scopes.agents[0] || 'main';
    const events = await readProofEvents({ activationRoot: config.activationRoot, agentId, limit });
    return { ok: true, plugin: PLUGIN_ID, limit: Math.min(100, Math.max(1, Number(limit || 20))), events };
}
async function writeProofForDecision(event, config, api, decision, phase, usedFileIdsRedacted = [], rejectedFiles = []) {
    if (!config.proofEvents)
        return null;
    const redactedTurn = redactedTurnFromPromptEvent(event, config);
    const proof = {
        schemaVersion: 'ocb.proof.event.v1',
        pluginVersion: PLUGIN_VERSION,
        profileId: redactedTurn.openclawProfile || redactedTurn.agentId,
        agentId: redactedTurn.agentId,
        sessionKeyHash: redactedTurn.sessionKeyHash,
        sessionIdHash: redactedTurn.sessionIdHash,
        runIdHash: redactedTurn.runIdHash,
        promptHash: redactedTurn.promptHash,
        eventId: eventId('proof'),
        timestamp: new Date().toISOString(),
        hookPhase: phase,
        slice: decision.slice,
        mode: config.mode,
        decisionKind: decision.kind,
        reasonCode: decision.reasonCode,
        usedMemoryIdsRedacted: usedFileIdsRedacted,
        rejectedFiles,
        rawTranscriptStored: false,
        rawUserTextStored: false,
        redactionApplied: true,
        hashesOnlyForUserText: true
    };
    try {
        return await appendProofEvent(proof, { activationRoot: config.activationRoot, agentId: redactedTurn.agentId, proofRetentionEvents: config.proofRetentionEvents });
    }
    catch (error) {
        api.logger?.warn?.({ error }, 'OpenClawBrain failed to write proof event');
        return null;
    }
}
async function writeTelemetryEvent(marker, event, config, api) {
    if (!config.enabled || !config.proofEvents)
        return;
    const agentId = agentIdFromEvent(event);
    const sanitized = {
        schemaVersion: 'ocb.proof.event.v1',
        pluginVersion: PLUGIN_VERSION,
        profileId: safeString(event.ctx?.profile ?? event.profile ?? agentId),
        agentId,
        sessionKeyHash: hashText(event.ctx?.sessionKey ?? event.sessionKey ?? ''),
        sessionIdHash: hashText(event.ctx?.sessionId ?? event.sessionId ?? ''),
        runIdHash: hashText(event.ctx?.runId ?? event.runId ?? ''),
        promptHash: hashText(''),
        eventId: eventId('telemetry'),
        timestamp: new Date().toISOString(),
        hookPhase: marker,
        slice: 'unknown',
        mode: config.mode,
        decisionKind: 'proof_only',
        reasonCode: marker,
        usedMemoryIdsRedacted: [],
        modelCallIdHash: hashText(event.modelCallId ?? event.callId ?? event.id ?? ''),
        rawTranscriptStored: false,
        rawUserTextStored: false,
        redactionApplied: true,
        hashesOnlyForUserText: true
    };
    try {
        await appendProofEvent(sanitized, { activationRoot: config.activationRoot, agentId, proofRetentionEvents: config.proofRetentionEvents });
    }
    catch (error) {
        api.logger?.warn?.({ error }, 'OpenClawBrain failed to write telemetry proof');
    }
}
async function writeGatewayStatus(marker, event, config, api) {
    const agentId = agentIdFromEvent(event);
    try {
        await ensureActivationRoot(config, agentId);
        await writeStatus(buildStatus(config, { agentId, lastDecisionKind: marker }), { activationRoot: config.activationRoot, agentId });
    }
    catch (error) {
        api.logger?.warn?.({ error }, 'OpenClawBrain failed to write gateway status');
    }
}
async function writeDecisionStatus(config, redactedTurn, decision) {
    await writeStatus(buildStatus(config, { agentId: redactedTurn.agentId, lastDecisionKind: decision.kind }), { activationRoot: config.activationRoot, agentId: redactedTurn.agentId });
}
function safeRegisterHook(api, name, handler) {
    try {
        api.on?.(name, handler);
    }
    catch (error) {
        api.logger?.debug?.({ hook: name, error }, 'OpenClawBrain hook unavailable; skipping');
    }
}
function safeRegisterOptionalHook(api, name, handler) {
    if (typeof api.supportsHook !== 'function' || api.supportsHook(name) !== true)
        return;
    safeRegisterHook(api, name, handler);
}
function writeJson(res, payload) {
    if (!res)
        return payload;
    res.statusCode = 200;
    res.setHeader?.('content-type', 'application/json');
    res.end?.(JSON.stringify(payload));
    return true;
}
function limitFromRequest(req = {}) {
    if (req.query?.limit)
        return req.query.limit;
    try {
        const url = new URL(req.url || 'http://local/plugins/openclawbrain/proof', 'http://local');
        return url.searchParams.get('limit') || 20;
    }
    catch {
        return 20;
    }
}
function agentIdFromEvent(event = {}) {
    return safeString(event.ctx?.agentId ?? event.agentId ?? event.agent_id ?? event.profileId ?? event.profile_id ?? event.session?.agentId ?? 'main') || 'main';
}
