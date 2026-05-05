import { DEFAULT_CONFIG, PLUGIN_ID, PLUGIN_VERSION, isAgentAllowed, normalizePluginConfig, resolveOpenClawBrainConfig } from './config.js';
import { buildInjectionText, ensureActivationRoot, readActivationContext } from './context-files.js';
import { CaptureOrchestrator } from './capture.js';
import { ContextSelector } from './context-selector.js';
import { FeedbackDistiller } from './feedback-distiller.js';
import { BackgroundLearner } from './learning.js';
import { JobQueue } from './job-queue.js';
import { LatencyController } from './latency-controller.js';
import { OllamaNativeLlmClient, OpenAICompatibleLlmClient, isOllamaLoopbackBaseUrl } from './llm-client.js';
import { MemoryPlanner } from './memory-planner.js';
import { MemoryOperationApplier } from './memory-operations.js';
import { MemoryStore } from './memory-store.js';
import { decidePolicy } from './policy.js';
import { appendProofEvent, readProofEvents, readStatus, writeStatus } from './proof-store.js';
import { RouteLearning } from './route-learning.js';
import { RouteTeacher, buildRouteGraphSnapshot } from './route-teacher.js';
import { auditPayload, buildMemoryCorpusSupplement, buildMemoryPromptSupplement, explainLastPayload, extractMemoryId, graphPayload, learnPayload, memoryPath, renderMemory, searchPayload } from './search.js';
import { buildStatus } from './status.js';
import { nativeSqliteSmokeTest } from './native-sqlite.js';
import { clipText, eventId, hashText, latestUserTextFromEvent, redactJsonValue, redactText, safeString } from './redact.js';
import { RouteFn } from './route-fn.js';
import { detectCaptureIntent, detectRetrievalIntent } from './capture-intent.js';
import { defaultScopeContext, memoryInScope, scopeContextFromPacket } from './scope.js';
export { normalizePluginConfig, resolveOpenClawBrainConfig } from './config.js';
export { redactText, hashText } from './redact.js';
export { decidePolicy, classifyTurn } from './policy.js';
export { readActivationContext } from './context-files.js';
export { appendProofEvent, readProofEvents, readStatus, writeStatus } from './proof-store.js';
export { buildStatus } from './status.js';
export { FakeLlmClient, OllamaNativeLlmClient, OpenAICompatibleLlmClient, isOllamaLoopbackBaseUrl } from './llm-client.js';
export { JsonParseError, JsonTimeoutError, JsonValidationError, runJsonWithValidation, validateWithGuard, withTimeout } from './llm-json.js';
export { CaptureOrchestrator, sanitizeToolEvent } from './capture.js';
export { FeedbackDistiller, validateFeedbackDistillation } from './feedback-distiller.js';
export { MemoryOperationApplier } from './memory-operations.js';
export { JobQueue } from './job-queue.js';
export { LatencyController } from './latency-controller.js';
export { RouteCache, RouteFn } from './route-fn.js';
export { RouteTeacher, buildRouteGraphSnapshot } from './route-teacher.js';
export { detectCaptureIntent, detectRetrievalIntent, classifySensitiveValue } from './capture-intent.js';
export { ContextSelector } from './context-selector.js';
export { MemoryPlanner } from './memory-planner.js';
export { nativeSqliteSmokeTest } from './native-sqlite.js';
export { BackgroundLearner } from './learning.js';
export { RouteLearning } from './route-learning.js';
export { auditPayload, buildMemoryCorpusSupplement, buildMemoryPromptSupplement, explainLastPayload, extractMemoryId, graphPayload, learnPayload, memoryPath, renderMemory, searchPayload } from './search.js';
export const openClawBrainPluginEntry = {
    id: PLUGIN_ID,
    name: 'OpenClawBrain',
    version: PLUGIN_VERSION,
    kind: 'memory',
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
    if (phase === 'before_prompt_build' && shouldAllowRouting(config) && (config.mode === 'balanced' || config.mode === 'aggressive')) {
        return handleV2PromptHook(event, config, api, phase);
    }
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
    if (config.hooks.allowPromptContext !== true) {
        const failClosed = { kind: 'stay_silent', slice: decision.slice, reasonCode: 'prompt_context_disabled' };
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
async function handleV2PromptHook(event = {}, config = normalizePluginConfig(), api = {}, phase = 'before_prompt_build') {
    const agentId = agentIdFromEvent(event);
    const store = new MemoryStore({ activationRoot: config.activationRoot, agentId });
    const queue = new JobQueue({ store });
    const packet = new CaptureOrchestrator().fromBeforePromptBuild(event, config);
    const routeFn = new RouteFn({ config, store });
    const contextSelector = new ContextSelector(config);
    const initialPlan = suppressSyntheticCapture(routeFn.plan(packet), packet);
    if (!shouldAllowCapture(config))
        initialPlan.enqueueCapture = false;
    const initialCandidates = initialPlan.shouldRetrieve
        ? retrieveCandidates(store, packet, initialPlan.retrievalPlan.queries, initialPlan.retrievalPlan.memoryTypes, initialPlan.retrievalPlan.maxCandidates)
        : [];
    const latency = new LatencyController(config).chooseTier({
        agentId: packet.agentId,
        sessionId: packet.sessionId,
        latestUserMessage: packet.latestUserMessageRedacted,
        recentRouteCacheHit: initialPlan.latencyReason === 'cached route plan',
        recentPolicyMatch: false,
        candidateCount: initialCandidates.length,
        candidateAmbiguity: initialCandidates.length > 0 ? Math.min(1, initialCandidates.length / Math.max(1, initialPlan.retrievalPlan.maxCandidates)) : 0,
        hasHighConfidenceCorrectionCandidate: initialPlan.route === 'high_confidence_correction_only',
        userExplicitlyReferencesMemory: /\b(as before|same as last time|remember|we discussed before)\b/i.test(packet.latestUserMessageRedacted),
        taskValueEstimate: estimateTaskValue(packet.latestUserMessageRedacted),
        configMode: config.mode,
    });
    const client = llmClientFromConfig(config);
    let plan = initialPlan;
    let selection = initialPlan.shouldRetrieve ? contextSelector.select({ packet, plan, candidates: initialCandidates, store }) : emptySelection();
    let syncPlannerUsed = false;
    if (latency.kind === 'sync_memory_planner' && client) {
        syncPlannerUsed = true;
        const planner = new MemoryPlanner({ config, routeFn, store, client });
        const planned = await planner.run(packet);
        plan = planned.routePlan;
        selection = planned.contextSelection ?? emptySelection();
    }
    else if (plan.enqueueCapture && shouldAllowCapture(config)) {
        queue.enqueueFeedbackDistillation(agentId, { packet, captureIntent: plan.captureIntent, retrievalIntent: plan.retrievalIntent }, { priority: 10 });
    }
    const routeDecision = store.insertRouteDecision({
        agentId: packet.agentId,
        sessionId: packet.sessionId,
        turnId: packet.turnId,
        runId: packet.runId,
        route: plan.route,
        confidence: plan.confidence,
        latencyTier: latency.kind,
        syncLlmUsed: syncPlannerUsed,
        syncLatencyMs: latency.maxSyncMs || undefined,
        fallbackUsed: latency.kind === 'sync_memory_planner' && !client,
        turnFrame: plan.turnFrame,
        retrievalPlan: plan.retrievalPlan,
        injectionPlan: plan.injectionPlan,
        selectedMemoryIds: selection.selectedMemoryIds,
        omittedMemoryIds: selection.omitted.map((it) => it.memoryId),
        model: client ? (config.llm.plannerModel || config.llm.routeModel || config.llm.feedbackModel || '') : undefined,
        promptVersion: latency.kind === 'sync_memory_planner' ? 'memory-planner-v1' : 'route-fn-v1',
        policySnapshotId: plan.policySnapshotId,
        reward: 0,
    });
    buildRouteGraphSnapshot(store, packet.agentId, routeDecision.id, plan.retrievalPlan.queries, initialCandidates, plan.retrievalPlan.graphDepth);
    store.insertCaptureAudit({
        agentId: packet.agentId,
        sessionId: packet.sessionId,
        turnId: packet.turnId,
        runId: packet.runId,
        retrievalIntent: plan.retrievalIntent,
        captureIntent: plan.captureIntent,
        captureJobCreated: plan.enqueueCapture && !syncPlannerUsed && shouldAllowCapture(config),
        distillerRan: false,
        fallbackRan: false,
        candidateCount: 0,
        storedCount: 0,
        rejectedCount: plan.captureIntent.shouldConsiderCapture ? 0 : 1,
        rejectionReasons: plan.captureIntent.shouldConsiderCapture ? [] : [plan.captureIntent.reason || 'no_capture_signal'],
        evidenceHash: String(packet.metadata.promptHash || ''),
    });
    await appendProofEvent({
        kind: latency.kind === 'sync_memory_planner' ? 'llm_route_decision' : 'route_decision',
        agentId: packet.agentId,
        routeDecisionId: routeDecision.id,
        rawTranscriptStored: false,
        payload: {
            latencyTier: latency.kind,
            route: routeDecision.route,
            confidence: routeDecision.confidence,
            selectedMemoryIds: selection.selectedMemoryIds,
            model: routeDecision.model,
            promptVersion: routeDecision.promptVersion,
        },
    }, { activationRoot: config.activationRoot, agentId: packet.agentId, proofRetentionEvents: config.proofRetentionEvents });
    if (!selection.shouldInject) {
        await writeStatus(buildStatus(config, { agentId: packet.agentId, lastDecisionKind: 'no_memory', routing: { activePolicySnapshotId: routeDecision.policySnapshotId || null } }), { activationRoot: config.activationRoot, agentId: packet.agentId });
        store.close();
        return {};
    }
    for (const [index, memoryId] of selection.selectedMemoryIds.entries()) {
        store.insertInjection({
            agentId: packet.agentId,
            memoryId,
            routeDecisionId: routeDecision.id,
            runId: packet.runId,
            turnId: packet.turnId,
            sessionId: packet.sessionId,
            query: plan.retrievalPlan.queries[0] || packet.latestUserMessageRedacted,
            rank: index + 1,
            score: selection.selected[index]?.confidence || 0,
        });
    }
    await writeStatus(buildStatus(config, {
        agentId: packet.agentId,
        lastDecisionKind: 'memory_injected',
        routing: { activePolicySnapshotId: routeDecision.policySnapshotId || null },
        latency: { syncPlannerEnabled: config.latency.syncPlannerEnabled === true },
    }), { activationRoot: config.activationRoot, agentId: packet.agentId });
    store.close();
    return { prependContext: `<openclawbrain_context>\n${selection.distilledContext}\n</openclawbrain_context>` };
}
function registerPromptHooks(api, resolve) {
    safeRegisterHook(api, 'before_prompt_build', async (event = {}, ctx = {}) => handleTurnHook(withHookContext(event, ctx), resolve(), api, 'before_prompt_build'));
    safeRegisterOptionalHook(api, 'agent_turn_prepare', async (event = {}, ctx = {}) => handleTurnHook(withHookContext(event, ctx), resolve(), api, 'agent_turn_prepare'));
}
function registerLifecycleHooks(api, resolve) {
    safeRegisterHook(api, 'model_call_started', async (event = {}, ctx = {}) => writeTelemetryEvent('model_call_started', withHookContext(event, ctx), resolve(), api));
    safeRegisterHook(api, 'model_call_ended', async (event = {}, ctx = {}) => writeTelemetryEvent('model_call_ended', withHookContext(event, ctx), resolve(), api));
    safeRegisterHook(api, 'gateway_start', async (event = {}, ctx = {}) => writeGatewayStatus('gateway_start', withHookContext(event, ctx), resolve(), api));
    safeRegisterHook(api, 'gateway_stop', async (event = {}, ctx = {}) => writeGatewayStatus('gateway_stop', withHookContext(event, ctx), resolve(), api));
    if (resolve().hooks.allowConversationAccess === true) {
        safeRegisterOptionalHook(api, 'agent_end', async (event = {}, ctx = {}) => handleAgentEnd(withHookContext(event, ctx), resolve(), api));
    }
    if (resolve().capture?.enabled === true || resolve().learning?.enabled === true) {
        safeRegisterOptionalHook(api, 'after_tool_call', async (event = {}, ctx = {}) => handleAfterToolCall(withHookContext(event, ctx), resolve(), api));
    }
}
function registerFirstClassSurfaces(api, resolve) {
    const serviceState = {};
    api.registerService?.({
        id: PLUGIN_ID,
        start: async () => {
            await writeGatewayStatus('service_start', {}, resolve(), api);
            const config = resolve();
            if (config.learning.enabled === true) {
                serviceState.timer = setInterval(() => {
                    void processBackgroundJobs(config, api).catch((error) => {
                        api.logger?.warn?.({ error }, 'OpenClawBrain background job processing failed');
                    });
                }, config.learning.intervalMs);
            }
        },
        stop: async () => {
            if (serviceState.timer)
                clearInterval(serviceState.timer);
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
        path: '/plugins/openclawbrain/doctor',
        auth: 'gateway',
        match: 'exact',
        replaceExisting: true,
        handler: async (_req, res) => writeJson(res, doctorPayload(resolve()))
    });
    api.registerHttpRoute?.({
        path: '/plugins/openclawbrain/proof',
        auth: 'gateway',
        match: 'prefix',
        replaceExisting: true,
        handler: async (req, res) => writeJson(res, await proofPayload(resolve(), limitFromRequest(req)))
    });
    api.registerHttpRoute?.({
        path: '/plugins/openclawbrain/graph',
        auth: 'gateway',
        match: 'prefix',
        replaceExisting: true,
        handler: async (req, res) => writeJson(res, graphPayload(resolve(), agentIdFromRequest(req, resolve()), limitFromRequest(req)))
    });
    api.registerHttpRoute?.({
        path: '/plugins/openclawbrain/learn',
        auth: 'gateway',
        match: 'prefix',
        replaceExisting: true,
        handler: async (req, res) => writeJson(res, learnPayload(resolve(), agentIdFromRequest(req, resolve()), limitFromRequest(req)))
    });
    api.registerHttpRoute?.({
        path: '/plugins/openclawbrain/search',
        auth: 'gateway',
        match: 'prefix',
        replaceExisting: true,
        handler: async (req, res) => writeJson(res, searchPayload(resolve(), agentIdFromRequest(req, resolve()), queryFromRequest(req), limitFromRequest(req)))
    });
    api.registerHttpRoute?.({
        path: '/plugins/openclawbrain/audit',
        auth: 'gateway',
        match: 'prefix',
        replaceExisting: true,
        handler: async (req, res) => writeJson(res, auditPayload(resolve(), agentIdFromRequest(req, resolve()), limitFromRequest(req)))
    });
    api.registerHttpRoute?.({
        path: '/plugins/openclawbrain/explain-last',
        auth: 'gateway',
        match: 'prefix',
        replaceExisting: true,
        handler: async (req, res) => writeJson(res, explainLastPayload(resolve(), agentIdFromRequest(req, resolve()), turnIdFromRequest(req)))
    });
    api.registerHttpRoute?.({
        path: '/plugins/openclawbrain/route-teacher',
        auth: 'gateway',
        match: 'prefix',
        replaceExisting: true,
        handler: async (req, res) => writeJson(res, routeTeacherPayload(resolve(), agentIdFromRequest(req, resolve()), limitFromRequest(req)))
    });
    api.registerHttpRoute?.({
        path: '/plugins/openclawbrain/route-counterfactuals',
        auth: 'gateway',
        match: 'prefix',
        replaceExisting: true,
        handler: async (req, res) => writeJson(res, routeCounterfactualPayload(resolve(), agentIdFromRequest(req, resolve()), decisionIdFromRequest(req), limitFromRequest(req)))
    });
    api.registerHttpRoute?.({
        path: '/plugins/openclawbrain/route-policy',
        auth: 'gateway',
        match: 'prefix',
        replaceExisting: true,
        handler: async (req, res) => writeJson(res, routePolicyPayload(resolve(), agentIdFromRequest(req, resolve()), limitFromRequest(req)))
    });
    try {
        api.registerMemoryCapability?.(buildMemoryCapability(resolve));
        api.registerMemoryPromptSupplement?.(buildMemoryPromptSupplement());
        api.registerMemoryCorpusSupplement?.(buildMemoryCorpusSupplement(resolve()));
    }
    catch (error) {
        api.logger?.debug?.({ error }, 'OpenClawBrain memory capability/supplements unavailable; skipping');
    }
}
export function buildMemoryCapability(resolve) {
    return {
        promptBuilder: buildMemoryPromptSupplement(),
        runtime: {
            async getMemorySearchManager({ agentId }) {
                const config = resolve();
                const normalizedAgentId = safeString(agentId || config.scopes?.agents?.[0] || 'main') || 'main';
                if (!isAgentAllowed(config, normalizedAgentId)) {
                    return { manager: null, error: `agent not allowed for OpenClawBrain memory: ${normalizedAgentId}` };
                }
                return { manager: createOpenClawBrainMemorySearchManager(config, normalizedAgentId) };
            },
            resolveMemoryBackendConfig() {
                return { backend: 'builtin' };
            },
            async closeAllMemorySearchManagers() {
                return undefined;
            },
        },
    };
}
function createOpenClawBrainMemorySearchManager(config, agentId) {
    const store = new MemoryStore({ activationRoot: config.activationRoot, agentId });
    const renderReadResult = (memory, from = 1, lines) => {
        const allLines = renderMemory(memory).split(/\n/);
        const safeFrom = Math.max(1, Number(from || 1));
        const start = safeFrom - 1;
        const count = Math.max(1, Number(lines || allLines.length));
        const selected = allLines.slice(start, start + count);
        const nextFrom = start + count < allLines.length ? start + count + 1 : undefined;
        return {
            text: selected.join('\n'),
            path: memoryPath(memory),
            from: safeFrom,
            lines: selected.length,
            truncated: nextFrom !== undefined,
            ...(nextFrom ? { nextFrom } : {}),
        };
    };
    return {
        async search(query, opts = {}) {
            const memories = store.searchMemories(safeString(query), agentId, {
                limit: opts.maxResults ?? 10,
                scopeContext: defaultScopeContext(agentId),
            });
            return memories.map((memory) => ({
                path: memoryPath(memory),
                startLine: 1,
                endLine: renderMemory(memory).split(/\n/).length,
                score: Number((memory.importance * memory.confidence).toFixed(3)),
                textScore: Number((memory.importance * memory.confidence).toFixed(3)),
                snippet: memory.content,
                source: 'memory',
                citation: `${memoryPath(memory)}#L1-L${renderMemory(memory).split(/\n/).length}`,
            }));
        },
        async readFile({ relPath, from, lines }) {
            const memory = store.getMemory(extractMemoryId(relPath));
            if (!memory || memory.deletedAt || memory.supersededBy || !memoryInScope(memory, defaultScopeContext(agentId))) {
                return { text: '', path: relPath, from: Math.max(1, Number(from || 1)), lines: 0, truncated: false };
            }
            return renderReadResult(memory, from, lines);
        },
        status() {
            const nodes = store.countMemories(agentId);
            const edges = store.countEdgesForAgent(agentId);
            return {
                backend: 'builtin',
                provider: 'openclawbrain',
                files: nodes,
                chunks: nodes,
                dirty: false,
                sources: ['memory'],
                sourceCounts: [{ source: 'memory', files: nodes, chunks: nodes }],
                custom: {
                    agentId,
                    plugin: PLUGIN_ID,
                    pluginVersion: PLUGIN_VERSION,
                    nodes,
                    edges,
                    captureAuditRows: store.countCaptureAudit(agentId),
                    routeDecisions: store.countRouteDecisions(agentId),
                },
            };
        },
        async sync() {
            return undefined;
        },
        getCachedEmbeddingAvailability() {
            return { ok: true, checked: true, cached: true };
        },
        async probeEmbeddingAvailability() {
            return { ok: true, checked: true };
        },
        async probeVectorAvailability() {
            return false;
        },
        async close() {
            store.close();
        },
    };
}
async function statusPayload(config, req = {}) {
    const agentId = safeString(req.query?.agentId ?? req.query?.agent ?? config.scopes.agents[0] ?? 'main') || 'main';
    if (!isAgentAllowed(config, agentId))
        return { ok: false, agentId, reason: 'agent_not_allowed' };
    const nativeSqlite = nativeSqliteSmokeTest();
    let persisted = null;
    let details = { nativeSqlite };
    try {
        persisted = await readStatus({ activationRoot: config.activationRoot, agentId });
    }
    catch {
        persisted = null;
    }
    try {
        const store = new MemoryStore({ activationRoot: config.activationRoot, agentId });
        details = {
            memory: {
                nodes: store.countMemories(agentId),
                edges: store.countEdgesForAgent(agentId),
                corrections: store.countMemories(agentId, 'correction'),
                preferences: store.countMemories(agentId, 'preference'),
                workflows: store.countMemories(agentId, 'workflow'),
                context: store.countMemories(agentId, 'context'),
                projectFacts: store.countMemories(agentId, 'project_fact'),
                toolConventions: store.countMemories(agentId, 'tool_convention'),
                routingRules: store.countMemories(agentId, 'routing_rule'),
                recallRules: store.countMemories(agentId, 'recall_rule'),
                captureAuditRows: store.countCaptureAudit(agentId),
            },
            routing: {
                activePolicySnapshotId: store.getActivePolicySnapshot(agentId)?.id || null,
                routeDecisions: store.countRouteDecisions(agentId),
                pendingOutcomes: store.getUnresolvedRouteDecisions(agentId).length,
                positiveExamples: store.countRouteExamples(agentId, 'positive'),
                negativeExamples: store.countRouteExamples(agentId, 'negative'),
                routeTeacherRuns: store.listRouteTeacherRuns(agentId, 100).length,
                routeTrainingExamplesV2: store.countRouteTrainingExamplesV2(agentId),
                activePolicySnapshotV2Id: store.getActivePolicySnapshotV2(agentId)?.id || null,
            },
            learning: {
                enabled: config.learning.enabled === true,
                queueDepth: store.getJobQueueDepth(agentId),
            },
            latency: {
                syncPlannerEnabled: config.latency.syncPlannerEnabled === true,
                syncPlannerCalls: store.countSyncPlannerCalls(agentId),
                syncPlannerFallbacks: store.countSyncPlannerFallbacks(agentId),
                avgSyncPlannerMs: store.averageSyncPlannerLatency(agentId),
                tier0Turns: store.countRouteDecisionsByLatencyTier(agentId, 'no_extra_llm'),
                tier1Turns: store.countRouteDecisionsByLatencyTier(agentId, 'cached_route'),
                tier2Turns: store.countRouteDecisionsByLatencyTier(agentId, 'sync_memory_planner'),
            },
            nativeSqlite,
        };
        store.close();
    }
    catch {
        details = { nativeSqlite };
    }
    return { ...buildStatus(config, { agentId, ...details }), persisted };
}
function doctorPayload(config) {
    const nativeSqlite = nativeSqliteSmokeTest();
    return {
        ok: nativeSqlite.ok,
        plugin: PLUGIN_ID,
        pluginVersion: PLUGIN_VERSION,
        enabled: config.enabled === true,
        checks: {
            nativeSqlite,
            rawTranscriptUploadDisabled: config.rawTranscriptUpload !== true,
            promptContextExplicitlyAllowed: config.hooks?.allowPromptContext === true,
        },
    };
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
async function handleAgentEnd(event = {}, config = {}, api = {}) {
    await writeTelemetryEvent('agent_end', event, config, api);
    if (!shouldAllowCapture(config) || config.hooks.allowConversationAccess !== true)
        return {};
    const agentId = agentIdFromEvent(event);
    const store = new MemoryStore({ activationRoot: config.activationRoot, agentId });
    const queue = new JobQueue({ store });
    const packet = new CaptureOrchestrator().fromAgentEnd(event, config);
    const captureIntent = suppressSyntheticCaptureIntent(detectCaptureIntent(packet), packet);
    const retrievalIntent = detectRetrievalIntent(packet);
    const learner = new BackgroundLearner({ store, config });
    try {
        learner.processAgentEnd(agentId, packet);
    }
    catch (error) {
        api.logger?.warn?.({ error }, 'OpenClawBrain agent_end outcome learning failed');
    }
    if (!captureIntent.shouldConsiderCapture && (packet.recentInjections.length === 0 || shouldSuppressCapture(packet))) {
        store.insertCaptureAudit({
            agentId: packet.agentId,
            sessionId: packet.sessionId,
            turnId: packet.turnId,
            runId: packet.runId,
            retrievalIntent,
            captureIntent,
            captureJobCreated: false,
            distillerRan: false,
            fallbackRan: false,
            candidateCount: 0,
            storedCount: 0,
            rejectedCount: 1,
            rejectionReasons: [captureIntent.intent === 'one_off' ? 'one_off_request' : captureIntent.reason || 'no_capture_signal'],
            evidenceHash: String(packet.metadata.promptHash || ''),
        });
        store.close();
        return {};
    }
    if (config.capture.agentEndMode === 'best_effort_async' && config.llm.enabled === true) {
        try {
            await runFeedbackDistillation(packet, config, store, { captureIntent, retrievalIntent });
        }
        catch (error) {
            api.logger?.warn?.({ error }, 'OpenClawBrain best-effort agent_end distillation failed');
        }
        finally {
            store.close();
        }
        return {};
    }
    queue.enqueueFeedbackDistillation(agentId, { packet, captureIntent, retrievalIntent }, { priority: 10 });
    store.close();
    return {};
}
async function handleAfterToolCall(event = {}, config = {}, api = {}) {
    if (config.learning?.enabled !== true || config.hooks.allowToolObservation !== true)
        return {};
    const agentId = agentIdFromEvent(event);
    const store = new MemoryStore({ activationRoot: config.activationRoot, agentId });
    const queue = new JobQueue({ store });
    const packet = new CaptureOrchestrator().fromAfterToolCall(event, config);
    if (!packet.runId || !packet.turnId) {
        store.close();
        return {};
    }
    queue.enqueueOutcomeClassification(agentId, { packet }, { priority: 5 });
    store.close();
    return {};
}
async function processBackgroundJobs(config = {}, api = {}) {
    if (config.capture?.enabled !== true && config.learning?.enabled !== true && config.routeLearning?.enabled !== true)
        return;
    const agents = Array.isArray(config.scopes?.agents) && config.scopes.agents.length ? config.scopes.agents : ['main'];
    for (const agentId of agents)
        await processBackgroundJobsForAgent(config, api, agentId);
}
async function processBackgroundJobsForAgent(config = {}, api = {}, agentId = 'main') {
    const store = new MemoryStore({ activationRoot: config.activationRoot, agentId });
    const queue = new JobQueue({ store });
    const job = queue.claimNext(undefined, agentId);
    const learner = new BackgroundLearner({ store, config });
    const routeLearning = new RouteLearning({ store, config });
    const routeTeacher = new RouteTeacher({ store, config, client: llmClientFromConfig(config) });
    try {
        let learningReport = null;
        if (job?.kind === 'feedback_distillation') {
            await runFeedbackDistillation(job.payload?.packet, config, store, {
                captureIntent: job.payload?.captureIntent,
                retrievalIntent: job.payload?.retrievalIntent,
            });
            queue.enqueueRouteLearning(agentId, { cause: 'feedback_distillation', turnId: job.payload?.packet?.turnId || '' }, { priority: 4 });
        }
        else if (job?.kind === 'outcome_classification') {
            learningReport = learner.processOutcomeClassification(agentId, job.payload?.packet);
        }
        else if (job?.kind === 'route_learning') {
            learningReport = { ...routeLearning.run(agentId), lastRunAt: new Date().toISOString() };
            queue.enqueueRouteTeacher(agentId, { cause: 'route_learning', at: new Date().toISOString() }, { priority: 3, delayMs: 0 });
        }
        else if (job?.kind === 'route_teacher') {
            learningReport = { ...(await routeTeacher.run(agentId)), lastRunAt: new Date().toISOString() };
        }
        else if (job) {
            throw new Error(`unsupported_job_kind:${job.kind}`);
        }
        if (!job) {
            learningReport = learner.runMaintenance(agentId);
            if (config.routeLearning?.teacher?.enabled !== false) {
                const teacherReport = await routeTeacher.run(agentId);
                learningReport = { ...learningReport, routeTeacher: teacherReport, lastRunAt: new Date().toISOString() };
            }
        }
        if (learningReport) {
            await writeStatus(buildStatus(config, {
                agentId,
                lastDecisionKind: 'learning_cycle',
                routing: { activePolicySnapshotId: store.getActivePolicySnapshot(agentId)?.id || null },
                routeLearning: { activePolicySnapshotV2Id: store.getActivePolicySnapshotV2(agentId)?.id || null },
                learning: {
                    enabled: config.learning.enabled === true,
                    queueDepth: store.getJobQueueDepth(agentId),
                    lastRunAt: learningReport.lastRunAt,
                },
            }), { activationRoot: config.activationRoot, agentId });
        }
        if (job)
            queue.complete(job.id);
    }
    catch (error) {
        if (job)
            queue.fail(job.id, error?.message || String(error), 5000);
        api.logger?.warn?.({ error, jobId: job?.id, kind: job?.kind }, 'OpenClawBrain background job failed');
    }
    finally {
        store.close();
    }
}
async function runFeedbackDistillation(packet, config, store, context = {}) {
    if (!shouldAllowCapture(config) || shouldSuppressCapture(packet))
        return null;
    const client = llmClientFromConfig(config);
    if (!client)
        return null;
    const distiller = new FeedbackDistiller({ client, config });
    const captureIntent = context.captureIntent ?? detectCaptureIntent(packet);
    const retrievalIntent = context.retrievalIntent ?? detectRetrievalIntent(packet);
    const result = await distiller.distill(packet, { captureIntent, retrievalIntent });
    const applied = new MemoryOperationApplier({ store, config }).applyDistillation(result.output, packet, { captureIntent });
    store.insertDistillationRun({
        agentId: packet.agentId,
        sessionId: packet.sessionId,
        turnId: packet.turnId,
        runId: packet.runId,
        phase: packet.sourceHook === 'agent_end' ? 'agent_end_feedback' : 'immediate_feedback',
        model: result.audit.model,
        promptVersion: 'feedback-distiller-v1',
        inputHash: result.audit.inputHash,
        redactedInputSummary: result.audit.redactedInputSummary,
        outputJson: config.privacy?.storeDistillationOutputs === false ? JSON.stringify({ stored: false, reason: 'storeDistillationOutputs=false' }) : JSON.stringify(redactJsonValue(result.output)),
        validationStatus: result.audit.validationStatus,
        validationError: result.audit.validationError || result.audit.parseError,
        latencyMs: result.audit.latencyMs,
    });
    store.insertCaptureAudit({
        agentId: packet.agentId,
        sessionId: packet.sessionId,
        turnId: packet.turnId,
        runId: packet.runId,
        retrievalIntent,
        captureIntent,
        captureJobCreated: true,
        distillerRan: true,
        distillerModel: result.audit.model,
        distillerLatencyMs: result.audit.latencyMs,
        fallbackRan: result.audit.validationStatus === 'fallback',
        candidateCount: result.output.memoryCandidates.length,
        storedCount: applied?.storedCandidates ?? 0,
        rejectedCount: (applied?.rejectedCandidates ?? 0) + (result.output.shouldStore ? 0 : 1),
        rejectionReasons: [...new Set([...(applied?.rejectionReasons ?? []), ...(result.output.audit.rejectionReasons ?? [result.output.audit.modelReasonCode])])],
        safeCandidatePreview: config.privacy?.storeDistillationOutputs === false ? undefined : redactText(result.output.audit.safeCandidatePreview || '', 500),
        evidenceHash: String(packet.metadata.promptHash || ''),
    });
    return applied;
}
function llmClientFromConfig(config) {
    if (config.llm?.enabled !== true)
        return null;
    const models = [config.llm.plannerModel, config.llm.routeModel, config.llm.feedbackModel, config.llm.learningModel].filter(Boolean);
    const allowed = new Set(Array.isArray(config.llm.allowedModels) ? config.llm.allowedModels : []);
    if (allowed.size > 0 && models.some((model) => !allowed.has(model)))
        return null;
    if (config.llm.baseUrl && !isLoopbackUrl(config.llm.baseUrl) && config.llm.allowRemoteLlm !== true)
        return null;
    if (config.llm.baseUrl && isOllamaLoopbackBaseUrl(config.llm.baseUrl))
        return new OllamaNativeLlmClient({ baseUrl: config.llm.baseUrl });
    if (config.llm.baseUrl)
        return new OpenAICompatibleLlmClient({ baseUrl: config.llm.baseUrl });
    return null;
}
function retrieveCandidates(store, packet, queries, memoryTypes, maxCandidates) {
    const agentId = packet.agentId;
    const scopeContext = scopeContextFromPacket(packet);
    const seen = new Set();
    const results = [];
    for (const query of queries) {
        const hits = store.searchMemories(query, agentId, { limit: maxCandidates, scopeContext });
        for (const hit of hits) {
            if (seen.has(hit.id))
                continue;
            seen.add(hit.id);
            results.push(hit);
            if (results.length >= maxCandidates)
                return results;
        }
    }
    for (const memoryType of memoryTypes) {
        const hits = store.listMemories(agentId, { type: memoryType, limit: maxCandidates, scopeContext });
        for (const hit of hits) {
            if (seen.has(hit.id))
                continue;
            seen.add(hit.id);
            results.push(hit);
            if (results.length >= maxCandidates)
                return results;
        }
    }
    return results;
}
function emptySelection() {
    return {
        shouldInject: false,
        confidence: 0,
        selectedMemoryIds: [],
        distilledContext: '',
        selected: [],
        omitted: [],
        audit: { promptBudgetUsedChars: 0, risk: 'low' },
    };
}
function estimateTaskValue(message) {
    const lower = message.toLowerCase();
    if (/\b(architecture|implementation|debug|production|repo|build|design|plan)\b/.test(lower))
        return 'high';
    if (/\b(install|dependency|test|setup|continue)\b/.test(lower))
        return 'medium';
    return 'low';
}
function withHookContext(event = {}, ctx = {}) {
    if (!ctx || typeof ctx !== 'object' || Object.keys(ctx).length === 0)
        return event || {};
    const base = event && typeof event === 'object' ? event : {};
    const existingCtx = base.ctx && typeof base.ctx === 'object' ? base.ctx : {};
    const mergedCtx = { ...ctx, ...existingCtx };
    return {
        ...base,
        ctx: mergedCtx,
        agentId: base.agentId ?? base.agent_id ?? mergedCtx.agentId,
        sessionKey: base.sessionKey ?? base.session_key ?? mergedCtx.sessionKey,
        sessionId: base.sessionId ?? base.session_id ?? mergedCtx.sessionId,
        runId: base.runId ?? base.run_id ?? mergedCtx.runId,
    };
}
function suppressSyntheticCapture(plan, packet) {
    const trigger = String(packet?.metadata?.trigger || '').toLowerCase();
    if (!shouldSuppressCapture(packet))
        return plan;
    return {
        ...plan,
        route: plan.route === 'capture_only' ? 'no_memory' : plan.route,
        enqueueCapture: false,
        captureIntent: {
            ...plan.captureIntent,
            shouldConsiderCapture: false,
            intent: 'one_off',
            confidence: Math.max(0.9, Number(plan.captureIntent?.confidence || 0)),
            reason: `System-generated ${trigger || 'non-user'} prompt; capture disabled`,
            matchedSignals: [],
        },
    };
}
function suppressSyntheticCaptureIntent(captureIntent, packet) {
    if (!shouldSuppressCapture(packet))
        return captureIntent;
    const trigger = String(packet?.metadata?.trigger || '').toLowerCase();
    return {
        ...captureIntent,
        shouldConsiderCapture: false,
        intent: 'one_off',
        confidence: Math.max(0.9, Number(captureIntent?.confidence || 0)),
        reason: `System-generated ${trigger || 'non-user'} prompt; capture disabled`,
        matchedSignals: [],
    };
}
function shouldSuppressCapture(packet) {
    const trigger = String(packet?.metadata?.trigger || '').toLowerCase();
    const sourceHook = String(packet?.sourceHook || '').toLowerCase();
    if (['heartbeat', 'cron', 'system', 'subagent'].includes(trigger))
        return true;
    if (sourceHook === 'after_tool_call')
        return true;
    return false;
}
function shouldAllowCapture(config) {
    return config.capture?.enabled === true && config.capture?.mode !== 'off' && config.memory?.captureMode !== 'off';
}
function shouldAllowRouting(config) {
    return config.routing?.enabled === true && config.routing?.mode !== 'off';
}
function isLoopbackUrl(value) {
    try {
        const url = new URL(value);
        if (!['http:', 'https:'].includes(url.protocol))
            return false;
        const host = url.hostname.replace(/\.$/, '').toLowerCase();
        if (host === 'localhost' || host === '::1' || host === '[::1]')
            return true;
        if (/^127(?:\.\d{1,3}){3}$/.test(host))
            return true;
        if (host === '::ffff:127.0.0.1' || host === '[::ffff:127.0.0.1]')
            return true;
        return false;
    }
    catch {
        return false;
    }
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
function decisionIdFromRequest(req = {}) {
    if (req.query?.decisionId)
        return safeString(req.query.decisionId);
    if (req.query?.routeDecisionId)
        return safeString(req.query.routeDecisionId);
    try {
        const url = new URL(req.url || 'http://local/plugins/openclawbrain/route-counterfactuals', 'http://local');
        return safeString(url.searchParams.get('decisionId') || url.searchParams.get('routeDecisionId') || '');
    }
    catch {
        return '';
    }
}
function routeTeacherPayload(config = {}, agentId = 'main', limit = 20) {
    const store = new MemoryStore({ activationRoot: config.activationRoot, agentId });
    try {
        const runs = store.listRouteTeacherRuns(agentId, Number(limit) || 20);
        return {
            ok: true,
            agentId,
            count: runs.length,
            runs: runs.map((run) => ({
                id: run.id,
                routeDecisionId: run.routeDecisionId,
                verdict: run.verdict,
                teacherRoute: run.teacherRoute,
                teacherMemoryIds: run.teacherMemoryIds,
                teacherQueries: run.teacherQueries,
                teacherGraphDepth: run.teacherGraphDepth,
                syncPlannerWorthIt: run.syncPlannerWorthIt,
                confidence: run.confidence,
                rationale: run.rationale,
                validated: run.validated,
                rejectionReason: run.rejectionReason,
                model: run.model,
                createdAt: run.createdAt,
            })),
        };
    }
    finally {
        store.close();
    }
}
function routeCounterfactualPayload(config = {}, agentId = 'main', decisionId = '', limit = 50) {
    const store = new MemoryStore({ activationRoot: config.activationRoot, agentId });
    try {
        const counterfactuals = store.listRouteCounterfactuals(agentId, decisionId || undefined, Number(limit) || 50);
        return { ok: true, agentId, routeDecisionId: decisionId || null, count: counterfactuals.length, counterfactuals };
    }
    finally {
        store.close();
    }
}
function routePolicyPayload(config = {}, agentId = 'main', limit = 20) {
    const store = new MemoryStore({ activationRoot: config.activationRoot, agentId });
    try {
        const active = store.getActivePolicySnapshotV2(agentId);
        const snapshots = store.listPolicySnapshotsV2(agentId, Number(limit) || 20);
        const examples = store.listRouteTrainingExamplesV2(agentId, 50);
        return {
            ok: true,
            agentId,
            active,
            snapshotCount: snapshots.length,
            snapshots: snapshots.map((snapshot) => ({
                id: snapshot.id,
                version: snapshot.version,
                status: snapshot.status,
                ruleCount: snapshot.rules.length,
                globalBudgets: snapshot.globalBudgets,
                evalSummary: snapshot.evalSummary,
                createdAt: snapshot.createdAt,
            })),
            exampleCount: examples.length,
            examples: examples.slice(0, 20),
        };
    }
    finally {
        store.close();
    }
}
function queryFromRequest(req = {}) {
    if (req.query?.query)
        return String(req.query.query);
    if (req.query?.q)
        return String(req.query.q);
    try {
        const url = new URL(req.url || 'http://local/plugins/openclawbrain/search', 'http://local');
        return url.searchParams.get('query') || url.searchParams.get('q') || '';
    }
    catch {
        return '';
    }
}
function turnIdFromRequest(req = {}) {
    if (req.query?.turnId)
        return safeString(req.query.turnId);
    if (req.query?.turn)
        return safeString(req.query.turn);
    try {
        const url = new URL(req.url || 'http://local/plugins/openclawbrain/explain-last', 'http://local');
        return safeString(url.searchParams.get('turnId') || url.searchParams.get('turn') || '');
    }
    catch {
        return '';
    }
}
function agentIdFromRequest(req = {}, config = {}) {
    if (req.query?.agentId)
        return safeString(req.query.agentId) || 'main';
    if (req.query?.agent)
        return safeString(req.query.agent) || 'main';
    try {
        const url = new URL(req.url || 'http://local/plugins/openclawbrain/status', 'http://local');
        return safeString(url.searchParams.get('agentId') || url.searchParams.get('agent') || config.scopes?.agents?.[0] || 'main') || 'main';
    }
    catch {
        return safeString(config.scopes?.agents?.[0] || 'main') || 'main';
    }
}
function agentIdFromEvent(event = {}) {
    return safeString(event.ctx?.agentId ?? event.agentId ?? event.agent_id ?? event.profileId ?? event.profile_id ?? event.session?.agentId ?? 'main') || 'main';
}
