import { DEFAULT_CONFIG, PLUGIN_ID, PLUGIN_VERSION, activationRootForAgent, isAgentAllowed, normalizePluginConfig, resolveOpenClawBrainConfig } from './config.js';
import { buildInjectionText, ensureActivationRoot, readActivationContext } from './context-files.js';
import { CaptureOrchestrator } from './capture.js';
import { ContextSelector } from './context-selector.js';
import { FeedbackDistiller } from './feedback-distiller.js';
import { BackgroundLearner } from './learning.js';
import { JobQueue } from './job-queue.js';
import { LatencyController } from './latency-controller.js';
import { FakeLlmClient, OpenAICompatibleLlmClient } from './llm-client.js';
import { MemoryPlanner } from './memory-planner.js';
import { MemoryOperationApplier } from './memory-operations.js';
import { MemoryStore } from './memory-store.js';
import { decidePolicy } from './policy.js';
import { appendProofEvent, readProofEvents, readStatus, writeStatus } from './proof-store.js';
import { RouteLearning } from './route-learning.js';
import { buildMemoryCorpusSupplement, buildMemoryPromptSupplement, graphPayload, learnPayload, searchPayload } from './search.js';
import { buildStatus } from './status.js';
import { nativeSqliteSmokeTest } from './native-sqlite.js';
import { clipText, eventId, hashText, latestUserTextFromEvent, redactText, safeString, shortHash } from './redact.js';
import { RouteFn } from './route-fn.js';

export { normalizePluginConfig, resolveOpenClawBrainConfig } from './config.js';
export { redactText, hashText } from './redact.js';
export { decidePolicy, classifyTurn } from './policy.js';
export { readActivationContext } from './context-files.js';
export { appendProofEvent, readProofEvents, readStatus, writeStatus } from './proof-store.js';
export { buildStatus } from './status.js';
export { FakeLlmClient, OpenAICompatibleLlmClient } from './llm-client.js';
export { JsonParseError, JsonTimeoutError, JsonValidationError, runJsonWithValidation, validateWithGuard, withTimeout } from './llm-json.js';
export { CaptureOrchestrator, sanitizeToolEvent } from './capture.js';
export { FeedbackDistiller, validateFeedbackDistillation } from './feedback-distiller.js';
export { MemoryOperationApplier } from './memory-operations.js';
export { JobQueue } from './job-queue.js';
export { LatencyController } from './latency-controller.js';
export { RouteCache, RouteFn } from './route-fn.js';
export { ContextSelector } from './context-selector.js';
export { MemoryPlanner } from './memory-planner.js';
export { nativeSqliteSmokeTest } from './native-sqlite.js';
export { BackgroundLearner } from './learning.js';
export { RouteLearning } from './route-learning.js';
export { buildMemoryCorpusSupplement, buildMemoryPromptSupplement, graphPayload, learnPayload, searchPayload } from './search.js';

export const openClawBrainPluginEntry = {
  id: PLUGIN_ID,
  name: 'OpenClawBrain',
  version: PLUGIN_VERSION,
  register(api: any = {}) {
    const resolve = () => resolveOpenClawBrainConfig(api);
    registerFirstClassSurfaces(api, resolve);
    registerPromptHooks(api, resolve);
    registerLifecycleHooks(api, resolve);
  }
};

export default openClawBrainPluginEntry;

export function redactedTurnFromPromptEvent(event: any = {}, config: any = DEFAULT_CONFIG) {
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

export async function handleTurnHook(event: any = {}, config: any = normalizePluginConfig(), api: any = {}, phase = 'before_prompt_build') {
  const agentId = agentIdFromEvent(event);
  if (config.rawTranscriptUpload === true) {
    await writeProofForDecision(event, config, api, { kind: 'stay_silent', slice: 'unknown', reasonCode: 'raw_transcript_upload_requested' }, phase, [], []);
    return {};
  }
  if (!config.enabled || config.mode === 'off') return {};
  if (!isAgentAllowed(config, agentId)) return {};

  if (phase === 'before_prompt_build' && config.routing?.enabled === true && (config.mode === 'balanced' || config.mode === 'aggressive')) {
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
  if (!injection) return {};
  return { prependContext: injection };
}

async function handleV2PromptHook(event: any = {}, config: any = normalizePluginConfig(), api: any = {}, phase = 'before_prompt_build') {
  const agentId = agentIdFromEvent(event);
  const store = new MemoryStore({ activationRoot: config.activationRoot, agentId });
  const queue = new JobQueue({ store });
  const packet = new CaptureOrchestrator().fromBeforePromptBuild(event, config);
  const routeFn = new RouteFn({ config, store });
  const contextSelector = new ContextSelector(config);
  const initialPlan = routeFn.plan(packet);
  const initialCandidates = initialPlan.shouldRetrieve
    ? retrieveCandidates(store, packet.agentId, initialPlan.retrievalPlan.queries, initialPlan.retrievalPlan.memoryTypes, initialPlan.retrievalPlan.maxCandidates)
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

  if (latency.kind === 'sync_memory_planner' && client) {
    const planner = new MemoryPlanner({ config, routeFn, store, client });
    const planned = await planner.run(packet);
    plan = planned.routePlan;
    selection = planned.contextSelection ?? emptySelection();
  } else if (plan.enqueueCapture) {
    queue.enqueueFeedbackDistillation(agentId, { packet }, { priority: 10 });
  }

  const routeDecision = store.insertRouteDecision({
    agentId: packet.agentId,
    sessionId: packet.sessionId,
    turnId: packet.turnId,
    runId: packet.runId,
    route: plan.route,
    confidence: plan.confidence,
    latencyTier: latency.kind,
    syncLlmUsed: latency.kind === 'sync_memory_planner' && Boolean(client),
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

function registerPromptHooks(api: any, resolve: any) {
  safeRegisterHook(api, 'before_prompt_build', async (event = {}) => handleTurnHook(event, resolve(), api, 'before_prompt_build'));
  safeRegisterOptionalHook(api, 'agent_turn_prepare', async (event = {}) => handleTurnHook(event, resolve(), api, 'agent_turn_prepare'));
}

function registerLifecycleHooks(api: any, resolve: any) {
  safeRegisterHook(api, 'model_call_started', async (event = {}) => writeTelemetryEvent('model_call_started', event, resolve(), api));
  safeRegisterHook(api, 'model_call_ended', async (event = {}) => writeTelemetryEvent('model_call_ended', event, resolve(), api));
  safeRegisterHook(api, 'gateway_start', async (event = {}) => writeGatewayStatus('gateway_start', event, resolve(), api));
  safeRegisterHook(api, 'gateway_stop', async (event = {}) => writeGatewayStatus('gateway_stop', event, resolve(), api));
  if (resolve().hooks.allowConversationAccess === true) {
    safeRegisterOptionalHook(api, 'agent_end', async (event = {}) => handleAgentEnd(event, resolve(), api));
  }
  if (resolve().capture?.enabled === true || resolve().learning?.enabled === true) {
    safeRegisterOptionalHook(api, 'after_tool_call', async (event = {}) => handleAfterToolCall(event, resolve(), api));
  }
}

function registerFirstClassSurfaces(api: any, resolve: any) {
  const serviceState: { timer?: NodeJS.Timeout } = {};
  api.registerService?.({
    id: PLUGIN_ID,
    start: async () => {
      await writeGatewayStatus('service_start', {}, resolve(), api);
      const config = resolve();
      if (config.learning.enabled === true) {
        serviceState.timer = setInterval(() => {
          void processBackgroundJobs(config, api).catch((error: any) => {
            api.logger?.warn?.({ error }, 'OpenClawBrain background job processing failed');
          });
        }, config.learning.intervalMs);
      }
    },
    stop: async () => {
      if (serviceState.timer) clearInterval(serviceState.timer);
      await writeGatewayStatus('service_stop', {}, resolve(), api);
    }
  });
  api.registerHttpRoute?.({
    path: '/plugins/openclawbrain/status',
    auth: 'gateway',
    match: 'exact',
    replaceExisting: true,
    handler: async (req: any, res: any) => writeJson(res, await statusPayload(resolve(), req))
  });
  api.registerHttpRoute?.({
    path: '/plugins/openclawbrain/doctor',
    auth: 'gateway',
    match: 'exact',
    replaceExisting: true,
    handler: async (_req: any, res: any) => writeJson(res, doctorPayload(resolve()))
  });
  api.registerHttpRoute?.({
    path: '/plugins/openclawbrain/proof',
    auth: 'gateway',
    match: 'prefix',
    replaceExisting: true,
    handler: async (req: any, res: any) => writeJson(res, await proofPayload(resolve(), limitFromRequest(req)))
  });
  api.registerHttpRoute?.({
    path: '/plugins/openclawbrain/graph',
    auth: 'gateway',
    match: 'prefix',
    replaceExisting: true,
    handler: async (req: any, res: any) => writeJson(res, graphPayload(resolve(), agentIdFromRequest(req, resolve()), limitFromRequest(req)))
  });
  api.registerHttpRoute?.({
    path: '/plugins/openclawbrain/learn',
    auth: 'gateway',
    match: 'prefix',
    replaceExisting: true,
    handler: async (req: any, res: any) => writeJson(res, learnPayload(resolve(), agentIdFromRequest(req, resolve()), limitFromRequest(req)))
  });
  api.registerHttpRoute?.({
    path: '/plugins/openclawbrain/search',
    auth: 'gateway',
    match: 'prefix',
    replaceExisting: true,
    handler: async (req: any, res: any) => writeJson(res, searchPayload(resolve(), agentIdFromRequest(req, resolve()), queryFromRequest(req), limitFromRequest(req)))
  });
  try {
    api.registerMemoryPromptSupplement?.(buildMemoryPromptSupplement());
    api.registerMemoryCorpusSupplement?.(buildMemoryCorpusSupplement(resolve()));
  } catch (error: any) {
    api.logger?.debug?.({ error }, 'OpenClawBrain memory supplements unavailable; skipping');
  }
}

async function statusPayload(config: any, req: any = {}) {
  const agentId = safeString(req.query?.agentId ?? req.query?.agent ?? config.scopes.agents[0] ?? 'main') || 'main';
  const nativeSqlite = nativeSqliteSmokeTest();
  let persisted = null;
  let details: any = { nativeSqlite };
  try {
    persisted = await readStatus({ activationRoot: config.activationRoot, agentId });
  } catch {
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
      },
      routing: {
        activePolicySnapshotId: store.getActivePolicySnapshot(agentId)?.id || null,
        routeDecisions: store.countRouteDecisions(agentId),
        pendingOutcomes: store.getUnresolvedRouteDecisions(agentId).length,
        positiveExamples: store.countRouteExamples(agentId, 'positive'),
        negativeExamples: store.countRouteExamples(agentId, 'negative'),
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
  } catch {
    details = { nativeSqlite };
  }
  return { ...buildStatus(config, { agentId, ...details }), persisted };
}

function doctorPayload(config: any) {
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

async function proofPayload(config: any, limit: any = 20) {
  const agentId = config.scopes.agents[0] || 'main';
  const events = await readProofEvents({ activationRoot: config.activationRoot, agentId, limit });
  return { ok: true, plugin: PLUGIN_ID, limit: Math.min(100, Math.max(1, Number(limit || 20))), events };
}

async function writeProofForDecision(event: any, config: any, api: any, decision: any, phase: string, usedFileIdsRedacted: any[] = [], rejectedFiles: any[] = []) {
  if (!config.proofEvents) return null;
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
  } catch (error: any) {
    api.logger?.warn?.({ error }, 'OpenClawBrain failed to write proof event');
    return null;
  }
}

async function writeTelemetryEvent(marker: string, event: any, config: any, api: any) {
  if (!config.enabled || !config.proofEvents) return;
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
  } catch (error: any) {
    api.logger?.warn?.({ error }, 'OpenClawBrain failed to write telemetry proof');
  }
}

async function writeGatewayStatus(marker: string, event: any, config: any, api: any) {
  const agentId = agentIdFromEvent(event);
  try {
    await ensureActivationRoot(config, agentId);
    await writeStatus(buildStatus(config, { agentId, lastDecisionKind: marker }), { activationRoot: config.activationRoot, agentId });
  } catch (error: any) {
    api.logger?.warn?.({ error }, 'OpenClawBrain failed to write gateway status');
  }
}

async function writeDecisionStatus(config: any, redactedTurn: any, decision: any) {
  await writeStatus(buildStatus(config, { agentId: redactedTurn.agentId, lastDecisionKind: decision.kind }), { activationRoot: config.activationRoot, agentId: redactedTurn.agentId });
}

async function handleAgentEnd(event: any = {}, config: any = {}, api: any = {}) {
  await writeTelemetryEvent('agent_end', event, config, api);
  if (!config.capture?.enabled || config.hooks.allowConversationAccess !== true) return {};
  const agentId = agentIdFromEvent(event);
  const store = new MemoryStore({ activationRoot: config.activationRoot, agentId });
  const queue = new JobQueue({ store });
  const packet = new CaptureOrchestrator().fromAgentEnd(event, config);
  const learner = new BackgroundLearner({ store, config });

  try {
    learner.processAgentEnd(agentId, packet);
  } catch (error: any) {
    api.logger?.warn?.({ error }, 'OpenClawBrain agent_end outcome learning failed');
  }

  if (config.capture.agentEndMode === 'best_effort_async' && config.llm.enabled === true) {
    try {
      await runFeedbackDistillation(packet, config, store);
    } catch (error: any) {
      api.logger?.warn?.({ error }, 'OpenClawBrain best-effort agent_end distillation failed');
    } finally {
      store.close();
    }
    return {};
  }

  queue.enqueueFeedbackDistillation(agentId, { packet }, { priority: 10 });
  store.close();
  return {};
}

async function handleAfterToolCall(event: any = {}, config: any = {}, api: any = {}) {
  const agentId = agentIdFromEvent(event);
  const store = new MemoryStore({ activationRoot: config.activationRoot, agentId });
  const queue = new JobQueue({ store });
  const packet = new CaptureOrchestrator().fromAfterToolCall(event, config);
  queue.enqueueOutcomeClassification(agentId, { packet }, { priority: 5 });
  store.close();
  return {};
}

async function processBackgroundJobs(config: any = {}, api: any = {}) {
  if (config.capture?.enabled !== true && config.learning?.enabled !== true) return;
  const agentId = config.scopes.agents[0] || 'main';
  const store = new MemoryStore({ activationRoot: config.activationRoot, agentId });
  const queue = new JobQueue({ store });
  const job = queue.claimNext();
  const learner = new BackgroundLearner({ store, config });
  const routeLearning = new RouteLearning({ store, config });
  try {
    let learningReport: any = null;
    if (job?.kind === 'feedback_distillation') {
      await runFeedbackDistillation(job.payload?.packet as any, config, store);
      queue.enqueueRouteLearning(agentId, { cause: 'feedback_distillation', turnId: (job.payload?.packet as any)?.turnId || '' }, { priority: 4 });
    } else if (job?.kind === 'outcome_classification') {
      learningReport = learner.processOutcomeClassification(agentId, job.payload?.packet as any);
    } else if (job?.kind === 'route_learning') {
      learningReport = { ...routeLearning.run(agentId), lastRunAt: new Date().toISOString() };
    }
    if (!job) {
      learningReport = learner.runMaintenance(agentId);
    }
    if (learningReport) {
      await writeStatus(buildStatus(config, {
        agentId,
        lastDecisionKind: 'learning_cycle',
        routing: { activePolicySnapshotId: store.getActivePolicySnapshot(agentId)?.id || null },
        learning: {
          enabled: config.learning.enabled === true,
          queueDepth: store.getJobQueueDepth(agentId),
          lastRunAt: learningReport.lastRunAt,
        },
      }), { activationRoot: config.activationRoot, agentId });
    }
    if (job) queue.complete(job.id);
  } catch (error: any) {
    if (job) queue.fail(job.id, error?.message || String(error), 5000);
    api.logger?.warn?.({ error, jobId: job?.id, kind: job?.kind }, 'OpenClawBrain background job failed');
  } finally {
    store.close();
  }
}

async function runFeedbackDistillation(packet: any, config: any, store: MemoryStore) {
  const client = llmClientFromConfig(config);
  if (!client) return null;
  const distiller = new FeedbackDistiller({ client, config });
  const result = await distiller.distill(packet);
  const applied = new MemoryOperationApplier({ store, config }).applyDistillation(result.output, packet);
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
    outputJson: JSON.stringify(result.output),
    validationStatus: result.audit.validationStatus,
    validationError: result.audit.validationError || result.audit.parseError,
    latencyMs: result.audit.latencyMs,
  });
  return applied;
}

function llmClientFromConfig(config: any) {
  if (config.llm?.enabled !== true) return null;
  if (config.llm.baseUrl) return new OpenAICompatibleLlmClient({ baseUrl: config.llm.baseUrl });
  return null;
}

function retrieveCandidates(store: MemoryStore, agentId: string, queries: string[], memoryTypes: string[], maxCandidates: number) {
  const seen = new Set<string>();
  const results: any[] = [];
  for (const query of queries) {
    const hits = store.searchMemories(query, agentId, { limit: maxCandidates });
    for (const hit of hits) {
      if (seen.has(hit.id)) continue;
      seen.add(hit.id);
      results.push(hit);
      if (results.length >= maxCandidates) return results;
    }
  }
  for (const memoryType of memoryTypes) {
    const hits = store.listMemories(agentId, { type: memoryType as any, limit: maxCandidates });
    for (const hit of hits) {
      if (seen.has(hit.id)) continue;
      seen.add(hit.id);
      results.push(hit);
      if (results.length >= maxCandidates) return results;
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
    audit: { promptBudgetUsedChars: 0, risk: 'low' as const },
  };
}

function estimateTaskValue(message: string): 'low' | 'medium' | 'high' {
  const lower = message.toLowerCase();
  if (/\b(architecture|implementation|debug|production|repo|build|design|plan)\b/.test(lower)) return 'high';
  if (/\b(install|dependency|test|setup|continue)\b/.test(lower)) return 'medium';
  return 'low';
}

function safeRegisterHook(api: any, name: string, handler: any) {
  try {
    api.on?.(name, handler);
  } catch (error: any) {
    api.logger?.debug?.({ hook: name, error }, 'OpenClawBrain hook unavailable; skipping');
  }
}

function safeRegisterOptionalHook(api: any, name: string, handler: any) {
  if (typeof api.supportsHook !== 'function' || api.supportsHook(name) !== true) return;
  safeRegisterHook(api, name, handler);
}

function writeJson(res: any, payload: any) {
  if (!res) return payload;
  res.statusCode = 200;
  res.setHeader?.('content-type', 'application/json');
  res.end?.(JSON.stringify(payload));
  return true;
}

function limitFromRequest(req: any = {}) {
  if (req.query?.limit) return req.query.limit;
  try {
    const url = new URL(req.url || 'http://local/plugins/openclawbrain/proof', 'http://local');
    return url.searchParams.get('limit') || 20;
  } catch {
    return 20;
  }
}

function queryFromRequest(req: any = {}) {
  if (req.query?.query) return String(req.query.query);
  try {
    const url = new URL(req.url || 'http://local/plugins/openclawbrain/search', 'http://local');
    return url.searchParams.get('query') || '';
  } catch {
    return '';
  }
}

function agentIdFromRequest(req: any = {}, config: any = {}) {
  if (req.query?.agentId) return safeString(req.query.agentId) || 'main';
  if (req.query?.agent) return safeString(req.query.agent) || 'main';
  try {
    const url = new URL(req.url || 'http://local/plugins/openclawbrain/status', 'http://local');
    return safeString(url.searchParams.get('agentId') || url.searchParams.get('agent') || config.scopes?.agents?.[0] || 'main') || 'main';
  } catch {
    return safeString(config.scopes?.agents?.[0] || 'main') || 'main';
  }
}

function agentIdFromEvent(event: any = {}) {
  return safeString(event.ctx?.agentId ?? event.agentId ?? event.agent_id ?? event.profileId ?? event.profile_id ?? event.session?.agentId ?? 'main') || 'main';
}
