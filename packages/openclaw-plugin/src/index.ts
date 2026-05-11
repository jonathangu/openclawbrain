import { DEFAULT_CONFIG, PLUGIN_ID, PLUGIN_VERSION, activationRootForAgent, isAgentAllowed, normalizePluginConfig, resolveOpenClawBrainConfig } from './config.js';
import { buildInjectionText, ensureActivationRoot, readActivationContext } from './context-files.js';
import { CaptureOrchestrator } from './capture.js';
import {
  buildCodexBridgeStatus,
  buildCodexHandoff,
  CodexBridgeStore,
  formatCodexStatus,
  formatCodexThreads,
  formatHandoffBrief,
  handleBrainCommand,
  normalizeCodexBridgeConfig,
  processCodexBridgeWatches,
} from './codex-continuity.js';
import { ContextSelector } from './context-selector.js';
import { FeedbackDistiller } from './feedback-distiller.js';
import { BackgroundLearner } from './learning.js';
import { JobQueue } from './job-queue.js';
import { LatencyController } from './latency-controller.js';
import { FakeLlmClient, OllamaNativeLlmClient, OpenAICompatibleLlmClient, isOllamaLoopbackBaseUrl } from './llm-client.js';
import { MemoryPlanner } from './memory-planner.js';
import { MemoryOperationApplier } from './memory-operations.js';
import { GraphMaintenanceEngine, graphMaintenancePayload } from './graph-maintenance.js';
import { authorityEventTypeForDecision } from './memory-authority.js';
import { MemoryStore } from './memory-store.js';
import { decidePolicy } from './policy.js';
import { appendProofEvent, readProofEvents, readStatus, writeStatus } from './proof-store.js';
import { RouteLearning } from './route-learning.js';
import { RouteTeacher, buildRouteGraphSnapshot } from './route-teacher.js';
import { auditPayload, buildMemoryCorpusSupplement, buildMemoryPromptSupplement, explainLastPayload, extractMemoryId, graphPayload, learnPayload, memoryPath, renderMemory, searchPayload } from './search.js';
import { buildStatus } from './status.js';
import { nativeSqliteSmokeTest } from './native-sqlite.js';
import { clipText, eventId, hashText, latestUserTextFromEvent, redactJsonValue, redactText, safeString, shortHash } from './redact.js';
import { RouteFn } from './route-fn.js';
import { maybeDistillAndStorePolicyV2, scorePolicySnapshotV2, validatePolicySnapshotV2 } from './route-policy-v2.js';
import { maybeDistillAndStorePolicyV3, scorePolicySnapshotV3, validatePolicySnapshotV3 } from './route-policy-v3.js';
import { detectRoutingModeV3 } from './route-policy-v3-routing-mode.js';
import { detectCaptureIntent, detectRetrievalIntent } from './capture-intent.js';
import { defaultScopeContext, memoryInScope, scopeContextFromPacket } from './scope.js';

export { normalizePluginConfig, resolveOpenClawBrainConfig } from './config.js';
export { redactText, hashText } from './redact.js';
export { decidePolicy, classifyTurn } from './policy.js';
export { readActivationContext } from './context-files.js';
export {
  buildCodexBridgeStatus,
  buildCodexHandoff,
  CodexBridgeStore,
  formatCodexStatus,
  formatCodexThreads,
  formatHandoffBrief,
  handleBrainCommand,
  normalizeCodexBridgeConfig,
  processCodexBridgeWatches,
} from './codex-continuity.js';
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

export const openClawBrainPluginEntry = {
  id: PLUGIN_ID,
  name: 'OpenClawBrain',
  version: PLUGIN_VERSION,
  kind: 'memory',
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
  const initialPlan = suppressSyntheticCapture(routeFn.plan(packet), packet);
  if (!shouldAllowCapture(config)) initialPlan.enqueueCapture = false;
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
  } else if (plan.enqueueCapture && shouldAllowCapture(config)) {
    queue.enqueueFeedbackDistillation(agentId, { packet, captureIntent: plan.captureIntent, retrievalIntent: plan.retrievalIntent }, { priority: 10 });
  }

  const routeDecision = store.insertRouteDecision({
    agentId: packet.agentId,
    routeFrameId: store.insertRouteFrame({
      agentId: packet.agentId,
      sessionKeyHash: hashText(packet.sessionKey || packet.sessionId || ''),
      turnHash: hashText(packet.latestUserMessageRedacted || ''),
      redactedTurnSummary: redactText(plan.turnFrame.summary || packet.latestUserMessageRedacted, 500),
      taskType: plan.turnFrame.taskType,
      turnSignals: extractRouteSignals(plan.turnFrame, packet.latestUserMessageRedacted),
      intentSignals: [plan.retrievalIntent.intent, plan.captureIntent.intent].filter(Boolean),
      safetySignals: [],
      projectHint: plan.turnFrame.routeHints.likelyNeedsProjectContext ? 'project_context' : undefined,
      repoHint: plan.turnFrame.activeObjects.find((object: any) => object.kind === 'repo')?.value,
      latencyBudgetMs: latency.maxSyncMs || 0,
    }).id,
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
    policyRuleId: plan.matchedPolicyRuleId,
    routingMode: plan.routingMode,
    rawPolicyScore: plan.rawPolicyScore,
    calibratedPolicyScore: plan.calibratedPolicyScore,
    policyThreshold: plan.policyThreshold,
    abstained: plan.abstained,
    fallbackSource: plan.fallbackSource,
    candidateCount: initialCandidates.length,
    reasonCode: plan.reasonCode || plan.latencyReason,
    injectionPayloadHash: selection.shouldInject ? hashText(selection.distilledContext) : undefined,
    reward: 0,
  });

  recordRouteAuthorityEvents(store, packet, routeDecision.id, selection);

  recordRouteShadowDecisionsV3(store, packet.agentId, routeDecision.id, plan.turnFrame, packet.latestUserMessageRedacted, plan.policySnapshotId, config);

  buildRouteGraphSnapshot(
    store,
    packet.agentId,
    routeDecision.id,
    plan.retrievalPlan.queries,
    initialCandidates,
    plan.retrievalPlan.graphDepth,
  );

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
      omittedMemoryIds: selection.omitted.map((it) => it.memoryId),
      authority: selection.audit.authority || [],
      policySnapshotId: routeDecision.policySnapshotId || null,
      policyRuleId: routeDecision.policyRuleId || null,
      candidateCount: initialCandidates.length,
      reasonCode: routeDecision.reasonCode || null,
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
  safeRegisterHook(api, 'before_prompt_build', async (event = {}, ctx = {}) => handleTurnHook(withHookContext(event, ctx), resolve(), api, 'before_prompt_build'));
  safeRegisterOptionalHook(api, 'agent_turn_prepare', async (event = {}, ctx = {}) => handleTurnHook(withHookContext(event, ctx), resolve(), api, 'agent_turn_prepare'));
}

function registerLifecycleHooks(api: any, resolve: any) {
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

function registerFirstClassSurfaces(api: any, resolve: any) {
  const serviceState: { timer?: NodeJS.Timeout; codexTimer?: NodeJS.Timeout } = {};
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
      if (config.codexBridge?.enabled === true) {
        serviceState.codexTimer = setInterval(() => {
          void processCodexBridgeWatches(resolve(), api).catch((error: any) => {
            api.logger?.warn?.({ error }, 'OpenClawBrain Codex bridge watch processing failed');
          });
        }, config.codexBridge.watchPollIntervalMs);
      }
    },
    stop: async () => {
      if (serviceState.timer) clearInterval(serviceState.timer);
      if (serviceState.codexTimer) clearInterval(serviceState.codexTimer);
      await writeGatewayStatus('service_stop', {}, resolve(), api);
    }
  });
  api.registerCommand?.({
    name: 'brain',
    description: 'OpenClawBrain memory and Codex continuity commands',
    acceptsArgs: true,
    requireAuth: true,
    handler: async (ctx: any) => handleBrainCommand(ctx, resolve(), api),
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
    path: '/plugins/openclawbrain/codex/status',
    auth: 'gateway',
    match: 'exact',
    replaceExisting: true,
    handler: async (req: any, res: any) => writeJson(res, await codexStatusPayload(resolve(), req))
  });
  api.registerHttpRoute?.({
    path: '/plugins/openclawbrain/codex/threads',
    auth: 'gateway',
    match: 'exact',
    replaceExisting: true,
    handler: async (req: any, res: any) => writeJson(res, await codexThreadsPayload(resolve(), req))
  });
  api.registerHttpRoute?.({
    path: '/plugins/openclawbrain/codex/handoff',
    auth: 'gateway',
    match: 'exact',
    replaceExisting: true,
    handler: async (req: any, res: any) => writeJson(res, await codexHandoffPayload(resolve(), req))
  });
  api.registerHttpRoute?.({
    path: '/plugins/openclawbrain/codex/watches',
    auth: 'gateway',
    match: 'exact',
    replaceExisting: true,
    handler: async (req: any, res: any) => writeJson(res, codexWatchesPayload(resolve(), req))
  });
  api.registerHttpRoute?.({
    path: '/plugins/openclawbrain/proof',
    auth: 'gateway',
    match: 'prefix',
    replaceExisting: true,
    handler: async (req: any, res: any) => writeJson(res, await proofPayload(resolve(), limitFromRequest(req)))
  });
  for (const [path, action] of [
    ['/plugins/openclawbrain/graph/health', 'health'],
    ['/plugins/openclawbrain/graph/dry-run', 'dry-run'],
    ['/plugins/openclawbrain/graph/proposals', 'proposals'],
    ['/plugins/openclawbrain/graph/apply', 'apply'],
    ['/plugins/openclawbrain/graph/reject', 'reject'],
    ['/plugins/openclawbrain/graph/stale', 'stale'],
    ['/plugins/openclawbrain/graph/clusters', 'clusters'],
    ['/plugins/openclawbrain/graph/tombstones', 'tombstones'],
    ['/plugins/openclawbrain/graph/explain', 'explain'],
  ] as Array<[string, string]>) {
    api.registerHttpRoute?.({
      path,
      auth: 'gateway',
      match: 'exact',
      replaceExisting: true,
      handler: async (req: any, res: any) => writeJson(res, graphMaintenancePayload(resolve(), req, action))
    });
  }
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
  api.registerHttpRoute?.({
    path: '/plugins/openclawbrain/audit',
    auth: 'gateway',
    match: 'prefix',
    replaceExisting: true,
    handler: async (req: any, res: any) => writeJson(res, auditPayload(resolve(), agentIdFromRequest(req, resolve()), limitFromRequest(req)))
  });
  api.registerHttpRoute?.({
    path: '/plugins/openclawbrain/explain-last',
    auth: 'gateway',
    match: 'prefix',
    replaceExisting: true,
    handler: async (req: any, res: any) => writeJson(res, explainLastPayload(resolve(), agentIdFromRequest(req, resolve()), turnIdFromRequest(req)))
  });
  api.registerHttpRoute?.({
    path: '/plugins/openclawbrain/route-teacher',
    auth: 'gateway',
    match: 'prefix',
    replaceExisting: true,
    handler: async (req: any, res: any) => writeJson(res, routeTeacherPayload(resolve(), agentIdFromRequest(req, resolve()), limitFromRequest(req)))
  });
  api.registerHttpRoute?.({
    path: '/plugins/openclawbrain/route-counterfactuals',
    auth: 'gateway',
    match: 'prefix',
    replaceExisting: true,
    handler: async (req: any, res: any) => writeJson(res, routeCounterfactualPayload(resolve(), agentIdFromRequest(req, resolve()), decisionIdFromRequest(req), limitFromRequest(req)))
  });
  api.registerHttpRoute?.({
    path: '/plugins/openclawbrain/route-policy',
    auth: 'gateway',
    match: 'prefix',
    replaceExisting: true,
    handler: async (req: any, res: any) => writeJson(res, routePolicyPayload(resolve(), agentIdFromRequest(req, resolve()), limitFromRequest(req)))
  });
  try {
    api.registerMemoryCapability?.(buildMemoryCapability(resolve));
    api.registerMemoryPromptSupplement?.(buildMemoryPromptSupplement());
    api.registerMemoryCorpusSupplement?.(buildMemoryCorpusSupplement(resolve()));
  } catch (error: any) {
    api.logger?.debug?.({ error }, 'OpenClawBrain memory capability/supplements unavailable; skipping');
  }
}

export function buildMemoryCapability(resolve: any) {
  return {
    promptBuilder: buildMemoryPromptSupplement(),
    runtime: {
      async getMemorySearchManager({ agentId }: { cfg?: any; agentId: string; purpose?: string }) {
        const config = resolve();
        const normalizedAgentId = safeString(agentId || config.scopes?.agents?.[0] || 'main') || 'main';
        if (!isAgentAllowed(config, normalizedAgentId)) {
          return { manager: null, error: `agent not allowed for OpenClawBrain memory: ${normalizedAgentId}` };
        }
        return { manager: createOpenClawBrainMemorySearchManager(config, normalizedAgentId) };
      },
      resolveMemoryBackendConfig() {
        return { backend: 'builtin' as const };
      },
      async closeAllMemorySearchManagers() {
        return undefined;
      },
    },
  };
}

function createOpenClawBrainMemorySearchManager(config: any, agentId: string) {
  const store = new MemoryStore({ activationRoot: config.activationRoot, agentId });
  const renderReadResult = (memory: any, from = 1, lines?: number) => {
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
    async search(query: string, opts: any = {}) {
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
        source: 'memory' as const,
        citation: `${memoryPath(memory)}#L1-L${renderMemory(memory).split(/\n/).length}`,
      }));
    },
    async readFile({ relPath, from, lines }: { relPath: string; from?: number; lines?: number }) {
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
        backend: 'builtin' as const,
        provider: 'openclawbrain',
        files: nodes,
        chunks: nodes,
        dirty: false,
        sources: ['memory' as const],
        sourceCounts: [{ source: 'memory' as const, files: nodes, chunks: nodes }],
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

async function statusPayload(config: any, req: any = {}) {
  const agentId = safeString(req.query?.agentId ?? req.query?.agent ?? config.scopes.agents[0] ?? 'main') || 'main';
  if (!isAgentAllowed(config, agentId)) return { ok: false, agentId, reason: 'agent_not_allowed' };
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
        projectFacts: store.countMemories(agentId, 'project_fact'),
        toolConventions: store.countMemories(agentId, 'tool_convention'),
        routingRules: store.countMemories(agentId, 'routing_rule'),
        recallRules: store.countMemories(agentId, 'recall_rule'),
        captureAuditRows: store.countCaptureAudit(agentId),
        authorityEvents: store.listMemoryAuthorityEvents(agentId, 500).length,
      },
      routing: {
        activePolicySnapshotId: store.getActivePolicySnapshot(agentId)?.id || null,
        routeDecisions: store.countRouteDecisions(agentId),
        pendingOutcomes: store.getUnresolvedRouteDecisions(agentId).length,
        positiveExamples: store.countRouteExamples(agentId, 'positive'),
        negativeExamples: store.countRouteExamples(agentId, 'negative'),
        routeTeacherRuns: store.listRouteTeacherRuns(agentId, 100).length,
        routeTrainingExamplesV2: store.countRouteTrainingExamplesV2(agentId),
        routeFramesV3: store.listRouteFramesV3(agentId, 500).length,
        routeActionPrototypesV3: store.listRouteActionPrototypesV3(agentId, 500).length,
        routePairExamplesV3: store.listRoutePairExamplesV3(agentId, 500).length,
        activePolicySnapshotV2Id: store.getActivePolicySnapshotV2(agentId)?.id || null,
        activePolicySnapshotV3Id: store.getActivePolicySnapshotV3(agentId)?.id || null,
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

async function codexStatusPayload(config: any, req: any = {}): Promise<any> {
  const agentId = agentIdFromRequest(req, config);
  if (!isAgentAllowed(config, agentId)) return { ok: false, agentId, reason: 'agent_not_allowed' };
  return buildCodexBridgeStatus(config, agentId);
}

async function codexThreadsPayload(config: any, req: any = {}) {
  const status = await codexStatusPayload(config, req);
  const query = safeString(req.query?.q ?? req.query?.query ?? '').toLowerCase();
  const threads = Array.isArray(status.latestThreads)
    ? status.latestThreads.filter((thread: any) => !query || `${thread.title} ${thread.cwd} ${thread.goal?.objective || ''}`.toLowerCase().includes(query))
    : [];
  return { ok: status.ok, source: status.source, stale: status.stale, staleReason: status.staleReason, threads };
}

async function codexHandoffPayload(config: any, req: any = {}) {
  const status = await codexStatusPayload(config, req);
  const threadId = safeString(req.query?.threadId ?? req.query?.thread ?? '');
  return buildCodexHandoff(status, threadId || undefined);
}

function codexWatchesPayload(config: any, req: any = {}) {
  const agentId = agentIdFromRequest(req, config);
  if (!isAgentAllowed(config, agentId)) return { ok: false, agentId, reason: 'agent_not_allowed' };
  const store = new CodexBridgeStore({ config, agentId });
  try {
    return {
      ok: true,
      agentId,
      watches: store.listWatches(agentId, { activeOnly: req.query?.active !== 'false' }),
      events: store.listEvents(agentId, limitFromRequest(req)),
    };
  } finally {
    store.close();
  }
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
  if (!shouldAllowCapture(config) || config.hooks.allowConversationAccess !== true) return {};
  const agentId = agentIdFromEvent(event);
  const store = new MemoryStore({ activationRoot: config.activationRoot, agentId });
  const queue = new JobQueue({ store });
  const packet = new CaptureOrchestrator().fromAgentEnd(event, config);
  const captureIntent = suppressSyntheticCaptureIntent(detectCaptureIntent(packet), packet);
  const retrievalIntent = detectRetrievalIntent(packet);
  const learner = new BackgroundLearner({ store, config });

  try {
    learner.processAgentEnd(agentId, packet);
  } catch (error: any) {
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
    } catch (error: any) {
      api.logger?.warn?.({ error }, 'OpenClawBrain best-effort agent_end distillation failed');
    } finally {
      store.close();
    }
    return {};
  }

  queue.enqueueFeedbackDistillation(agentId, { packet, captureIntent, retrievalIntent }, { priority: 10 });
  store.close();
  return {};
}

async function handleAfterToolCall(event: any = {}, config: any = {}, api: any = {}) {
  if (config.learning?.enabled !== true || config.hooks.allowToolObservation !== true) return {};
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

async function processBackgroundJobs(config: any = {}, api: any = {}) {
  if (config.capture?.enabled !== true && config.learning?.enabled !== true && config.routeLearning?.enabled !== true) return;
  const agents = Array.isArray(config.scopes?.agents) && config.scopes.agents.length ? config.scopes.agents : ['main'];
  for (const agentId of agents) await processBackgroundJobsForAgent(config, api, agentId);
}

async function processBackgroundJobsForAgent(config: any = {}, api: any = {}, agentId = 'main') {
  const store = new MemoryStore({ activationRoot: config.activationRoot, agentId });
  const queue = new JobQueue({ store });
  const job = queue.claimNext(undefined, agentId);
  const learner = new BackgroundLearner({ store, config });
  const routeLearning = new RouteLearning({ store, config });
  const routeTeacher = new RouteTeacher({ store, config, client: llmClientFromConfig(config) });
  try {
    let learningReport: any = null;
    if (job?.kind === 'feedback_distillation') {
      await runFeedbackDistillation(job.payload?.packet as any, config, store, {
        captureIntent: job.payload?.captureIntent as any,
        retrievalIntent: job.payload?.retrievalIntent as any,
      });
      queue.enqueueRouteLearning(agentId, { cause: 'feedback_distillation', turnId: (job.payload?.packet as any)?.turnId || '' }, { priority: 4 });
    } else if (job?.kind === 'outcome_classification') {
      learningReport = learner.processOutcomeClassification(agentId, job.payload?.packet as any);
    } else if (job?.kind === 'route_learning') {
      learningReport = { ...routeLearning.run(agentId), lastRunAt: new Date().toISOString() };
      queue.enqueueRouteTeacher(agentId, { cause: 'route_learning', at: new Date().toISOString() }, { priority: 3, delayMs: 0 });
    } else if (job?.kind === 'route_teacher') {
      learningReport = { ...(await routeTeacher.run(agentId)), lastRunAt: new Date().toISOString() };
    } else if (job) {
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
        routeLearning: {
          activePolicySnapshotV2Id: store.getActivePolicySnapshotV2(agentId)?.id || null,
          activePolicySnapshotV3Id: store.getActivePolicySnapshotV3(agentId)?.id || null,
        },
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

async function runFeedbackDistillation(packet: any, config: any, store: MemoryStore, context: any = {}) {
  if (!shouldAllowCapture(config) || shouldSuppressCapture(packet)) return null;
  const client = llmClientFromConfig(config);
  if (!client) return null;
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

function llmClientFromConfig(config: any) {
  if (config.llm?.enabled !== true) return null;
  const models = [config.llm.plannerModel, config.llm.routeModel, config.llm.feedbackModel, config.llm.learningModel].filter(Boolean);
  const allowed = new Set(Array.isArray(config.llm.allowedModels) ? config.llm.allowedModels : []);
  if (allowed.size > 0 && models.some((model) => !allowed.has(model))) return null;
  if (config.llm.baseUrl && !isLoopbackUrl(config.llm.baseUrl) && config.llm.allowRemoteLlm !== true) return null;
  if (config.llm.baseUrl && isOllamaLoopbackBaseUrl(config.llm.baseUrl)) return new OllamaNativeLlmClient({ baseUrl: config.llm.baseUrl });
  if (config.llm.baseUrl) return new OpenAICompatibleLlmClient({ baseUrl: config.llm.baseUrl });
  return null;
}

function retrieveCandidates(store: MemoryStore, packet: any, queries: string[], memoryTypes: string[], maxCandidates: number) {
  const agentId = packet.agentId;
  const scopeContext = scopeContextFromPacket(packet);
  const seen = new Set<string>();
  const results: any[] = [];
  for (const query of queries) {
    const hits = store.searchMemories(query, agentId, { limit: maxCandidates, scopeContext });
    for (const hit of hits) {
      if (seen.has(hit.id)) continue;
      seen.add(hit.id);
      results.push(hit);
      if (results.length >= maxCandidates) return results;
    }
  }
  for (const memoryType of memoryTypes) {
    const hits = store.listMemories(agentId, { type: memoryType as any, limit: maxCandidates, scopeContext });
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
    audit: { promptBudgetUsedChars: 0, risk: 'low' as const, authority: [] },
  };
}

function estimateTaskValue(message: string): 'low' | 'medium' | 'high' {
  const lower = message.toLowerCase();
  if (/\b(architecture|implementation|debug|production|repo|build|design|plan)\b/.test(lower)) return 'high';
  if (/\b(install|dependency|test|setup|continue)\b/.test(lower)) return 'medium';
  return 'low';
}

function withHookContext(event: any = {}, ctx: any = {}) {
  if (!ctx || typeof ctx !== 'object' || Object.keys(ctx).length === 0) return event || {};
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

function suppressSyntheticCapture(plan: any, packet: any) {
  const trigger = String(packet?.metadata?.trigger || '').toLowerCase();
  if (!shouldSuppressCapture(packet)) return plan;
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

function suppressSyntheticCaptureIntent(captureIntent: any, packet: any) {
  if (!shouldSuppressCapture(packet)) return captureIntent;
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

function shouldSuppressCapture(packet: any) {
  const trigger = String(packet?.metadata?.trigger || '').toLowerCase();
  const sourceHook = String(packet?.sourceHook || '').toLowerCase();
  if (['heartbeat', 'cron', 'system', 'subagent'].includes(trigger)) return true;
  if (sourceHook === 'after_tool_call') return true;
  return false;
}

function shouldAllowCapture(config: any) {
  return config.capture?.enabled === true && config.capture?.mode !== 'off' && config.memory?.captureMode !== 'off';
}

function shouldAllowRouting(config: any) {
  return config.routing?.enabled === true && config.routing?.mode !== 'off';
}

function extractRouteSignals(turnFrame: any, text = '') {
  const lower = `${text} ${turnFrame?.summary || ''} ${turnFrame?.userGoal || ''} ${(turnFrame?.impliedNeeds || []).join(' ')}`.toLowerCase();
  const signals = new Set<string>();
  for (const token of ['test', 'build', 'install', 'dependency', 'write', 'draft', 'plan', 'debug', 'thanks', 'ok', 'remember', 'actually', 'instead']) {
    if (lower.includes(token)) signals.add(token);
  }
  if (turnFrame?.routeHints?.likelyNeedsWorkflow) signals.add('workflow');
  if (turnFrame?.routeHints?.likelyNeedsPreferences) signals.add('preference');
  if (turnFrame?.routeHints?.likelyNeedsCorrections) signals.add('correction');
  if (turnFrame?.routeHints?.likelyNeedsProjectContext) signals.add('project_context');
  return [...signals].slice(0, 12);
}

function recordRouteAuthorityEvents(store: MemoryStore, packet: any, routeDecisionId: string, selection: any) {
  const authority = Array.isArray(selection?.audit?.authority) ? selection.audit.authority : [];
  for (const resolution of authority) {
    store.insertMemoryAuthorityEvent({
      agentId: packet.agentId,
      memoryId: resolution.memoryId,
      eventType: authorityEventTypeForDecision(resolution.decision),
      source: 'route_decision',
      turnId: packet.turnId,
      routeId: routeDecisionId,
      evidenceId: String(packet.metadata?.promptHash || ''),
      reason: (resolution.reasons || []).join('; '),
    });
  }
}

function isLoopbackUrl(value: string) {
  try {
    const url = new URL(value);
    if (!['http:', 'https:'].includes(url.protocol)) return false;
    const host = url.hostname.replace(/\.$/, '').toLowerCase();
    if (host === 'localhost' || host === '::1' || host === '[::1]') return true;
    if (/^127(?:\.\d{1,3}){3}$/.test(host)) return true;
    if (host === '::ffff:127.0.0.1' || host === '[::ffff:127.0.0.1]') return true;
    return false;
  } catch {
    return false;
  }
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

function decisionIdFromRequest(req: any = {}) {
  if (req.query?.decisionId) return safeString(req.query.decisionId);
  if (req.query?.routeDecisionId) return safeString(req.query.routeDecisionId);
  try {
    const url = new URL(req.url || 'http://local/plugins/openclawbrain/route-counterfactuals', 'http://local');
    return safeString(url.searchParams.get('decisionId') || url.searchParams.get('routeDecisionId') || '');
  } catch {
    return '';
  }
}

function routeTeacherPayload(config: any = {}, agentId = 'main', limit = 20) {
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
  } finally {
    store.close();
  }
}

function routeCounterfactualPayload(config: any = {}, agentId = 'main', decisionId = '', limit = 50) {
  const store = new MemoryStore({ activationRoot: config.activationRoot, agentId });
  try {
    const counterfactuals = store.listRouteCounterfactuals(agentId, decisionId || undefined, Number(limit) || 50);
    return { ok: true, agentId, routeDecisionId: decisionId || null, count: counterfactuals.length, counterfactuals };
  } finally {
    store.close();
  }
}

function routePolicyPayload(config: any = {}, agentId = 'main', limit = 20) {
  const store = new MemoryStore({ activationRoot: config.activationRoot, agentId });
  try {
    const active = store.getActivePolicySnapshotV2(agentId);
    const activeV3 = store.getActivePolicySnapshotV3(agentId);
    const snapshots = store.listPolicySnapshotsV2(agentId, Number(limit) || 20);
    const snapshotsV3 = store.listPolicySnapshotsV3(agentId, Number(limit) || 20);
    const examples = store.listRouteTrainingExamplesV2(agentId, 50);
    const framesV3 = store.listRouteFramesV3(agentId, 50);
    const prototypesV3 = store.listRouteActionPrototypesV3(agentId, 50);
    const shadowDecisionsV3 = store.listRouteShadowDecisionsV3(agentId, 50);
    const calibrationExamplesV3 = store.listRouteCalibrationExamplesV3(agentId, 50);
    const evalCasesV3 = store.listRouteEvalCasesV3?.(agentId, 50) || [];
    const evalCaseLabelsV3 = store.listRouteEvalCaseLabelsV3?.(agentId, 100) || [];
    const candidateReportsV3 = store.listRoutePolicyCandidateReportsV3(agentId, 20);
    const familyStatsV3 = store.listRouteActionFamilyStatsV3(agentId, 20);
    return {
      ok: true,
      agentId,
      active,
      activeV3,
      snapshotCount: snapshots.length,
      snapshotCountV3: snapshotsV3.length,
      snapshots: snapshots.map((snapshot) => ({
        id: snapshot.id,
        version: snapshot.version,
        status: snapshot.status,
        ruleCount: snapshot.rules.length,
        globalBudgets: snapshot.globalBudgets,
        evalSummary: snapshot.evalSummary,
        createdAt: snapshot.createdAt,
      })),
      snapshotsV3: snapshotsV3.map((snapshot) => ({
        id: snapshot.id,
        version: snapshot.version,
        status: snapshot.status,
        ruleCount: snapshot.rules.length,
        actionPriorCount: Object.keys(snapshot.actionPriors || {}).length,
        globalBudgets: snapshot.globalBudgets,
        evalSummary: snapshot.evalSummary,
        createdAt: snapshot.createdAt,
      })),
      exampleCount: examples.length,
      examples: examples.slice(0, 20),
      routeFramesV3Count: framesV3.length,
      routeFramesV3: framesV3.slice(0, 20),
      routeActionPrototypeCountV3: prototypesV3.length,
      routeActionPrototypesV3: prototypesV3.slice(0, 20),
      routeShadowDecisionCountV3: shadowDecisionsV3.length,
      routeShadowDecisionsV3: shadowDecisionsV3.slice(0, 20),
      routeCalibrationExampleCountV3: calibrationExamplesV3.length,
      routeCalibrationExamplesV3: calibrationExamplesV3.slice(0, 20),
      routeEvalCaseCountV3: evalCasesV3.length,
      routeEvalCasesV3: evalCasesV3.slice(0, 20),
      routeEvalCaseLabelCountV3: evalCaseLabelsV3.length,
      routeEvalCaseLabelsV3: evalCaseLabelsV3.slice(0, 30),
      routePolicyCandidateReportCountV3: candidateReportsV3.length,
      routePolicyCandidateReportsV3: candidateReportsV3.slice(0, 10),
      routeActionFamilyStatsCountV3: familyStatsV3.length,
      routeActionFamilyStatsV3: familyStatsV3.slice(0, 10),
    };
  } finally {
    store.close();
  }
}

function recordRouteShadowDecisionsV3(store: any, agentId: string, routeDecisionId: string, turnFrame: any, message: string, controllingSnapshotId: string | undefined, config: any) {
  if (config.routeLearning?.policyV3?.enabled === false || config.routeLearning?.policyV3?.storeShadowDecisions === false) return;
  const limit = Math.max(0, Number(config.routeLearning?.policyV3?.maxShadowSnapshots ?? 3));
  if (limit === 0) return;
  const snapshots = (store.listPolicySnapshotsV3(agentId, Math.max(10, limit * 4)) || [])
    .filter((snapshot: any) => snapshot?.version === 'route-policy-v3')
    .filter((snapshot: any) => snapshot.id !== controllingSnapshotId)
    .filter((snapshot: any) => snapshot.status === 'shadow')
    .slice(0, limit);
  for (const snapshot of snapshots) {
    const match = scorePolicySnapshotV3(snapshot, turnFrame, message, { requireActive: false });
    store.insertRouteShadowDecisionV3({
      agentId,
      routeDecisionId,
      snapshotId: snapshot.id,
      snapshotStatus: snapshot.status,
      proposedRoute: match.rule?.route || 'no_memory',
      proposedActionId: match.rule?.actionId,
      proposedRuleId: match.rule?.id,
      rawScore: Number(match.rawScore || 0),
      calibratedScore: Number(match.calibratedScore || match.score || 0),
      threshold: Number(match.threshold || 0),
      abstained: Boolean(match.abstained || !match.matched),
      routingMode: detectRoutingModeV3(turnFrame, message),
      reasonCode: match.reasonCode || 'shadow_policy_v3',
    });
  }
}

function queryFromRequest(req: any = {}) {
  if (req.query?.query) return String(req.query.query);
  if (req.query?.q) return String(req.query.q);
  try {
    const url = new URL(req.url || 'http://local/plugins/openclawbrain/search', 'http://local');
    return url.searchParams.get('query') || url.searchParams.get('q') || '';
  } catch {
    return '';
  }
}

function turnIdFromRequest(req: any = {}) {
  if (req.query?.turnId) return safeString(req.query.turnId);
  if (req.query?.turn) return safeString(req.query.turn);
  try {
    const url = new URL(req.url || 'http://local/plugins/openclawbrain/explain-last', 'http://local');
    return safeString(url.searchParams.get('turnId') || url.searchParams.get('turn') || '');
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
