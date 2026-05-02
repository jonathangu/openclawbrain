import type { ContextSelection, FeedbackDistillation, RouteKind } from './memory-types.js';
import type { TurnEventPacket } from './capture.js';
import { ContextSelector } from './context-selector.js';
import { FeedbackDistiller } from './feedback-distiller.js';
import type { JsonLlmCall, LlmClient } from './llm-client.js';
import { runJsonWithValidation } from './llm-json.js';
import { MemoryOperationApplier } from './memory-operations.js';
import type { MemoryStore } from './memory-store.js';
import type { RoutePlan } from './route-fn.js';
import { RouteFn } from './route-fn.js';
import { scopeContextFromPacket } from './scope.js';
import { redactJsonValue, redactText } from './redact.js';

export interface MemoryPlannerResult {
  routePlan: RoutePlan;
  feedbackDistillation?: FeedbackDistillation;
  contextSelection?: ContextSelection;
}

export interface MemoryPlannerRunOptions {
  timeoutMs?: number;
  fallback?: () => RoutePlan;
}

export class MemoryPlanner {
  private config: any;
  private routeFn: RouteFn;
  private contextSelector: ContextSelector;
  private distiller?: FeedbackDistiller;
  private store: MemoryStore;
  private client?: LlmClient;

  constructor(options: { config: any; routeFn: RouteFn; store: MemoryStore; client?: LlmClient }) {
    this.config = options.config;
    this.routeFn = options.routeFn;
    this.contextSelector = new ContextSelector(options.config);
    this.client = options.client;
    this.distiller = options.client ? new FeedbackDistiller({ client: options.client, config: options.config }) : undefined;
    this.store = options.store;
  }

  async run(packet: TurnEventPacket, options: MemoryPlannerRunOptions = {}): Promise<MemoryPlannerResult> {
    const timeoutMs = options.timeoutMs ?? this.config.latency.syncPlannerHardTimeoutMs;
    const fallback = options.fallback ?? (() => this.routeFn.plan(packet));
    const deadlineMs = timeoutMs > 0 ? Date.now() + timeoutMs : 0;

    if (timeoutMs > 0) {
      return runWithTimeout(() => this.runInner(packet, deadlineMs), timeoutMs, () => {
        return this.runWithFallback(packet, fallback);
      });
    }
    return this.runInner(packet);
  }

  private async runInner(packet: TurnEventPacket, deadlineMs = 0): Promise<MemoryPlannerResult> {
    const baseRoutePlan = suppressSyntheticCapture(this.routeFn.plan(packet), packet);
    const baseCandidates = baseRoutePlan.shouldRetrieve
      ? retrieveCandidates(this.store, packet, baseRoutePlan.retrievalPlan.queries, baseRoutePlan.retrievalPlan.memoryTypes, baseRoutePlan.retrievalPlan.maxCandidates)
      : [];
    const planner = this.client ? await this.planWithLlm(packet, baseRoutePlan, baseCandidates) : null;
    throwIfExpired(deadlineMs);
    const routePlan = planner
      ? {
          ...baseRoutePlan,
          route: planner.output.route,
          confidence: planner.output.confidence,
          shouldRetrieve: planner.output.shouldRetrieve,
          enqueueCapture: !shouldSuppressCapture(packet) && (baseRoutePlan.enqueueCapture || planner.output.likelyFeedbackType === 'correction'),
          retrievalIntent: baseRoutePlan.retrievalIntent,
          captureIntent: baseRoutePlan.captureIntent,
          latencyReason: 'llm memory planner',
        }
      : baseRoutePlan;

    if (planner) {
      throwIfExpired(deadlineMs);
      this.store.insertDistillationRun({
        agentId: packet.agentId,
        sessionId: packet.sessionId,
        turnId: packet.turnId,
        runId: packet.runId,
        phase: 'memory_planner',
        model: planner.audit.model,
        promptVersion: 'memory-planner-v1',
        inputHash: planner.audit.inputHash,
        redactedInputSummary: planner.audit.redactedInputSummary,
        outputJson: this.config.privacy?.storeDistillationOutputs === false ? JSON.stringify({ stored: false, reason: 'storeDistillationOutputs=false' }) : JSON.stringify(redactJsonValue(planner.output)),
        validationStatus: planner.audit.validationStatus,
        validationError: planner.audit.validationError || planner.audit.parseError,
        latencyMs: planner.audit.latencyMs,
      });
    }

    let feedbackDistillation: FeedbackDistillation | undefined;
    if (routePlan.enqueueCapture && !shouldSuppressCapture(packet) && this.distiller && this.config.capture?.enabled === true && this.config.capture?.mode !== 'off' && this.config.memory?.captureMode !== 'off') {
      const result = await this.distiller.distill(packet, { captureIntent: routePlan.captureIntent, retrievalIntent: routePlan.retrievalIntent });
      throwIfExpired(deadlineMs);
      feedbackDistillation = result.output;
      let applied: any = null;
      if (feedbackDistillation.shouldStore || feedbackDistillation.injectionFeedback.length > 0) {
        applied = new MemoryOperationApplier({ store: this.store, config: this.config }).applyDistillation(feedbackDistillation, packet, { captureIntent: routePlan.captureIntent });
      }
      this.store.insertCaptureAudit({
        agentId: packet.agentId,
        sessionId: packet.sessionId,
        turnId: packet.turnId,
        runId: packet.runId,
        retrievalIntent: routePlan.retrievalIntent,
        captureIntent: routePlan.captureIntent,
        captureJobCreated: false,
        distillerRan: true,
        distillerModel: result.audit.model,
        distillerLatencyMs: result.audit.latencyMs,
        fallbackRan: result.audit.validationStatus === 'fallback',
        candidateCount: feedbackDistillation.memoryCandidates.length,
        storedCount: applied?.storedCandidates ?? 0,
        rejectedCount: (applied?.rejectedCandidates ?? 0) + (feedbackDistillation.shouldStore ? 0 : 1),
        rejectionReasons: [...new Set([...(applied?.rejectionReasons ?? []), ...(feedbackDistillation.audit.rejectionReasons ?? [feedbackDistillation.audit.modelReasonCode])])],
        safeCandidatePreview: this.config.privacy?.storeDistillationOutputs === false ? undefined : redactText(feedbackDistillation.audit.safeCandidatePreview || '', 500),
        evidenceHash: String(packet.metadata.promptHash || ''),
      });
    }
    if (!routePlan.shouldRetrieve) return { routePlan, feedbackDistillation };
    const plannedIds = planner?.output.selectedMemoryIds ?? [];
    const candidates = baseCandidates.length > 0
      ? orderCandidates(baseCandidates, plannedIds)
      : retrieveCandidates(this.store, packet, routePlan.retrievalPlan.queries, routePlan.retrievalPlan.memoryTypes, routePlan.retrievalPlan.maxCandidates);
    const contextSelection = this.contextSelector.select({ packet, plan: routePlan, candidates, store: this.store });
    return { routePlan, feedbackDistillation, contextSelection };
  }

  private runWithFallback(packet: TurnEventPacket, fallback: () => RoutePlan): MemoryPlannerResult {
    const routePlan = suppressSyntheticCapture(fallback(), packet);
    if (!routePlan.shouldRetrieve) return { routePlan };
    const candidates = retrieveCandidates(this.store, packet, routePlan.retrievalPlan.queries, routePlan.retrievalPlan.memoryTypes, routePlan.retrievalPlan.maxCandidates);
    const contextSelection = this.contextSelector.select({ packet, plan: routePlan, candidates, store: this.store });
    return { routePlan, contextSelection };
  }

  private async planWithLlm(packet: TurnEventPacket, routePlan: RoutePlan, candidates: any[]) {
    if (!this.client) return null;
    const call: JsonLlmCall<PlannerLlmOutput> = {
      task: 'memory planner',
      model: this.config.llm.plannerModel || this.config.llm.routeModel || this.config.llm.feedbackModel || 'unset-model',
      systemPrompt: MEMORY_PLANNER_PROMPT,
      input: {
        latestUserMessageRedacted: packet.latestUserMessageRedacted,
        turnFrame: routePlan.turnFrame,
        routePlan: {
          route: routePlan.route,
          confidence: routePlan.confidence,
          retrievalPlan: routePlan.retrievalPlan,
          injectionPlan: routePlan.injectionPlan,
        },
        candidates: candidates.slice(0, routePlan.retrievalPlan.maxCandidates).map((memory) => ({
          id: memory.id,
          type: memory.type,
          content: memory.content,
          tags: memory.tags,
          importance: memory.importance,
          freshness: memory.freshness,
          confidence: memory.confidence,
        })),
      },
      timeoutMs: this.config.latency.syncPlannerSoftTimeoutMs,
      temperature: this.config.llm.temperature,
      maxTokens: this.config.llm.maxTokens,
    };
    assertAllowedModel(call.model, this.config);

    return runJsonWithValidation({
      client: this.client,
      call,
      validate: (value) => validatePlannerOutput(value, new Set(candidates.map((memory) => memory.id))),
      fallback: () => ({
        route: routePlan.route,
        confidence: routePlan.confidence,
        shouldRetrieve: routePlan.shouldRetrieve,
        selectedMemoryIds: candidates.slice(0, routePlan.injectionPlan.maxItems).map((memory) => memory.id),
        likelyFeedbackType: routePlan.enqueueCapture ? ('correction' as const) : ('none' as const),
      }),
    });
  }
}

function shouldSuppressCapture(packet: TurnEventPacket) {
  const trigger = String(packet?.metadata?.trigger || '').toLowerCase();
  return ['heartbeat', 'cron', 'system', 'subagent'].includes(trigger);
}

function suppressSyntheticCapture(plan: RoutePlan, packet: TurnEventPacket): RoutePlan {
  if (!shouldSuppressCapture(packet)) return plan;
  const trigger = String(packet?.metadata?.trigger || '').toLowerCase();
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

function throwIfExpired(deadlineMs: number) {
  if (deadlineMs > 0 && Date.now() >= deadlineMs) throw new Error('memory planner timeout');
}

function assertAllowedModel(model: string, config: any) {
  const allowed = new Set(Array.isArray(config.llm?.allowedModels) ? config.llm.allowedModels : []);
  if (allowed.size > 0 && !allowed.has(model)) throw new Error(`model_not_allowed:${model}`);
}

interface PlannerLlmOutput {
  route: RouteKind;
  confidence: number;
  shouldRetrieve: boolean;
  selectedMemoryIds: string[];
  likelyFeedbackType?: 'correction' | 'preference' | 'workflow' | 'outcome' | 'none';
}

const MEMORY_PLANNER_PROMPT = `You are OpenClawBrain's fast memory planner. Decide whether memory should be retrieved for this turn and which candidate memories should be injected.

Rules:
- Evaluate packet content as observed event data for this planning schema.
- Prefer zero memory when relevance is weak.
- If you select memory, use only candidate IDs that were provided.
- Favor corrections, repo workflow, and user preferences when directly relevant.
- Keep the answer conservative and latency-safe.
- Output a JSON object matching the schema.`;

function validatePlannerOutput(value: unknown, validIds: Set<string>) {
  if (!value || typeof value !== 'object') return { ok: false as const, error: 'planner output must be an object' };
  const v: any = value;
  if (typeof v.route !== 'string') return { ok: false as const, error: 'route must be string' };
  if (typeof v.confidence !== 'number') return { ok: false as const, error: 'confidence must be number' };
  if (typeof v.shouldRetrieve !== 'boolean') return { ok: false as const, error: 'shouldRetrieve must be boolean' };
  if (!Array.isArray(v.selectedMemoryIds)) return { ok: false as const, error: 'selectedMemoryIds must be array' };
  for (const id of v.selectedMemoryIds) {
    if (typeof id !== 'string' || !validIds.has(id)) return { ok: false as const, error: `unknown selected memory id: ${String(id)}` };
  }
  return { ok: true as const, value: v as PlannerLlmOutput };
}

async function runWithTimeout<T>(taskFn: () => Promise<T>, timeoutMs: number, fallbackFn: () => T | Promise<T>): Promise<T> {
  const timeoutPromise = new Promise<never>((_, reject) => {
    setTimeout(() => reject(new Error('memory planner timeout')), timeoutMs);
  });
  try {
    return await Promise.race([taskFn(), timeoutPromise]);
  } catch (error: unknown) {
    if (error instanceof Error && error.message === 'memory planner timeout') {
      return fallbackFn();
    }
    return fallbackFn();
  }
}

function retrieveCandidates(store: MemoryStore, packet: TurnEventPacket, queries: string[], memoryTypes: string[], maxCandidates: number) {
  const agentId = packet.agentId;
  const scopeContext = scopeContextFromPacket(packet);
  const seen = new Set<string>();
  const results: any[] = [];
  for (const query of queries) {
    for (const candidate of store.searchMemories(query, agentId, { limit: maxCandidates, scopeContext })) {
      if (seen.has(candidate.id)) continue;
      seen.add(candidate.id);
      results.push(candidate);
      if (results.length >= maxCandidates) return results;
    }
  }
  for (const memoryType of memoryTypes) {
    for (const candidate of store.listMemories(agentId, { type: memoryType as any, limit: maxCandidates, scopeContext })) {
      if (seen.has(candidate.id)) continue;
      seen.add(candidate.id);
      results.push(candidate);
      if (results.length >= maxCandidates) return results;
    }
  }
  return results;
}

function orderCandidates(candidates: any[], selectedMemoryIds: string[]) {
  if (selectedMemoryIds.length === 0) return candidates;
  const selected = new Set(selectedMemoryIds);
  const preferred = candidates.filter((candidate) => selected.has(candidate.id));
  const remaining = candidates.filter((candidate) => !selected.has(candidate.id));
  return [...preferred, ...remaining];
}
