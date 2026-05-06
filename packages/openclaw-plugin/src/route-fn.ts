import type { InjectionPlan, MemoryType, RetrievalPlan, RouteDecision, RouteKind, TurnFrame } from './memory-types.js';
import type { TurnEventPacket } from './capture.js';
import { hashText } from './redact.js';
import { detectCaptureIntent, detectRetrievalIntent, type CaptureIntentResult, type RetrievalIntentResult } from './capture-intent.js';
import { scorePolicySnapshotV2 } from './route-policy-v2.js';
import { scorePolicySnapshotV3 } from './route-policy-v3.js';

export interface RouteFingerprint {
  agentId: string;
  scopeKey?: string;
  taskTypeHint?: string;
  topicKeys: string[];
  explicitMemoryReference: boolean;
  explicitCorrectionCue: boolean;
  captureIntent?: string;
  retrievalIntent?: string;
}

export interface CachedRoutePlan {
  route: RouteKind;
  retrievalPlan: RetrievalPlan;
  injectionPlan: InjectionPlan;
  confidence: number;
  expiresAt: string;
  sourceRouteDecisionId?: string;
  policySnapshotId?: string;
  matchedPolicyRuleId?: string;
  retrievalIntent?: RetrievalIntentResult;
  captureIntent?: CaptureIntentResult;
}

export interface RoutePlan {
  route: RouteKind;
  confidence: number;
  turnFrame: TurnFrame;
  retrievalPlan: RetrievalPlan;
  injectionPlan: InjectionPlan;
  shouldRetrieve: boolean;
  enqueueCapture: boolean;
  retrievalIntent: RetrievalIntentResult;
  captureIntent: CaptureIntentResult;
  latencyReason: string;
  policySnapshotId?: string;
  matchedPolicyRuleId?: string;
  reasonCode?: string;
}

export class RouteCache {
  private cache = new Map<string, CachedRoutePlan>();

  get(fingerprint: RouteFingerprint): CachedRoutePlan | null {
    const key = fingerprintKey(fingerprint);
    const hit = this.cache.get(key);
    if (!hit) return null;
    if (Date.parse(hit.expiresAt) < Date.now()) {
      this.cache.delete(key);
      return null;
    }
    return hit;
  }

  set(fingerprint: RouteFingerprint, plan: CachedRoutePlan) {
    this.cache.set(fingerprintKey(fingerprint), plan);
  }

  invalidate(predicate?: (key: string, value: CachedRoutePlan) => boolean) {
    if (!predicate) {
      this.cache.clear();
      return;
    }
    for (const [key, value] of this.cache.entries()) {
      if (predicate(key, value)) this.cache.delete(key);
    }
  }
}

export class RouteFn {
  private config: any;
  private cache: RouteCache;
  private store?: any;

  constructor(options: { config: any; cache?: RouteCache; store?: any }) {
    this.config = options.config;
    this.cache = options.cache ?? new RouteCache();
    this.store = options.store;
  }

  fingerprint(packet: TurnEventPacket): RouteFingerprint {
    const message = packet.latestUserMessageRedacted.toLowerCase();
    const captureIntent = detectCaptureIntent(packet);
    const retrievalIntent = detectRetrievalIntent(packet);
    return {
      agentId: packet.agentId,
      scopeKey: packet.sessionId || packet.sessionKey || undefined,
      taskTypeHint: String(packet.metadata.turnType || ''),
      topicKeys: extractTopicKeys(message),
      explicitMemoryReference: /\b(as before|like i said|same as last time|we discussed before|remember)\b/i.test(packet.latestUserMessageRedacted),
      explicitCorrectionCue: /\b(actually|instead|no,|don't|do not|wrong|use .* instead)\b/i.test(packet.latestUserMessageRedacted),
      captureIntent: captureIntent.intent,
      retrievalIntent: retrievalIntent.intent,
    };
  }

  plan(packet: TurnEventPacket): RoutePlan {
    const fingerprint = this.fingerprint(packet);
    const cached = this.cache.get(fingerprint);
    const turnFrame = turnFrameFromPacket(packet);
    const captureIntent = detectCaptureIntent(packet);
    const retrievalIntent = detectRetrievalIntent(packet);
    if (cached) {
      return {
        route: cached.route,
        confidence: cached.confidence,
        turnFrame,
        retrievalPlan: cached.retrievalPlan,
        injectionPlan: cached.injectionPlan,
        shouldRetrieve: cached.route === 'retrieve_memory' || cached.route === 'retrieve_and_distill' || cached.route === 'high_confidence_correction_only',
        enqueueCapture: captureIntent.shouldConsiderCapture,
        retrievalIntent: cached.retrievalIntent ?? retrievalIntent,
        captureIntent: cached.captureIntent ?? captureIntent,
        latencyReason: 'cached route plan',
        policySnapshotId: cached.policySnapshotId ?? this.store?.getActivePolicySnapshotV3?.(packet.agentId)?.id ?? this.store?.getActivePolicySnapshotV2?.(packet.agentId)?.id ?? this.store?.getActivePolicySnapshot?.(packet.agentId)?.id,
        matchedPolicyRuleId: cached.matchedPolicyRuleId,
        reasonCode: cached.matchedPolicyRuleId ? `cached_policy_rule:${cached.matchedPolicyRuleId}` : 'cached_route_plan',
      };
    }

    const policySnapshot = this.loadPolicySnapshot(packet);
    const plan = heuristicRoutePlan(packet, turnFrame, this.config, policySnapshot, retrievalIntent, captureIntent);
    this.cache.set(fingerprint, {
      route: plan.route,
      retrievalPlan: plan.retrievalPlan,
      injectionPlan: plan.injectionPlan,
      confidence: plan.confidence,
      retrievalIntent: plan.retrievalIntent,
      captureIntent: plan.captureIntent,
      policySnapshotId: plan.policySnapshotId,
      matchedPolicyRuleId: plan.matchedPolicyRuleId,
      expiresAt: new Date(Date.now() + 5 * 60 * 1000).toISOString(),
    });
    return plan;
  }

  private loadPolicySnapshot(packet: TurnEventPacket) {
    if (!this.store) return null;
    try {
      return this.store.getActivePolicySnapshotV3?.(packet.agentId) ?? this.store.getActivePolicySnapshotV2?.(packet.agentId) ?? this.store.getActivePolicySnapshot(packet.agentId);
    } catch {
      return null;
    }
  }
}

function heuristicRoutePlan(packet: TurnEventPacket, turnFrame: TurnFrame, config: any, policySnapshot: any, retrievalIntent: RetrievalIntentResult, captureIntent: CaptureIntentResult): RoutePlan {
  const message = packet.latestUserMessageRedacted.toLowerCase();
  const explicitCorrectionCue = /\b(actually|instead|wrong|no,)\b/i.test(packet.latestUserMessageRedacted);
  const planningLike = /\b(plan|design|architecture|file-by-file|implementation)\b/.test(message);

  const policyMatch = policySnapshot?.version === 'route-policy-v3'
    ? scorePolicySnapshotV3(policySnapshot, turnFrame, packet.latestUserMessageRedacted)
    : scorePolicySnapshotV2(policySnapshot?.version === 'route-policy-v2' ? policySnapshot : null, turnFrame, packet.latestUserMessageRedacted);
  if (policyMatch.matched && policyMatch.rule && policyMatch.score >= Number(config.routing?.minRouteConfidence ?? 0.7) && !explicitCorrectionCue) {
    return routePlanFromPolicyRule(packet, turnFrame, config, policySnapshot, policyMatch.rule, policyMatch.score, retrievalIntent, captureIntent, policyMatch.reasonCode);
  }

  let route: RouteKind = 'no_memory';
  let confidence = Math.max(retrievalIntent.confidence, captureIntent.confidence);
  if (explicitCorrectionCue && retrievalIntent.shouldRetrieve) {
    route = 'high_confidence_correction_only';
    confidence = Math.max(confidence, 0.9);
  } else if (retrievalIntent.shouldRetrieve && captureIntent.shouldConsiderCapture) {
    route = 'retrieve_and_distill';
  } else if (retrievalIntent.shouldRetrieve) {
    route = 'retrieve_memory';
    confidence = Math.max(confidence, planningLike ? 0.82 : 0.72);
  } else if (captureIntent.shouldConsiderCapture) {
    route = 'capture_only';
    confidence = Math.max(confidence, captureIntent.confidence);
  }

  const policyBoost = policySnapshot ? applyPolicySnapshot(packet, turnFrame, policySnapshot) : null;
  if (policyBoost && policyBoost.route && !explicitCorrectionCue && retrievalIntent.intent !== 'no_retrieval') {
    route = policyBoost.route;
    confidence = Math.max(confidence, policyBoost.confidence);
  }

  const heuristicQueries = buildQueries(packet, retrievalIntent);
  const policyQueries = policyBoost?.queries ?? [];
  const allQueries = [...new Set([...heuristicQueries, ...policyQueries])];

  const heuristicMemoryTypes = memoryTypesForTurn(route, retrievalIntent, captureIntent, message, planningLike);
  const policyMemoryTypes = policyBoost?.memoryTypes ?? [];
  const allMemoryTypes = [...new Set([...heuristicMemoryTypes, ...policyMemoryTypes])] as MemoryType[];

  const retrievalPlan: RetrievalPlan = {
    queries: allQueries,
    memoryTypes: allMemoryTypes,
    requiredTags: [],
    excludedTags: retrievalIntent.includeRecallRules ? [] : ['recall_value'],
    graphDepth: policyBoost?.graphDepth ?? (planningLike || policyBoost ? 1 : 0),
    maxCandidates: config.routing.maxCandidateMemories,
  };

  const injectionPlan: InjectionPlan = {
    maxItems: config.routing.maxInjectedMemories,
    maxChars: config.routing.maxInjectedChars,
    preferredFormat: explicitCorrectionCue ? 'rules' : planningLike ? 'bullets' : retrievalIntent.intent === 'recall_value_request' ? 'rules' : 'none',
  };

  return {
    route,
    confidence,
    turnFrame,
    retrievalPlan,
    injectionPlan,
    shouldRetrieve: retrievalIntent.shouldRetrieve
      || route === 'retrieve_memory'
      || route === 'retrieve_and_distill'
      || route === 'high_confidence_correction_only'
      || policyBoost?.route === 'retrieve_memory'
      || policyBoost?.route === 'retrieve_and_distill'
      || policyBoost?.route === 'high_confidence_correction_only',
    enqueueCapture: captureIntent.shouldConsiderCapture,
    retrievalIntent,
    captureIntent,
    latencyReason: policySnapshot ? 'heuristic with policy snapshot' : 'heuristic uncached route',
    policySnapshotId: policySnapshot?.id,
    matchedPolicyRuleId: policyBoost?.matchedPolicyRuleId,
    reasonCode: policyBoost?.matchedPolicyRuleId
      ? `policy_boost:${policyBoost.matchedPolicyRuleId}`
      : policyBoost?.reasonCode || 'heuristic_uncached_route',
  };
}

function routePlanFromPolicyRule(packet: TurnEventPacket, turnFrame: TurnFrame, config: any, policySnapshot: any, rule: any, score: number, retrievalIntent: RetrievalIntentResult, captureIntent: CaptureIntentResult, reasonCode: string): RoutePlan {
  const route = rule.route as RouteKind;
  const shouldRetrieve = route === 'retrieve_memory' || route === 'retrieve_and_distill' || route === 'high_confidence_correction_only';
  const maxItems = Math.min(Number(config.routing.maxInjectedMemories ?? 8), Number(policySnapshot?.globalBudgets?.maxInjectedMemories ?? config.routing.maxInjectedMemories ?? 8));
  const maxChars = Math.min(Number(config.routing.maxInjectedChars ?? 2500), Number(policySnapshot?.globalBudgets?.maxInjectedChars ?? config.routing.maxInjectedChars ?? 2500));
  return {
    route,
    confidence: score,
    turnFrame,
    retrievalPlan: {
      queries: shouldRetrieve ? [...new Set([...(rule.queries ?? []), ...buildQueries(packet, retrievalIntent)])].slice(0, 8) : [],
      memoryTypes: shouldRetrieve ? ([...new Set(rule.memoryTypes ?? [])] as MemoryType[]) : [],
      requiredTags: [],
      excludedTags: retrievalIntent.includeRecallRules ? [] : ['recall_value'],
      graphDepth: shouldRetrieve ? (rule.graphDepth ?? policySnapshot?.globalBudgets?.defaultGraphDepth ?? 0) : 0,
      maxCandidates: Math.min(Number(config.routing.maxCandidateMemories ?? 40), 40),
    },
    injectionPlan: {
      maxItems,
      maxChars,
      preferredFormat: shouldRetrieve ? (route === 'high_confidence_correction_only' ? 'rules' : 'bullets') : 'none',
    },
    shouldRetrieve,
    enqueueCapture: captureIntent.shouldConsiderCapture,
    retrievalIntent,
    captureIntent,
    latencyReason: policySnapshot?.version === 'route-policy-v3' ? 'policy-v3 distilled rule' : 'policy-v2 deterministic rule',
    policySnapshotId: policySnapshot?.id,
    matchedPolicyRuleId: rule.id,
    reasonCode,
  };
}

function applyPolicySnapshot(packet: TurnEventPacket, turnFrame: TurnFrame, policySnapshot: any) {
  const boost = { route: null as RouteKind | null, confidence: 0, memoryTypes: [] as MemoryType[], queries: [] as string[], graphDepth: undefined as undefined | 0 | 1 | 2, matchedPolicyRuleId: undefined as string | undefined, reasonCode: undefined as string | undefined };
  if (policySnapshot?.version === 'route-policy-v3' && Array.isArray(policySnapshot.rules)) {
    const match = scorePolicySnapshotV3(policySnapshot, turnFrame, packet.latestUserMessageRedacted);
    const rule = match.rule;
    if (!match.matched || !rule) {
      boost.reasonCode = match.reasonCode;
      return boost;
    }
    boost.route = rule.route;
    boost.confidence = match.score || Number(rule.confidence || 0.7);
    boost.memoryTypes = Array.isArray(rule.memoryTypes) ? rule.memoryTypes : [];
    boost.queries = Array.isArray(rule.queries) ? rule.queries : [];
    boost.graphDepth = rule.graphDepth ?? 0;
    boost.matchedPolicyRuleId = rule.id;
    boost.reasonCode = match.reasonCode;
    return boost;
  }
  if (policySnapshot?.version === 'route-policy-v2' && Array.isArray(policySnapshot.rules)) {
    const match = scorePolicySnapshotV2(policySnapshot, turnFrame, packet.latestUserMessageRedacted);
    const rule = match.rule;
    if (!match.matched || !rule) {
      boost.reasonCode = match.reasonCode;
      return boost;
    }
    boost.route = rule.route;
    boost.confidence = match.score || Number(rule.confidence || 0.7);
    boost.memoryTypes = Array.isArray(rule.memoryTypes) ? rule.memoryTypes : [];
    boost.queries = Array.isArray(rule.queries) ? rule.queries : [];
    boost.graphDepth = rule.graphDepth ?? 0;
    boost.matchedPolicyRuleId = rule.id;
    boost.reasonCode = match.reasonCode;
    return boost;
  }
  if (!policySnapshot?.policyText) return boost;
  const policy = String(policySnapshot.policyText).toLowerCase();
  const taskType = turnFrame.taskType;
  const taskTypeLine = policy.split('\n').find(line => line.includes(taskType));
  if (!taskTypeLine) return boost;
  if (/retrieve|memory|pull/.test(taskTypeLine) && taskTypeLine.includes(taskType)) {
    boost.route = 'retrieve_memory';
    boost.confidence = 0.78;
  }
  if (/no memory|prefer no|skip memory/.test(taskTypeLine)) {
    boost.route = 'no_memory';
    boost.confidence = 0.7;
  }
  const typeMatches = taskTypeLine.match(/\b(correction|preference|workflow|context|project_fact|tool_convention|routing_rule|agent_assignment|recall_rule|outcome)\b/gi);
  if (typeMatches) boost.memoryTypes = [...new Set(typeMatches.map(t => t.toLowerCase() as MemoryType))];
  if (/planning/.test(taskType)) boost.queries.push('implementation planning architecture preferences workflow');
  if (/coding/.test(taskType) && /install|dependency|package/.test(policy)) boost.queries.push('package manager correction workflow repo setup');
  return boost;
}

function policyRuleMatches(rule: any, turnFrame: TurnFrame, message: string) {
  const match = rule?.match ?? {};
  if (match.taskType && match.taskType !== turnFrame.taskType) return false;
  const signals = Array.isArray(match.turnSignals) ? match.turnSignals : [];
  if (signals.length === 0) return true;
  return signals.some((signal: string) => message.includes(String(signal).toLowerCase()) || turnFrame.impliedNeeds.join(' ').toLowerCase().includes(String(signal).toLowerCase()));
}

function memoryTypesForTurn(route: RouteKind, retrievalIntent: RetrievalIntentResult, captureIntent: CaptureIntentResult, lower: string, planningLike: boolean): MemoryType[] {
  if (retrievalIntent.intent === 'recall_value_request') return ['recall_rule'];
  if (route === 'high_confidence_correction_only') return ['correction'];
  const types = new Set<MemoryType>(retrievalIntent.memoryTypes);
  if (planningLike) ['correction', 'preference', 'workflow', 'project_fact', 'tool_convention', 'routing_rule', 'agent_assignment', 'outcome', 'context'].forEach((t) => types.add(t as MemoryType));
  if (/\b(install|dependency|dependencies|pnpm|npm|yarn|build|test|setup)\b/.test(lower)) ['correction', 'workflow', 'tool_convention'].forEach((t) => types.add(t as MemoryType));
  if (captureIntent.intent === 'routing_rule') types.add('routing_rule');
  if (captureIntent.intent === 'recall_rule') types.add('recall_rule');
  if (types.size === 0) ['preference', 'context'].forEach((t) => types.add(t as MemoryType));
  return [...types];
}

function buildQueries(packet: TurnEventPacket, retrievalIntent?: RetrievalIntentResult): string[] {
  const text = packet.latestUserMessageRedacted.trim();
  const lower = text.toLowerCase();
  const queries = [retrievalIntent?.query || text];
  if (/\b(plan|architecture|implementation|file-by-file)\b/.test(lower)) queries.push('implementation planning architecture preferences workflow');
  if (/\b(install|dependency|dependencies|pnpm|npm|yarn|build|test|setup)\b/.test(lower)) queries.push('package manager correction workflow repo setup');
  if (/\b(actually|instead|wrong|no,)\b/.test(lower)) queries.push('recent correction preference update');
  if (/\b(codeword|phrase|recall)\b/.test(lower)) queries.push('recall rule codeword phrase');
  return [...new Set(queries.map((q) => q.trim()).filter(Boolean))];
}

function turnFrameFromPacket(packet: TurnEventPacket): TurnFrame {
  const message = packet.latestUserMessageRedacted;
  const lower = message.toLowerCase();
  const taskType = /\b(debug|error|failing|broken)\b/.test(lower)
    ? 'debugging'
    : /\b(plan|design|architecture|implementation)\b/.test(lower)
      ? 'planning'
      : /\b(code|build|test|install|dependency|repo|file)\b/.test(lower)
        ? 'coding'
        : /\b(write|draft|summarize)\b/.test(lower)
          ? 'writing'
          : /\b(actually|instead|wrong|no,)\b/.test(lower)
            ? 'correction'
            : /\b(i prefer|remember that|going forward|from now on)\b/.test(lower)
              ? 'preference_update'
              : 'other';
  return {
    summary: message.slice(0, 240),
    userGoal: message.slice(0, 240),
    taskType,
    activeObjects: extractTopicKeys(lower).slice(0, 6).map((value) => ({ kind: 'concept' as const, value })),
    impliedNeeds: [
      /\b(plan|architecture|implementation)\b/.test(lower) ? 'Need prior architecture context' : '',
      /\b(pnpm|npm|yarn|install|dependency)\b/.test(lower) ? 'Need package-manager corrections' : '',
      /\b(if i ask|when i ask|codeword|phrase)\b/.test(lower) ? 'Need recall-rule handling' : '',
    ].filter(Boolean),
    memoryQuestions: [/\bremember|before|same as last time|codeword|phrase\b/.test(lower) ? 'What prior context or recall rule is being referenced?' : ''].filter(Boolean),
    constraints: [],
    routeHints: {
      likelyNeedsCorrections: /\b(actually|instead|wrong|pnpm|npm)\b/.test(lower),
      likelyNeedsPreferences: /\b(file-by-file|concrete|plan|prefer)\b/.test(lower),
      likelyNeedsWorkflow: /\b(install|build|test|setup|going forward|always|never)\b/.test(lower),
      likelyNeedsProjectContext: /\b(openclawbrain|repo|architecture|implementation|project)\b/.test(lower),
    },
  };
}

function extractTopicKeys(message: string): string[] {
  const matches = message.match(/[a-z][a-z0-9_-]{2,}/g) || [];
  const stop = new Set(['this', 'that', 'with', 'from', 'have', 'like', 'said', 'same', 'last', 'time', 'what', 'when', 'then', 'them', 'your', 'into']);
  return [...new Set(matches.filter((word) => !stop.has(word)).slice(0, 12))];
}

function fingerprintKey(fingerprint: RouteFingerprint): string {
  return hashText(JSON.stringify(fingerprint));
}
