import type { InjectionPlan, RetrievalPlan, RouteDecision, RouteKind, TurnFrame } from './memory-types.js';
import type { TurnEventPacket } from './capture.js';
import { hashText } from './redact.js';

export interface RouteFingerprint {
  agentId: string;
  scopeKey?: string;
  taskTypeHint?: string;
  topicKeys: string[];
  explicitMemoryReference: boolean;
  explicitCorrectionCue: boolean;
}

export interface CachedRoutePlan {
  route: RouteKind;
  retrievalPlan: RetrievalPlan;
  injectionPlan: InjectionPlan;
  confidence: number;
  expiresAt: string;
  sourceRouteDecisionId?: string;
}

export interface RoutePlan {
  route: RouteKind;
  confidence: number;
  turnFrame: TurnFrame;
  retrievalPlan: RetrievalPlan;
  injectionPlan: InjectionPlan;
  shouldRetrieve: boolean;
  enqueueCapture: boolean;
  latencyReason: string;
  policySnapshotId?: string;
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
    return {
      agentId: packet.agentId,
      scopeKey: packet.sessionId || packet.sessionKey || undefined,
      taskTypeHint: String(packet.metadata.turnType || ''),
      topicKeys: extractTopicKeys(message),
      explicitMemoryReference: /\b(as before|like i said|same as last time|we discussed before|remember)\b/i.test(packet.latestUserMessageRedacted),
      explicitCorrectionCue: /\b(actually|instead|no,|don't|do not|wrong|use .* instead)\b/i.test(packet.latestUserMessageRedacted),
    };
  }

  plan(packet: TurnEventPacket): RoutePlan {
    const fingerprint = this.fingerprint(packet);
    const cached = this.cache.get(fingerprint);
    const turnFrame = turnFrameFromPacket(packet);
    if (cached) {
      return {
        route: cached.route,
        confidence: cached.confidence,
        turnFrame,
        retrievalPlan: cached.retrievalPlan,
        injectionPlan: cached.injectionPlan,
        shouldRetrieve: cached.route === 'retrieve_memory' || cached.route === 'retrieve_and_distill' || cached.route === 'high_confidence_correction_only',
        enqueueCapture: fingerprint.explicitCorrectionCue || fingerprint.explicitMemoryReference,
        latencyReason: 'cached route plan',
        policySnapshotId: this.store?.getActivePolicySnapshot?.(packet.agentId)?.id,
      };
    }

    const policySnapshot = this.loadPolicySnapshot(packet);
    const plan = heuristicRoutePlan(packet, turnFrame, this.config, policySnapshot);
    this.cache.set(fingerprint, {
      route: plan.route,
      retrievalPlan: plan.retrievalPlan,
      injectionPlan: plan.injectionPlan,
      confidence: plan.confidence,
      expiresAt: new Date(Date.now() + 5 * 60 * 1000).toISOString(),
    });
    return plan;
  }

  private loadPolicySnapshot(packet: TurnEventPacket) {
    if (!this.store) return null;
    try {
      return this.store.getActivePolicySnapshot(packet.agentId);
    } catch {
      return null;
    }
  }
}

function heuristicRoutePlan(packet: TurnEventPacket, turnFrame: TurnFrame, config: any, policySnapshot?: any): RoutePlan {
  const message = packet.latestUserMessageRedacted.toLowerCase();
  const explicitCorrectionCue = /\b(actually|instead|wrong|no,)\b/i.test(packet.latestUserMessageRedacted);
  const explicitMemoryReference = /\b(as before|same as last time|remember|we discussed before)\b/i.test(packet.latestUserMessageRedacted);
  const installLike = /\b(install|dependency|dependencies|pnpm|npm|yarn|build|test|setup)\b/.test(message);
  const planningLike = /\b(plan|design|architecture|file-by-file|implementation)\b/.test(message);

  let route: RouteKind = 'no_memory';
  let confidence = 0.55;
  if (explicitCorrectionCue) {
    route = 'high_confidence_correction_only';
    confidence = 0.9;
  } else if (planningLike || explicitMemoryReference || installLike) {
    route = 'retrieve_memory';
    confidence = planningLike ? 0.82 : 0.72;
  }

  const policyBoost = policySnapshot ? applyPolicySnapshot(packet, turnFrame, policySnapshot) : null;
  if (policyBoost && policyBoost.route && !explicitCorrectionCue) {
    route = policyBoost.route;
    confidence = Math.max(confidence, policyBoost.confidence);
  }

  const heuristicQueries = buildQueries(packet);
  const policyQueries = policyBoost?.queries ?? [];
  const allQueries = [...new Set([...heuristicQueries, ...policyQueries])];

  const heuristicMemoryTypes = route === 'high_confidence_correction_only'
    ? ['correction']
    : planningLike
      ? ['correction', 'preference', 'workflow', 'context']
      : installLike
        ? ['correction', 'workflow']
        : ['preference', 'context'];
  const policyMemoryTypes = policyBoost?.memoryTypes ?? [];
  const allMemoryTypes = [...new Set([...heuristicMemoryTypes, ...policyMemoryTypes])] as any;

  const retrievalPlan: RetrievalPlan = {
    queries: allQueries,
    memoryTypes: allMemoryTypes,
    requiredTags: [],
    excludedTags: [],
    graphDepth: planningLike || policyBoost ? 1 : 0,
    maxCandidates: config.routing.maxCandidateMemories,
  };

  const injectionPlan: InjectionPlan = {
    maxItems: config.routing.maxInjectedMemories,
    maxChars: config.routing.maxInjectedChars,
    preferredFormat: explicitCorrectionCue ? 'rules' : planningLike ? 'bullets' : 'none',
  };

  return {
    route,
    confidence,
    turnFrame,
    retrievalPlan,
    injectionPlan,
    shouldRetrieve: route !== 'no_memory',
    enqueueCapture: explicitCorrectionCue || explicitMemoryReference,
    latencyReason: policySnapshot ? 'heuristic with policy snapshot' : 'heuristic uncached route',
    policySnapshotId: policySnapshot?.id,
  };
}

function applyPolicySnapshot(packet: TurnEventPacket, turnFrame: TurnFrame, policySnapshot: any) {
  const boost = { route: null as RouteKind | null, confidence: 0, memoryTypes: [] as string[], queries: [] as string[] };
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
  const typeMatches = taskTypeLine.match(/\b(correction|preference|workflow|context)\b/gi);
  if (typeMatches) boost.memoryTypes = [...new Set(typeMatches.map(t => t.toLowerCase()))];
  if (/planning/.test(taskType)) boost.queries.push('implementation planning architecture preferences workflow');
  if (/coding/.test(taskType) && /install|dependency|package/.test(policy)) boost.queries.push('package manager correction workflow repo setup');
  return boost;
}

function buildPolicyEnrichedRoute(packet: TurnEventPacket, turnFrame: TurnFrame, config: any, route: RouteKind, confidence: number, policyBoost: any): RoutePlan {
  const explicitCorrectionCue = /\b(actually|instead|wrong|no,)\b/i.test(packet.latestUserMessageRedacted);
  const heuristicQueries = buildQueries(packet);
  const allQueries = [...new Set([...heuristicQueries, ...policyBoost.queries])];
  return {
    route,
    confidence,
    turnFrame,
    retrievalPlan: {
      queries: allQueries,
      memoryTypes: policyBoost.memoryTypes as any,
      requiredTags: [],
      excludedTags: [],
      graphDepth: 1,
      maxCandidates: config.routing.maxCandidateMemories,
    },
    injectionPlan: {
      maxItems: config.routing.maxInjectedMemories,
      maxChars: config.routing.maxInjectedChars,
      preferredFormat: explicitCorrectionCue ? 'rules' : 'bullets',
    },
    shouldRetrieve: route !== 'no_memory',
    enqueueCapture: explicitCorrectionCue || /\b(as before|same as last time|remember|we discussed before)\b/i.test(packet.latestUserMessageRedacted),
    latencyReason: 'heuristic with policy snapshot',
  };
}

function buildQueries(packet: TurnEventPacket): string[] {
  const text = packet.latestUserMessageRedacted.trim();
  const lower = text.toLowerCase();
  const queries = [text];
  if (/\b(plan|architecture|implementation|file-by-file)\b/.test(lower)) queries.push('implementation planning architecture preferences workflow');
  if (/\b(install|dependency|dependencies|pnpm|npm|yarn|build|test|setup)\b/.test(lower)) queries.push('package manager correction workflow repo setup');
  if (/\b(actually|instead|wrong|no,)\b/.test(lower)) queries.push('recent correction preference update');
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
            : 'other';
  return {
    summary: message.slice(0, 240),
    userGoal: message.slice(0, 240),
    taskType,
    activeObjects: extractTopicKeys(lower).slice(0, 6).map((value) => ({ kind: 'concept' as const, value })),
    impliedNeeds: [
      /\b(plan|architecture|implementation)\b/.test(lower) ? 'Need prior architecture context' : '',
      /\b(pnpm|npm|yarn|install|dependency)\b/.test(lower) ? 'Need package-manager corrections' : '',
    ].filter(Boolean),
    memoryQuestions: [/\bremember|before|same as last time\b/.test(lower) ? 'What prior context is being referenced?' : ''].filter(Boolean),
    constraints: [],
    routeHints: {
      likelyNeedsCorrections: /\b(actually|instead|wrong|pnpm|npm)\b/.test(lower),
      likelyNeedsPreferences: /\b(file-by-file|concrete|plan)\b/.test(lower),
      likelyNeedsWorkflow: /\b(install|build|test|setup)\b/.test(lower),
      likelyNeedsProjectContext: /\b(openclawbrain|repo|architecture|implementation)\b/.test(lower),
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
