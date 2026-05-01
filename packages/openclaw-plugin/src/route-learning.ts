import type { InjectionEvent, InjectionOutcome, RouteDecision, RouteExample, RoutePolicySnapshot } from './memory-types.js';
import type { MemoryStore } from './memory-store.js';

export interface RouteLearningRunReport {
  resolvedDecisions: number;
  examplesCreated: number;
  memoryUpdates: number;
  snapshotId?: string;
}

export class RouteLearning {
  private store: MemoryStore;
  private config: any;

  constructor(options: { store: MemoryStore; config: any }) {
    this.store = options.store;
    this.config = options.config;
  }

  run(agentId: string): RouteLearningRunReport {
    let resolvedDecisions = 0;
    let examplesCreated = 0;
    let memoryUpdates = 0;

    for (const decision of this.store.getUnresolvedRouteDecisions(agentId)) {
      const injections = this.store.getInjectionsForRouteDecision(decision.id);
      const outcome = classifyRouteOutcome(decision, injections);
      if (!outcome) continue;
      this.store.resolveRouteDecision(decision.id, outcome.outcome, outcome.reward);
      resolvedDecisions += 1;
      memoryUpdates += applyMemoryUpdates(this.store, injections);
      if (!this.store.hasRouteExampleForDecision(agentId, decision.id)) {
        this.store.insertRouteExample(buildRouteExample(agentId, decision, injections, outcome.outcome, outcome.reward));
        examplesCreated += 1;
      }
    }

    const snapshot = maybeBuildPolicySnapshot(this.store, agentId, this.config);
    return {
      resolvedDecisions,
      examplesCreated,
      memoryUpdates,
      snapshotId: snapshot?.id,
    };
  }
}

function classifyRouteOutcome(decision: RouteDecision, injections: InjectionEvent[]) {
  if (injections.length === 0) {
    if (decision.selectedMemoryIds.length === 0) return { outcome: 'no_signal', reward: 0 };
    return null;
  }
  if (injections.some((injection) => injection.outcome === 'pending')) return null;
  const outcomes = injections.map((injection) => injection.outcome);
  if (outcomes.some((outcome) => outcome === 'harmful' || outcome === 'user_corrected')) {
    return { outcome: 'corrected_after_injection', reward: -1.0 };
  }
  if (outcomes.some((outcome) => outcome === 'tool_failure')) {
    return { outcome: 'tool_failure', reward: -0.7 };
  }
  if (outcomes.some((outcome) => outcome === 'tool_success')) {
    return { outcome: 'tool_success', reward: 0.8 };
  }
  if (outcomes.some((outcome) => outcome === 'helped' || outcome === 'accepted')) {
    return { outcome: 'helpful_context', reward: 1.0 };
  }
  if (outcomes.every((outcome) => outcome === 'ignored' || outcome === 'unknown')) {
    return { outcome: 'irrelevant_context', reward: -0.4 };
  }
  return { outcome: 'no_signal', reward: 0 };
}

function buildRouteExample(agentId: string, decision: RouteDecision, injections: InjectionEvent[], outcome: string, reward: number): Omit<RouteExample, 'id' | 'createdAt'> {
  return {
    agentId,
    turnFrame: decision.turnFrame,
    routeDecision: {
      route: decision.route,
      confidence: decision.confidence,
      retrievalPlan: decision.retrievalPlan,
      injectionPlan: decision.injectionPlan,
      selectedMemoryIds: decision.selectedMemoryIds,
      model: decision.model,
      promptVersion: decision.promptVersion,
    },
    outcome,
    reward,
    lesson: buildLesson(decision, injections, outcome),
    tags: [
      `route:${decision.route}`,
      `task:${decision.turnFrame.taskType}`,
      `route_decision:${decision.id}`,
      ...decision.retrievalPlan.memoryTypes.map((type) => `memory_type:${type}`),
    ],
  };
}

function buildLesson(decision: RouteDecision, injections: InjectionEvent[], outcome: string) {
  const types = decision.retrievalPlan.memoryTypes.join(', ') || 'none';
  if (outcome === 'helpful_context' || outcome === 'tool_success') {
    return `Prefer ${decision.route} for ${decision.turnFrame.taskType} turns when ${types} memories are relevant.`;
  }
  if (outcome === 'corrected_after_injection') {
    return `Tighten ${decision.route} for ${decision.turnFrame.taskType} turns; injected memories caused correction.`;
  }
  if (outcome === 'irrelevant_context') {
    return `Reduce memory injection for ${decision.turnFrame.taskType} turns when retrieved ${types} memories are not used.`;
  }
  if (outcome === 'tool_failure') {
    return `Do not trust ${decision.route} alone for ${decision.turnFrame.taskType}; associated workflow memory did not help.`;
  }
  if (injections.length === 0) {
    return `No memory signal for ${decision.turnFrame.taskType}; keep default latency-safe fallback.`;
  }
  return `Outcome for ${decision.turnFrame.taskType} turn was weak; keep the route conservative.`;
}

function applyMemoryUpdates(store: MemoryStore, injections: InjectionEvent[]) {
  let updated = 0;
  for (const injection of injections) {
    const patch = scorePatchForOutcome(injection.outcome);
    if (!patch) continue;
    const result = store.adjustMemoryScore(injection.memoryId, patch);
    if (result) updated += 1;
  }
  return updated;
}

function scorePatchForOutcome(outcome: InjectionOutcome) {
  switch (outcome) {
    case 'helped':
    case 'accepted':
      return { importanceDelta: 0.08, confidenceDelta: 0.03, usefulCountDelta: 1, useCountDelta: 1 };
    case 'tool_success':
      return { importanceDelta: 0.06, confidenceDelta: 0.02, usefulCountDelta: 1, useCountDelta: 1 };
    case 'ignored':
      return { importanceDelta: -0.02, useCountDelta: 1 };
    case 'assistant_failed_to_use':
      return { importanceDelta: 0.02, useCountDelta: 1 };
    case 'user_corrected':
      return { importanceDelta: -0.08, confidenceDelta: -0.06, useCountDelta: 1 };
    case 'harmful':
    case 'tool_failure':
      return { importanceDelta: -0.2, confidenceDelta: -0.15, useCountDelta: 1 };
    default:
      return null;
  }
}

function maybeBuildPolicySnapshot(store: MemoryStore, agentId: string, config: any): RoutePolicySnapshot | null {
  const examples = store.getRouteExamples(agentId, Math.max(config.learning.maxPositiveExamples, config.learning.maxNegativeExamples) * 2);
  const positive = examples.filter((example) => example.reward > 0).slice(0, config.learning.maxPositiveExamples);
  const negative = examples.filter((example) => example.reward < 0).slice(0, config.learning.maxNegativeExamples);
  const existing = store.getActivePolicySnapshot(agentId);
  if (!existing && examples.length === 0) return null;
  if (!existing && examples.length < config.learning.minExamplesForPolicyUpdate) return null;
  if (existing && examples.length < config.learning.minExamplesForPolicyUpdate && positive.length === 0 && negative.length === 0) {
    return existing;
  }

  const lines = [
    `Route policy snapshot:`,
    positive.length > 0
      ? `- Prefer memory retrieval for successful patterns: ${summarizeExamples(positive)}.`
      : `- No strong positive route examples yet; keep memory routing conservative.`,
    negative.length > 0
      ? `- Avoid weak patterns: ${summarizeExamples(negative)}.`
      : `- No strong negative route examples yet.`,
    `- Keep synchronous planner usage bounded; fall back to cached/local routing when confidence is high.`,
  ];

  return store.insertPolicySnapshot({
    agentId,
    policyText: lines.join('\n'),
    examples: [...positive, ...negative].map((example) => example.id),
    model: config.llm?.learningModel || 'deterministic',
    promptVersion: 'route-learning-v1',
    active: true,
  });
}

function summarizeExamples(examples: RouteExample[]) {
  const counts = new Map<string, number>();
  for (const example of examples) {
    const key = `${example.routeDecision.route || 'unknown'} on ${example.turnFrame.taskType}`;
    counts.set(key, (counts.get(key) || 0) + 1);
  }
  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3)
    .map(([key, count]) => `${key} (${count})`)
    .join(', ');
}
