import type { LlmClient } from './llm-client.js';
import type { MemoryStore } from './memory-store.js';
import { hashText, redactJsonValue, redactText } from './redact.js';
import type {
  MemoryNode,
  MemoryType,
  RouteCounterfactual,
  RouteCounterfactualKind,
  RouteDecision,
  RouteGraphSnapshot,
  RouteKind,
  RouteTeacherRun,
  RouteTeacherVerdict,
  RouteTrainingExampleKind,
  TaskType,
} from './memory-types.js';
import { maybeDistillAndStorePolicyV2 } from './route-policy-v2.js';
import { ingestRouteLearningArtifactsV3, maybeDistillAndStorePolicyV3 } from './route-policy-v3.js';

const ROUTES: RouteKind[] = ['no_memory', 'capture_only', 'retrieve_memory', 'retrieve_and_distill', 'high_confidence_correction_only'];
const MEMORY_TYPES: MemoryType[] = ['correction', 'preference', 'workflow', 'project_fact', 'tool_convention', 'routing_rule', 'agent_assignment', 'recall_rule', 'outcome', 'context'];
const PROMPT_VERSION = 'route-teacher-v1';

export interface RouteTeacherReport {
  teacherRuns: number;
  counterfactuals: number;
  examples: number;
  policySnapshotId?: string;
  policySnapshotV3Id?: string;
  routeFramesV3?: number;
  pairExamplesV3?: number;
}

export class RouteTeacher {
  private store: MemoryStore;
  private config: any;
  private client?: LlmClient | null;

  constructor(options: { store: MemoryStore; config: any; client?: LlmClient | null }) {
    this.store = options.store;
    this.config = options.config;
    this.client = options.client ?? null;
  }

  async run(agentId: string): Promise<RouteTeacherReport> {
    if (this.config.routeLearning?.enabled === false || this.config.routeLearning?.teacher?.enabled === false) {
      return { teacherRuns: 0, counterfactuals: 0, examples: 0 };
    }

    const maxRuns = Math.max(0, Number(this.config.routeLearning?.teacher?.maxRunsPerCycle ?? 5));
    if (maxRuns === 0) return { teacherRuns: 0, counterfactuals: 0, examples: 0 };

    let teacherRuns = 0;
    let counterfactuals = 0;
    let examples = 0;
    let routeFramesV3 = 0;
    let pairExamplesV3 = 0;

    const candidates = this.store.getResolvedRouteDecisions(agentId, 100)
      .filter((decision) => !this.store.hasRouteTeacherRunForDecision(decision.id))
      .filter((decision) => Math.abs(decision.reward || 0) >= Number(this.config.routeLearning?.teacher?.minResolvedRewardMagnitude ?? 0))
      .slice(0, maxRuns);

    for (const decision of candidates) {
      const result = await this.teachDecision(agentId, decision);
      teacherRuns += result.teacherRuns;
      counterfactuals += result.counterfactuals;
      examples += result.examples;
      routeFramesV3 += result.routeFramesV3 ?? 0;
      pairExamplesV3 += result.pairExamplesV3 ?? 0;
    }

    const distillation = maybeDistillAndStorePolicyV2(this.store, agentId, this.config);
    const distillationV3 = maybeDistillAndStorePolicyV3(this.store, agentId, this.config);
    return {
      teacherRuns,
      counterfactuals,
      examples,
      routeFramesV3,
      pairExamplesV3,
      policySnapshotId: distillation.snapshot?.id,
      policySnapshotV3Id: distillationV3.snapshot?.id,
    };
  }

  async teachDecision(agentId: string, decision: RouteDecision): Promise<RouteTeacherReport> {
    const snapshot = this.store.getRouteGraphSnapshot(decision.id) ?? fallbackGraphSnapshot(this.store, agentId, decision);
    const input = teacherInput(decision, snapshot);
    const inputHash = hashText(JSON.stringify(input));
    const model = this.config.llm?.learningModel || this.config.llm?.routeModel || 'deterministic-route-teacher';

    let raw: any = null;
    let usedModel = model;
    if (this.client && this.config.llm?.enabled !== false) {
      try {
        raw = await this.client.runJson({
          task: 'openclawbrain_route_teacher',
          model,
          systemPrompt: routeTeacherSystemPrompt(),
          input,
          schema: routeTeacherSchema(),
          temperature: 0,
          maxTokens: Math.max(800, Number(this.config.llm?.maxTokens ?? 1200)),
          metadata: { routeDecisionId: decision.id, agentId },
        });
      } catch {
        raw = null;
      }
    }
    if (!raw) {
      usedModel = 'deterministic-route-teacher';
      raw = deterministicTeacherOutput(decision, snapshot);
    }

    const parsed = parseTeacherOutput(raw);
    const validation = validateTeacherOutput(parsed, snapshot, decision);
    const safeOutput = validation.valid ? parsed : deterministicTeacherOutput(decision, snapshot);
    const finalValidation = validateTeacherOutput(safeOutput, snapshot, decision);

    const run = this.store.insertRouteTeacherRun({
      agentId,
      routeDecisionId: decision.id,
      model: usedModel,
      promptVersion: PROMPT_VERSION,
      inputHash,
      outputHash: hashText(JSON.stringify(redactJsonValue(safeOutput))),
      verdict: safeOutput.verdict,
      teacherRoute: safeOutput.teacherRoute,
      teacherMemoryIds: safeOutput.teacherMemoryIds,
      teacherQueries: safeOutput.teacherQueries,
      teacherGraphDepth: safeOutput.teacherGraphDepth,
      syncPlannerWorthIt: safeOutput.syncPlannerWorthIt,
      confidence: safeOutput.confidence,
      rationale: redactText(safeOutput.rationale || finalValidation.reason || '', 500),
      validated: finalValidation.valid,
      rejectionReason: finalValidation.valid ? validation.reason : (validation.reason || finalValidation.reason),
    });

    const storedCounterfactuals: RouteCounterfactual[] = [];
    let counterfactuals = 0;
    for (const cf of normalizeCounterfactuals(safeOutput.counterfactuals, decision, snapshot)) {
      storedCounterfactuals.push(this.store.insertRouteCounterfactual({ ...cf, agentId, routeTeacherRunId: run.id, routeDecisionId: decision.id }));
      counterfactuals += 1;
    }

    const storedLessons = [];
    let examples = 0;
    for (const lesson of normalizeLessons(safeOutput.lessons, decision, snapshot, run)) {
      storedLessons.push(this.store.insertRouteTrainingExampleV2({ ...lesson, agentId, routeDecisionId: decision.id, routeTeacherRunId: run.id }));
      examples += 1;
    }

    const ingest = ingestRouteLearningArtifactsV3(
      this.store,
      agentId,
      decision,
      this.store.getRouteFrame(decision.routeFrameId || ''),
      run,
      storedCounterfactuals,
      storedLessons,
      this.config,
    );

    return { teacherRuns: 1, counterfactuals, examples, routeFramesV3: ingest.frameId ? 1 : 0, pairExamplesV3: ingest.pairExamples };
  }
}

function teacherInput(decision: RouteDecision, snapshot: RouteGraphSnapshot) {
  return {
    routeDecisionId: decision.id,
    turnFrame: redactJsonValue(decision.turnFrame),
    actualRoute: decision.route,
    actualOutcome: decision.outcome,
    reward: decision.reward,
    selectedMemoryIds: decision.selectedMemoryIds,
    omittedMemoryIds: decision.omittedMemoryIds,
    retrievalPlan: redactJsonValue(decision.retrievalPlan),
    injectionPlan: redactJsonValue(decision.injectionPlan),
    latencyTier: decision.latencyTier,
    syncLlmUsed: decision.syncLlmUsed,
    graphSnapshot: redactJsonValue(snapshot),
  };
}

function deterministicTeacherOutput(decision: RouteDecision, snapshot: RouteGraphSnapshot): any {
  const selected = new Set(decision.selectedMemoryIds || []);
  const available = snapshot.candidateSummaries || [];
  const availableUseful = available.filter((candidate) => candidate.score >= 0.45 || candidate.type === 'correction' || candidate.type === 'workflow');
  const topAlternate = availableUseful.find((candidate) => !selected.has(candidate.id));
  const hadInjection = selected.size > 0;
  const negative = (decision.reward || 0) < 0;
  const positive = (decision.reward || 0) > 0;
  let verdict: RouteTeacherVerdict = 'unknown';
  let teacherRoute: RouteKind = decision.route;
  let teacherMemoryIds = [...selected];
  let confidence = Math.max(0.55, Math.min(0.92, Math.abs(decision.reward || 0) || decision.confidence || 0.55));
  let rationale = 'No strong route learning signal; keep deterministic routing conservative.';

  if (!hadInjection && topAlternate && negative) {
    verdict = 'missed_recall';
    teacherRoute = 'retrieve_memory';
    teacherMemoryIds = [topAlternate.id];
    confidence = 0.82;
    rationale = `A ${topAlternate.type} memory was available but not injected before a negative outcome.`;
  } else if (hadInjection && positive) {
    verdict = 'correct_route';
    teacherRoute = decision.route === 'no_memory' ? 'retrieve_memory' : decision.route;
    confidence = 0.84;
    rationale = 'Injected memory aligned with a positive observed outcome.';
  } else if (hadInjection && negative) {
    verdict = 'over_injected';
    teacherRoute = topAlternate ? 'retrieve_memory' : 'no_memory';
    teacherMemoryIds = topAlternate ? [topAlternate.id] : [];
    confidence = 0.78;
    rationale = topAlternate ? 'Injected memory failed; a different graph candidate may be better.' : 'Injected memory correlated with a negative outcome; prefer silence for this shape.';
  } else if (!hadInjection && !topAlternate && decision.outcome === 'no_signal') {
    verdict = 'should_stay_silent';
    teacherRoute = 'no_memory';
    teacherMemoryIds = [];
    confidence = 0.76;
    rationale = 'No useful memory candidates were available and no later signal justified retrieval.';
  } else if (!hadInjection && topAlternate) {
    verdict = 'missed_recall';
    teacherRoute = 'retrieve_memory';
    teacherMemoryIds = [topAlternate.id];
    confidence = 0.68;
    rationale = `A plausible ${topAlternate.type} memory was available for this turn shape.`;
  }

  const teacherMemoryTypes = [...new Set(teacherMemoryIds.map((id) => available.find((candidate) => candidate.id === id)?.type).filter(Boolean))];
  const teacherQueries = decision.retrievalPlan?.queries?.length ? decision.retrievalPlan.queries : queryTemplatesFor(decision.turnFrame.taskType, teacherMemoryTypes as MemoryType[]);
  const teacherGraphDepth = clampGraphDepth(Math.max(Number(decision.retrievalPlan?.graphDepth ?? 0), topAlternate?.graphDistance ?? 0));

  return {
    verdict,
    teacherRoute,
    teacherMemoryIds,
    teacherQueries,
    teacherGraphDepth,
    syncPlannerWorthIt: decision.latencyTier === 'sync_memory_planner' ? positive : verdict === 'missed_recall' && available.length > 4,
    confidence,
    rationale,
    counterfactuals: buildCounterfactuals(decision, snapshot, teacherMemoryIds, teacherMemoryTypes as MemoryType[], verdict, confidence),
    lessons: [lessonFor(decision, verdict, teacherRoute, teacherMemoryTypes as MemoryType[], teacherQueries, teacherGraphDepth, confidence)],
  };
}

function buildCounterfactuals(decision: RouteDecision, snapshot: RouteGraphSnapshot, teacherMemoryIds: string[], teacherMemoryTypes: MemoryType[], verdict: RouteTeacherVerdict, confidence: number) {
  const cfs: any[] = [
    {
      kind: 'no_memory',
      memoryIds: [],
      memoryTypes: [],
      graphDepth: 0,
      estimatedOutcome: decision.selectedMemoryIds.length === 0 ? 'likely_neutral' : (decision.reward > 0 ? 'likely_missed' : 'likely_neutral'),
      confidence: 0.65,
      rationale: 'Baseline without memory for this turn shape.',
    },
  ];
  if (decision.selectedMemoryIds.length > 0) {
    cfs.push({
      kind: 'actual_injection',
      memoryIds: decision.selectedMemoryIds,
      memoryTypes: memoryTypesForIds(snapshot, decision.selectedMemoryIds),
      graphDepth: clampGraphDepth(decision.retrievalPlan?.graphDepth ?? 0),
      estimatedOutcome: decision.reward > 0 ? 'likely_helpful' : decision.reward < 0 ? 'likely_noise' : 'unknown',
      confidence: Math.max(0.55, Math.min(0.9, Math.abs(decision.reward || 0) || confidence)),
      rationale: 'Observed route and injection path.',
    });
  }
  if (teacherMemoryIds.length > 0) {
    cfs.push({
      kind: 'top_k_alternate',
      memoryIds: teacherMemoryIds,
      memoryTypes: teacherMemoryTypes,
      graphDepth: clampGraphDepth(Math.max(0, ...teacherMemoryIds.map((id) => snapshot.candidateSummaries.find((candidate) => candidate.id === id)?.graphDistance ?? 0))),
      estimatedOutcome: verdict === 'missed_recall' || verdict === 'correct_route' ? 'likely_helpful' : 'likely_neutral',
      confidence,
      rationale: 'Best graph-grounded alternate selected by route teacher.',
    });
  }
  if (verdict === 'should_stay_silent' || verdict === 'over_injected') {
    cfs.push({
      kind: 'stay_silent',
      memoryIds: [],
      memoryTypes: [],
      graphDepth: 0,
      estimatedOutcome: verdict === 'should_stay_silent' ? 'likely_helpful' : 'likely_neutral',
      confidence,
      rationale: 'Silence is preferable for this route shape.',
    });
  }
  if (teacherMemoryTypes.includes('correction')) {
    cfs.push({ kind: 'correction_only', memoryIds: teacherMemoryIds, memoryTypes: ['correction'], graphDepth: 1, estimatedOutcome: 'likely_helpful', confidence, rationale: 'Correction memory directly changes future behavior.' });
  }
  if (teacherMemoryTypes.includes('workflow')) {
    cfs.push({ kind: 'workflow_only', memoryIds: teacherMemoryIds, memoryTypes: ['workflow'], graphDepth: 1, estimatedOutcome: 'likely_helpful', confidence, rationale: 'Workflow memory can guide tool/action sequence.' });
  }
  return cfs;
}

function lessonFor(decision: RouteDecision, verdict: RouteTeacherVerdict, route: RouteKind, memoryTypes: MemoryType[], queries: string[], graphDepth: 0 | 1 | 2, confidence: number): any {
  const taskType = decision.turnFrame.taskType;
  const turnSignals = signalsForTurn(decision);
  let kind: RouteTrainingExampleKind = 'prefer_route';
  if (verdict === 'should_stay_silent') kind = 'correct_silence';
  else if (verdict === 'over_injected' || verdict === 'latency_waste') kind = 'avoid_route';
  else if (verdict === 'missed_recall') kind = 'missed_recall';
  return {
    exampleKind: kind,
    taskType,
    turnSignals,
    route,
    memoryTypes,
    queryTemplates: queries.length ? queries.slice(0, 5) : queryTemplatesFor(taskType, memoryTypes),
    graphDepth,
    confidence,
    supportCount: kind === 'avoid_route' ? 0 : 1,
    harmCount: kind === 'avoid_route' ? 1 : 0,
    source: 'teacher',
    evidenceIds: [decision.id],
  };
}

function normalizeLessons(rawLessons: any[], decision: RouteDecision, snapshot: RouteGraphSnapshot, run: RouteTeacherRun) {
  const lessons = Array.isArray(rawLessons) && rawLessons.length ? rawLessons : [lessonFor(decision, run.verdict, run.teacherRoute, memoryTypesForIds(snapshot, run.teacherMemoryIds), run.teacherQueries, run.teacherGraphDepth, run.confidence)];
  return lessons.map((lesson) => {
    const memoryTypes = sanitizeMemoryTypes(lesson.memoryTypes ?? memoryTypesForIds(snapshot, run.teacherMemoryIds));
    return {
      exampleKind: sanitizeExampleKind(lesson.kind ?? lesson.exampleKind, run.verdict),
      taskType: sanitizeTaskType(lesson.taskType ?? decision.turnFrame.taskType),
      turnSignals: sanitizeStrings(lesson.turnSignals ?? signalsForTurn(decision), 8),
      route: sanitizeRoute(lesson.route ?? run.teacherRoute),
      memoryTypes,
      queryTemplates: sanitizeStrings(lesson.queryTemplates ?? lesson.queries ?? run.teacherQueries ?? queryTemplatesFor(decision.turnFrame.taskType, memoryTypes), 8),
      graphDepth: clampGraphDepth(lesson.graphDepth ?? run.teacherGraphDepth ?? 0),
      confidence: clamp01(lesson.confidence ?? run.confidence),
      supportCount: Number(lesson.supportCount ?? (run.verdict === 'over_injected' ? 0 : 1)),
      harmCount: Number(lesson.harmCount ?? (run.verdict === 'over_injected' ? 1 : 0)),
      source: 'teacher' as const,
      evidenceIds: sanitizeStrings(lesson.evidenceIds ?? [decision.id, run.id], 20),
    };
  }).filter((lesson) => lesson.confidence >= 0.5);
}

function normalizeCounterfactuals(rawCounterfactuals: any[], decision: RouteDecision, snapshot: RouteGraphSnapshot) {
  const cfs = Array.isArray(rawCounterfactuals) && rawCounterfactuals.length
    ? rawCounterfactuals
    : buildCounterfactuals(decision, snapshot, [], [], 'unknown', 0.55);
  const validIds = new Set(snapshot.candidateMemoryIds);
  return cfs.map((cf) => {
    const ids = sanitizeStrings(cf.memoryIds ?? [], 20).filter((id) => validIds.has(id) || decision.selectedMemoryIds.includes(id));
    return {
      kind: sanitizeCounterfactualKind(cf.kind),
      memoryIds: ids,
      memoryTypes: sanitizeMemoryTypes(cf.memoryTypes ?? memoryTypesForIds(snapshot, ids)),
      graphDepth: clampGraphDepth(cf.graphDepth ?? 0),
      estimatedOutcome: sanitizeEstimatedOutcome(cf.estimatedOutcome),
      confidence: clamp01(cf.confidence ?? 0.5),
      rationale: redactText(String(cf.rationale || 'No rationale supplied.'), 500),
    };
  }).filter((cf) => cf.confidence >= 0.4);
}

function validateTeacherOutput(output: any, snapshot: RouteGraphSnapshot, decision: RouteDecision) {
  if (!output || typeof output !== 'object') return { valid: false, reason: 'teacher_output_not_object' };
  if (!ROUTES.includes(output.teacherRoute)) return { valid: false, reason: 'unsupported_teacher_route' };
  if (!output.verdict) return { valid: false, reason: 'missing_verdict' };
  if (typeof output.confidence !== 'number' || output.confidence < 0 || output.confidence > 1) return { valid: false, reason: 'bad_confidence' };
  const validIds = new Set([...snapshot.candidateMemoryIds, ...decision.selectedMemoryIds]);
  for (const id of output.teacherMemoryIds ?? []) {
    if (!validIds.has(id)) return { valid: false, reason: `unknown_memory_id:${id}` };
  }
  if (Number(output.teacherGraphDepth ?? 0) > 2) return { valid: false, reason: 'graph_depth_too_high' };
  return { valid: true, reason: '' };
}

function parseTeacherOutput(raw: any) {
  if (typeof raw === 'string') {
    try { return JSON.parse(raw); } catch { return null; }
  }
  return raw;
}

function fallbackGraphSnapshot(store: MemoryStore, agentId: string, decision: RouteDecision): RouteGraphSnapshot {
  const memories = [...new Set([...decision.selectedMemoryIds, ...decision.omittedMemoryIds])]
    .map((id) => store.getMemory(id))
    .filter(Boolean) as MemoryNode[];
  const summaries = memories.map((memory) => memoryToCandidateSummary(store, memory, 0));
  return store.insertRouteGraphSnapshot({
    agentId,
    routeDecisionId: decision.id,
    querySet: decision.retrievalPlan.queries,
    candidateMemoryIds: summaries.map((summary) => summary.id),
    candidateSummaries: summaries,
    graphStats: { nodeCountSeen: summaries.length, edgeCountSeen: 0, maxDepth: clampGraphDepth(decision.retrievalPlan.graphDepth ?? 0) },
  });
}

export function buildRouteGraphSnapshot(store: MemoryStore, agentId: string, routeDecisionId: string, queries: string[], candidates: MemoryNode[], graphDepth: 0 | 1 | 2): RouteGraphSnapshot {
  const summaries: RouteGraphSnapshot['candidateSummaries'] = [];
  const seen = new Set<string>();
  for (const candidate of candidates) {
    if (seen.has(candidate.id)) continue;
    seen.add(candidate.id);
    summaries.push(memoryToCandidateSummary(store, candidate, 0));
    if (graphDepth > 0) {
      for (const linked of store.getConnectedMemories(candidate.id, graphDepth, agentId)) {
        if (seen.has(linked.id)) continue;
        seen.add(linked.id);
        summaries.push(memoryToCandidateSummary(store, linked, 1));
      }
    }
  }
  return store.insertRouteGraphSnapshot({
    agentId,
    routeDecisionId,
    querySet: queries,
    candidateMemoryIds: summaries.map((summary) => summary.id),
    candidateSummaries: summaries,
    graphStats: {
      nodeCountSeen: summaries.length,
      edgeCountSeen: summaries.reduce((sum, summary) => sum + summary.linkedMemoryIds.length, 0),
      maxDepth: graphDepth,
    },
  });
}

function memoryToCandidateSummary(store: MemoryStore, memory: MemoryNode, graphDistance: number) {
  const edges = store.getEdges(memory.id);
  return {
    id: memory.id,
    type: memory.type,
    scope: `${memory.scopeKind}:${memory.scopeKey ?? ''}`,
    redactedContent: redactText(memory.content, 280),
    score: Number((memory.importance * 0.4 + memory.confidence * 0.4 + memory.freshness * 0.2).toFixed(3)),
    freshness: memory.freshness,
    graphDistance,
    linkedMemoryIds: edges.map((edge) => edge.fromId === memory.id ? edge.toId : edge.fromId).filter(Boolean).slice(0, 20),
  };
}

function memoryTypesForIds(snapshot: RouteGraphSnapshot, ids: string[]): MemoryType[] {
  return [...new Set(ids.map((id) => snapshot.candidateSummaries.find((candidate) => candidate.id === id)?.type).filter(Boolean) as MemoryType[])];
}

function queryTemplatesFor(taskType: TaskType, memoryTypes: MemoryType[]) {
  const queries = new Set<string>();
  if (taskType === 'coding' || memoryTypes.includes('workflow')) queries.add('repo package manager test workflow build setup');
  if (taskType === 'writing' || memoryTypes.includes('preference')) queries.add('writing style tone preference');
  if (memoryTypes.includes('correction')) queries.add('durable correction do instead');
  if (memoryTypes.includes('context')) queries.add('project context routing guidance');
  if (queries.size === 0) queries.add(`${taskType} memory preference workflow context`);
  return [...queries];
}

function signalsForTurn(decision: RouteDecision) {
  const text = `${decision.turnFrame.summary} ${decision.turnFrame.userGoal}`.toLowerCase();
  const signals = new Set<string>();
  for (const token of ['test', 'build', 'install', 'dependency', 'write', 'draft', 'plan', 'debug', 'thanks', 'ok', 'remember', 'actually', 'instead']) {
    if (text.includes(token)) signals.add(token);
  }
  if (decision.turnFrame.routeHints.likelyNeedsWorkflow) signals.add('workflow');
  if (decision.turnFrame.routeHints.likelyNeedsPreferences) signals.add('preference');
  if (decision.turnFrame.routeHints.likelyNeedsCorrections) signals.add('correction');
  if (decision.turnFrame.routeHints.likelyNeedsProjectContext) signals.add('project_context');
  return [...signals].slice(0, 8);
}

function sanitizeRoute(value: any): RouteKind {
  return ROUTES.includes(value) ? value : 'retrieve_memory';
}

function sanitizeTaskType(value: any): TaskType {
  const allowed: TaskType[] = ['coding', 'planning', 'debugging', 'writing', 'preference_update', 'correction', 'general_question', 'other'];
  return allowed.includes(value) ? value : 'other';
}

function sanitizeMemoryTypes(values: any): MemoryType[] {
  return [...new Set((Array.isArray(values) ? values : []).filter((value) => MEMORY_TYPES.includes(value)))].slice(0, 8);
}

function sanitizeStrings(values: any, limit: number): string[] {
  return (Array.isArray(values) ? values : [])
    .map((value) => redactText(String(value || '').trim(), 180))
    .filter(Boolean)
    .slice(0, limit);
}

function sanitizeExampleKind(value: any, verdict: RouteTeacherVerdict): RouteTrainingExampleKind {
  const allowed: RouteTrainingExampleKind[] = ['prefer_route', 'avoid_route', 'missed_recall', 'correct_silence', 'avoid_sync_planner', 'prefer_sync_planner', 'prefer_memory_type', 'avoid_memory_type', 'prefer_graph_depth', 'avoid_graph_depth'];
  if (allowed.includes(value)) return value;
  if (verdict === 'should_stay_silent') return 'correct_silence';
  if (verdict === 'missed_recall') return 'missed_recall';
  if (verdict === 'over_injected' || verdict === 'latency_waste') return 'avoid_route';
  return 'prefer_route';
}

function sanitizeCounterfactualKind(value: any): RouteCounterfactualKind {
  const allowed: RouteCounterfactualKind[] = ['no_memory', 'actual_injection', 'top_k_alternate', 'broader_graph', 'correction_only', 'workflow_only', 'preference_only', 'context_only', 'stay_silent', 'sync_planner'];
  return allowed.includes(value) ? value : 'top_k_alternate';
}

function sanitizeEstimatedOutcome(value: any): RouteCounterfactual['estimatedOutcome'] {
  const allowed: RouteCounterfactual['estimatedOutcome'][] = ['likely_helpful', 'likely_neutral', 'likely_noise', 'likely_harmful', 'likely_missed', 'unknown'];
  return allowed.includes(value) ? value : 'unknown';
}

function clampGraphDepth(value: any): 0 | 1 | 2 {
  const n = Math.max(0, Math.min(2, Number(value || 0)));
  return n >= 2 ? 2 : n >= 1 ? 1 : 0;
}

function clamp01(value: any) {
  return Math.max(0, Math.min(1, Number(value || 0)));
}

function routeTeacherSystemPrompt() {
  return [
    'You are the OpenClawBrain route teacher.',
    'Judge whether a runtime route_fn decision should have retrieved memory, stayed silent, used a wider graph, changed memory types, or used a sync planner.',
    'Use only the provided redacted turn frame and graph snapshot. Do not invent memory ids.',
    'Return strict JSON only. LLM decides semantic meaning; code enforces trust boundaries.',
  ].join('\n');
}

function routeTeacherSchema() {
  return {
    type: 'object',
    required: ['verdict', 'teacherRoute', 'teacherMemoryIds', 'teacherQueries', 'teacherGraphDepth', 'syncPlannerWorthIt', 'confidence', 'rationale', 'counterfactuals', 'lessons'],
  };
}
