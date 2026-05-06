import type {
  MemoryType,
  RouteKind,
  RoutePolicyRuleV2,
  RoutePolicySnapshotV2,
  RouteTrainingExampleV2,
  TaskType,
  TurnFrame,
} from './memory-types.js';
import { hashText, redactText } from './redact.js';

const ROUTES: RouteKind[] = ['no_memory', 'capture_only', 'retrieve_memory', 'retrieve_and_distill', 'high_confidence_correction_only'];
const RETRIEVAL_ROUTES = new Set<RouteKind>(['retrieve_memory', 'retrieve_and_distill', 'high_confidence_correction_only']);
const MEMORY_TYPES: MemoryType[] = ['correction', 'preference', 'workflow', 'project_fact', 'tool_convention', 'routing_rule', 'agent_assignment', 'recall_rule', 'outcome', 'context'];
const POSITIVE_KINDS = new Set(['prefer_route', 'missed_recall', 'prefer_memory_type', 'prefer_graph_depth', 'prefer_sync_planner']);
const SUPPRESSION_KINDS = new Set(['correct_silence', 'avoid_route', 'avoid_memory_type', 'avoid_graph_depth', 'avoid_sync_planner']);

export interface RoutePolicyValidationReport {
  ok: boolean;
  status: 'active' | 'shadow' | 'rejected';
  reason: string;
  errors: string[];
  warnings: string[];
  projectedSyncPlannerRate: number;
  noisyInjectionRate: number;
  harmRate: number;
}

export interface RoutePolicyMatchResult {
  matched: boolean;
  rule?: RoutePolicyRuleV2;
  score: number;
  reasonCode: string;
}

export interface RoutePolicyDistillationReport {
  snapshot?: RoutePolicySnapshotV2;
  validation?: RoutePolicyValidationReport;
  examplesConsidered: number;
  rulesGenerated: number;
}

export function maybeDistillAndStorePolicyV2(store: any, agentId: string, config: any): RoutePolicyDistillationReport {
  if (config.routeLearning?.policyV2?.enabled === false) {
    return { examplesConsidered: 0, rulesGenerated: 0 };
  }
  const examples = store.listRouteTrainingExamplesV2(agentId, 300) as RouteTrainingExampleV2[];
  const minExamples = Number(config.routeLearning?.policyV2?.minExamples ?? 3);
  const existing = store.getActivePolicySnapshotV2(agentId) as RoutePolicySnapshotV2 | null;
  if (examples.length < minExamples) {
    return { snapshot: existing ?? undefined, examplesConsidered: examples.length, rulesGenerated: existing?.rules.length ?? 0 };
  }

  const rules = distillPolicyRulesV2(examples, config);
  if (rules.length === 0) {
    const rejected = store.insertPolicySnapshotV2(buildSnapshot(agentId, [], examples, config, 'rejected', {
      cases: examples.length,
      wins: 0,
      ties: 0,
      misses: examples.filter((ex) => ex.exampleKind === 'missed_recall').length,
      noisyInjections: examples.filter((ex) => ex.exampleKind === 'avoid_route' || ex.exampleKind === 'avoid_memory_type').length,
      harms: examples.reduce((sum, ex) => sum + ex.harmCount, 0),
      p95LatencyMs: 0,
      activationDecision: 'rejected',
      activationStatusReason: 'no_valid_rules',
      validationErrors: ['no_valid_rules'],
    } as any));
    return { snapshot: rejected, examplesConsidered: examples.length, rulesGenerated: 0, validation: validatePolicySnapshotV2(rejected, config, existing) };
  }

  const draft = buildSnapshot(agentId, rules, examples, config, 'candidate');
  const validation = validatePolicySnapshotV2(draft, config, existing);
  const identical = existing && stablePolicyBody(existing) === stablePolicyBody(draft);
  if (identical) return { snapshot: existing, validation, examplesConsidered: examples.length, rulesGenerated: rules.length };

  const finalStatus = validation.ok
    ? (config.routeLearning?.policyV2?.shadowBeforeActivate === true ? 'shadow' : 'active')
    : 'rejected';
  const stored = store.insertPolicySnapshotV2({
    ...draft,
    status: finalStatus,
    evalSummary: {
      ...(draft.evalSummary ?? {}),
      activationDecision: finalStatus,
      activationStatusReason: validation.reason,
      validationErrors: validation.errors,
      validationWarnings: validation.warnings,
      projectedSyncPlannerRate: validation.projectedSyncPlannerRate,
      noisyInjectionRate: validation.noisyInjectionRate,
      harmRate: validation.harmRate,
    } as any,
  });
  return { snapshot: stored, validation: { ...validation, status: finalStatus }, examplesConsidered: examples.length, rulesGenerated: rules.length };
}

export function distillPolicyRulesV2(examples: RouteTrainingExampleV2[], config: any): RoutePolicyRuleV2[] {
  const minConfidence = Number(config.routeLearning?.policyV2?.minRuleConfidence ?? 0.6);
  const minSupport = Number(config.routeLearning?.policyV2?.minSupport ?? 1);
  const groups = new Map<string, RouteTrainingExampleV2[]>();
  for (const raw of examples) {
    const example = canonicalExample(raw, config);
    if (example.confidence < 0.5) continue;
    const sortedTypes = [...example.memoryTypes].sort().join(',');
    const key = `${example.exampleKind}:${example.taskType}:${example.route}:${sortedTypes}:${example.graphDepth}`;
    const group = groups.get(key) ?? [];
    group.push(example);
    groups.set(key, group);
  }

  const rules: RoutePolicyRuleV2[] = [];
  for (const [key, group] of groups.entries()) {
    const [kind, taskType, route] = key.split(':');
    const support = group.reduce((sum, ex) => sum + Math.max(0, Number(ex.supportCount || 0)), 0);
    const harm = group.reduce((sum, ex) => sum + Math.max(0, Number(ex.harmCount || 0)), 0);
    const confidence = clamp01(Math.max(...group.map((ex) => ex.confidence)) + support * 0.025 - harm * 0.1);
    const harmRate = harm / Math.max(1, support + harm);
    if (confidence < minConfidence) continue;
    if (support < minSupport && !SUPPRESSION_KINDS.has(kind)) continue;
    if (harmRate > Number(config.routeLearning?.policyV2?.maxRuleHarmRate ?? 0.34) && !SUPPRESSION_KINDS.has(kind)) continue;

    const positive = POSITIVE_KINDS.has(kind);
    const suppression = SUPPRESSION_KINDS.has(kind);
    const memoryTypes = positive ? unique(group.flatMap((ex) => ex.memoryTypes)).slice(0, 6) : [];
    const queries = positive ? unique(group.flatMap((ex) => ex.queryTemplates).map((value) => redactText(value, 120))).slice(0, 8) : [];
    const turnSignals = stableSignals(group.flatMap((ex) => ex.turnSignals)).slice(0, 8);
    const evidenceIds = unique(group.flatMap((ex) => ex.evidenceIds)).slice(0, 30);
    const effectiveRoute = suppression ? 'no_memory' : sanitizeRoute(route);
    const graphDepth = suppression ? 0 : clampGraphDepth(Math.max(...group.map((ex) => ex.graphDepth)), config);
    const syncPlanner = kind === 'prefer_sync_planner'
      ? 'allowed'
      : kind === 'avoid_sync_planner' || suppression
        ? 'no'
        : 'never_unless_ambiguous';

    const rule: RoutePolicyRuleV2 = {
      id: hashText(`${key}:${support}:${harm}:${turnSignals.join('|')}:${queries.join('|')}`).slice(7, 23),
      priority: suppression ? 100 : 50,
      match: { taskType: taskType as TaskType, turnSignals },
      route: effectiveRoute,
      memoryTypes,
      queries,
      graphDepth,
      syncPlanner,
      confidence,
      evidenceIds,
      stats: { support, harm, harmRate, kind },
      reason: suppression ? `suppression_from_${kind}` : `distilled_from_${kind}`,
    } as RoutePolicyRuleV2;
    rules.push(rule);
  }

  return rules
    .filter((rule) => validateRuleShape(rule, config).length === 0)
    .sort((a: any, b: any) => Number(b.priority || 0) - Number(a.priority || 0) || b.confidence - a.confidence)
    .slice(0, Math.max(3, Number(config.routeLearning?.policyV2?.maxRules ?? 25)));
}

export function validatePolicySnapshotV2(snapshot: Partial<RoutePolicySnapshotV2>, config: any = {}, existing?: RoutePolicySnapshotV2 | null): RoutePolicyValidationReport {
  const errors: string[] = [];
  const warnings: string[] = [];
  if (snapshot.version !== 'route-policy-v2') errors.push('unsupported_policy_version');
  const rules = Array.isArray(snapshot.rules) ? snapshot.rules : [];
  if (rules.length === 0) errors.push('no_rules');
  for (const rule of rules) errors.push(...validateRuleShape(rule, config));

  const syncRules = rules.filter((rule) => rule.syncPlanner === 'allowed' || rule.syncPlanner === 'prefer').length;
  const projectedSyncPlannerRate = rules.length ? syncRules / rules.length : 0;
  const maxSyncPlannerRate = Number(snapshot.globalBudgets?.maxSyncPlannerRate ?? config.routeLearning?.policyV2?.maxSyncPlannerRate ?? 0.05);
  if (projectedSyncPlannerRate > maxSyncPlannerRate) errors.push(`sync_planner_rate_exceeds_budget:${projectedSyncPlannerRate.toFixed(3)}>${maxSyncPlannerRate}`);

  const evalSummary: any = snapshot.evalSummary ?? {};
  const cases = Number(evalSummary.cases || 0);
  const noisyInjectionRate = cases ? Number(evalSummary.noisyInjections || 0) / cases : 0;
  const harmRate = cases ? Number(evalSummary.harms || 0) / cases : 0;
  const maxNoisyRate = Number(config.routeLearning?.policyV2?.maxNoisyInjectionRate ?? 0.05);
  const maxHarmRate = Number(config.routeLearning?.policyV2?.maxHarmRate ?? 0.1);
  if (noisyInjectionRate > maxNoisyRate) errors.push(`noisy_injection_rate_exceeds_gate:${noisyInjectionRate.toFixed(3)}>${maxNoisyRate}`);
  if (harmRate > maxHarmRate) errors.push(`harm_rate_exceeds_gate:${harmRate.toFixed(3)}>${maxHarmRate}`);

  if (existing?.evalSummary) {
    const oldHarms = Number((existing.evalSummary as any).harms || 0) / Math.max(1, Number((existing.evalSummary as any).cases || 0));
    if (harmRate > oldHarms && Number(evalSummary.wins || 0) <= 0) {
      errors.push('candidate_harm_rate_worse_without_win');
    }
  }

  const reason = errors.length === 0 ? 'passed_activation_gates' : errors[0];
  return { ok: errors.length === 0, status: errors.length === 0 ? 'active' : 'rejected', reason, errors, warnings, projectedSyncPlannerRate, noisyInjectionRate, harmRate };
}

export function scorePolicySnapshotV2(snapshot: RoutePolicySnapshotV2 | null | undefined, turnFrame: TurnFrame, message = ''): RoutePolicyMatchResult {
  if (!snapshot || snapshot.version !== 'route-policy-v2' || snapshot.status !== 'active' || !Array.isArray(snapshot.rules)) {
    return { matched: false, score: 0, reasonCode: 'no_active_policy_v2' };
  }
  const haystack = `${message} ${turnFrame.summary} ${turnFrame.userGoal} ${turnFrame.impliedNeeds.join(' ')} ${turnFrame.memoryQuestions.join(' ')}`.toLowerCase();
  let best: RoutePolicyMatchResult = { matched: false, score: 0, reasonCode: 'no_matching_policy_rule' };
  for (const rule of snapshot.rules) {
    const match = scoreRule(rule, turnFrame, haystack);
    if (!match.matched) continue;
    if (match.score > best.score) best = match;
  }
  return best;
}

function scoreRule(rule: RoutePolicyRuleV2, turnFrame: TurnFrame, haystack: string): RoutePolicyMatchResult {
  const match = rule.match ?? {};
  if (match.taskType && match.taskType !== turnFrame.taskType) return { matched: false, rule, score: 0, reasonCode: 'task_type_mismatch' };
  const signals = Array.isArray(match.turnSignals) ? match.turnSignals.map((value) => String(value).toLowerCase()).filter(Boolean) : [];
  const overlap = signals.filter((signal) => haystack.includes(signal)).length;
  if (signals.length > 0 && overlap === 0) return { matched: false, rule, score: 0, reasonCode: 'turn_signal_mismatch' };
  const priority = Number((rule as any).priority || 0) / 200;
  const support = Number((rule as any).stats?.support || 0);
  const supportBonus = Math.min(0.1, support * 0.01);
  const signalBonus = signals.length ? Math.min(0.12, overlap * 0.03) : 0;
  const routeHintBonus = routeHintCompatibility(rule, turnFrame);
  const score = clamp01(Number(rule.confidence || 0) + priority + supportBonus + signalBonus + routeHintBonus);
  return { matched: true, rule, score, reasonCode: `policy_rule:${rule.id}` };
}

function routeHintCompatibility(rule: RoutePolicyRuleV2, turnFrame: TurnFrame) {
  if (!RETRIEVAL_ROUTES.has(rule.route)) return 0;
  let bonus = 0;
  if (rule.memoryTypes.includes('correction') && turnFrame.routeHints.likelyNeedsCorrections) bonus += 0.03;
  if (rule.memoryTypes.includes('preference') && turnFrame.routeHints.likelyNeedsPreferences) bonus += 0.03;
  if (rule.memoryTypes.includes('workflow') && turnFrame.routeHints.likelyNeedsWorkflow) bonus += 0.03;
  if ((rule.memoryTypes.includes('project_fact') || rule.memoryTypes.includes('context')) && turnFrame.routeHints.likelyNeedsProjectContext) bonus += 0.03;
  return bonus;
}

function buildSnapshot(agentId: string, rules: RoutePolicyRuleV2[], examples: RouteTrainingExampleV2[], config: any, status: 'candidate' | 'active' | 'shadow' | 'rejected', evalOverride?: any): Omit<RoutePolicySnapshotV2, 'id' | 'createdAt'> {
  const cases = examples.length;
  const noisyInjections = examples.filter((ex) => ex.exampleKind === 'avoid_route' || ex.exampleKind === 'avoid_memory_type').length;
  const harms = examples.reduce((sum, ex) => sum + Math.max(0, Number(ex.harmCount || 0)), 0);
  return {
    agentId,
    version: 'route-policy-v2',
    status,
    rules,
    globalBudgets: {
      maxSyncPlannerRate: Number(config.routeLearning?.policyV2?.maxSyncPlannerRate ?? 0.05),
      maxInjectedMemories: Number(config.routing?.maxInjectedMemories ?? 8),
      maxInjectedChars: Number(config.routing?.maxInjectedChars ?? 2500),
      defaultGraphDepth: clampGraphDepth(config.routeLearning?.counterfactuals?.maxGraphDepth ?? 1, config),
    },
    evalSummary: evalOverride ?? ({
      cases,
      wins: examples.filter((ex) => ex.exampleKind === 'prefer_route' || ex.exampleKind === 'missed_recall' || ex.exampleKind === 'prefer_memory_type' || ex.exampleKind === 'prefer_graph_depth').length,
      ties: examples.filter((ex) => ex.exampleKind === 'correct_silence').length,
      misses: examples.filter((ex) => ex.exampleKind === 'missed_recall').length,
      noisyInjections,
      harms,
      p95LatencyMs: 0,
    } as any),
    exampleIds: examples.map((ex) => ex.id).slice(0, 150),
    model: config.llm?.learningModel || 'deterministic-route-policy-v2',
    promptVersion: 'route-policy-v2-distiller-v2',
  };
}

function validateRuleShape(rule: any, config: any) {
  const errors: string[] = [];
  if (!rule || typeof rule !== 'object') return ['rule_not_object'];
  if (!rule.id) errors.push('rule_missing_id');
  if (!ROUTES.includes(rule.route)) errors.push(`unsupported_route:${rule.route}`);
  if (typeof rule.confidence !== 'number' || rule.confidence < 0 || rule.confidence > 1) errors.push(`bad_rule_confidence:${rule.id}`);
  const graphDepth = Number(rule.graphDepth ?? 0);
  const maxGraphDepth = Number(config.routeLearning?.counterfactuals?.maxGraphDepth ?? 2);
  if (graphDepth < 0 || graphDepth > maxGraphDepth || graphDepth > 2) errors.push(`graph_depth_out_of_bounds:${rule.id}`);
  for (const type of rule.memoryTypes ?? []) {
    if (!MEMORY_TYPES.includes(type)) errors.push(`unknown_memory_type:${type}`);
  }
  const signals = Array.isArray(rule.match?.turnSignals) ? rule.match.turnSignals : [];
  if (rule.route !== 'no_memory' && !rule.match?.taskType && signals.length === 0) errors.push(`broad_retrieval_rule:${rule.id}`);
  if (RETRIEVAL_ROUTES.has(rule.route) && (!Array.isArray(rule.memoryTypes) || rule.memoryTypes.length === 0) && (!Array.isArray(rule.queries) || rule.queries.length === 0)) {
    errors.push(`retrieval_rule_without_types_or_queries:${rule.id}`);
  }
  if (!['no', 'never_unless_ambiguous', 'allowed', 'prefer'].includes(rule.syncPlanner)) errors.push(`bad_sync_planner:${rule.id}`);
  return errors;
}

function canonicalExample(example: RouteTrainingExampleV2, config: any): RouteTrainingExampleV2 {
  return {
    ...example,
    confidence: clamp01(Number(example.confidence || 0)),
    memoryTypes: unique((example.memoryTypes ?? []).filter((type) => MEMORY_TYPES.includes(type))).slice(0, 8),
    queryTemplates: unique((example.queryTemplates ?? []).map((query) => redactText(query, 120)).filter(Boolean)).slice(0, 8),
    graphDepth: clampGraphDepth(example.graphDepth, config),
    turnSignals: stableSignals(example.turnSignals ?? []).slice(0, 8),
    route: sanitizeRoute(example.route),
  };
}

function stableSignals(values: string[]) {
  return unique(values.map((value) => String(value || '').toLowerCase().replace(/[^a-z0-9_-]+/g, '_').replace(/^_+|_+$/g, '')).filter((value) => value.length >= 2 && value.length <= 40));
}

function sanitizeRoute(value: any): RouteKind {
  return ROUTES.includes(value) ? value : 'retrieve_memory';
}

function clampGraphDepth(value: any, config: any): 0 | 1 | 2 {
  const max = Math.min(2, Math.max(0, Number(config.routeLearning?.counterfactuals?.maxGraphDepth ?? 2)));
  const n = Math.max(0, Math.min(max, Number(value || 0)));
  return n >= 2 ? 2 : n >= 1 ? 1 : 0;
}

function unique<T>(values: T[]): T[] {
  return [...new Set(values)];
}

function clamp01(value: number) {
  return Math.max(0, Math.min(1, value));
}

function stablePolicyBody(snapshot: RoutePolicySnapshotV2 | Omit<RoutePolicySnapshotV2, 'id' | 'createdAt'>) {
  return JSON.stringify({ rules: snapshot.rules, globalBudgets: snapshot.globalBudgets, exampleIds: snapshot.exampleIds });
}
