import type { MemoryType, RouteKind, RoutePolicyRuleV3 } from './memory-types.js';

export interface RoutePolicyV3CompactnessSummary {
  beforeMerge: number;
  afterMerge: number;
  afterPrune: number;
  duplicateGroups: number;
  mergedAway: number;
  dominatedPruned: number;
  avgSignalsPerRule: number;
  avgQueriesPerRule: number;
  maxRulesPerRoute: number;
}

const SYNONYM_MAP: Record<string, string> = {
  tests: 'test',
  testing: 'test',
  builds: 'build',
  building: 'build',
  dependencies: 'dependency',
  deps: 'dependency',
  packages: 'package',
  installation: 'install',
  setup: 'install',
  bootstrap: 'install',
  workflows: 'workflow',
  command: 'cmd',
  commands: 'cmd',
  scripts: 'script',
  corrections: 'correction',
  fixing: 'fix',
  planner: 'planning',
  architecture: 'design',
  repo: 'project',
  workspace: 'project',
};

export function normalizeQueryTemplateFamilyV3(values: any[], limit = 8): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const value of values || []) {
    const normalized = normalizeTemplateV3(String(value || ''));
    if (!normalized || seen.has(normalized)) continue;
    seen.add(normalized);
    out.push(normalized);
    if (out.length >= limit) break;
  }
  return out;
}

export function normalizeSignalFamilyV3(values: any[], limit = 10): string[] {
  const tokens = tokenizeV3(values || []);
  return [...new Set(tokens)].slice(0, limit);
}

export function canonicalActionKeyV3(rule: Pick<RoutePolicyRuleV3, 'route' | 'memoryTypes' | 'graphDepth' | 'syncPlanner' | 'queries' | 'match'>): string {
  return JSON.stringify({
    route: rule.route,
    memoryTypes: [...new Set(rule.memoryTypes || [])].sort(),
    graphDepth: Number(rule.graphDepth || 0),
    syncPlanner: String(rule.syncPlanner || 'no'),
    taskType: String(rule.match?.taskType || ''),
    projectHint: String(rule.match?.projectHint || ''),
    repoHintPresent: Boolean(rule.match?.repoHintPresent),
    queries: normalizeQueryTemplateFamilyV3(rule.queries || [], 8),
    signals: normalizeSignalFamilyV3(rule.match?.turnSignals || [], 10),
  });
}

export function mergeRuleCandidatesV3(rules: RoutePolicyRuleV3[], config: any = {}) {
  const groups = new Map<string, RoutePolicyRuleV3[]>();
  for (const rule of rules) {
    const key = canonicalActionKeyV3(rule);
    const bucket = groups.get(key) || [];
    bucket.push(rule);
    groups.set(key, bucket);
  }
  const merged = [...groups.values()].map((group) => mergeGroupV3(group, config));
  return {
    rules: merged,
    duplicateGroups: [...groups.values()].filter((group) => group.length > 1).length,
    mergedAway: Math.max(0, rules.length - merged.length),
  };
}

export function pruneDominatedRulesV3(rules: RoutePolicyRuleV3[]) {
  const kept: RoutePolicyRuleV3[] = [];
  const pruned: string[] = [];
  const ordered = [...rules].sort((a, b) => Number(b.priority || 0) - Number(a.priority || 0) || b.confidence - a.confidence);
  for (const rule of ordered) {
    const dominatedBy = kept.find((other) => dominatesRuleV3(other, rule));
    if (dominatedBy) {
      pruned.push(rule.id);
      continue;
    }
    kept.push(rule);
  }
  return { rules: kept, prunedRuleIds: pruned };
}

export function compactnessSummaryV3(beforeMergeRules: RoutePolicyRuleV3[], afterMergeRules: RoutePolicyRuleV3[], finalRules: RoutePolicyRuleV3[], duplicateGroups: number, dominatedPruned: number): RoutePolicyV3CompactnessSummary {
  const byRoute = new Map<RouteKind, number>();
  for (const rule of finalRules) {
    byRoute.set(rule.route, (byRoute.get(rule.route) || 0) + 1);
  }
  return {
    beforeMerge: beforeMergeRules.length,
    afterMerge: afterMergeRules.length,
    afterPrune: finalRules.length,
    duplicateGroups,
    mergedAway: Math.max(0, beforeMergeRules.length - afterMergeRules.length),
    dominatedPruned,
    avgSignalsPerRule: finalRules.length ? Number((finalRules.reduce((sum, rule) => sum + ((rule.match?.turnSignals || []).length), 0) / finalRules.length).toFixed(3)) : 0,
    avgQueriesPerRule: finalRules.length ? Number((finalRules.reduce((sum, rule) => sum + ((rule.queries || []).length), 0) / finalRules.length).toFixed(3)) : 0,
    maxRulesPerRoute: Math.max(0, ...byRoute.values()),
  };
}

function mergeGroupV3(group: RoutePolicyRuleV3[], config: any): RoutePolicyRuleV3 {
  const sorted = [...group].sort((a, b) => b.confidence - a.confidence || Number(b.priority || 0) - Number(a.priority || 0));
  const head = sorted[0];
  const maxSignals = Math.max(4, Number(config.routeLearning?.policyV3?.maxRuleSignals ?? 8));
  const queries = normalizeQueryTemplateFamilyV3(sorted.flatMap((rule) => rule.queries || []), 8);
  const signals = normalizeSignalFamilyV3(sorted.flatMap((rule) => rule.match?.turnSignals || []), maxSignals);
  const supports = sorted.map((rule) => Number(rule.priors?.support || 0));
  const harms = sorted.map((rule) => Number(rule.priors?.harm || 0));
  const banditCounts = sorted.map((rule) => Number(rule.priors?.banditCount || 0));
  const totalBandit = banditCounts.reduce((sum, value) => sum + value, 0);
  const representativeId = head.actionId;
  return {
    ...head,
    actionId: representativeId,
    match: {
      ...head.match,
      turnSignals: signals,
      projectHint: head.match?.projectHint || sorted.map((rule) => rule.match?.projectHint).find(Boolean),
      repoHintPresent: sorted.some((rule) => Boolean(rule.match?.repoHintPresent)) || undefined,
    },
    queries,
    confidence: Number((sorted.reduce((sum, rule) => sum + rule.confidence, 0) / Math.max(1, sorted.length)).toFixed(4)),
    evidenceIds: [...new Set(sorted.flatMap((rule) => rule.evidenceIds || []))].slice(0, 60),
    priors: {
      support: supports.reduce((sum, value) => sum + value, 0),
      harm: harms.reduce((sum, value) => sum + value, 0),
      banditCount: totalBandit,
      banditMeanReward: totalBandit > 0
        ? Number((sorted.reduce((sum, rule) => sum + (Number(rule.priors?.banditMeanReward || 0) * Number(rule.priors?.banditCount || 0)), 0) / totalBandit).toFixed(4))
        : Number((sorted.reduce((sum, rule) => sum + Number(rule.priors?.banditMeanReward || 0), 0) / Math.max(1, sorted.length)).toFixed(4)),
      pairWinRate: Number((sorted.reduce((sum, rule) => sum + Number(rule.priors?.pairWinRate || 0.5), 0) / Math.max(1, sorted.length)).toFixed(4)),
    },
    reason: sorted.length > 1
      ? `merged_rule_group:${sorted.map((rule) => rule.id).join(',')}`
      : head.reason,
  };
}

function dominatesRuleV3(dominant: RoutePolicyRuleV3, candidate: RoutePolicyRuleV3): boolean {
  if (dominant.id === candidate.id) return false;
  if (dominant.route !== candidate.route) return false;
  if (dominant.graphDepth !== candidate.graphDepth) return false;
  if (dominant.syncPlanner !== candidate.syncPlanner) return false;
  if (sortedKey(dominant.memoryTypes || []) !== sortedKey(candidate.memoryTypes || [])) return false;
  if (String(dominant.match?.taskType || '') !== String(candidate.match?.taskType || '')) return false;
  if (String(dominant.match?.projectHint || '') !== String(candidate.match?.projectHint || '')) return false;
  if (Boolean(dominant.match?.repoHintPresent) !== Boolean(candidate.match?.repoHintPresent)) return false;
  if (dominant.confidence + 0.02 < candidate.confidence) return false;

  const dominantSignals = new Set(normalizeSignalFamilyV3(dominant.match?.turnSignals || [], 12));
  const candidateSignals = normalizeSignalFamilyV3(candidate.match?.turnSignals || [], 12);
  if (candidateSignals.length === 0) return false;
  const dominantCoversCandidate = candidateSignals.every((signal) => dominantSignals.has(signal));
  const dominantQueryKey = sortedKey(normalizeQueryTemplateFamilyV3(dominant.queries || [], 8));
  const candidateQueryKey = sortedKey(normalizeQueryTemplateFamilyV3(candidate.queries || [], 8));
  return dominantCoversCandidate && dominantQueryKey === candidateQueryKey;
}

function normalizeTemplateV3(value: string): string {
  const tokens = tokenizeV3([value]);
  if (tokens.length === 0) return '';
  const compact = tokens.filter((token) => !(token === 'project' && tokens.length > 2));
  return (compact.length ? compact : tokens).join(' ');
}

function tokenizeV3(values: any[]): string[] {
  const tokens = String(values.flatMap((value) => String(value || '').split(/[^a-z0-9_-]+/i)).join(' ')).toLowerCase()
    .split(/\s+/)
    .map((token) => token.trim())
    .filter(Boolean)
    .map((token) => SYNONYM_MAP[token] || token)
    .filter((token) => token.length >= 2 && token.length <= 40);
  return [...new Set(tokens)];
}

function sortedKey(values: (string | MemoryType)[]): string {
  return [...new Set((values || []).map((value) => String(value || '')))].sort().join('|');
}
