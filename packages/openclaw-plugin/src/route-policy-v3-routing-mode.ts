import type { MemoryType, RouteActionPrototypeV3, RouteFrameV3, RouteKind, TurnFrame } from './memory-types.js';

export type RoutingModeV3 = 'exact_correction' | 'workflow_exact' | 'semantic_planning' | 'casual_silence' | 'mixed';

export interface RoutingWeightsV3 {
  sparse: number;
  dense: number;
  bandit: number;
  risk: number;
  signalBonus: number;
}

export function detectRoutingModeV3(
  frame: Pick<RouteFrameV3, 'taskType' | 'turnSignals' | 'routeHintFlags' | 'redactedTurnSummary'> | TurnFrame,
  message = '',
): RoutingModeV3 {
  const taskType = String(frame.taskType || 'other');
  const summary = 'redactedTurnSummary' in frame
    ? String(frame.redactedTurnSummary || '')
    : `${frame.summary || ''} ${frame.userGoal || ''}`;
  const signals = [
    ...(('turnSignals' in frame ? frame.turnSignals : []) || []),
    ...(('routeHintFlags' in frame ? frame.routeHintFlags : []) || []),
  ].map((value) => String(value || '').toLowerCase());
  const haystack = `${message} ${summary} ${signals.join(' ')}`.toLowerCase();

  if (/\b(actually|instead|wrong|fix|correct|correction|pnpm|npm|yarn)\b/.test(haystack) || signals.includes('needs_correction')) {
    return 'exact_correction';
  }
  if (/\b(test|tests|build|install|dependency|dependencies|setup|script|command|repo)\b/.test(haystack) || signals.includes('needs_workflow')) {
    return 'workflow_exact';
  }
  if (taskType === 'planning' || /\b(plan|design|architecture|file-by-file|roadmap|strategy|implementation)\b/.test(haystack)) {
    return 'semantic_planning';
  }
  if (taskType === 'other' && /\b(thanks|thank you|ok|okay|cool|great|nice)\b/.test(haystack)) {
    return 'casual_silence';
  }
  return 'mixed';
}

export function hybridWeightsForRoutingModeV3(mode: RoutingModeV3): RoutingWeightsV3 {
  switch (mode) {
    case 'exact_correction':
      return { sparse: 0.64, dense: 0.16, bandit: 0.12, risk: 0.08, signalBonus: 0.06 };
    case 'workflow_exact':
      return { sparse: 0.58, dense: 0.22, bandit: 0.12, risk: 0.08, signalBonus: 0.05 };
    case 'semantic_planning':
      return { sparse: 0.34, dense: 0.42, bandit: 0.16, risk: 0.08, signalBonus: 0.03 };
    case 'casual_silence':
      return { sparse: 0.62, dense: 0.1, bandit: 0.08, risk: 0.2, signalBonus: 0.03 };
    default:
      return { sparse: 0.46, dense: 0.28, bandit: 0.16, risk: 0.1, signalBonus: 0.04 };
  }
}

export function calibrationThresholdAdjustmentForModeV3(mode: RoutingModeV3): number {
  switch (mode) {
    case 'exact_correction':
      return 0.04;
    case 'semantic_planning':
      return 0.02;
    case 'casual_silence':
      return -0.03;
    default:
      return 0;
  }
}

export function prototypeRiskPenaltyV3(prototype: Pick<RouteActionPrototypeV3, 'route' | 'memoryTypes' | 'graphDepth' | 'syncPlanner' | 'harmPrior'>, mode: RoutingModeV3): number {
  let penalty = 0;
  if (prototype.syncPlanner === 'allowed' || prototype.syncPlanner === 'prefer') penalty += 0.04;
  if (prototype.graphDepth >= 2) penalty += 0.03;
  penalty += Math.min(0.08, Math.max(0, Number(prototype.harmPrior || 0)) * 0.04);
  if (mode === 'casual_silence' && prototype.route !== 'no_memory') penalty += 0.08;
  if (mode === 'exact_correction' && !prototype.memoryTypes.includes('correction' as MemoryType) && prototype.route !== 'no_memory') penalty += 0.05;
  if (mode === 'semantic_planning' && prototype.route === 'no_memory') penalty += 0.03;
  return penalty;
}

export function routeModeDiagnosticFamilyV3(mode: RoutingModeV3, route: RouteKind): string {
  return `${mode}:${route}`;
}
