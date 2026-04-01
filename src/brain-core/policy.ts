/**
 * Softmax routing policy over action sets.
 *
 * Implements P_ρ(a|s) from the paper:
 * P_ρ(a_j | s_t) = exp(score(a_j) / τ) / Σ_k exp(score(a_k) / τ)
 *
 * Policy is ALWAYS stochastic (samples from softmax, never argmax).
 * Temperature τ controls exploration vs exploitation:
 *   - Learning: τ = 1.0 (explore)
 *   - Serving:  τ = 0.1 (exploit, nearly deterministic)
 */

import type {
  BrainNode,
  TraversalAction,
  TraversalState,
  PolicyParams,
  TrustLevel,
  TrajectoryCandidateScoreBreakdown,
  TrajectoryPolicyStateSnapshot,
} from "./types.js";
import { DEFAULT_POLICY_PARAMS } from "./types.js";
import { BrainGraph, cosineSimilarity } from "./graph.js";

const TRUST_EVIDENCE_SCORE: Record<TrustLevel, number> = {
  human: 1.0,
  self: 0.6,
  scanner: 0.3,
  teacher: 0.15,
};

function getPendingNodeIds(state: TraversalState): string[] {
  return state.pendingNodeIds ?? [];
}

function computeRemainingExpansionSlots(state: TraversalState): number {
  return Math.max(1, state.maxHops - state.expansionCount);
}

function computeActiveFrontierSize(state: TraversalState): number {
  return state.frontier.length + getPendingNodeIds(state).length;
}

function computeEffectiveBudgetRemaining(state: TraversalState): number {
  return Math.max(0, state.budgetRemaining - state.reservedTokenCost);
}

function computeBudgetUsedFraction(state: TraversalState): number {
  const totalBudget = Math.max(0, state.initialBudget);
  const effectiveBudgetRemaining = computeEffectiveBudgetRemaining(state);
  return totalBudget > 0 ? 1 - effectiveBudgetRemaining / totalBudget : 0;
}

export function computePolicyStateSnapshot(state: TraversalState): TrajectoryPolicyStateSnapshot {
  const effectiveBudgetRemaining = computeEffectiveBudgetRemaining(state);
  const budgetUsedFraction = computeBudgetUsedFraction(state);
  const remainingExpansionSlots = computeRemainingExpansionSlots(state);
  const activeFrontierSize = computeActiveFrontierSize(state);
  const frontierBacklogPressure = Math.min(1, activeFrontierSize / remainingExpansionSlots);
  const frontierCapacity = Math.max(
    1,
    state.maxFrontierSize ?? Math.max(activeFrontierSize, remainingExpansionSlots),
  );
  const frontierSaturation = Math.min(1, activeFrontierSize / frontierCapacity);
  const frontierPressure = Math.max(frontierBacklogPressure, frontierSaturation);

  return {
    effectiveBudgetRemaining,
    budgetUsedFraction,
    frontierBacklogPressure,
    frontierSaturation,
    frontierPressure,
    pressureLevel: Math.min(1, (budgetUsedFraction + frontierPressure) / 2),
    remainingExpansionSlots,
    activeFrontierSize,
    pendingSelectionCount: getPendingNodeIds(state).length,
  };
}

function computeBranchOpportunityCostSignal(
  targetNode: BrainNode,
  state: TraversalState,
  graph: BrainGraph,
  policyState: TrajectoryPolicyStateSnapshot,
): {
  opportunityCostSignal: number;
  tokenCostFraction: number;
  downstreamOpportunityCount: number;
  downstreamPressure: number;
} {
  const tokenCostFraction = policyState.effectiveBudgetRemaining > 0
    ? Math.min(1, targetNode.tokenCount / policyState.effectiveBudgetRemaining)
    : 1;
  const downstreamOpportunityCount = graph.getNeighbors(targetNode.id).filter((neighborId) => (
    neighborId !== state.sourceNodeId
    && !state.visited.has(neighborId)
    && !state.frontier.includes(neighborId)
    && !getPendingNodeIds(state).includes(neighborId)
  )).length;
  const frontierRoom = Math.max(
    1,
    (state.maxFrontierSize ?? policyState.remainingExpansionSlots) - policyState.activeFrontierSize,
  );
  const downstreamPressure = Math.max(
    Math.min(1, downstreamOpportunityCount / policyState.remainingExpansionSlots),
    Math.min(1, downstreamOpportunityCount / frontierRoom),
  );
  return {
    opportunityCostSignal: policyState.pressureLevel * ((tokenCostFraction * 0.55) + (downstreamPressure * 0.45)),
    tokenCostFraction,
    downstreamOpportunityCount,
    downstreamPressure,
  };
}

function computeLocalRedundancySignal(
  targetNode: BrainNode,
  state: TraversalState,
  graph: BrainGraph,
): number {
  if (!targetNode?.embedding) {
    return 0;
  }

  const comparisonNodeIds = new Set<string>([
    ...state.frontier,
    ...state.fired,
    ...state.visited,
    ...getPendingNodeIds(state),
  ]);
  if (state.sourceNodeId) {
    comparisonNodeIds.add(state.sourceNodeId);
  }
  comparisonNodeIds.delete(targetNode.id);

  let maxSimilarity = 0;
  for (const nodeId of comparisonNodeIds) {
    const comparisonNode = graph.getNode(nodeId);
    if (!comparisonNode?.embedding) {
      continue;
    }
    maxSimilarity = Math.max(
      maxSimilarity,
      Math.max(0, cosineSimilarity(targetNode.embedding, comparisonNode.embedding)),
    );
  }

  return maxSimilarity;
}

function computeNearbyEvidenceQualitySignal(targetNode: BrainNode, graph: BrainGraph): number {
  const trustSignal = TRUST_EVIDENCE_SCORE[targetNode.trust];
  const supportiveIncomingEdgeCount = graph.getIncomingEdges(targetNode.id).filter((edge) => (
    edge.kind !== "inhibitory" && edge.weight >= 0
  )).length;
  const structuralSupportSignal = supportiveIncomingEdgeCount / (supportiveIncomingEdgeCount + 1);
  return trustSignal * 0.6 + structuralSupportSignal * 0.4;
}

function explainActionScore(
  action: TraversalAction,
  state: TraversalState,
  graph: BrainGraph,
  params: PolicyParams,
): { score: number; scoreBreakdown: TrajectoryCandidateScoreBreakdown } {
  const policyState = computePolicyStateSnapshot(state);

  if (action.type === "stop_local") {
    const expansionFraction = state.maxHops > 0 ? state.expansionCount / state.maxHops : 0;
    const learnedStopWeight = graph.getStopLocalWeight(state.sourceNodeId);
    const budgetPressureContribution = params.budgetPressure * policyState.budgetUsedFraction;
    const hopPressureContribution = params.hopPressure * expansionFraction;
    const frontierPressureContribution = params.frontierPressure * policyState.frontierPressure;
    const score = learnedStopWeight
      + params.stopBias
      + budgetPressureContribution
      + hopPressureContribution
      + frontierPressureContribution;

    return {
      score,
      scoreBreakdown: {
        totalScore: score,
        pressureLevel: policyState.pressureLevel,
        learnedStopWeight,
        stopBias: params.stopBias,
        budgetPressureContribution,
        hopPressureContribution,
        frontierPressureContribution,
      },
    };
  }

  const targetNode = graph.getNode(action.targetNodeId);
  if (!targetNode) {
    return {
      score: -Infinity,
      scoreBreakdown: { totalScore: -Infinity },
    };
  }

  if (state.sourceNodeId === null) {
    const seedPrior = action.seedScore ?? 0;
    const learnedSeedWeight = graph.getSeedWeight(action.targetNodeId);
    const score = seedPrior + learnedSeedWeight;
    return {
      score,
      scoreBreakdown: {
        totalScore: score,
        seedPrior,
        learnedSeedWeight,
      },
    };
  }

  const edge = graph.getEdge(state.sourceNodeId, action.targetNodeId);
  const edgeScore = edge ? edge.weight * edge.prior : 0;

  let relevance = 0;
  if (targetNode.embedding && state.queryEmbedding.length > 0) {
    relevance = cosineSimilarity(state.queryEmbedding, targetNode.embedding);
  }

  const kindBias = edge ? (params.edgeKindBias[edge.kind] ?? 0) : 0;
  const evidenceQualityBonus = params.evidenceQualityBias * computeNearbyEvidenceQualitySignal(targetNode, graph);
  const branchOpportunity = computeBranchOpportunityCostSignal(targetNode, state, graph, policyState);
  const opportunityCostPenalty = params.branchOpportunityCost * branchOpportunity.opportunityCostSignal;
  const redundancySimilarity = computeLocalRedundancySignal(targetNode, state, graph);
  const redundancyPressureMultiplier = 0.75 + policyState.pressureLevel;
  const redundancyPenalty =
    params.localRedundancyPenalty * redundancySimilarity * redundancyPressureMultiplier;
  const score = edgeScore + relevance + kindBias + evidenceQualityBonus
    - opportunityCostPenalty - redundancyPenalty;

  return {
    score,
    scoreBreakdown: {
      totalScore: score,
      pressureLevel: policyState.pressureLevel,
      edgeScore,
      relevance,
      kindBias,
      evidenceQualityBonus,
      opportunityCostPenalty,
      tokenCostFraction: branchOpportunity.tokenCostFraction,
      downstreamOpportunityCount: branchOpportunity.downstreamOpportunityCount,
      downstreamPressure: branchOpportunity.downstreamPressure,
      redundancyPenalty,
      redundancySimilarity,
      redundancyPressureMultiplier,
    },
  };
}

/**
 * Score a single action given current state and graph.
 *
 * For stop_local: score combines learned source-local preference with budget/hop/frontier pressure.
 * For traverse: score combines learned edge signal, query relevance, evidence quality,
 * and penalties for redundant or high-opportunity-cost local branches.
 */
export function scoreAction(
  action: TraversalAction,
  state: TraversalState,
  graph: BrainGraph,
  params: PolicyParams = DEFAULT_POLICY_PARAMS,
): number {
  return explainActionScore(action, state, graph, params).score;
}

/**
 * Compute softmax distribution over the full action set.
 *
 * Returns sorted candidates with their scores and probabilities.
 * Numerically stable: subtract max score before exp.
 */
export function softmaxPolicy(
  actions: TraversalAction[],
  state: TraversalState,
  graph: BrainGraph,
  params: PolicyParams = DEFAULT_POLICY_PARAMS,
): Array<{
  action: TraversalAction;
  score: number;
  probability: number;
  scoreBreakdown?: TrajectoryCandidateScoreBreakdown;
}> {
  if (actions.length === 0) return [];

  const scored = actions.map((action) => ({
    action,
    ...explainActionScore(action, state, graph, params),
  }));

  // Numerically stable softmax
  const maxScore = Math.max(...scored.map((s) => s.score));
  const tau = params.temperature;

  const expScores = scored.map((s) => ({
    ...s,
    expScore: Math.exp((s.score - maxScore) / tau),
  }));

  const sumExp = expScores.reduce((sum, s) => sum + s.expScore, 0);

  return expScores.map((s) => ({
    action: s.action,
    score: s.score,
    probability: sumExp > 0 ? s.expScore / sumExp : 1 / actions.length,
    scoreBreakdown: s.scoreBreakdown,
  }));
}

/**
 * Sample an action from the softmax distribution.
 *
 * Stochastic — NEVER argmax. Even at low temperature, this samples
 * from the distribution. This is required for the paper's REINFORCE
 * update to have valid gradients.
 */
export function sampleAction(
  distribution: Array<{ action: TraversalAction; probability: number }>,
): { action: TraversalAction; probability: number; index: number } {
  if (distribution.length === 0) {
    return { action: { type: "stop_local" }, probability: 1.0, index: 0 };
  }

  const r = Math.random();
  let cumulative = 0;

  for (let i = 0; i < distribution.length; i++) {
    cumulative += distribution[i].probability;
    if (r <= cumulative) {
      return {
        action: distribution[i].action,
        probability: distribution[i].probability,
        index: i,
      };
    }
  }

  // Fallback: numerical precision edge case
  const last = distribution.length - 1;
  return {
    action: distribution[last].action,
    probability: distribution[last].probability,
    index: last,
  };
}

/**
 * Compute log probability of a chosen action.
 * Used in REINFORCE gradient: ∂logP_ρ(a|s)/∂ρ
 */
export function logProbability(probability: number): number {
  return Math.log(Math.max(probability, 1e-10));
}
