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
} from "./types.js";
import { DEFAULT_POLICY_PARAMS } from "./types.js";
import { BrainGraph, cosineSimilarity } from "./graph.js";

const TRUST_EVIDENCE_SCORE: Record<TrustLevel, number> = {
  human: 1.0,
  self: 0.6,
  scanner: 0.3,
  teacher: 0.15,
};

function computeEffectiveBudgetRemaining(state: TraversalState): number {
  return Math.max(0, state.budgetRemaining - state.reservedTokenCost);
}

function computeBudgetUsedFraction(state: TraversalState): number {
  const totalBudget = Math.max(0, state.initialBudget);
  const effectiveBudgetRemaining = computeEffectiveBudgetRemaining(state);
  return totalBudget > 0 ? 1 - effectiveBudgetRemaining / totalBudget : 0;
}

function computeFrontierPressure(state: TraversalState): number {
  const remainingExpansionSlots = Math.max(1, state.maxHops - state.expansionCount);
  return Math.min(1, state.frontier.length / remainingExpansionSlots);
}

function computeBranchOpportunityCostSignal(
  targetNode: BrainNode,
  state: TraversalState,
  graph: BrainGraph,
): number {
  const effectiveBudgetRemaining = computeEffectiveBudgetRemaining(state);
  const budgetUsedFraction = computeBudgetUsedFraction(state);
  const frontierPressure = computeFrontierPressure(state);
  const tokenCostFraction = effectiveBudgetRemaining > 0
    ? Math.min(1, targetNode.tokenCount / effectiveBudgetRemaining)
    : 1;
  const downstreamOpportunityCount = graph.getNeighbors(targetNode.id).filter((neighborId) => (
    neighborId !== state.sourceNodeId
    && !state.visited.has(neighborId)
    && !state.frontier.includes(neighborId)
  )).length;
  const downstreamBranchFactor = downstreamOpportunityCount / (downstreamOpportunityCount + 2);
  const pressureLevel = (budgetUsedFraction + frontierPressure) / 2;
  return pressureLevel * ((tokenCostFraction + downstreamBranchFactor) / 2);
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
  if (action.type === "stop_local") {
    const budgetUsedFraction = computeBudgetUsedFraction(state);
    const expansionFraction = state.maxHops > 0 ? state.expansionCount / state.maxHops : 0;
    const frontierPressure = computeFrontierPressure(state);
    return graph.getStopLocalWeight(state.sourceNodeId)
      + params.stopBias
      + params.budgetPressure * budgetUsedFraction
      + params.hopPressure * expansionFraction
      + params.frontierPressure * frontierPressure;
  }

  // Traverse action
  const targetNode = graph.getNode(action.targetNodeId);
  if (!targetNode) return -Infinity;

  if (state.sourceNodeId === null) {
    const seedPrior = action.seedScore ?? 0;
    const learnedSeedWeight = graph.getSeedWeight(action.targetNodeId);
    return seedPrior + learnedSeedWeight;
  }

  // Find edge from current position to target
  const edge = graph.getEdge(state.sourceNodeId, action.targetNodeId);

  // Base score from edge weight and prior
  const edgeScore = edge ? edge.weight * edge.prior : 0;

  // Query relevance via embedding cosine similarity
  let relevance = 0;
  if (targetNode.embedding && state.queryEmbedding.length > 0) {
    relevance = cosineSimilarity(state.queryEmbedding, targetNode.embedding);
  }

  // Edge kind bias
  const kindBias = edge ? (params.edgeKindBias[edge.kind] ?? 0) : 0;
  const opportunityCostPenalty =
    params.branchOpportunityCost * computeBranchOpportunityCostSignal(targetNode, state, graph);
  const redundancyPenalty =
    params.localRedundancyPenalty * computeLocalRedundancySignal(targetNode, state, graph);
  const evidenceQualityBonus =
    params.evidenceQualityBias * computeNearbyEvidenceQualitySignal(targetNode, graph);

  return edgeScore + relevance + kindBias + evidenceQualityBonus
    - opportunityCostPenalty - redundancyPenalty;
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
): Array<{ action: TraversalAction; score: number; probability: number }> {
  if (actions.length === 0) return [];

  const scored = actions.map((action) => ({
    action,
    score: scoreAction(action, state, graph, params),
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
