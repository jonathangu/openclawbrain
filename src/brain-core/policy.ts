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
  TraversalAction,
  TraversalState,
  PolicyParams,
} from "./types.js";
import { DEFAULT_POLICY_PARAMS, START_NODE_ID as START } from "./types.js";
import { BrainGraph, cosineSimilarity } from "./graph.js";

/**
 * Score a single action given current state and graph.
 *
 * For STOP: score increases with budget depletion and hop count.
 * For traverse: score = edge.weight * edge.prior + cos(query, target) + bias
 */
export function scoreAction(
  action: TraversalAction,
  state: TraversalState,
  graph: BrainGraph,
  params: PolicyParams = DEFAULT_POLICY_PARAMS,
): number {
  if (action.type === "stop") {
    const totalBudget = state.budgetRemaining + state.fired.reduce((sum, id) => {
      const node = graph.getNode(id);
      return sum + (node?.tokenCount ?? 0);
    }, 0);
    const budgetUsedFraction = totalBudget > 0 ? 1 - state.budgetRemaining / totalBudget : 0;
    const hopFraction = state.maxHops > 0 ? state.hopCount / state.maxHops : 0;
    return params.stopBias
      + params.budgetPressure * budgetUsedFraction
      + params.hopPressure * hopFraction;
  }

  // Traverse action
  const targetNode = graph.getNode(action.targetNodeId);
  if (!targetNode) return -Infinity;

  // Find edge from current position to target
  const edge = state.currentNodeId
    ? graph.getEdge(state.currentNodeId, action.targetNodeId)
    : graph.getEdge(START, action.targetNodeId);

  // Base score from edge weight and prior
  const edgeScore = edge ? edge.weight * edge.prior : 0;

  // Query relevance via embedding cosine similarity
  let relevance = 0;
  if (targetNode.embedding && state.queryEmbedding.length > 0) {
    relevance = cosineSimilarity(state.queryEmbedding, targetNode.embedding);
  }

  // Edge kind bias
  const kindBias = edge ? (params.edgeKindBias[edge.kind] ?? 0) : 0;
  const seedPrior = state.currentNodeId === null ? (action.seedScore ?? 0) : 0;

  return edgeScore + relevance + kindBias + seedPrior;
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
    return { action: { type: "stop" }, probability: 1.0, index: 0 };
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
