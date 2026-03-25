/**
 * REINFORCE update rule implementing Lemma 6.1 from the paper.
 *
 * Paper's correct update direction:
 *   ∂/∂ρ v_ρ(s_t) = E[z · Σ_{l=t}^{T} ∂logP_ρ(a_l|s_l)/∂ρ]
 *
 * Key insight: The sum goes over the ENTIRE trajectory from t to T.
 * This assigns credit to every routing decision that led to the outcome,
 * not just the last step. Williams (1992) equation (1) is one-step only.
 *
 * For the softmax policy over edge weights:
 *   ∂logP(a_j|s)/∂w_j = 1 - P(a_j|s)     (for the chosen action's weight)
 *   ∂logP(a_j|s)/∂w_k = -P(a_k|s)         (for other actions' weights)
 *
 * We update only the chosen action's parameter at each step, using:
 *   Δw_j = learningRate × (z - baseline) × (1 - P(a_j|s))
 *
 * The full-trajectory sum is achieved by accumulating updates across all steps.
 */

import type { Episode, PolicyWeightUpdate } from "./types.js";
import { START_NODE_ID } from "./types.js";
import type { BrainGraph } from "./graph.js";

/**
 * Compute REINFORCE weight updates from a completed episode.
 *
 * Implements Lemma 6.1: full-trajectory credit assignment.
 * Every step l from 0 to T contributes to the gradient.
 */
export function computeReinforceUpdates(
  episode: Episode,
  learningRate: number,
  baseline: number,
): PolicyWeightUpdate[] {
  if (episode.reward === null) return [];

  const advantage = episode.reward - baseline;
  if (Math.abs(advantage) < 1e-8) return [];

  const updates: Map<string, PolicyWeightUpdate> = new Map();

  // Sum over all selection substeps l = 0 to T (full-trajectory, not one-step)
  for (const expansion of episode.trajectory) {
    for (const substep of expansion.substeps) {
      const sourceId = substep.stateSnapshot.sourceNodeId ?? START_NODE_ID;

      // ∂logP(a_l|s_l)/∂ρ for the softmax = (1 - P(a_l|s_l))
      const gradLogP = 1 - substep.chosenActionProbability;

      // Δρ ∝ (z - baseline) × ∂logP/∂ρ
      const delta = learningRate * advantage * gradLogP;
      if (Math.abs(delta) < 1e-12) {
        continue;
      }

      if (substep.chosenAction.type === "stop_local") {
        const key = `stop→${sourceId}`;
        const existing = updates.get(key);
        if (existing && existing.kind === "stop_local") {
          existing.delta += delta;
        } else {
          updates.set(key, { kind: "stop_local", sourceNodeId: sourceId, delta });
        }
        continue;
      }

      const targetId = substep.chosenAction.targetNodeId;

      if (sourceId === START_NODE_ID) {
        const key = `seed→${targetId}`;
        const existing = updates.get(key);
        if (existing && existing.kind === "seed") {
          existing.delta += delta;
        } else {
          updates.set(key, { kind: "seed", nodeId: targetId, delta });
        }
        continue;
      }

      // Accumulate: the full-trajectory sum means each substep adds to the gradient.
      const key = `${sourceId}→${targetId}`;
      const existing = updates.get(key);
      if (existing && existing.kind === "edge") {
        existing.delta += delta;
      } else {
        updates.set(key, { kind: "edge", source: sourceId, target: targetId, delta });
      }
    }
  }

  return [...updates.values()];
}

/**
 * Update running baseline via exponential moving average.
 *
 * baseline_{n+1} = α × z_n + (1 - α) × baseline_n
 *
 * The baseline reduces variance in the REINFORCE estimate
 * without introducing bias (standard variance reduction technique).
 */
export function updateBaseline(
  currentBaseline: number,
  newReward: number,
  alpha: number,
): number {
  return alpha * newReward + (1 - alpha) * currentBaseline;
}

/**
 * Apply computed weight updates to graph edges.
 *
 * After applying, edge weights may become negative (inhibitory).
 * This is intentional — the paper allows signed outcomes.
 */
export function applyWeightUpdates(
  graph: BrainGraph,
  updates: PolicyWeightUpdate[],
): void {
  for (const update of updates) {
    if (update.kind === "seed") {
      const nextWeight = Math.max(-10, Math.min(10, graph.getSeedWeight(update.nodeId) + update.delta));
      graph.setSeedWeight(update.nodeId, nextWeight);
      continue;
    }

    if (update.kind === "stop_local") {
      const nextWeight = Math.max(
        -10,
        Math.min(10, graph.getStopLocalWeight(update.sourceNodeId) + update.delta),
      );
      graph.setStopLocalWeight(update.sourceNodeId, nextWeight);
      continue;
    }

    const edge = graph.getEdge(update.source, update.target);
    if (!edge) continue;

    // Update weight: w_new = w_old + Δw
    edge.weight += update.delta;

    // Clamp to prevent numerical explosion
    edge.weight = Math.max(-10, Math.min(10, edge.weight));
  }
}
