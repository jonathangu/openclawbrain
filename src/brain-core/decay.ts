/**
 * Weight decay toward structural priors.
 *
 * Without evidence reinforcing them, learned edge weights
 * should converge back to their structural/semantic priors.
 *
 * Decay formula: w_new = w * rate + prior * (1 - rate)
 *
 * rate = 1.0 → no decay (weight unchanged)
 * rate = 0.0 → instant snap to prior
 * rate = 0.995 → slow decay (default)
 */

import type { BrainEdge } from "./types.js";
import type { BrainGraph } from "./graph.js";

/**
 * Decay a single weight toward its prior.
 */
export function decayWeight(weight: number, prior: number, rate: number): number {
  return weight * rate + prior * (1 - rate);
}

/**
 * Batch decay all edge weights in the graph toward their priors.
 */
export function decayAllWeights(graph: BrainGraph, rate: number, now: number): void {
  for (const node of graph.getAllNodes()) {
    const edges = graph.getOutgoingEdges(node.id);
    for (const edge of edges) {
      edge.weight = decayWeight(edge.weight, edge.prior, rate);
      edge.decayedAt = now;
    }
  }
}
