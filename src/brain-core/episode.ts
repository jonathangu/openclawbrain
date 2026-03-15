/**
 * Episode recording and replay.
 */

import { randomUUID } from "node:crypto";
import type { Episode, TrajectoryStep, PolicyParams } from "./types.js";
import { DEFAULT_POLICY_PARAMS } from "./types.js";
import type { BrainGraph } from "./graph.js";
import type { TraverseResult } from "./traverse.js";
import { softmaxPolicy } from "./policy.js";

export function recordEpisode(params: {
  traversalResult: TraverseResult;
  queryText: string;
  queryEmbedding: Float32Array | null;
  conversationId: number | null;
  packVersion: number | null;
}): Episode {
  return {
    id: `be_${randomUUID().slice(0, 12)}`,
    conversationId: params.conversationId,
    queryText: params.queryText,
    queryEmbedding: params.queryEmbedding,
    trajectory: params.traversalResult.trajectory,
    firedNodes: params.traversalResult.firedNodes.map((n) => n.nodeId),
    vetoedNodes: params.traversalResult.vetoedNodes.map((v) => v.nodeId),
    contextChars: params.traversalResult.contextChars,
    reward: null,
    rewardSource: null,
    packVersion: params.packVersion,
    createdAt: Date.now(),
  };
}

/**
 * Replay an episode against a (possibly mutated) graph.
 * Returns what the policy WOULD produce with updated weights.
 * Used for replay-gate validation before pack promotion.
 */
export function replayEpisode(
  episode: Episode,
  graph: BrainGraph,
  policyParams: PolicyParams = DEFAULT_POLICY_PARAMS,
): { firedNodes: string[]; wouldChange: boolean } {
  if (!episode.queryEmbedding || episode.trajectory.length === 0) {
    return { firedNodes: [], wouldChange: false };
  }

  const fired: string[] = [];
  let changed = false;

  for (const step of episode.trajectory) {
    if (step.chosenAction.type !== "traverse") continue;

    const actions = step.candidates.map((c) => c.action);
    const state = {
      currentNodeId: step.stateSnapshot.currentNodeId,
      queryEmbedding: episode.queryEmbedding,
      visited: new Set<string>(),
      fired,
      budgetRemaining: step.stateSnapshot.budgetRemaining,
      hopCount: step.stateSnapshot.hopCount,
      maxHops: 8,
    };

    const newDist = softmaxPolicy(actions, state, graph, policyParams);
    const originalChoice = step.chosenAction;
    const newTopAction = newDist.reduce((best, d) => d.probability > best.probability ? d : best, newDist[0]);

    if (newTopAction.action.type !== originalChoice.type ||
        (newTopAction.action.type === "traverse" && originalChoice.type === "traverse" &&
         newTopAction.action.targetNodeId !== originalChoice.targetNodeId)) {
      changed = true;
    }

    if (step.chosenAction.type === "traverse") {
      fired.push(step.chosenAction.targetNodeId);
    }
  }

  return { firedNodes: fired, wouldChange: changed };
}
