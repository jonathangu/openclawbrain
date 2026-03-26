/**
 * Episode recording and replay.
 */

import { randomUUID } from "node:crypto";
import type { Episode, PolicyParams } from "./types.js";
import { DEFAULT_POLICY_PARAMS } from "./types.js";
import type { BrainGraph } from "./graph.js";
import type { TraverseResult } from "./traverse.js";
import { softmaxPolicy } from "./policy.js";
import { isForcedStopSubstep } from "./trajectory-stop.js";

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

  for (const expansion of episode.trajectory) {
    for (const substep of expansion.substeps) {
      if (isForcedStopSubstep(substep)) {
        continue;
      }

      const actions = substep.candidates.map((candidate) => candidate.action);
      const state = {
        sourceNodeId: expansion.sourceNodeId,
        queryEmbedding: episode.queryEmbedding,
        frontier: [...substep.stateSnapshot.frontierNodeIds],
        visited: new Set<string>(),
        fired,
        budgetRemaining: substep.stateSnapshot.budgetRemaining,
        initialBudget: substep.stateSnapshot.initialBudget,
        reservedTokenCost: substep.stateSnapshot.reservedTokenCost,
        expansionCount: substep.stateSnapshot.expansionIndex,
        maxHops: substep.stateSnapshot.maxHops,
      };

      const newDist = softmaxPolicy(actions, state, graph, policyParams);
      const originalChoice = substep.chosenAction;
      const newTopAction = newDist.reduce(
        (best, candidate) => candidate.probability > best.probability ? candidate : best,
        newDist[0],
      );

      if (
        newTopAction.action.type !== originalChoice.type
        || (
          newTopAction.action.type === "traverse"
          && originalChoice.type === "traverse"
          && newTopAction.action.targetNodeId !== originalChoice.targetNodeId
        )
      ) {
        changed = true;
      }

      if (substep.chosenAction.type === "traverse") {
        fired.push(substep.chosenAction.targetNodeId);
      }
    }
  }

  return { firedNodes: fired, wouldChange: changed };
}
