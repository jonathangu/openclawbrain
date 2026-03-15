/**
 * Full traversal loop implementing the paper's finite-time game.
 *
 * Algorithm:
 * 1. Seed phase: select start nodes by embedding similarity
 * 2. Loop: expand candidates → compute softmax policy → sample → fire/veto
 * 3. Terminal: STOP chosen, budget exhausted, max hops, or dead end
 *
 * Paper assumptions honored:
 * - Assumption 1: game ends in finite time (maxHops bound)
 * - Assumption 2: reward only at terminal state (not intermediate)
 * - Stochastic policy P_ρ(a|s) via softmax sampling
 */

import type {
  TraversalState,
  TraversalAction,
  TrajectoryStep,
  PolicyParams,
  NodeKind,
  SeedScore,
} from "./types.js";
import { DEFAULT_POLICY_PARAMS, START_NODE_ID as START_NODE } from "./types.js";
import type { BrainGraph } from "./graph.js";
import { softmaxPolicy, sampleAction } from "./policy.js";

export interface TraverseOptions {
  graph: BrainGraph;
  queryEmbedding: Float32Array;
  queryText: string;
  maxHops: number;
  budgetChars: number;
  temperature: number;
  maxSeeds: number;
  semanticThreshold: number;
  policyParams?: Partial<PolicyParams>;
}

export interface TraverseResult {
  firedNodes: Array<{ nodeId: string; kind: NodeKind; content: string; tokenCount: number }>;
  vetoedNodes: Array<{ nodeId: string; reason: string }>;
  trajectory: TrajectoryStep[];
  seedScores: SeedScore[];
  contextChars: number;
  footer: string;
}

/**
 * Execute a full graph traversal.
 *
 * Returns the fired nodes, vetoed nodes, full trajectory (for REINFORCE),
 * seed scores, and a human-readable footer.
 */
export function traverse(options: TraverseOptions): TraverseResult {
  const {
    graph,
    queryEmbedding,
    maxHops,
    budgetChars,
    temperature,
    maxSeeds,
    semanticThreshold,
  } = options;

  const params: PolicyParams = {
    ...DEFAULT_POLICY_PARAMS,
    temperature,
    ...options.policyParams,
  };

  // Step 1: Seed selection
  const seedCandidates = graph.seedByEmbedding(queryEmbedding, maxSeeds, semanticThreshold);

  if (seedCandidates.length === 0) {
    return {
      firedNodes: [],
      vetoedNodes: [],
      trajectory: [],
      seedScores: [],
      contextChars: 0,
      footer: "Brain · 0 seeds · no traversal",
    };
  }

  // Initialize traversal state
  const state: TraversalState = {
    currentNodeId: null,
    queryEmbedding,
    visited: new Set(),
    fired: [],
    budgetRemaining: budgetChars,
    hopCount: 0,
    maxHops,
  };

  const trajectory: TrajectoryStep[] = [];
  const firedNodes: Array<{ nodeId: string; kind: NodeKind; content: string; tokenCount: number }> = [];
  const vetoedNodes: Array<{ nodeId: string; reason: string }> = [];
  let seedScores: SeedScore[] = [];

  // Step 2: Traversal loop
  for (let hop = 0; hop < maxHops; hop++) {
    // Compute action set
    const actions = graph.getActionSet(
      state.currentNodeId,
      state.visited,
      state.currentNodeId === null ? seedCandidates : undefined,
    );

    if (actions.length === 0) break; // Dead end
    if (actions.length === 1 && actions[0].type === "stop") {
      // Only STOP available — record step and terminate
      const step: TrajectoryStep = {
        stateSnapshot: {
          currentNodeId: state.currentNodeId,
          hopCount: state.hopCount,
          budgetRemaining: state.budgetRemaining,
          visitedCount: state.visited.size,
          firedCount: state.fired.length,
        },
        candidates: [{ action: { type: "stop" }, score: 0, probability: 1.0 }],
        chosenAction: { type: "stop" },
        chosenActionProbability: 1.0,
        stopProbability: 1.0,
      };
      trajectory.push(step);
      break;
    }

    // Compute softmax distribution
    const distribution = softmaxPolicy(actions, state, graph, params);

    // Sample action (stochastic)
    const sampled = sampleAction(distribution);

    if (state.currentNodeId === null) {
      seedScores = seedCandidates.map((seed) => {
        const traverseEntry = distribution.find(
          (entry) => entry.action.type === "traverse" && entry.action.targetNodeId === seed.nodeId,
        );
        const seedEdge = graph.getEdge(START_NODE, seed.nodeId);
        return {
          nodeId: seed.nodeId,
          priorScore: seed.score,
          learnedScore: seedEdge ? seedEdge.weight * seedEdge.prior : 0,
          policyScore: traverseEntry?.score ?? seed.score,
          probability: traverseEntry?.probability ?? 0,
          chosen: sampled.action.type === "traverse" && sampled.action.targetNodeId === seed.nodeId,
        };
      });
    }

    // Find STOP probability for trace
    const stopEntry = distribution.find((d) => d.action.type === "stop");
    const stopProb = stopEntry?.probability ?? 0;

    // Record trajectory step
    const step: TrajectoryStep = {
      stateSnapshot: {
        currentNodeId: state.currentNodeId,
        hopCount: state.hopCount,
        budgetRemaining: state.budgetRemaining,
        visitedCount: state.visited.size,
        firedCount: state.fired.length,
      },
      candidates: distribution.map((d) => ({
        action: d.action,
        score: d.score,
        probability: d.probability,
      })),
      chosenAction: sampled.action,
      chosenActionProbability: sampled.probability,
      stopProbability: stopProb,
    };
    trajectory.push(step);

    // Execute action
    if (sampled.action.type === "stop") {
      break; // Terminal: STOP chosen
    }

    const targetNodeId = sampled.action.targetNodeId;
    state.visited.add(targetNodeId);
    state.hopCount++;

    // Inhibitory veto check
    if (state.currentNodeId && graph.isVetoed(state.currentNodeId, targetNodeId)) {
      const reason = graph.getVetoReason(state.currentNodeId, targetNodeId) ?? "inhibitory";
      vetoedNodes.push({ nodeId: targetNodeId, reason });
      state.currentNodeId = targetNodeId; // Still move to the node, but don't fire it
      continue;
    }

    // Fire node: add to context
    const targetNode = graph.getNode(targetNodeId);
    if (targetNode) {
      state.fired.push(targetNodeId);
      state.budgetRemaining -= targetNode.tokenCount;

      firedNodes.push({
        nodeId: targetNode.id,
        kind: targetNode.kind,
        content: targetNode.content,
        tokenCount: targetNode.tokenCount,
      });
    }

    state.currentNodeId = targetNodeId;

    // Terminal: budget exhausted
    if (state.budgetRemaining <= 0) break;
  }

  const contextChars = firedNodes.reduce((sum, n) => sum + n.content.length, 0);

  const chosenSeed = seedScores.find((seed) => seed.chosen)?.nodeId ?? "none";
  const footer = `Brain · ${seedScores.length} seeds · start ${chosenSeed} · ${state.hopCount} hops · ${firedNodes.length} fired · ${vetoedNodes.length} veto · ${contextChars} chars`;

  return {
    firedNodes,
    vetoedNodes,
    trajectory,
    seedScores,
    contextChars,
    footer,
  };
}
