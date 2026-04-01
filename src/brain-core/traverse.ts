/**
 * Frontier traversal with sequential local subset selection.
 *
 * Algorithm:
 * 1. Seed phase: expand the implicit __START__ source into zero, one, or many seeds.
 * 2. Frontier loop: expand pending source nodes in FIFO order.
 * 3. Local loop: each source repeatedly samples { traverse(target), stop_local }.
 * 4. Commit accepted targets to the outer frontier, subject to budget and hard caps.
 *
 * Paper assumptions honored:
 * - Finite game via maxHops, maxFanoutPerNode, maxFrontierSize, and budget caps
 * - Terminal reward only at episode end
 * - Stochastic policy P_ρ(a|s) via softmax sampling
 */

import type {
  BrainInterruptionMetadata,
  InterruptionAccounting,
  TraversalState,
  TraversalAction,
  TrajectoryExpansion,
  TrajectoryProposalOutcome,
  TrajectoryStopReason,
  TrajectorySubstep,
  PolicyParams,
  NodeKind,
  SeedScore,
} from "./types.js";
import { DEFAULT_POLICY_PARAMS } from "./types.js";
import type { BrainGraph } from "./graph.js";
import { computePolicyStateSnapshot, softmaxPolicy, sampleAction } from "./policy.js";

const DEFAULT_MAX_FANOUT_PER_NODE = 4;
const DEFAULT_MAX_FRONTIER_SIZE = 32;

export interface TraverseOptions {
  graph: BrainGraph;
  queryEmbedding: Float32Array;
  queryText: string;
  maxHops: number;
  budgetChars: number;
  temperature: number;
  maxSeeds: number;
  semanticThreshold: number;
  maxFanoutPerNode?: number;
  maxFrontierSize?: number;
  deadlineAtMs?: number | null;
  policyParams?: Partial<PolicyParams>;
}

export interface TraverseResult {
  firedNodes: Array<{ nodeId: string; kind: NodeKind; content: string; tokenCount: number }>;
  vetoedNodes: Array<{ nodeId: string; reason: string }>;
  trajectory: TrajectoryExpansion[];
  seedScores: SeedScore[];
  contextChars: number;
  footer: string;
  interruption?: BrainInterruptionMetadata | null;
  interruptionAccounting?: InterruptionAccounting | null;
}

function isDeadlineExceeded(deadlineAtMs?: number | null): boolean {
  return typeof deadlineAtMs === "number" && Number.isFinite(deadlineAtMs) && Date.now() >= deadlineAtMs;
}

function createTraversalInterruption(servedPartial: boolean): BrainInterruptionMetadata {
  return {
    interrupted: true,
    stage: "traversal",
    reason: "deadline_during_traversal",
    servedPartial,
  };
}

function initializeSeedScores(
  graph: BrainGraph,
  seedCandidates: Array<{ nodeId: string; score: number }>,
): SeedScore[] {
  return seedCandidates.map((seed) => {
    const learnedSeedWeight = graph.getSeedWeight(seed.nodeId);
    const seedPolicyScore = seed.score + learnedSeedWeight;
    return {
      nodeId: seed.nodeId,
      priorScore: seed.score,
      learnedSeedWeight,
      initialPolicyScore: seedPolicyScore,
      initialProbability: 0,
      latestPolicyScore: seedPolicyScore,
      latestProbability: 0,
      selected: false,
      selectionSubstepIndex: null,
    };
  });
}

function updateSeedScoresForSubstep(
  seedScores: SeedScore[],
  distribution: Array<{ action: TraversalAction; score: number; probability: number }>,
  sampledAction: TraversalAction,
  selectionIndex: number,
): void {
  for (const score of seedScores) {
    const traverseEntry = distribution.find(
      (entry) => entry.action.type === "traverse" && entry.action.targetNodeId === score.nodeId,
    );
    if (!traverseEntry) {
      continue;
    }

    if (selectionIndex === 0) {
      score.initialPolicyScore = traverseEntry.score;
      score.initialProbability = traverseEntry.probability;
    }

    score.latestPolicyScore = traverseEntry.score;
    score.latestProbability = traverseEntry.probability;

    if (
      sampledAction.type === "traverse"
      && sampledAction.targetNodeId === score.nodeId
      && !score.selected
    ) {
      score.selected = true;
      score.selectionSubstepIndex = selectionIndex;
    }
  }
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
    deadlineAtMs,
  } = options;

  const params: PolicyParams = {
    ...DEFAULT_POLICY_PARAMS,
    temperature,
    ...options.policyParams,
  };
  const maxFanoutPerNode = Math.max(0, options.maxFanoutPerNode ?? DEFAULT_MAX_FANOUT_PER_NODE);
  const maxFrontierSize = Math.max(0, options.maxFrontierSize ?? DEFAULT_MAX_FRONTIER_SIZE);

  // Step 1: Seed selection
  const seedCandidates = graph.seedByEmbedding(queryEmbedding, maxSeeds, semanticThreshold);

  if (seedCandidates.length === 0) {
    return {
      firedNodes: [],
      vetoedNodes: [],
      trajectory: [],
      seedScores: [],
      contextChars: 0,
      footer: "Brain · 0 seed candidates · no traversal",
      interruption: null,
    };
  }

  const state: TraversalState = {
    sourceNodeId: null,
    queryEmbedding,
    frontier: [],
    visited: new Set(),
    fired: [],
    budgetRemaining: budgetChars,
    initialBudget: budgetChars,
    reservedTokenCost: 0,
    expansionCount: 0,
    maxHops,
    pendingNodeIds: [],
    maxFrontierSize,
  };

  const trajectory: TrajectoryExpansion[] = [];
  const firedNodes: Array<{ nodeId: string; kind: NodeKind; content: string; tokenCount: number }> = [];
  const vetoedNodes: Array<{ nodeId: string; reason: string }> = [];
  const seedScores = initializeSeedScores(graph, seedCandidates);
  let interruption: BrainInterruptionMetadata | null = null;

  const expandSource = (
    sourceNodeId: string | null,
    frontierBefore: string[],
  ): TrajectoryExpansion => {
    const expansionIndex = state.expansionCount;
    const budgetBefore = state.budgetRemaining;
    const substeps: TrajectorySubstep[] = [];
    const selectedTargets: string[] = [];
    const acceptedTargets: string[] = [];
    const vetoedTargets: Array<{ targetNodeId: string; reason: string }> = [];
    const proposalOutcomes: TrajectoryProposalOutcome[] = [];
    let terminationReason: TrajectoryStopReason | null = null;

    state.sourceNodeId = sourceNodeId;
    state.reservedTokenCost = 0;
    state.pendingNodeIds = [];

    const buildStateSnapshot = (selectionIndex: number): TrajectorySubstep["stateSnapshot"] => ({
      sourceNodeId,
      expansionIndex,
      selectionIndex,
      budgetRemaining: state.budgetRemaining,
      initialBudget: state.initialBudget,
      reservedTokenCost: state.reservedTokenCost,
      maxHops: state.maxHops,
      maxFrontierSize: state.maxFrontierSize,
      frontierSize: state.frontier.length,
      frontierNodeIds: [...state.frontier],
      visitedCount: state.visited.size,
      firedCount: state.fired.length,
      pendingSelectionCount: state.pendingNodeIds?.length ?? 0,
      pendingTargetNodeIds: [...(state.pendingNodeIds ?? [])],
      policyState: computePolicyStateSnapshot(state),
    });

    const buildTrajectoryCandidates = (
      distribution: Array<{
        action: TraversalAction;
        score: number;
        probability: number;
        scoreBreakdown?: TrajectorySubstep["candidates"][number]["scoreBreakdown"];
      }>,
    ): TrajectorySubstep["candidates"] => (
      distribution.map((entry) => ({
        action: entry.action,
        score: entry.score,
        probability: entry.probability,
        priorScore: entry.action.type === "traverse" && sourceNodeId === null ? entry.action.seedScore : undefined,
        learnedSeedWeight: entry.action.type === "traverse" && sourceNodeId === null
          ? graph.getSeedWeight(entry.action.targetNodeId)
          : undefined,
        scoreBreakdown: entry.scoreBreakdown,
      }))
    );

    const appendForcedStopSubstep = (selectionIndex: number, stopReason: TrajectoryStopReason): void => {
      const stopDistribution = softmaxPolicy([{ type: "stop_local" }], state, graph, params);
      const stopProbability = stopDistribution[0]?.probability ?? 1;
      substeps.push({
        stateSnapshot: buildStateSnapshot(selectionIndex),
        candidates: buildTrajectoryCandidates(stopDistribution),
        chosenAction: { type: "stop_local" },
        chosenActionProbability: stopProbability,
        stopProbability,
        stopTruth: "forced",
        stopReason,
      });
      terminationReason = stopReason;
    };

    for (let selectionIndex = 0; ; selectionIndex++) {
      if (isDeadlineExceeded(deadlineAtMs)) {
        appendForcedStopSubstep(selectionIndex, "interrupt");
        interruption ??= createTraversalInterruption(firedNodes.length > 0);
        break;
      }

      const fanoutCapReached = selectedTargets.length >= maxFanoutPerNode;
      const frontierCapReached = state.frontier.length + selectedTargets.length >= maxFrontierSize;
      const budgetCapReached = state.reservedTokenCost >= state.budgetRemaining;
      let forcedStopReason: TrajectoryStopReason | null = null;
      let actions: TraversalAction[];
      if (fanoutCapReached) {
        actions = [{ type: "stop_local" }];
        forcedStopReason = "fanout_cap";
      } else if (frontierCapReached) {
        actions = [{ type: "stop_local" }];
        forcedStopReason = "frontier_cap";
      } else if (budgetCapReached) {
        actions = [{ type: "stop_local" }];
        forcedStopReason = "selection_budget_exhausted";
      } else {
        const affordableBudget = Math.max(0, state.budgetRemaining - state.reservedTokenCost);
        actions = graph.getActionSet(sourceNodeId, state.visited, {
          seeds: sourceNodeId === null ? seedCandidates : undefined,
          excludedTargets: new Set(selectedTargets),
        }).filter((action) => {
          if (action.type !== "traverse") {
            return true;
          }
          const targetNode = graph.getNode(action.targetNodeId);
          return !!targetNode && targetNode.tokenCount <= affordableBudget;
        });

        if (!actions.some((action) => action.type === "traverse")) {
          actions = [{ type: "stop_local" }];
          forcedStopReason = "no_traversable_candidates";
        }
      }

      const distribution = softmaxPolicy(actions, state, graph, params);
      const sampled = sampleAction(distribution);

      if (sourceNodeId === null) {
        updateSeedScoresForSubstep(seedScores, distribution, sampled.action, selectionIndex);
      }

      const stopProbability = distribution.find((entry) => entry.action.type === "stop_local")?.probability ?? 0;
      substeps.push({
        stateSnapshot: buildStateSnapshot(selectionIndex),
        candidates: buildTrajectoryCandidates(distribution),
        chosenAction: sampled.action,
        chosenActionProbability: sampled.probability,
        stopProbability,
        stopTruth: sampled.action.type === "stop_local"
          ? (forcedStopReason ? "forced" : "chosen")
          : undefined,
        stopReason: sampled.action.type === "stop_local"
          ? (forcedStopReason ?? "policy_stop")
          : undefined,
      });

      if (sampled.action.type === "stop_local") {
        terminationReason = forcedStopReason ?? "policy_stop";
        break;
      }

      const targetNode = graph.getNode(sampled.action.targetNodeId);
      if (!targetNode) {
        proposalOutcomes.push({
          targetNodeId: sampled.action.targetNodeId,
          outcome: "dropped",
          reason: "missing_target_node",
        });
        appendForcedStopSubstep(selectionIndex + 1, "missing_target_node");
        break;
      }

      selectedTargets.push(targetNode.id);
      state.pendingNodeIds?.push(targetNode.id);
      state.reservedTokenCost += targetNode.tokenCount;
    }

    for (let index = 0; index < selectedTargets.length; index++) {
      if (isDeadlineExceeded(deadlineAtMs)) {
        for (const droppedTargetNodeId of selectedTargets.slice(index)) {
          proposalOutcomes.push({
            targetNodeId: droppedTargetNodeId,
            outcome: "dropped",
            reason: "interrupt",
          });
        }
        terminationReason ??= "interrupt";
        interruption ??= createTraversalInterruption(firedNodes.length > 0);
        break;
      }

      const targetNodeId = selectedTargets[index];
      const targetNode = graph.getNode(targetNodeId);
      if (!targetNode) {
        proposalOutcomes.push({
          targetNodeId,
          outcome: "dropped",
          reason: "missing_target_node",
        });
        continue;
      }

      if (state.visited.has(targetNodeId)) {
        proposalOutcomes.push({
          targetNodeId,
          outcome: "dropped",
          reason: "already_visited",
        });
        continue;
      }

      if (sourceNodeId !== null && graph.isVetoed(sourceNodeId, targetNodeId)) {
        const reason = graph.getVetoReason(sourceNodeId, targetNodeId) ?? "inhibitory";
        vetoedTargets.push({ targetNodeId, reason });
        vetoedNodes.push({ nodeId: targetNodeId, reason });
        proposalOutcomes.push({
          targetNodeId,
          outcome: "vetoed",
          reason,
        });
        continue;
      }

      if (state.frontier.length >= maxFrontierSize) {
        for (const droppedTargetNodeId of selectedTargets.slice(index)) {
          proposalOutcomes.push({
            targetNodeId: droppedTargetNodeId,
            outcome: "dropped",
            reason: "frontier_cap",
          });
        }
        terminationReason ??= "frontier_cap";
        break;
      }

      if (targetNode.tokenCount > state.budgetRemaining) {
        for (const droppedTargetNodeId of selectedTargets.slice(index)) {
          proposalOutcomes.push({
            targetNodeId: droppedTargetNodeId,
            outcome: "dropped",
            reason: "selection_budget_exhausted",
          });
        }
        terminationReason ??= "selection_budget_exhausted";
        break;
      }

      state.visited.add(targetNodeId);
      state.frontier.push(targetNodeId);
      state.fired.push(targetNodeId);
      state.budgetRemaining -= targetNode.tokenCount;
      acceptedTargets.push(targetNodeId);
      proposalOutcomes.push({
        targetNodeId,
        outcome: "accepted",
        reason: "accepted",
      });
      firedNodes.push({
        nodeId: targetNode.id,
        kind: targetNode.kind,
        content: targetNode.content,
        tokenCount: targetNode.tokenCount,
      });
    }

    const expansion: TrajectoryExpansion = {
      sourceNodeId,
      expansionIndex,
      frontierBefore,
      frontierAfter: [...state.frontier],
      budgetBefore,
      budgetAfter: state.budgetRemaining,
      substeps,
      selectedTargets,
      acceptedTargets,
      vetoedTargets,
      proposalOutcomes,
      terminationReason,
    };

    state.sourceNodeId = null;
    state.reservedTokenCost = 0;
    state.pendingNodeIds = [];
    state.expansionCount++;
    return expansion;
  };

  if (maxHops > 0) {
    trajectory.push(expandSource(null, []));
  }

  while (
    state.frontier.length > 0
    && state.expansionCount < maxHops
    && state.budgetRemaining > 0
  ) {
    if (isDeadlineExceeded(deadlineAtMs)) {
      interruption ??= createTraversalInterruption(firedNodes.length > 0);
      break;
    }
    const frontierBefore = [...state.frontier];
    const nextSourceNodeId = state.frontier.shift();
    if (!nextSourceNodeId) {
      break;
    }
    trajectory.push(expandSource(nextSourceNodeId, frontierBefore));
  }

  const contextChars = firedNodes.reduce((sum, node) => sum + node.content.length, 0);
  const selectedSeedIds = seedScores.filter((seed) => seed.selected).map((seed) => seed.nodeId);

  // Compute interruption accounting for bounded-anytime serving truth surfaces
  const budgetUsed = budgetChars - state.budgetRemaining;
  const budgetUtilization = budgetChars > 0 ? budgetUsed / budgetChars : 0;
  let interruptionAccounting: InterruptionAccounting | null = null;

  if (interruption) {
    const droppedProposalReasons: Record<string, number> = {};
    let droppedProposalCount = 0;
    for (const expansion of trajectory) {
      for (const outcome of expansion.proposalOutcomes ?? []) {
        if (outcome.outcome === "dropped") {
          droppedProposalCount += 1;
          droppedProposalReasons[outcome.reason] = (droppedProposalReasons[outcome.reason] ?? 0) + 1;
        }
      }
    }

    interruptionAccounting = {
      droppedFrontierNodeIds: [...state.frontier],
      completedExpansionCount: state.expansionCount,
      maxExpansions: maxHops,
      budgetUsed,
      budgetTotal: budgetChars,
      budgetUtilization,
      droppedProposalCount,
      droppedProposalReasons,
    };
  }

  const footerParts = [
    "Brain",
    `${seedScores.length} seed candidates`,
    `${selectedSeedIds.length} seed picks`,
    `${trajectory.length} expansions`,
    `${firedNodes.length} fired`,
    `${vetoedNodes.length} veto`,
    `${contextChars} chars`,
  ];

  if (interruption) {
    footerParts.push("INTERRUPTED");
    if (interruptionAccounting) {
      if (interruptionAccounting.droppedFrontierNodeIds.length > 0) {
        footerParts.push(`${interruptionAccounting.droppedFrontierNodeIds.length} frontier dropped`);
      }
      if (interruptionAccounting.droppedProposalCount > 0) {
        footerParts.push(`${interruptionAccounting.droppedProposalCount} proposals dropped`);
      }
      footerParts.push(`${Math.round(budgetUtilization * 100)}% budget used`);
    }
    footerParts.push(interruption.servedPartial ? "partial" : "empty");
  }

  const footer = footerParts.join(" · ");

  return {
    firedNodes,
    vetoedNodes,
    trajectory,
    seedScores,
    contextChars,
    footer,
    interruption,
    interruptionAccounting,
  };
}
