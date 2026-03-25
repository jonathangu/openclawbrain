import { describe, expect, it } from "vitest";
import { BrainGraph } from "../../src/brain-core/graph.js";
import { softmaxPolicy } from "../../src/brain-core/policy.js";
import { applyWeightUpdates, computeReinforceUpdates } from "../../src/brain-core/update.js";
import { DEFAULT_POLICY_PARAMS } from "../../src/brain-core/types.js";
import type {
  BrainNode,
  Episode,
  TrajectoryExpansion,
  TrajectoryStep,
  TraversalAction,
  TraversalState,
} from "../../src/brain-core/types.js";

function makeNode(id: string, embedding = new Float32Array([1, 0, 0])): BrainNode {
  return {
    id,
    kind: "chunk",
    content: `content of ${id}`,
    embedding,
    sourceUri: null,
    trust: "scanner",
    tags: [],
    tokenCount: 100,
    metadata: {},
    createdAt: Date.now(),
    updatedAt: Date.now(),
  };
}

function makeSeedState(queryEmbedding = new Float32Array([1, 0, 0])): TraversalState {
  return {
    sourceNodeId: null,
    queryEmbedding,
    frontier: [],
    visited: new Set(),
    fired: [],
    budgetRemaining: 1000,
    initialBudget: 1000,
    reservedTokenCost: 0,
    expansionCount: 0,
    maxHops: 8,
  };
}

function makeSeedEpisode(nodeId: string, probability: number, reward: number): Episode {
  const step: TrajectoryStep = {
    stateSnapshot: {
      sourceNodeId: null,
      expansionIndex: 0,
      selectionIndex: 0,
      budgetRemaining: 1000,
      initialBudget: 1000,
      reservedTokenCost: 0,
      maxHops: 8,
      frontierSize: 0,
      frontierNodeIds: [],
      visitedCount: 0,
      firedCount: 0,
    },
    candidates: [
      {
        action: { type: "traverse", targetNodeId: nodeId, seedScore: 0.8 } as TraversalAction,
        score: 0.8,
        probability,
        priorScore: 0.8,
        learnedSeedWeight: 0,
      },
      {
        action: { type: "stop_local" },
        score: 0.1,
        probability: 1 - probability,
      },
    ],
    chosenAction: { type: "traverse", targetNodeId: nodeId, seedScore: 0.8 },
    chosenActionProbability: probability,
    stopProbability: 1 - probability,
  };
  const expansion: TrajectoryExpansion = {
    sourceNodeId: null,
    expansionIndex: 0,
    frontierBefore: [],
    frontierAfter: [nodeId],
    budgetBefore: 1000,
    budgetAfter: 900,
    substeps: [step],
    selectedTargets: [nodeId],
    acceptedTargets: [nodeId],
    vetoedTargets: [],
  };

  return {
    id: "be_seed_test",
    conversationId: null,
    queryText: "seed query",
    queryEmbedding: new Float32Array([1, 0, 0]),
    trajectory: [expansion],
    firedNodes: [nodeId],
    vetoedNodes: [],
    contextChars: 0,
    reward,
    rewardSource: "human",
    packVersion: 1,
    createdAt: Date.now(),
  };
}

describe("seed policy", () => {
  it("positive reward strengthens the chosen seed weight", () => {
    const graph = new BrainGraph();
    graph.addNode(makeNode("a"));

    const updates = computeReinforceUpdates(makeSeedEpisode("a", 0.6, 1.0), 0.1, 0.0);
    applyWeightUpdates(graph, updates);

    expect(graph.getSeedWeight("a")).toBeGreaterThan(0);
  });

  it("negative reward weakens the chosen seed weight", () => {
    const graph = new BrainGraph();
    graph.addNode(makeNode("a"));
    graph.setSeedWeight("a", 0.5);

    const updates = computeReinforceUpdates(makeSeedEpisode("a", 0.6, -1.0), 0.1, 0.0);
    applyWeightUpdates(graph, updates);

    expect(graph.getSeedWeight("a")).toBeLessThan(0.5);
  });

  it("STOP can win at seed phase for low-signal queries", () => {
    const graph = new BrainGraph();
    graph.addNode(makeNode("a", new Float32Array([0, 1, 0])));

    const actions: TraversalAction[] = [
      { type: "traverse", targetNodeId: "a", seedScore: 0.05 },
      { type: "stop_local" },
    ];
    const dist = softmaxPolicy(actions, makeSeedState(new Float32Array([1, 0, 0])), graph, {
      ...DEFAULT_POLICY_PARAMS,
      stopBias: 0.4,
      temperature: 0.2,
    });

    const stop = dist.find((entry) => entry.action.type === "stop_local");
    const seed = dist.find((entry) => entry.action.type === "traverse");
    expect(stop?.probability ?? 0).toBeGreaterThan(seed?.probability ?? 0);
  });
});
