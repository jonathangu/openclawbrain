import { describe, it, expect } from "vitest";
import { BrainGraph } from "../../src/brain-core/graph.js";
import { replayEpisode } from "../../src/brain-core/replay.js";
import type { BrainEdge, BrainNode, Episode, TrajectoryExpansion, TrajectoryStep } from "../../src/brain-core/types.js";

function makeNode(id: string): BrainNode {
  return {
    id,
    kind: "chunk",
    content: id,
    embedding: new Float32Array([1, 0, 0]),
    sourceUri: "test.md",
    trust: "scanner",
    tags: [],
    tokenCount: 10,
    metadata: {},
    createdAt: Date.now(),
    updatedAt: Date.now(),
  };
}

function makeEdge(source: string, target: string, weight: number): BrainEdge {
  return {
    source,
    target,
    kind: "learned",
    weight,
    prior: 0.5,
    metadata: {},
    decayedAt: Date.now(),
    createdAt: Date.now(),
  };
}

function makeStep(source: string, target: string, alternative: string, probability: number): TrajectoryStep {
  return {
    stateSnapshot: {
      sourceNodeId: source,
      expansionIndex: 0,
      selectionIndex: 0,
      budgetRemaining: 100,
      initialBudget: 100,
      reservedTokenCost: 0,
      maxHops: 4,
      frontierSize: 0,
      frontierNodeIds: [],
      visitedCount: 0,
      firedCount: 0,
    },
    candidates: [
      { action: { type: "traverse", targetNodeId: target }, score: 1, probability },
      { action: { type: "traverse", targetNodeId: alternative }, score: 0.9, probability: 0.15 },
      { action: { type: "stop_local" }, score: 0, probability: 1 - probability - 0.15 },
    ],
    chosenAction: { type: "traverse", targetNodeId: target },
    chosenActionProbability: probability,
    stopProbability: 1 - probability - 0.15,
  };
}

function makeExpansion(source: string, target: string, alternative: string, probability: number): TrajectoryExpansion {
  const substep = makeStep(source, target, alternative, probability);
  return {
    sourceNodeId: source,
    expansionIndex: 0,
    frontierBefore: [source],
    frontierAfter: [],
    budgetBefore: 100,
    budgetAfter: 90,
    substeps: [substep],
    selectedTargets: [target],
    acceptedTargets: [target],
    vetoedTargets: [],
  };
}

describe("replay", () => {
  it("flags human-positive regressions when a different route would now win", () => {
    const graph = new BrainGraph();
    graph.addNode(makeNode("a"));
    graph.addNode(makeNode("b"));
    graph.addNode(makeNode("c"));
    graph.addEdge(makeEdge("a", "b", 0.1));
    graph.addEdge(makeEdge("a", "c", 2));

    const episode: Episode = {
      id: "ep",
      conversationId: 1,
      queryText: "test",
      queryEmbedding: new Float32Array([1, 0, 0]),
      trajectory: [makeExpansion("a", "b", "c", 0.8)],
      firedNodes: ["b"],
      vetoedNodes: [],
      contextChars: 0,
      reward: 1,
      rewardSource: "human",
      packVersion: 1,
      createdAt: Date.now(),
    };

    const replay = replayEpisode(episode, graph);
    expect(replay.wouldChange).toBe(true);
  });

  it("ignores explicitly forced stop_local substeps when deciding whether replay would change", () => {
    const graph = new BrainGraph();
    graph.addNode(makeNode("a"));
    graph.addNode(makeNode("b"));
    graph.addNode(makeNode("c"));
    graph.addEdge(makeEdge("a", "b", 0.1));
    graph.addEdge(makeEdge("a", "c", 2));

    const forcedStopEpisode: Episode = {
      id: "ep-forced-stop",
      conversationId: 1,
      queryText: "test",
      queryEmbedding: new Float32Array([1, 0, 0]),
      trajectory: [{
        sourceNodeId: "a",
        expansionIndex: 0,
        frontierBefore: ["a"],
        frontierAfter: [],
        budgetBefore: 100,
        budgetAfter: 100,
        substeps: [
          {
            stateSnapshot: {
              sourceNodeId: "a",
              expansionIndex: 0,
              selectionIndex: 0,
              budgetRemaining: 100,
              initialBudget: 100,
              reservedTokenCost: 0,
              maxHops: 4,
              frontierSize: 0,
              frontierNodeIds: [],
              visitedCount: 0,
              firedCount: 0,
            },
            candidates: [
              { action: { type: "traverse", targetNodeId: "b" }, score: 0.2, probability: 0.2 },
              { action: { type: "traverse", targetNodeId: "c" }, score: 0.8, probability: 0.7 },
              { action: { type: "stop_local" }, score: 0, probability: 0.1 },
            ],
            chosenAction: { type: "stop_local" },
            chosenActionProbability: 0.1,
            stopProbability: 0.1,
            stopTruth: "forced",
            stopReason: "frontier_cap",
          },
        ],
        selectedTargets: [],
        acceptedTargets: [],
        vetoedTargets: [],
        terminationReason: "frontier_cap",
      }],
      firedNodes: [],
      vetoedNodes: [],
      contextChars: 0,
      reward: 1,
      rewardSource: "human",
      packVersion: 1,
      createdAt: Date.now(),
    };

    const replay = replayEpisode(forcedStopEpisode, graph);
    expect(replay.wouldChange).toBe(false);
    expect(replay.firedNodes).toEqual([]);
  });
});
