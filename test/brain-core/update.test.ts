import { describe, it, expect } from "vitest";
import { computeReinforceUpdates, updateBaseline, applyWeightUpdates } from "../../src/brain-core/update.js";
import { BrainGraph } from "../../src/brain-core/graph.js";
import { START_NODE_ID } from "../../src/brain-core/types.js";
import type { Episode, TrajectoryExpansion, TrajectoryStep, BrainNode, BrainEdge } from "../../src/brain-core/types.js";

function makeNode(id: string): BrainNode {
  return {
    id, kind: "chunk", content: `content of ${id}`,
    embedding: new Float32Array([1, 0, 0]), sourceUri: null,
    trust: "scanner", tags: [], tokenCount: 100, metadata: {},
    createdAt: Date.now(), updatedAt: Date.now(),
  };
}

function makeEdge(source: string, target: string, weight = 0.5): BrainEdge {
  return {
    source, target, kind: "learned", weight, prior: 0.5,
    metadata: {}, decayedAt: Date.now(), createdAt: Date.now(),
  };
}

function makeStep(sourceId: string | null, targetId: string, prob: number, expansionIndex = 0): TrajectoryStep {
  return {
    stateSnapshot: {
      sourceNodeId: sourceId,
      expansionIndex,
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
      { action: { type: "traverse", targetNodeId: targetId }, score: 1, probability: prob },
      { action: { type: "stop_local" }, score: -1, probability: 1 - prob },
    ],
    chosenAction: { type: "traverse", targetNodeId: targetId },
    chosenActionProbability: prob,
    stopProbability: 1 - prob,
  };
}

function makeExpansion(sourceId: string | null, targetId: string, prob: number, expansionIndex = 0): TrajectoryExpansion {
  const substep = makeStep(sourceId, targetId, prob, expansionIndex);
  return {
    sourceNodeId: sourceId,
    expansionIndex,
    frontierBefore: sourceId === null ? [] : [sourceId],
    frontierAfter: [],
    budgetBefore: 1000,
    budgetAfter: 900,
    substeps: [substep],
    selectedTargets: [targetId],
    acceptedTargets: [targetId],
    vetoedTargets: [],
  };
}

function makeEpisode(trajectory: TrajectoryExpansion[], reward: number | null): Episode {
  return {
    id: "test-ep", conversationId: null, queryText: "test",
    queryEmbedding: null, trajectory, firedNodes: [], vetoedNodes: [],
    contextChars: 0, reward, rewardSource: reward !== null ? "self" : null,
    packVersion: null, createdAt: Date.now(),
  };
}

describe("update (REINFORCE, Lemma 6.1)", () => {
  it("positive reward strengthens chosen edges", () => {
    const episode = makeEpisode([makeExpansion("a", "b", 0.6)], 1.0);
    const updates = computeReinforceUpdates(episode, 0.1, 0.0);

    expect(updates.length).toBe(1);
    expect(updates[0]).toMatchObject({ kind: "edge", source: "a", target: "b" });
    expect(updates[0].delta).toBeGreaterThan(0); // Positive reward → strengthen
  });

  it("negative reward weakens chosen edges", () => {
    const episode = makeEpisode([makeExpansion("a", "b", 0.6)], -1.0);
    const updates = computeReinforceUpdates(episode, 0.1, 0.0);

    expect(updates.length).toBe(1);
    expect(updates[0].delta).toBeLessThan(0); // Negative reward → weaken
  });

  it("baseline reduces update magnitude", () => {
    const episode = makeEpisode([makeExpansion("a", "b", 0.6)], 0.5);

    const updatesNoBaseline = computeReinforceUpdates(episode, 0.1, 0.0);
    const updatesWithBaseline = computeReinforceUpdates(episode, 0.1, 0.4);

    // With baseline closer to reward, advantage is smaller → smaller update
    expect(Math.abs(updatesWithBaseline[0].delta)).toBeLessThan(
      Math.abs(updatesNoBaseline[0].delta),
    );
  });

  it("full-trajectory credit: ALL steps get credit, not just last", () => {
    // Episode with 3 steps: a→b, b→c, c→d
    const trajectory = [
      makeExpansion("a", "b", 0.5, 0),
      makeExpansion("b", "c", 0.5, 1),
      makeExpansion("c", "d", 0.5, 2),
    ];
    const episode = makeEpisode(trajectory, 1.0);
    const updates = computeReinforceUpdates(episode, 0.1, 0.0);

    // All 3 edges should receive updates (full-trajectory sum)
    expect(updates.length).toBe(3);

    const edges = updates
      .filter((u): u is Extract<(typeof updates)[number], { kind: "edge" }> => u.kind === "edge")
      .map((u) => `${u.source}→${u.target}`);
    expect(edges).toContain("a→b");
    expect(edges).toContain("b→c");
    expect(edges).toContain("c→d");

    // All updates should be positive (positive reward, zero baseline)
    for (const u of updates) {
      expect(u.delta).toBeGreaterThan(0);
    }
  });

  it("updates seed-phase transitions through explicit seed weights", () => {
    const episode = makeEpisode([makeExpansion(null, "b", 0.6)], 1.0);
    const updates = computeReinforceUpdates(episode, 0.1, 0.0);

    expect(updates).toEqual([
      expect.objectContaining({
        kind: "seed",
        nodeId: "b",
      }),
    ]);
    expect(updates[0]?.delta).toBeGreaterThan(0);
  });

  it("zero advantage produces no updates", () => {
    const episode = makeEpisode([makeExpansion("a", "b", 0.5)], 0.5);
    const updates = computeReinforceUpdates(episode, 0.1, 0.5); // baseline = reward
    expect(updates.length).toBe(0);
  });

  it("null reward produces no updates", () => {
    const episode = makeEpisode([makeExpansion("a", "b", 0.5)], null);
    const updates = computeReinforceUpdates(episode, 0.1, 0.0);
    expect(updates.length).toBe(0);
  });

  it("emits learned updates for chosen stop_local substeps", () => {
    const stopExpansion: TrajectoryExpansion = {
      sourceNodeId: "a",
      expansionIndex: 0,
      frontierBefore: ["a"],
      frontierAfter: [],
      budgetBefore: 1000,
      budgetAfter: 1000,
      substeps: [
        {
          stateSnapshot: {
            sourceNodeId: "a",
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
            { action: { type: "traverse", targetNodeId: "b" }, score: 0.2, probability: 0.2 },
            { action: { type: "stop_local" }, score: 0.8, probability: 0.8 },
          ],
          chosenAction: { type: "stop_local" },
          chosenActionProbability: 0.8,
          stopProbability: 0.8,
        },
      ],
      selectedTargets: [],
      acceptedTargets: [],
      vetoedTargets: [],
    };

    const updates = computeReinforceUpdates(makeEpisode([stopExpansion], 1.0), 0.1, 0.0);
    expect(updates).toEqual([
      expect.objectContaining({
        kind: "stop_local",
        sourceNodeId: "a",
      }),
    ]);
    expect(updates[0]?.delta).toBeGreaterThan(0);
  });

  it("negative reward weakens chosen stop_local weights", () => {
    const stopExpansion: TrajectoryExpansion = {
      sourceNodeId: null,
      expansionIndex: 0,
      frontierBefore: [],
      frontierAfter: [],
      budgetBefore: 1000,
      budgetAfter: 1000,
      substeps: [
        {
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
            { action: { type: "traverse", targetNodeId: "b", seedScore: 0.2 }, score: 0.2, probability: 0.2 },
            { action: { type: "stop_local" }, score: 0.8, probability: 0.8 },
          ],
          chosenAction: { type: "stop_local" },
          chosenActionProbability: 0.8,
          stopProbability: 0.8,
        },
      ],
      selectedTargets: [],
      acceptedTargets: [],
      vetoedTargets: [],
    };

    const updates = computeReinforceUpdates(makeEpisode([stopExpansion], -1.0), 0.1, 0.0);
    expect(updates).toEqual([
      expect.objectContaining({
        kind: "stop_local",
        sourceNodeId: START_NODE_ID,
      }),
    ]);
    expect(updates[0]?.delta).toBeLessThan(0);
  });

  it("does not emit learned stop_local updates for explicitly forced stops", () => {
    const stopExpansion: TrajectoryExpansion = {
      sourceNodeId: "a",
      expansionIndex: 0,
      frontierBefore: ["a"],
      frontierAfter: [],
      budgetBefore: 1000,
      budgetAfter: 1000,
      substeps: [
        {
          stateSnapshot: {
            sourceNodeId: "a",
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
            { action: { type: "traverse", targetNodeId: "b" }, score: 0.2, probability: 0.2 },
            { action: { type: "stop_local" }, score: 0.8, probability: 0.8 },
          ],
          chosenAction: { type: "stop_local" },
          chosenActionProbability: 0.8,
          stopProbability: 0.8,
          stopTruth: "forced",
          stopReason: "frontier_cap",
        },
      ],
      selectedTargets: [],
      acceptedTargets: [],
      vetoedTargets: [],
      terminationReason: "frontier_cap",
    };

    const updates = computeReinforceUpdates(makeEpisode([stopExpansion], 1.0), 0.1, 0.0);
    expect(updates).toEqual([]);
  });

  it("does not emit fake stop_local updates when STOP_LOCAL is forced", () => {
    const updates = computeReinforceUpdates(makeEpisode([{
      sourceNodeId: "a",
      expansionIndex: 0,
      frontierBefore: ["a"],
      frontierAfter: [],
      budgetBefore: 1000,
      budgetAfter: 1000,
      substeps: [
        {
          stateSnapshot: {
            sourceNodeId: "a",
            expansionIndex: 0,
            selectionIndex: 0,
            budgetRemaining: 1000,
            initialBudget: 1000,
            reservedTokenCost: 1000,
            maxHops: 8,
            frontierSize: 0,
            frontierNodeIds: [],
            visitedCount: 0,
            firedCount: 0,
          },
          candidates: [
            { action: { type: "stop_local" }, score: 0.8, probability: 1 },
          ],
          chosenAction: { type: "stop_local" },
          chosenActionProbability: 1,
          stopProbability: 1,
        },
      ],
      selectedTargets: [],
      acceptedTargets: [],
      vetoedTargets: [],
    }], 1.0), 0.1, 0.0);
    expect(updates).toEqual([]);
  });

  describe("updateBaseline", () => {
    it("moves baseline toward new reward", () => {
      const newBaseline = updateBaseline(0.0, 1.0, 0.1);
      expect(newBaseline).toBeCloseTo(0.1);
    });

    it("converges over many updates", () => {
      let baseline = 0.0;
      for (let i = 0; i < 100; i++) {
        baseline = updateBaseline(baseline, 0.5, 0.1);
      }
      expect(baseline).toBeCloseTo(0.5, 1);
    });
  });

  describe("applyWeightUpdates", () => {
    it("modifies edge weight in graph", () => {
      const graph = new BrainGraph();
      graph.addNode(makeNode("a"));
      graph.addNode(makeNode("b"));
      graph.addEdge(makeEdge("a", "b", 0.5));

      applyWeightUpdates(graph, [{ kind: "edge", source: "a", target: "b", delta: 0.2 }]);

      const edge = graph.getEdge("a", "b");
      expect(edge?.weight).toBeCloseTo(0.7);
    });

    it("creates valid updates for seed weights too", () => {
      const graph = new BrainGraph();
      graph.addNode(makeNode("b"));
      graph.setSeedWeight("b", 0.5);

      applyWeightUpdates(graph, [{ kind: "seed", nodeId: "b", delta: 0.2 }]);

      expect(graph.getSeedWeight("b")).toBeCloseTo(0.7);
    });

    it("creates valid updates for stop_local weights too", () => {
      const graph = new BrainGraph();
      graph.addNode(makeNode("a"));
      graph.setStopLocalWeight("a", 0.5);

      applyWeightUpdates(graph, [{ kind: "stop_local", sourceNodeId: "a", delta: 0.2 }]);

      expect(graph.getStopLocalWeight("a")).toBeCloseTo(0.7);
    });

    it("clamps weights to [-10, 10]", () => {
      const graph = new BrainGraph();
      graph.addNode(makeNode("a"));
      graph.addNode(makeNode("b"));
      graph.addEdge(makeEdge("a", "b", 9.5));

      applyWeightUpdates(graph, [{ kind: "edge", source: "a", target: "b", delta: 5.0 }]);

      const edge = graph.getEdge("a", "b");
      expect(edge?.weight).toBe(10);
    });
  });
});
