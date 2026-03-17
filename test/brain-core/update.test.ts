import { describe, it, expect } from "vitest";
import { computeReinforceUpdates, updateBaseline, applyWeightUpdates } from "../../src/brain-core/update.js";
import { BrainGraph } from "../../src/brain-core/graph.js";
import type { Episode, TrajectoryStep, BrainNode, BrainEdge } from "../../src/brain-core/types.js";

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

function makeStep(sourceId: string | null, targetId: string, prob: number): TrajectoryStep {
  return {
    stateSnapshot: {
      currentNodeId: sourceId, hopCount: 0,
      budgetRemaining: 1000, visitedCount: 0, firedCount: 0,
    },
    candidates: [
      { action: { type: "traverse", targetNodeId: targetId }, score: 1, probability: prob },
      { action: { type: "stop" }, score: -1, probability: 1 - prob },
    ],
    chosenAction: { type: "traverse", targetNodeId: targetId },
    chosenActionProbability: prob,
    stopProbability: 1 - prob,
  };
}

function makeEpisode(trajectory: TrajectoryStep[], reward: number | null): Episode {
  return {
    id: "test-ep", conversationId: null, queryText: "test",
    queryEmbedding: null, trajectory, firedNodes: [], vetoedNodes: [],
    contextChars: 0, reward, rewardSource: reward !== null ? "self" : null,
    packVersion: null, createdAt: Date.now(),
  };
}

describe("update (REINFORCE, Lemma 6.1)", () => {
  it("positive reward strengthens chosen edges", () => {
    const episode = makeEpisode([makeStep("a", "b", 0.6)], 1.0);
    const updates = computeReinforceUpdates(episode, 0.1, 0.0);

    expect(updates.length).toBe(1);
    expect(updates[0]).toMatchObject({ kind: "edge", source: "a", target: "b" });
    expect(updates[0].delta).toBeGreaterThan(0); // Positive reward → strengthen
  });

  it("negative reward weakens chosen edges", () => {
    const episode = makeEpisode([makeStep("a", "b", 0.6)], -1.0);
    const updates = computeReinforceUpdates(episode, 0.1, 0.0);

    expect(updates.length).toBe(1);
    expect(updates[0].delta).toBeLessThan(0); // Negative reward → weaken
  });

  it("baseline reduces update magnitude", () => {
    const episode = makeEpisode([makeStep("a", "b", 0.6)], 0.5);

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
      makeStep("a", "b", 0.5),
      makeStep("b", "c", 0.5),
      makeStep("c", "d", 0.5),
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
    const episode = makeEpisode([makeStep(null, "b", 0.6)], 1.0);
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
    const episode = makeEpisode([makeStep("a", "b", 0.5)], 0.5);
    const updates = computeReinforceUpdates(episode, 0.1, 0.5); // baseline = reward
    expect(updates.length).toBe(0);
  });

  it("null reward produces no updates", () => {
    const episode = makeEpisode([makeStep("a", "b", 0.5)], null);
    const updates = computeReinforceUpdates(episode, 0.1, 0.0);
    expect(updates.length).toBe(0);
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
