import { describe, it, expect } from "vitest";
import { scoreAction, softmaxPolicy, sampleAction, logProbability } from "../../src/brain-core/policy.js";
import { BrainGraph } from "../../src/brain-core/graph.js";
import { DEFAULT_POLICY_PARAMS } from "../../src/brain-core/types.js";
import type { TraversalState, BrainNode, BrainEdge } from "../../src/brain-core/types.js";

function makeNode(id: string, embedding?: Float32Array): BrainNode {
  return {
    id,
    kind: "chunk",
    content: `content of ${id}`,
    embedding: embedding ?? new Float32Array([1, 0, 0]),
    sourceUri: null,
    trust: "scanner",
    tags: [],
    tokenCount: 100,
    metadata: {},
    createdAt: Date.now(),
    updatedAt: Date.now(),
  };
}

function makeEdge(source: string, target: string, weight = 0.5): BrainEdge {
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

function makeState(sourceNodeId: string | null): TraversalState {
  return {
    sourceNodeId,
    queryEmbedding: new Float32Array([1, 0, 0]),
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

describe("policy", () => {
  describe("scoreAction", () => {
    it("stop_local score increases with budget depletion", () => {
      const graph = new BrainGraph();
      const lowBudget = { ...makeState(null), budgetRemaining: 100, expansionCount: 5 };
      const highBudget = { ...makeState(null), budgetRemaining: 900, expansionCount: 1 };

      const scoreLow = scoreAction({ type: "stop_local" }, lowBudget, graph);
      const scoreHigh = scoreAction({ type: "stop_local" }, highBudget, graph);

      expect(scoreLow).toBeGreaterThan(scoreHigh);
    });

    it("stop_local score incorporates learned source-local stop weights", () => {
      const graph = new BrainGraph();
      graph.addNode(makeNode("a"));
      graph.setStopLocalWeight(null, 0.4);
      graph.setStopLocalWeight("a", 1.2);

      const seedScore = scoreAction({ type: "stop_local" }, makeState(null), graph);
      const localScore = scoreAction({ type: "stop_local" }, makeState("a"), graph);

      expect(localScore).toBeGreaterThan(seedScore);
    });

    it("traverse score incorporates edge weight and embedding similarity", () => {
      const graph = new BrainGraph();
      graph.addNode(makeNode("a"));
      graph.addNode(makeNode("b", new Float32Array([1, 0, 0]))); // same as query
      graph.addNode(makeNode("c", new Float32Array([0, 1, 0]))); // orthogonal to query
      graph.addEdge(makeEdge("a", "b", 1.0));
      graph.addEdge(makeEdge("a", "c", 0.1));

      const state = makeState("a");
      const scoreB = scoreAction({ type: "traverse", targetNodeId: "b" }, state, graph);
      const scoreC = scoreAction({ type: "traverse", targetNodeId: "c" }, state, graph);

      expect(scoreB).toBeGreaterThan(scoreC);
    });

    it("learns seed-head preference from explicit seed weights", () => {
      const graph = new BrainGraph();
      graph.addNode(makeNode("b", new Float32Array([1, 0, 0])));
      graph.addNode(makeNode("c", new Float32Array([1, 0, 0])));
      graph.setSeedWeight("b", 1.5);

      const state = makeState(null);
      const scoreB = scoreAction({ type: "traverse", targetNodeId: "b", seedScore: 0.8 }, state, graph);
      const scoreC = scoreAction({ type: "traverse", targetNodeId: "c", seedScore: 0.8 }, state, graph);

      expect(scoreB).toBeGreaterThan(scoreC);
    });
  });

  describe("softmaxPolicy", () => {
    it("returns valid probability distribution summing to 1", () => {
      const graph = new BrainGraph();
      graph.addNode(makeNode("a"));
      graph.addNode(makeNode("b"));
      graph.addEdge(makeEdge("a", "b"));

      const state = makeState("a");
      const actions = graph.getActionSet("a", new Set());
      const dist = softmaxPolicy(actions, state, graph);

      const sum = dist.reduce((s, d) => s + d.probability, 0);
      expect(sum).toBeCloseTo(1.0, 5);
      for (const d of dist) {
        expect(d.probability).toBeGreaterThan(0);
        expect(d.probability).toBeLessThanOrEqual(1);
      }
    });

    it("higher temperature produces more uniform distribution", () => {
      const graph = new BrainGraph();
      graph.addNode(makeNode("a"));
      graph.addNode(makeNode("b"));
      graph.addNode(makeNode("c", new Float32Array([0, 1, 0])));
      graph.addEdge(makeEdge("a", "b", 2.0));
      graph.addEdge(makeEdge("a", "c", 0.1));

      const state = makeState("a");
      const actions = graph.getActionSet("a", new Set());

      const lowTemp = softmaxPolicy(actions, state, graph, { ...DEFAULT_POLICY_PARAMS, temperature: 0.1 });
      const highTemp = softmaxPolicy(actions, state, graph, { ...DEFAULT_POLICY_PARAMS, temperature: 5.0 });

      // With high temp, probabilities should be more uniform
      const lowTempMax = Math.max(...lowTemp.map((d) => d.probability));
      const highTempMax = Math.max(...highTemp.map((d) => d.probability));
      expect(lowTempMax).toBeGreaterThan(highTempMax);
    });
  });

  describe("sampleAction", () => {
    it("returns a valid action from the distribution", () => {
      const dist = [
        { action: { type: "traverse" as const, targetNodeId: "b" }, probability: 0.7 },
        { action: { type: "stop_local" as const }, probability: 0.3 },
      ];

      const result = sampleAction(dist);
      expect(result.action.type === "traverse" || result.action.type === "stop_local").toBe(true);
      expect(result.probability).toBeGreaterThan(0);
    });

    it("samples stochastically (not always argmax)", () => {
      const dist = [
        { action: { type: "traverse" as const, targetNodeId: "b" }, probability: 0.6 },
        { action: { type: "stop_local" as const }, probability: 0.4 },
      ];

      // Run many samples — both actions should appear
      const counts = { traverse: 0, stop_local: 0 };
      for (let i = 0; i < 200; i++) {
        const result = sampleAction(dist);
        counts[result.action.type]++;
      }

      expect(counts.traverse).toBeGreaterThan(0);
      expect(counts.stop_local).toBeGreaterThan(0);
    });
  });

  describe("logProbability", () => {
    it("returns negative value for probabilities < 1", () => {
      expect(logProbability(0.5)).toBeLessThan(0);
      expect(logProbability(0.1)).toBeLessThan(logProbability(0.5));
    });

    it("returns 0 for probability = 1", () => {
      expect(logProbability(1.0)).toBeCloseTo(0);
    });
  });
});
