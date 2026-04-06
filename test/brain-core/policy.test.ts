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

function makeToolNode(id: string, embedding?: Float32Array): BrainNode {
  return {
    ...makeNode(id, embedding),
    kind: "toolcard",
    tags: ["candidate_type:tool", "action_kind:tool"],
    metadata: {
      candidate_type: "tool",
      action_kind: "tool",
    },
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

    it("stop_local score increases under frontier backlog pressure", () => {
      const graph = new BrainGraph();
      const lowPressure = { ...makeState("a"), frontier: ["b"], expansionCount: 1, maxHops: 8 };
      const highPressure = {
        ...makeState("a"),
        frontier: ["b", "c", "d"],
        expansionCount: 6,
        maxHops: 8,
      };

      const scoreLow = scoreAction({ type: "stop_local" }, lowPressure, graph);
      const scoreHigh = scoreAction({ type: "stop_local" }, highPressure, graph);

      expect(scoreHigh).toBeGreaterThan(scoreLow);
    });

    it("stop_local score increases when pending local picks consume frontier room", () => {
      const graph = new BrainGraph();
      const relaxedState = {
        ...makeState("a"),
        expansionCount: 2,
        maxHops: 6,
        maxFrontierSize: 3,
      };
      const pressuredState = {
        ...relaxedState,
        pendingNodeIds: ["b", "c"],
      };

      const relaxedScore = scoreAction({ type: "stop_local" }, relaxedState, graph);
      const pressuredScore = scoreAction({ type: "stop_local" }, pressuredState, graph);

      expect(pressuredScore).toBeGreaterThan(relaxedScore);
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

    it("penalizes high-cost branch opportunities when budget and frontier are tight", () => {
      const graph = new BrainGraph();
      graph.addNode(makeNode("a"));
      graph.addNode(makeNode("b", new Float32Array([1, 0, 0])));
      graph.addNode(makeNode("c", new Float32Array([1, 0, 0])));
      graph.addNode(makeNode("d", new Float32Array([1, 0, 0])));
      graph.addNode(makeNode("e", new Float32Array([1, 0, 0])));
      graph.addNode(makeNode("f", new Float32Array([1, 0, 0])));
      graph.addNode({ ...makeNode("g", new Float32Array([1, 0, 0])), tokenCount: 360 });
      graph.addNode({ ...makeNode("h", new Float32Array([1, 0, 0])), tokenCount: 90 });
      graph.addEdge(makeEdge("a", "g", 1.2));
      graph.addEdge(makeEdge("a", "h", 1.2));
      graph.addEdge(makeEdge("g", "d", 1));
      graph.addEdge(makeEdge("g", "e", 1));
      graph.addEdge(makeEdge("g", "f", 1));

      const pressuredState = {
        ...makeState("a"),
        budgetRemaining: 400,
        initialBudget: 1000,
        frontier: ["b", "c"],
        expansionCount: 6,
        maxHops: 8,
      };

      const scoreHighCost = scoreAction({ type: "traverse", targetNodeId: "g" }, pressuredState, graph);
      const scoreLowCost = scoreAction({ type: "traverse", targetNodeId: "h" }, pressuredState, graph);

      expect(scoreLowCost).toBeGreaterThan(scoreHighCost);
    });

    it("penalizes locally redundant targets against nearby selected evidence", () => {
      const graph = new BrainGraph();
      graph.addNode(makeNode("source", new Float32Array([1, 0, 0])));
      graph.addNode(makeNode("frontier-1", new Float32Array([0, 1, 0])));
      graph.addNode(makeNode("redundant", new Float32Array([0, 1, 0])));
      graph.addNode(makeNode("novel", new Float32Array([0, 0, 1])));
      graph.addEdge(makeEdge("source", "redundant", 1));
      graph.addEdge(makeEdge("source", "novel", 1));

      const state = {
        ...makeState("source"),
        frontier: ["frontier-1"],
        fired: ["frontier-1"],
      };

      const redundantScore = scoreAction({ type: "traverse", targetNodeId: "redundant" }, state, graph);
      const novelScore = scoreAction({ type: "traverse", targetNodeId: "novel" }, state, graph);

      expect(novelScore).toBeGreaterThan(redundantScore);
    });

    it("penalizes targets that are redundant with already-pending local selections", () => {
      const graph = new BrainGraph();
      graph.addNode(makeNode("source", new Float32Array([1, 0, 0])));
      graph.addNode(makeNode("picked", new Float32Array([0, 1, 0])));
      graph.addNode(makeNode("redundant", new Float32Array([0, 1, 0])));
      graph.addNode(makeNode("novel", new Float32Array([0, 0, 1])));
      graph.addEdge(makeEdge("source", "redundant", 1));
      graph.addEdge(makeEdge("source", "novel", 1));

      const state = {
        ...makeState("source"),
        pendingNodeIds: ["picked"],
      };

      const redundantScore = scoreAction({ type: "traverse", targetNodeId: "redundant" }, state, graph);
      const novelScore = scoreAction({ type: "traverse", targetNodeId: "novel" }, state, graph);

      expect(novelScore).toBeGreaterThan(redundantScore);
    });

    it("rewards higher-quality nearby evidence when other signals are similar", () => {
      const graph = new BrainGraph();
      graph.addNode(makeNode("a"));
      graph.addNode({ ...makeNode("human-backed"), trust: "human" });
      graph.addNode({ ...makeNode("lower-trust"), trust: "teacher" });
      graph.addNode(makeNode("support-1"));
      graph.addNode(makeNode("support-2"));
      graph.addEdge(makeEdge("a", "human-backed", 0.6));
      graph.addEdge(makeEdge("a", "lower-trust", 0.6));
      graph.addEdge(makeEdge("support-1", "human-backed", 0.2));
      graph.addEdge(makeEdge("support-2", "human-backed", 0.2));

      const state = makeState("a");
      const humanBackedScore = scoreAction({ type: "traverse", targetNodeId: "human-backed" }, state, graph);
      const lowerTrustScore = scoreAction({ type: "traverse", targetNodeId: "lower-trust" }, state, graph);

      expect(humanBackedScore).toBeGreaterThan(lowerTrustScore);
    });

    it("raises branch opportunity cost when pending picks leave little frontier room", () => {
      const graph = new BrainGraph();
      graph.addNode(makeNode("a"));
      graph.addNode(makeNode("queued"));
      graph.addNode({ ...makeNode("wide"), tokenCount: 200 });
      graph.addNode({ ...makeNode("narrow"), tokenCount: 200 });
      graph.addNode(makeNode("wide-1"));
      graph.addNode(makeNode("wide-2"));
      graph.addEdge(makeEdge("a", "wide", 1));
      graph.addEdge(makeEdge("a", "narrow", 1));
      graph.addEdge(makeEdge("wide", "wide-1", 1));
      graph.addEdge(makeEdge("wide", "wide-2", 1));

      const relaxedState = {
        ...makeState("a"),
        budgetRemaining: 600,
        initialBudget: 1000,
        expansionCount: 2,
        maxHops: 4,
        maxFrontierSize: 2,
      };
      const pressuredState = {
        ...relaxedState,
        pendingNodeIds: ["queued"],
      };

      const relaxedScore = scoreAction({ type: "traverse", targetNodeId: "wide" }, relaxedState, graph);
      const pressuredScore = scoreAction({ type: "traverse", targetNodeId: "wide" }, pressuredState, graph);

      expect(pressuredScore).toBeLessThan(relaxedScore);
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

    it("uses traverse seed priors to break otherwise flat candidate ties", () => {
      const graph = new BrainGraph();
      graph.addNode(makeNode("source", new Float32Array([1, 0, 0])));
      graph.addNode(makeNode("high", new Float32Array([1, 0, 0])));
      graph.addNode(makeNode("low", new Float32Array([1, 0, 0])));

      const state = makeState("source");
      const highScore = scoreAction({ type: "traverse", targetNodeId: "high", seedScore: 0.9 }, state, graph);
      const lowScore = scoreAction({ type: "traverse", targetNodeId: "low", seedScore: 0.1 }, state, graph);

      expect(highScore).toBeGreaterThan(lowScore);
    });

    it("adds graph-visible tool-action priors when scoring toolcard candidates", () => {
      const graph = new BrainGraph();
      graph.addNode(makeNode("source", new Float32Array([1, 0, 0])));
      graph.addNode(makeToolNode("tool:proof", new Float32Array([1, 0, 0])));
      graph.addNode(makeNode("doc:proof", new Float32Array([1, 0, 0])));
      graph.addEdge(makeEdge("source", "tool:proof", 0.5));
      graph.addEdge(makeEdge("source", "doc:proof", 0.5));
      graph.setToolActionPrior("source", "tool:proof", 1.25);

      const state = makeState("source");
      const toolScore = scoreAction({ type: "traverse", targetNodeId: "tool:proof" }, state, graph);
      const docScore = scoreAction({ type: "traverse", targetNodeId: "doc:proof" }, state, graph);

      expect(toolScore).toBeGreaterThan(docScore);
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

    it("includes precomputed tool-action priors in the graph-visible action set", () => {
      const graph = new BrainGraph();
      graph.addNode(makeNode("source"));
      graph.addNode(makeToolNode("tool:proof"));
      graph.setToolActionPrior("source", "tool:proof", 0.8);

      const actions = graph.getActionSet("source", new Set());
      expect(actions.some((action) => action.type === "traverse" && action.targetNodeId === "tool:proof")).toBe(true);
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
