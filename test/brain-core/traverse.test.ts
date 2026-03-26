import { afterEach, describe, expect, it, vi } from "vitest";
import { BrainGraph } from "../../src/brain-core/graph.js";
import { traverse } from "../../src/brain-core/traverse.js";
import type { BrainEdge, BrainNode } from "../../src/brain-core/types.js";

function makeNode(id: string, embedding: Float32Array, tokenCount = 100): BrainNode {
  return {
    id,
    kind: "chunk",
    content: `node ${id}`,
    embedding,
    sourceUri: "test.md",
    trust: "scanner",
    tags: [],
    tokenCount,
    metadata: {},
    createdAt: Date.now(),
    updatedAt: Date.now(),
  };
}

function makeEdge(source: string, target: string, kind: BrainEdge["kind"] = "learned", weight = 1): BrainEdge {
  return {
    source,
    target,
    kind,
    weight,
    prior: 1,
    metadata: {},
    decayedAt: Date.now(),
    createdAt: Date.now(),
  };
}

function baseOptions(graph: BrainGraph) {
  return {
    graph,
    queryEmbedding: new Float32Array([1, 0, 0]),
    queryText: "hello",
    maxHops: 4,
    budgetChars: 1000,
    temperature: 0.1,
    maxSeeds: 5,
    semanticThreshold: 0.1,
  };
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("traverse", () => {
  it("supports zero-edge selection via stop_local", () => {
    const graph = new BrainGraph();
    graph.addNode(makeNode("a", new Float32Array([1, 0, 0])));
    vi.spyOn(Math, "random").mockReturnValue(0.99);

    const result = traverse({
      ...baseOptions(graph),
      policyParams: {
        stopBias: 5,
      },
    });

    expect(result.firedNodes).toEqual([]);
    expect(result.trajectory).toHaveLength(1);
    expect(result.trajectory[0]).toMatchObject({
      sourceNodeId: null,
      selectedTargets: [],
      acceptedTargets: [],
      terminationReason: "policy_stop",
    });
    expect(result.trajectory[0]?.substeps.at(-1)).toMatchObject({
      chosenAction: { type: "stop_local" },
      stopTruth: "chosen",
      stopReason: "policy_stop",
    });
    expect(result.footer).toContain("seed picks");
    expect(result.footer).toContain("expansions");
  });

  it("supports one-edge selection as a strict subset of the new semantics", () => {
    const graph = new BrainGraph();
    graph.addNode(makeNode("a", new Float32Array([1, 0, 0])));
    vi.spyOn(Math, "random").mockReturnValue(0);

    const result = traverse({
      ...baseOptions(graph),
      maxHops: 2,
    });

    expect(result.seedScores.filter((seed) => seed.selected).map((seed) => seed.nodeId)).toEqual(["a"]);
    expect(result.trajectory.map((expansion) => expansion.sourceNodeId)).toEqual([null, "a"]);
    expect(result.trajectory[0]?.acceptedTargets).toEqual(["a"]);
    expect(result.trajectory[1]?.acceptedTargets).toEqual([]);
    expect(result.trajectory[1]?.substeps.at(-1)).toMatchObject({
      chosenAction: { type: "stop_local" },
      stopTruth: "forced",
      stopReason: "no_traversable_candidates",
    });
    expect(result.trajectory[1]?.terminationReason).toBe("no_traversable_candidates");
    expect(result.firedNodes.map((node) => node.nodeId)).toEqual(["a"]);
  });

  it("supports many-edge selection from one source node", () => {
    const graph = new BrainGraph();
    graph.addNode(makeNode("a", new Float32Array([1, 0, 0])));
    graph.addNode(makeNode("b", new Float32Array([0, 1, 0])));
    graph.addNode(makeNode("c", new Float32Array([0, 1, 0])));
    graph.addEdge(makeEdge("a", "b"));
    graph.addEdge(makeEdge("a", "c"));
    vi.spyOn(Math, "random").mockReturnValue(0);

    const result = traverse(baseOptions(graph));
    const branchExpansion = result.trajectory.find((expansion) => expansion.sourceNodeId === "a");

    expect(branchExpansion).toBeDefined();
    expect(branchExpansion?.selectedTargets).toEqual(["b", "c"]);
    expect(branchExpansion?.acceptedTargets).toEqual(["b", "c"]);
    expect(branchExpansion?.proposalOutcomes).toEqual([
      { targetNodeId: "b", outcome: "accepted", reason: "accepted" },
      { targetNodeId: "c", outcome: "accepted", reason: "accepted" },
    ]);
    expect(branchExpansion?.substeps.map((substep) => substep.chosenAction.type)).toEqual([
      "traverse",
      "traverse",
      "stop_local",
    ]);
    expect(branchExpansion?.substeps.at(-1)).toMatchObject({
      stopTruth: "forced",
      stopReason: "no_traversable_candidates",
    });
    expect(result.firedNodes.map((node) => node.nodeId)).toEqual(["a", "b", "c"]);
    expect(result.footer).not.toContain("hops");
  });

  it("executes the frontier in FIFO order when branching", () => {
    const graph = new BrainGraph();
    graph.addNode(makeNode("a", new Float32Array([1, 0, 0])));
    graph.addNode(makeNode("d", new Float32Array([1, 0, 0])));
    graph.addNode(makeNode("b", new Float32Array([0, 1, 0])));
    graph.addNode(makeNode("e", new Float32Array([0, 1, 0])));
    graph.addEdge(makeEdge("a", "b"));
    graph.addEdge(makeEdge("d", "e"));
    vi.spyOn(Math, "random").mockReturnValue(0);

    const result = traverse({
      ...baseOptions(graph),
      maxHops: 3,
    });

    expect(result.trajectory.map((expansion) => expansion.sourceNodeId)).toEqual([null, "a", "d"]);
    expect(result.firedNodes.map((node) => node.nodeId)).toEqual(["a", "d", "b", "e"]);
  });

  it("enforces maxFanoutPerNode and maxFrontierSize hard caps", () => {
    const fanoutGraph = new BrainGraph();
    fanoutGraph.addNode(makeNode("a", new Float32Array([1, 0, 0])));
    fanoutGraph.addNode(makeNode("b", new Float32Array([0, 1, 0])));
    fanoutGraph.addNode(makeNode("c", new Float32Array([0, 1, 0])));
    fanoutGraph.addNode(makeNode("d", new Float32Array([0, 1, 0])));
    fanoutGraph.addEdge(makeEdge("a", "b"));
    fanoutGraph.addEdge(makeEdge("a", "c"));
    fanoutGraph.addEdge(makeEdge("a", "d"));
    vi.spyOn(Math, "random").mockReturnValue(0);

    const fanoutResult = traverse({
      ...baseOptions(fanoutGraph),
      maxFanoutPerNode: 2,
    });
    const fanoutExpansion = fanoutResult.trajectory.find((expansion) => expansion.sourceNodeId === "a");
    expect(fanoutExpansion?.acceptedTargets).toEqual(["b", "c"]);
    expect(fanoutExpansion?.substeps).toHaveLength(3);
    expect(fanoutExpansion?.substeps.at(-1)).toMatchObject({
      chosenAction: { type: "stop_local" },
      stopTruth: "forced",
      stopReason: "fanout_cap",
    });
    expect(fanoutExpansion?.terminationReason).toBe("fanout_cap");

    const frontierGraph = new BrainGraph();
    frontierGraph.addNode(makeNode("a", new Float32Array([1, 0, 0])));
    frontierGraph.addNode(makeNode("d", new Float32Array([1, 0, 0])));
    frontierGraph.addNode(makeNode("e", new Float32Array([1, 0, 0])));

    const frontierResult = traverse({
      ...baseOptions(frontierGraph),
      maxHops: 1,
      maxFrontierSize: 2,
    });
    expect(frontierResult.trajectory[0]?.acceptedTargets).toEqual(["a", "d"]);
    expect(frontierResult.trajectory[0]?.substeps.at(-1)).toMatchObject({
      chosenAction: { type: "stop_local" },
      stopTruth: "forced",
      stopReason: "frontier_cap",
    });
    expect(frontierResult.trajectory[0]?.terminationReason).toBe("frontier_cap");
    expect(frontierResult.firedNodes.map((node) => node.nodeId)).toEqual(["a", "d"]);
  });

  it("keeps veto behavior proposal-local so another parent can still accept the target", () => {
    const graph = new BrainGraph();
    graph.addNode(makeNode("a", new Float32Array([1, 0, 0])));
    graph.addNode(makeNode("d", new Float32Array([1, 0, 0])));
    graph.addNode(makeNode("c", new Float32Array([0, 1, 0])));
    graph.addEdge(makeEdge("a", "c", "inhibitory", -1));
    graph.addEdge(makeEdge("d", "c"));
    vi.spyOn(Math, "random").mockReturnValue(0);

    const result = traverse({
      ...baseOptions(graph),
      maxHops: 3,
    });

    expect(result.trajectory.map((expansion) => expansion.sourceNodeId)).toEqual([null, "a", "d"]);
    expect(result.trajectory[1]?.vetoedTargets).toEqual([{ targetNodeId: "c", reason: "inhibitory edge" }]);
    expect(result.trajectory[1]?.proposalOutcomes).toEqual([
      { targetNodeId: "c", outcome: "vetoed", reason: "inhibitory edge" },
    ]);
    expect(result.trajectory[2]?.acceptedTargets).toEqual(["c"]);
    expect(result.vetoedNodes).toEqual([{ nodeId: "c", reason: "inhibitory edge" }]);
    expect(result.firedNodes.map((node) => node.nodeId)).toEqual(["a", "d", "c"]);
  });

  it("records forced missing-target stops instead of silently breaking", () => {
    const graph = new BrainGraph();
    graph.addNode(makeNode("a", new Float32Array([1, 0, 0])));
    graph.addNode(makeNode("b", new Float32Array([0, 1, 0])));
    graph.addEdge(makeEdge("a", "b"));
    vi.spyOn(Math, "random").mockReturnValue(0);

    const originalGetNode = graph.getNode.bind(graph);
    let bCalls = 0;
    vi.spyOn(graph, "getNode").mockImplementation((nodeId: string) => {
      const node = originalGetNode(nodeId);
      if (nodeId !== "b") {
        return node;
      }
      bCalls += 1;
      return bCalls === 3 ? undefined : node;
    });

    const result = traverse({
      ...baseOptions(graph),
      maxHops: 2,
    });
    const branchExpansion = result.trajectory.find((expansion) => expansion.sourceNodeId === "a");

    expect(branchExpansion?.acceptedTargets).toEqual([]);
    expect(branchExpansion?.proposalOutcomes).toEqual([
      { targetNodeId: "b", outcome: "dropped", reason: "missing_target_node" },
    ]);
    expect(branchExpansion?.substeps.at(-1)).toMatchObject({
      chosenAction: { type: "stop_local" },
      stopTruth: "forced",
      stopReason: "missing_target_node",
    });
    expect(branchExpansion?.terminationReason).toBe("missing_target_node");
    expect(result.firedNodes.map((node) => node.nodeId)).toEqual(["a"]);
  });

  it("records dropped proposal outcomes when commit-time budget checks reject a selected target", () => {
    const graph = new BrainGraph();
    graph.addNode(makeNode("a", new Float32Array([1, 0, 0])));
    graph.addNode(makeNode("b", new Float32Array([0, 1, 0]), 100));
    graph.addNode(makeNode("c", new Float32Array([0, 1, 0]), 100));
    graph.addEdge(makeEdge("a", "b"));
    graph.addEdge(makeEdge("a", "c"));
    vi.spyOn(Math, "random").mockReturnValue(0);

    const originalGetNode = graph.getNode.bind(graph);
    let cCalls = 0;
    vi.spyOn(graph, "getNode").mockImplementation((nodeId: string) => {
      const node = originalGetNode(nodeId);
      if (!node || nodeId !== "c") {
        return node;
      }
      cCalls += 1;
      if (cCalls >= 4) {
        return { ...node, tokenCount: 950 };
      }
      return { ...node, tokenCount: 800 };
    });

    const result = traverse({
      ...baseOptions(graph),
      maxHops: 2,
      budgetChars: 1000,
    });
    const branchExpansion = result.trajectory.find((expansion) => expansion.sourceNodeId === "a");

    expect(branchExpansion?.selectedTargets).toEqual(["b", "c"]);
    expect(branchExpansion?.acceptedTargets).toEqual(["b"]);
    expect(branchExpansion?.proposalOutcomes).toEqual([
      { targetNodeId: "b", outcome: "accepted", reason: "accepted" },
      { targetNodeId: "c", outcome: "dropped", reason: "selection_budget_exhausted" },
    ]);
    expect(branchExpansion?.terminationReason).toBe("selection_budget_exhausted");
    expect(branchExpansion?.substeps.at(-1)).toMatchObject({
      chosenAction: { type: "stop_local" },
      stopTruth: "forced",
      stopReason: "selection_budget_exhausted",
    });
    expect(result.firedNodes.map((node) => node.nodeId)).toEqual(["a", "b"]);
  });
});
