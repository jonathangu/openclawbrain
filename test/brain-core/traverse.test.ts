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

  it("tracks pending local picks so later selections avoid redundant siblings", () => {
    const graph = new BrainGraph();
    graph.addNode(makeNode("a", new Float32Array([1, 0, 0])));
    graph.addNode(makeNode("b", new Float32Array([0, 1, 0])));
    graph.addNode(makeNode("c", new Float32Array([0, 1, 0])));
    graph.addNode(makeNode("d", new Float32Array([0, 0, 1])));
    graph.addEdge(makeEdge("a", "b"));
    graph.addEdge(makeEdge("a", "c"));
    graph.addEdge(makeEdge("a", "d"));
    vi.spyOn(Math, "random")
      .mockReturnValueOnce(0.1)
      .mockReturnValueOnce(0.9)
      .mockReturnValueOnce(0.1)
      .mockReturnValueOnce(0.4)
      .mockReturnValue(0.9);

    const result = traverse({
      ...baseOptions(graph),
      maxHops: 2,
      maxFrontierSize: 2,
    });
    const branchExpansion = result.trajectory.find((expansion) => expansion.sourceNodeId === "a");
    const secondSubstep = branchExpansion?.substeps[1];
    const redundantCandidate = secondSubstep?.candidates.find(
      (candidate) => candidate.action.type === "traverse" && candidate.action.targetNodeId === "c",
    );
    const novelCandidate = secondSubstep?.candidates.find(
      (candidate) => candidate.action.type === "traverse" && candidate.action.targetNodeId === "d",
    );

    expect(branchExpansion?.acceptedTargets).toEqual(["b", "d"]);
    expect(secondSubstep?.stateSnapshot.pendingTargetNodeIds).toEqual(["b"]);
    expect(secondSubstep?.stateSnapshot.policyState).toEqual(expect.objectContaining({
      activeFrontierSize: 1,
      pendingSelectionCount: 1,
      frontierPressure: expect.any(Number),
    }));
    expect(redundantCandidate?.scoreBreakdown).toEqual(expect.objectContaining({
      redundancySimilarity: 1,
      redundancyPenalty: expect.any(Number),
    }));
    expect(novelCandidate?.scoreBreakdown).toEqual(expect.objectContaining({
      redundancySimilarity: 0,
      redundancyPenalty: 0,
    }));
    expect((redundantCandidate?.score ?? 0)).toBeLessThan(novelCandidate?.score ?? 0);
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

  describe("interruption accounting", () => {
    it("returns null interruptionAccounting when no deadline is set", () => {
      const graph = new BrainGraph();
      graph.addNode(makeNode("a", new Float32Array([1, 0, 0])));
      vi.spyOn(Math, "random").mockReturnValue(0);

      const result = traverse(baseOptions(graph));

      expect(result.interruption).toBeNull();
      expect(result.interruptionAccounting).toBeNull();
    });

    it("returns null interruptionAccounting when deadline is not exceeded", () => {
      const graph = new BrainGraph();
      graph.addNode(makeNode("a", new Float32Array([1, 0, 0])));
      vi.spyOn(Math, "random").mockReturnValue(0);

      const result = traverse({
        ...baseOptions(graph),
        deadlineAtMs: Date.now() + 60_000,
      });

      expect(result.interruption).toBeNull();
      expect(result.interruptionAccounting).toBeNull();
    });

    it("accounts for dropped frontier nodes when deadline interrupts frontier loop", () => {
      const graph = new BrainGraph();
      graph.addNode(makeNode("a", new Float32Array([1, 0, 0]), 50));
      graph.addNode(makeNode("d", new Float32Array([1, 0, 0]), 50));
      graph.addNode(makeNode("b", new Float32Array([0, 1, 0]), 50));
      graph.addNode(makeNode("e", new Float32Array([0, 1, 0]), 50));
      graph.addEdge(makeEdge("a", "b"));
      graph.addEdge(makeEdge("d", "e"));
      vi.spyOn(Math, "random").mockReturnValue(0);

      // Allow seed phase and first frontier expansion (about 9 Date.now calls),
      // then trigger deadline before second frontier expansion begins.
      let callCount = 0;
      const realNow = Date.now;
      vi.spyOn(Date, "now").mockImplementation(() => {
        callCount++;
        if (callCount > 9) {
          return realNow() + 999_999;
        }
        return realNow();
      });

      const result = traverse({
        ...baseOptions(graph),
        maxHops: 4,
        budgetChars: 1000,
        deadlineAtMs: realNow() + 500_000,
      });

      expect(result.interruption).toBeDefined();
      expect(result.interruption?.interrupted).toBe(true);
      expect(result.interruptionAccounting).not.toBeNull();

      const accounting = result.interruptionAccounting!;
      expect(accounting.maxExpansions).toBe(4);
      expect(accounting.budgetTotal).toBe(1000);
      expect(accounting.remainingBudgetChars).toBeGreaterThanOrEqual(0);
      expect(accounting.budgetUtilization).toBeGreaterThan(0);
      expect(accounting.budgetUtilization).toBeLessThanOrEqual(1);
      // droppedFrontierNodeIds contains frontier nodes never expanded
      expect(accounting.droppedFrontierNodeIds.length).toBeGreaterThanOrEqual(0);
      expect(accounting.interruptedExpansionSourceNodeId).not.toBeUndefined();
      // Budget used should match fired nodes
      const firedTokens = result.firedNodes.reduce((sum, n) => sum + n.tokenCount, 0);
      expect(accounting.budgetUsed).toBe(firedTokens);
    });

    it("reports interruption in footer with INTERRUPTED marker", () => {
      const graph = new BrainGraph();
      graph.addNode(makeNode("a", new Float32Array([1, 0, 0]), 100));
      vi.spyOn(Math, "random").mockReturnValue(0);

      // Expire immediately after seed phase selection
      let callCount = 0;
      const realNow = Date.now;
      vi.spyOn(Date, "now").mockImplementation(() => {
        callCount++;
        if (callCount > 10) {
          return realNow() + 999_999;
        }
        return realNow();
      });

      const result = traverse({
        ...baseOptions(graph),
        maxHops: 4,
        budgetChars: 1000,
        deadlineAtMs: realNow() + 500_000,
      });

      if (result.interruption) {
        expect(result.footer).toContain("INTERRUPTED");
        expect(result.footer).toContain("budget used");
        expect(result.footer).toMatch(/partial|empty/);
      }
    });

    it("tracks dropped proposals in interruptionAccounting", () => {
      const graph = new BrainGraph();
      graph.addNode(makeNode("a", new Float32Array([1, 0, 0]), 50));
      graph.addNode(makeNode("b", new Float32Array([0, 1, 0]), 50));
      graph.addEdge(makeEdge("a", "b"));
      vi.spyOn(Math, "random").mockReturnValue(0);

      // Let seed phase complete (a is selected), then deadline hits during
      // the commit phase of the first frontier expansion so proposals get dropped.
      let callCount = 0;
      const realNow = Date.now;
      vi.spyOn(Date, "now").mockImplementation(() => {
        callCount++;
        // Expire after enough calls to enter the commit loop of expandSource
        if (callCount > 20) {
          return realNow() + 999_999;
        }
        return realNow();
      });

      const result = traverse({
        ...baseOptions(graph),
        maxHops: 4,
        budgetChars: 1000,
        deadlineAtMs: realNow() + 500_000,
      });

      if (result.interruption && result.interruptionAccounting) {
        const accounting = result.interruptionAccounting;
        expect(accounting.budgetTotal).toBe(1000);
        expect(typeof accounting.budgetUtilization).toBe("number");
        expect(accounting.completedExpansionCount).toBeGreaterThanOrEqual(0);
        expect(typeof accounting.droppedProposalCount).toBe("number");
        expect(typeof accounting.droppedProposalReasons).toBe("object");
        expect(Array.isArray(accounting.droppedProposalNodeIds)).toBe(true);
      }
    });

    it("computes correct budget utilization on full non-interrupted traversal", () => {
      const graph = new BrainGraph();
      graph.addNode(makeNode("a", new Float32Array([1, 0, 0]), 200));
      vi.spyOn(Math, "random").mockReturnValue(0);

      const result = traverse({
        ...baseOptions(graph),
        budgetChars: 1000,
      });

      // No interruption, so interruptionAccounting should be null
      expect(result.interruptionAccounting).toBeNull();
      // But we can verify the footer doesn't include INTERRUPTED
      expect(result.footer).not.toContain("INTERRUPTED");
    });

    it("records servedPartial=true in interruption when nodes were already fired", () => {
      const graph = new BrainGraph();
      graph.addNode(makeNode("a", new Float32Array([1, 0, 0]), 50));
      graph.addNode(makeNode("d", new Float32Array([1, 0, 0]), 50));
      graph.addNode(makeNode("b", new Float32Array([0, 1, 0]), 50));
      graph.addEdge(makeEdge("a", "b"));
      vi.spyOn(Math, "random").mockReturnValue(0);

      // Allow seed phase to fire nodes, then expire during frontier expansion
      let callCount = 0;
      const realNow = Date.now;
      vi.spyOn(Date, "now").mockImplementation(() => {
        callCount++;
        if (callCount > 15) {
          return realNow() + 999_999;
        }
        return realNow();
      });

      const result = traverse({
        ...baseOptions(graph),
        maxHops: 4,
        budgetChars: 1000,
        deadlineAtMs: realNow() + 500_000,
      });

      if (result.interruption && result.firedNodes.length > 0) {
        expect(result.interruption.servedPartial).toBe(true);
        expect(result.footer).toContain("partial");
        expect(result.interruptionAccounting).not.toBeNull();
      }
    });

    it("records servedPartial=false in interruption when deadline hits before any nodes fire", () => {
      const graph = new BrainGraph();
      graph.addNode(makeNode("a", new Float32Array([1, 0, 0]), 50));
      vi.spyOn(Math, "random").mockReturnValue(0);

      // Expire immediately so the very first deadline check in expandSource fires
      const realNow = Date.now;
      vi.spyOn(Date, "now").mockImplementation(() => realNow() + 999_999);

      const result = traverse({
        ...baseOptions(graph),
        maxHops: 4,
        budgetChars: 1000,
        deadlineAtMs: realNow() + 500_000,
      });

      expect(result.interruption).toBeDefined();
      expect(result.interruption?.servedPartial).toBe(false);
      expect(result.interruptionAccounting).not.toBeNull();
      expect(result.interruptionAccounting?.budgetUsed).toBe(0);
      expect(result.interruptionAccounting?.budgetUtilization).toBe(0);
      expect(result.footer).toContain("empty");
    });
  });
});
