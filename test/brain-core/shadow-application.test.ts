import { describe, expect, it } from "vitest";
import { BrainGraph } from "../../src/brain-core/graph.js";
import {
  applyShadowMutationProposal,
  applyShadowMutationProposalToState,
  createShadowCandidateState,
  resetShadowCandidateState,
  revertShadowMutationApplication,
} from "../../src/brain-core/shadow-application.js";
import type { MutationProposal, BrainNode } from "../../src/brain-core/types.js";

function makeNode(id: string, kind: BrainNode["kind"] = "chunk"): BrainNode {
  const now = Date.now();
  return {
    id,
    kind,
    content: id,
    embedding: null,
    sourceUri: null,
    trust: "scanner",
    tags: [id],
    tokenCount: id.length,
    metadata: {},
    createdAt: now,
    updatedAt: now,
  };
}

function makeGraph(): BrainGraph {
  const graph = new BrainGraph();
  graph.addNode(makeNode("a"));
  graph.addNode(makeNode("b"));
  graph.addNode(makeNode("c"));
  graph.addEdge({
    source: "a",
    target: "b",
    kind: "learned",
    weight: 0.75,
    prior: 0.5,
    metadata: { seed: true },
    decayedAt: Date.now(),
    createdAt: Date.now(),
  });
  return graph;
}

function makeProposal(kind: MutationProposal["kind"], proposal: Record<string, unknown>): MutationProposal {
  return {
    id: `mut_${kind}`,
    kind,
    proposal,
    evidence: null,
    expectedGain: 0.1,
    status: "pending",
    createdAt: Date.now(),
    resolvedAt: null,
  };
}

describe("shadow mutation application", () => {
  it("applies prune proposals to a candidate graph and can revert them", () => {
    const baseGraph = makeGraph();
    const candidateGraph = baseGraph.clone();

    const application = applyShadowMutationProposal(
      candidateGraph,
      makeProposal("prune", { source: "a", target: "b", edgeKind: "learned" }),
    );

    expect(application.applied).toBe(true);
    expect(application.reversible).toBe(true);
    expect(application.operations).toHaveLength(1);
    expect(baseGraph.getOutgoingEdges("a")).toHaveLength(1);
    expect(candidateGraph.getOutgoingEdges("a")).toHaveLength(0);

    revertShadowMutationApplication(candidateGraph, application);

    expect(candidateGraph.getOutgoingEdges("a")).toHaveLength(1);
    expect(candidateGraph.getOutgoingEdges("a")[0]).toMatchObject({
      source: "a",
      target: "b",
      kind: "learned",
    });
  });

  it("tracks shadow candidate state and rolls it back in reverse order", () => {
    const baseGraph = makeGraph();
    const state = createShadowCandidateState(baseGraph);

    const connectApplication = applyShadowMutationProposalToState(
      state,
      makeProposal("connect", { nodeA: "b", nodeB: "c" }),
    );
    const injectApplication = applyShadowMutationProposalToState(
      state,
      makeProposal("inject", {
        nodeKind: "episode_anchor",
        content: "shadow note",
        firedNodes: ["a", "b"],
      }),
    );

    expect(connectApplication.applied).toBe(true);
    expect(injectApplication.applied).toBe(true);
    expect(state.candidateGraph.getOutgoingEdges("b")).toHaveLength(1);
    expect(state.candidateGraph.getAllNodes().some((node) => node.id.startsWith("shadow_inject_"))).toBe(true);

    resetShadowCandidateState(state);

    expect(state.applications).toHaveLength(0);
    expect(state.candidateGraph.getOutgoingEdges("a")).toHaveLength(1);
    expect(state.candidateGraph.getOutgoingEdges("b")).toHaveLength(0);
    expect(state.candidateGraph.getAllNodes().some((node) => node.id.startsWith("shadow_inject_"))).toBe(false);
  });

  it("rejects unsupported mutation kinds without changing the candidate graph", () => {
    const baseGraph = makeGraph();
    const candidateGraph = baseGraph.clone();

    const application = applyShadowMutationProposal(
      candidateGraph,
      makeProposal("split", { nodeId: "a" }),
    );

    expect(application.applied).toBe(false);
    expect(application.reason).toContain("unsupported mutation kind");
    expect(candidateGraph.getOutgoingEdges("a")).toHaveLength(1);
    expect(candidateGraph.getAllNodes()).toHaveLength(baseGraph.getAllNodes().length);
  });
});
