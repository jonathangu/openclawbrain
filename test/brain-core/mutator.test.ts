import { describe, it, expect, vi } from "vitest";
import { BrainGraph } from "../../src/brain-core/graph.js";
import { BrainMutator } from "../../src/brain-core/mutator.js";
import type { BrainEdge, BrainNode, Episode } from "../../src/brain-core/types.js";

function makeNode(id: string, embedding: Float32Array): BrainNode {
  return {
    id,
    kind: "chunk",
    content: id,
    embedding,
    sourceUri: "test.md",
    trust: "scanner",
    tags: [],
    tokenCount: 20,
    metadata: {},
    createdAt: Date.now(),
    updatedAt: Date.now(),
  };
}

function makeEdge(source: string, target: string, weight = 0.5, prior = 0.5): BrainEdge {
  return {
    source,
    target,
    kind: "learned",
    weight,
    prior,
    metadata: {},
    decayedAt: Date.now(),
    createdAt: Date.now(),
  };
}

function makeEpisode(firedNodes: string[], reward: number): Episode {
  return {
    id: `ep-${firedNodes.join("-")}`,
    conversationId: 1,
    queryText: "test",
    queryEmbedding: null,
    trajectory: [],
    firedNodes,
    vetoedNodes: [],
    contextChars: 0,
    reward,
    rewardSource: "self",
    packVersion: 1,
    createdAt: Date.now(),
  };
}

describe("mutator", () => {
  it("proposes connection mutations for repeated positive co-firing", () => {
    const graph = new BrainGraph();
    graph.addNode(makeNode("a", new Float32Array([1, 0, 0])));
    graph.addNode(makeNode("b", new Float32Array([1, 0, 0])));

    const mutator = new BrainMutator(
      {
        insertEdge: vi.fn(),
        deleteNode: vi.fn(),
        deleteEdge: vi.fn(),
        resolveMutation: vi.fn(),
      },
      graph,
      { info: vi.fn() },
    );

    const proposals = mutator.proposeMutations([
      makeEpisode(["a", "b"], 1),
      makeEpisode(["a", "b"], 1),
      makeEpisode(["a", "b"], 1),
    ]);

    expect(proposals.some((proposal) => proposal.kind === "connect")).toBe(true);
  });

  it("persists prune mutations through the persistence callback", () => {
    const graph = new BrainGraph();
    graph.addNode(makeNode("a", new Float32Array([1, 0, 0])));
    graph.addNode(makeNode("b", new Float32Array([1, 0, 0])));
    graph.addEdge(makeEdge("a", "b", 0.01, 0.01));

    const persistence = {
      insertEdge: vi.fn(),
      deleteNode: vi.fn(),
      deleteEdge: vi.fn(),
      resolveMutation: vi.fn(),
    };
    const mutator = new BrainMutator(persistence, graph, { info: vi.fn() });
    const proposal = mutator.proposeMutations([]).find((entry) => entry.kind === "prune");
    expect(proposal).toBeTruthy();

    mutator.applyMutation(proposal!);
    expect(persistence.deleteEdge).toHaveBeenCalled();
  });
});
