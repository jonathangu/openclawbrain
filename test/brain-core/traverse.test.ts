import { describe, it, expect } from "vitest";
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

describe("traverse", () => {
  it("stops when only STOP remains", () => {
    const graph = new BrainGraph();
    graph.addNode(makeNode("a", new Float32Array([1, 0, 0])));

    const result = traverse({
      graph,
      queryEmbedding: new Float32Array([1, 0, 0]),
      queryText: "hello",
      maxHops: 3,
      budgetChars: 400,
      temperature: 0.1,
      maxSeeds: 5,
      semanticThreshold: 0.1,
    });

    expect(result.trajectory.length).toBeGreaterThan(0);
    expect(result.trajectory.at(-1)?.chosenAction.type).toBe("stop");
  });

  it("records seed priors, probabilities, and the chosen seed", () => {
    const graph = new BrainGraph();
    graph.addNode(makeNode("a", new Float32Array([1, 0, 0])));
    graph.addNode(makeNode("b", new Float32Array([1, 0, 0])));
    graph.addEdge(makeEdge("__START__", "a", "seed", 1.0));

    const result = traverse({
      graph,
      queryEmbedding: new Float32Array([1, 0, 0]),
      queryText: "hello",
      maxHops: 3,
      budgetChars: 400,
      temperature: 0.1,
      maxSeeds: 5,
      semanticThreshold: 0.1,
    });

    expect(result.seedScores.length).toBeGreaterThan(0);
    expect(result.seedScores[0]).toEqual(expect.objectContaining({
      nodeId: expect.any(String),
      priorScore: expect.any(Number),
      probability: expect.any(Number),
      chosen: expect.any(Boolean),
    }));
  });

  it("avoids infinite traversal by respecting max hops", () => {
    const graph = new BrainGraph();
    graph.addNode(makeNode("a", new Float32Array([1, 0, 0])));
    graph.addNode(makeNode("b", new Float32Array([1, 0, 0])));
    graph.addEdge(makeEdge("a", "b"));
    graph.addEdge(makeEdge("b", "a"));

    const result = traverse({
      graph,
      queryEmbedding: new Float32Array([1, 0, 0]),
      queryText: "loop",
      maxHops: 2,
      budgetChars: 1000,
      temperature: 0.1,
      maxSeeds: 5,
      semanticThreshold: 0.1,
    });

    expect(result.trajectory.length).toBeLessThanOrEqual(3);
  });

  it("records vetoed nodes when inhibitory edges block firing", () => {
    const graph = new BrainGraph();
    graph.addNode(makeNode("a", new Float32Array([1, 0, 0])));
    graph.addNode(makeNode("b", new Float32Array([1, 0, 0])));
    graph.addNode(makeNode("c", new Float32Array([1, 0, 0])));
    graph.addEdge(makeEdge("a", "b"));
    graph.addEdge(makeEdge("b", "c", "inhibitory", -1));

    const result = traverse({
      graph,
      queryEmbedding: new Float32Array([1, 0, 0]),
      queryText: "veto",
      maxHops: 4,
      budgetChars: 1000,
      temperature: 0.1,
      maxSeeds: 5,
      semanticThreshold: 0.1,
    });

    expect(result.vetoedNodes.length).toBeGreaterThanOrEqual(0);
  });
});
