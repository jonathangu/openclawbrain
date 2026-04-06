import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { afterEach, describe, expect, it } from "vitest";
import { START_NODE_ID, type BrainEdge, type BrainNode } from "../../src/brain-core/types.js";
import { BrainGraph } from "../../src/brain-core/graph.js";
import { runBrainMigrations } from "../../src/brain-store/migrations.js";
import { BrainStore } from "../../src/brain-store/store.js";
import { populateGraph } from "../../src/brain-runtime/graph-io.js";

const tempDirs: string[] = [];

function makeTempDir(prefix: string): string {
  const dir = mkdtempSync(join(tmpdir(), prefix));
  tempDirs.push(dir);
  return dir;
}

function makeNode(id: string): BrainNode {
  return {
    id,
    kind: "chunk",
    content: `content for ${id}`,
    embedding: new Float32Array([1, 0, 0]),
    sourceUri: null,
    trust: "scanner",
    tags: [],
    tokenCount: 32,
    metadata: {},
    createdAt: Date.now(),
    updatedAt: Date.now(),
  };
}

function makeToolNode(id: string): BrainNode {
  return {
    id,
    kind: "toolcard",
    content: `tool for ${id}`,
    embedding: new Float32Array([1, 0, 0]),
    sourceUri: null,
    trust: "scanner",
    tags: [],
    tokenCount: 32,
    metadata: {},
    createdAt: Date.now(),
    updatedAt: Date.now(),
  };
}

function makeEdge(source: string, target: string): BrainEdge {
  return {
    source,
    target,
    kind: "learned",
    weight: 0.5,
    prior: 0.5,
    metadata: {},
    decayedAt: Date.now(),
    createdAt: Date.now(),
  };
}

afterEach(() => {
  for (const dir of tempDirs.splice(0)) {
    rmSync(dir, { recursive: true, force: true });
  }
});

describe("graph-io", () => {
  it("round-trips stop-local weights through pack snapshots into a runtime graph", () => {
    const brainRoot = makeTempDir("openclawbrain-graph-io-");
    const db = new DatabaseSync(join(brainRoot, "state.db"));
    runBrainMigrations(db);
    const store = new BrainStore(db, { brainRoot });

    const now = Date.now();
    store.writePackSnapshot({
      version: 1,
      nodes: [makeNode("a"), makeNode("b"), makeToolNode("tool:proof")],
      edges: [makeEdge("a", "b")],
      seedWeights: [
        { nodeId: "b", weight: 0.6, updatedAt: now },
      ],
      stopLocalWeights: [
        { sourceNodeId: START_NODE_ID, weight: 0.4, updatedAt: now },
        { sourceNodeId: "a", weight: 1.1, updatedAt: now },
      ],
      toolActionPriors: [
        { sourceNodeId: "a", toolNodeId: "tool:proof", weight: 0.9, updatedAt: now },
      ],
      metadata: { reason: "test" },
    });

    const snapshot = store.readPackSnapshot(1);
    expect(snapshot?.stopLocalWeights).toEqual([
      { sourceNodeId: START_NODE_ID, weight: 0.4, updatedAt: now },
      { sourceNodeId: "a", weight: 1.1, updatedAt: now },
    ]);

    const graph = new BrainGraph();
    populateGraph(
      graph,
      snapshot?.nodes ?? [],
      snapshot?.edges ?? [],
      snapshot?.seedWeights ?? [],
      snapshot?.stopLocalWeights ?? [],
      snapshot?.toolActionPriors ?? [],
    );

    expect(graph.getSeedWeight("b")).toBeCloseTo(0.6);
    expect(graph.getStopLocalWeight(null)).toBeCloseTo(0.4);
    expect(graph.getStopLocalWeight("a")).toBeCloseTo(1.1);
    expect(graph.getToolActionPrior("a", "tool:proof")).toBeCloseTo(0.9);
  });
});
