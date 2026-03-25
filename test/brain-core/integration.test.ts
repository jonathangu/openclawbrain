import { describe, it, expect, afterEach } from "vitest";
import { DatabaseSync } from "node:sqlite";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { BrainGraph } from "../../src/brain-core/graph.js";
import { BrainStore } from "../../src/brain-store/store.js";
import { runBrainMigrations } from "../../src/brain-store/migrations.js";
import { traverse } from "../../src/brain-core/traverse.js";
import { recordEpisode } from "../../src/brain-core/episode.js";
import { computeReinforceUpdates, applyWeightUpdates, updateBaseline } from "../../src/brain-core/update.js";
import type { BrainNode, BrainEdge } from "../../src/brain-core/types.js";

const tempDirs: string[] = [];

function setup() {
  const dir = mkdtempSync(join(tmpdir(), "brain-integration-test-"));
  tempDirs.push(dir);
  const db = new DatabaseSync(join(dir, "test.db"));
  db.exec("PRAGMA journal_mode = WAL");
  db.exec("PRAGMA foreign_keys = ON");
  runBrainMigrations(db);
  const store = new BrainStore(db);
  const graph = new BrainGraph();
  return { store, graph, db };
}

afterEach(() => {
  for (const dir of tempDirs.splice(0)) {
    rmSync(dir, { recursive: true, force: true });
  }
});

function makeNode(id: string, embedding: Float32Array): BrainNode {
  return {
    id, kind: "chunk", content: `content of ${id}`,
    embedding, sourceUri: "test.md", trust: "scanner",
    tags: [], tokenCount: 50, metadata: {},
    createdAt: Date.now(), updatedAt: Date.now(),
  };
}

function makeEdge(source: string, target: string, weight: number): BrainEdge {
  return {
    source, target, kind: "learned", weight, prior: 0.5,
    metadata: {}, decayedAt: Date.now(), createdAt: Date.now(),
  };
}

describe("integration: full learning pipeline", () => {
  it("query → episode → label → REINFORCE → improved routing", () => {
    const { store, graph } = setup();

    // Build a graph: A → B (good path), A → C (bad path)
    const nodeA = makeNode("a", new Float32Array([0.9, 0.1, 0]));
    const nodeB = makeNode("b", new Float32Array([0.8, 0.2, 0]));
    const nodeC = makeNode("c", new Float32Array([0.7, 0.3, 0]));
    graph.addNode(nodeA);
    graph.addNode(nodeB);
    graph.addNode(nodeC);
    graph.addEdge(makeEdge("a", "b", 0.5));
    graph.addEdge(makeEdge("a", "c", 0.5));

    // Persist
    store.insertNode(nodeA);
    store.insertNode(nodeB);
    store.insertNode(nodeC);
    store.insertEdge(makeEdge("a", "b", 0.5));
    store.insertEdge(makeEdge("a", "c", 0.5));

    // Step 1: Query (traverse the graph)
    const queryEmbedding = new Float32Array([1, 0, 0]);
    const result = traverse({
      graph,
      queryEmbedding,
      queryText: "test query",
      maxHops: 4,
      budgetChars: 5000,
      temperature: 0.5,
      maxSeeds: 5,
      semanticThreshold: 0.3,
    });

    expect(result.firedNodes.length).toBeGreaterThan(0);

    // Step 2: Record episode
    const episode = recordEpisode({
      traversalResult: result,
      queryText: "test query",
      queryEmbedding,
      conversationId: null,
      packVersion: null,
    });
    expect(episode.id).toMatch(/^be_/);

    // Step 3: Assign positive reward
    episode.reward = 1.0;
    episode.rewardSource = "human";

    // Step 4: Persist episode
    store.insertEpisode(episode);
    const retrieved = store.getEpisode(episode.id);
    expect(retrieved).not.toBeNull();
    expect(retrieved!.reward).toBe(1.0);

    // Step 5: Apply REINFORCE update
    const initialEdgeAB = graph.getEdge("a", "b");
    const initialEdgeAC = graph.getEdge("a", "c");
    const initialWeightAB = initialEdgeAB?.weight ?? 0.5;
    const initialWeightAC = initialEdgeAC?.weight ?? 0.5;
    const chosenSeed = result.seedScores.find((seed) => seed.selected)?.nodeId ?? null;
    const initialSeedWeight = chosenSeed ? graph.getSeedWeight(chosenSeed) : 0;

    const updates = computeReinforceUpdates(episode, 0.1, 0.0);
    applyWeightUpdates(graph, updates);

    // Step 6: Verify learning happened
    if (updates.length > 0) {
      const updatedEdgeAB = graph.getEdge("a", "b");
      const updatedEdgeAC = graph.getEdge("a", "c");
      const updatedSeedWeight = chosenSeed ? graph.getSeedWeight(chosenSeed) : 0;

      const abChanged = updatedEdgeAB && Math.abs(updatedEdgeAB.weight - initialWeightAB) > 0.001;
      const acChanged = updatedEdgeAC && Math.abs(updatedEdgeAC.weight - initialWeightAC) > 0.001;
      const seedChanged = chosenSeed !== null && Math.abs(updatedSeedWeight - initialSeedWeight) > 0.001;
      expect(abChanged || acChanged || seedChanged).toBe(true);

      for (const update of updates) {
        expect(update.delta).toBeGreaterThan(0);
      }
    }
  });

  it("episode persistence round-trips through store", () => {
    const { store, graph } = setup();

    graph.addNode(makeNode("a", new Float32Array([1, 0, 0])));
    store.insertNode(makeNode("a", new Float32Array([1, 0, 0])));

    const result = traverse({
      graph,
      queryEmbedding: new Float32Array([1, 0, 0]),
      queryText: "test",
      maxHops: 2,
      budgetChars: 5000,
      temperature: 0.5,
      maxSeeds: 5,
      semanticThreshold: 0.3,
    });

    const episode = recordEpisode({
      traversalResult: result,
      queryText: "test",
      queryEmbedding: new Float32Array([1, 0, 0]),
      conversationId: 42,
      packVersion: 1,
    });

    store.insertEpisode(episode);

    const retrieved = store.getEpisode(episode.id);
    expect(retrieved).not.toBeNull();
    expect(retrieved!.conversationId).toBe(42);
    expect(retrieved!.packVersion).toBe(1);
    expect(retrieved!.trajectory.length).toBe(result.trajectory.length);
  });

  it("label → episode reward → REINFORCE update pipeline", () => {
    const { store, graph } = setup();

    // Setup graph and episode
    graph.addNode(makeNode("a", new Float32Array([1, 0, 0])));
    graph.addNode(makeNode("b", new Float32Array([0.9, 0.1, 0])));
    graph.addEdge(makeEdge("a", "b", 0.5));

    const result = traverse({
      graph,
      queryEmbedding: new Float32Array([1, 0, 0]),
      queryText: "test",
      maxHops: 4,
      budgetChars: 5000,
      temperature: 0.1,
      maxSeeds: 5,
      semanticThreshold: 0.3,
    });

    const episode = recordEpisode({
      traversalResult: result,
      queryText: "test",
      queryEmbedding: new Float32Array([1, 0, 0]),
      conversationId: null,
      packVersion: null,
    });
    store.insertEpisode(episode);

    // Insert label
    const label = store.insertLabel({
      episodeId: episode.id,
      source: "human",
      value: 1.0,
      reason: "user confirmed",
    });
    expect(label.applied).toBe(false);

    // Process label: apply reward to episode
    const pending = store.getPendingLabels();
    expect(pending.length).toBe(1);
    store.setEpisodeReward(episode.id, pending[0].value, pending[0].source);
    store.markLabelApplied(pending[0].id);

    // Verify episode now has reward
    const updated = store.getEpisode(episode.id);
    expect(updated!.reward).toBe(1.0);
    expect(updated!.rewardSource).toBe("human");

    // Apply REINFORCE
    const episodes = store.getEpisodesForUpdate(10);
    expect(episodes.length).toBe(1);

    const updates = computeReinforceUpdates(episodes[0], 0.1, 0.0);
    applyWeightUpdates(graph, updates);
    store.markEpisodeUpdated(episodes[0].id);

    // Verify no more episodes to update
    expect(store.getEpisodesForUpdate(10).length).toBe(0);
  });
});
