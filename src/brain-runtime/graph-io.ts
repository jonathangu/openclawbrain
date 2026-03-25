import { computeHealth } from "../brain-core/health.js";
import type { BrainConfig, BrainNode, SeedWeight, StopLocalWeight } from "../brain-core/types.js";
import type { BrainGraph } from "../brain-core/graph.js";
import { PackManager } from "../brain-core/pack.js";
import type { BrainStore } from "../brain-store/store.js";

export function flattenEdges(graph: BrainGraph) {
  return graph.getAllEdges();
}

export function flattenSeedWeights(graph: BrainGraph): SeedWeight[] {
  const now = Date.now();
  return graph.getAllSeedWeights().map((seedWeight) => ({
    nodeId: seedWeight.nodeId,
    weight: seedWeight.weight,
    updatedAt: now,
  }));
}

export function flattenStopLocalWeights(graph: BrainGraph): StopLocalWeight[] {
  const now = Date.now();
  return graph.getAllStopLocalWeights().map((stopLocalWeight) => ({
    sourceNodeId: stopLocalWeight.sourceNodeId,
    weight: stopLocalWeight.weight,
    updatedAt: now,
  }));
}

export function populateGraph(
  graph: BrainGraph,
  nodes: BrainNode[],
  edges: ReturnType<typeof flattenEdges>,
  seedWeights: SeedWeight[] = [],
  stopLocalWeights: StopLocalWeight[] = [],
): void {
  graph.clear();
  for (const node of nodes) {
    graph.addNode(node);
  }
  for (const edge of edges) {
    graph.addEdge(edge);
  }
  for (const seedWeight of seedWeights) {
    graph.setSeedWeight(seedWeight.nodeId, seedWeight.weight);
  }
  for (const stopLocalWeight of stopLocalWeights) {
    graph.setStopLocalWeight(stopLocalWeight.sourceNodeId, stopLocalWeight.weight);
  }
}

export function reloadGraphFromStore(store: BrainStore, graph: BrainGraph): void {
  populateGraph(
    graph,
    store.getAllNodes(),
    store.loadAllEdges(),
    store.loadAllSeedWeights(),
    store.loadAllStopLocalWeights(),
  );
}

export function promoteGraphSnapshot(params: {
  store: BrainStore;
  graph: BrainGraph;
  packManager: PackManager;
  config: BrainConfig;
  reason: string;
  metadata: Record<string, unknown>;
}): number | null {
  if (params.graph.nodeCount() === 0) {
    return null;
  }

  const health = computeHealth(
    params.graph,
    params.store.getRecentEpisodes(params.config.replayEpisodeCount),
    params.store.getCurrentPackVersion() ?? 0,
  );
  const pack = params.packManager.buildCandidate(health);
  params.store.writePackSnapshot({
    version: pack.version,
    nodes: params.graph.getAllNodes(),
    edges: flattenEdges(params.graph),
    seedWeights: flattenSeedWeights(params.graph),
    stopLocalWeights: flattenStopLocalWeights(params.graph),
    metadata: {
      reason: params.reason,
      ...params.metadata,
    },
  });
  params.packManager.promote(pack.version);
  return pack.version;
}
