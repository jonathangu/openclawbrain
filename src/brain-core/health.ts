/**
 * Graph health metrics for promotion gates and operator visibility.
 */

import type { HealthMetrics, NodeKind, EdgeKind, Episode } from "./types.js";
import type { BrainGraph } from "./graph.js";

const NODE_KINDS: NodeKind[] = ["chunk", "workflow", "correction", "toolcard", "episode_anchor", "summary_bridge"];
const EDGE_KINDS: EdgeKind[] = ["sibling", "semantic", "learned", "seed", "inhibitory", "bridge"];

export function computeFiredPerQuery(episodes: Episode[]): number {
  if (episodes.length === 0) return 0;
  const totalFired = episodes.reduce((sum, ep) => sum + ep.firedNodes.length, 0);
  return totalFired / episodes.length;
}

export function computeDormantPercent(graph: BrainGraph, episodes: Episode[]): number {
  const totalNodes = graph.nodeCount();
  if (totalNodes === 0) return 0;

  const firedSet = new Set<string>();
  for (const ep of episodes) {
    for (const id of ep.firedNodes) firedSet.add(id);
  }

  const dormant = graph.getDormantNodes(firedSet);
  return dormant.length / totalNodes;
}

export function computeInhibitoryPercent(graph: BrainGraph): number {
  const totalEdges = graph.edgeCount();
  if (totalEdges === 0) return 0;

  let inhibCount = 0;
  for (const edge of graph.getAllEdges()) {
    if (edge.kind === "inhibitory" || edge.weight < 0) inhibCount++;
  }
  return inhibCount / totalEdges;
}

export function computeAvgPathLength(episodes: Episode[]): number {
  if (episodes.length === 0) return 0;
  const totalHops = episodes.reduce((sum, ep) => sum + ep.trajectory.length, 0);
  return totalHops / episodes.length;
}

export function computeAvgReward(episodes: Episode[]): number {
  const rewarded = episodes.filter((ep) => ep.reward !== null);
  if (rewarded.length === 0) return 0;
  const totalReward = rewarded.reduce((sum, ep) => sum + (ep.reward ?? 0), 0);
  return totalReward / rewarded.length;
}

export function computeChurn(
  previous: HealthMetrics | null,
  current: HealthMetrics,
): number {
  if (!previous) return 0;
  return Math.abs(current.avgReward - previous.avgReward)
    + Math.abs(current.firedPerQuery - previous.firedPerQuery) * 0.1
    + Math.abs(current.dormantPercent - previous.dormantPercent) * 0.1;
}

export function computeHealth(
  graph: BrainGraph,
  recentEpisodes: Episode[],
  packVersion: number,
  previousHealth?: HealthMetrics | null,
): HealthMetrics {
  const nodesByKind = {} as Record<NodeKind, number>;
  for (const kind of NODE_KINDS) nodesByKind[kind] = 0;
  for (const node of graph.getAllNodes()) nodesByKind[node.kind]++;

  const edgesByKind = {} as Record<EdgeKind, number>;
  for (const kind of EDGE_KINDS) edgesByKind[kind] = 0;
  for (const edge of graph.getAllEdges()) {
    edgesByKind[edge.kind]++;
  }

  // Cross-file edge percent: edges connecting nodes from different sourceUris
  let crossFileCount = 0;
  let totalEdgeCount = 0;
  for (const edge of graph.getAllEdges()) {
    totalEdgeCount++;
    const sourceNode = graph.getNode(edge.source);
    const targetNode = graph.getNode(edge.target);
    if (sourceNode && targetNode && sourceNode.sourceUri !== targetNode.sourceUri) {
      crossFileCount++;
    }
  }

  const health: HealthMetrics = {
    nodeCount: graph.nodeCount(),
    edgeCount: graph.edgeCount(),
    nodesByKind,
    edgesByKind,
    firedPerQuery: computeFiredPerQuery(recentEpisodes),
    dormantPercent: computeDormantPercent(graph, recentEpisodes),
    inhibitoryPercent: computeInhibitoryPercent(graph),
    orphanCount: graph.getOrphanNodes().length,
    avgPathLength: computeAvgPathLength(recentEpisodes),
    avgReward: computeAvgReward(recentEpisodes),
    crossFileEdgePercent: totalEdgeCount > 0 ? crossFileCount / totalEdgeCount : 0,
    churn: computeChurn(previousHealth ?? null, {
      // Temporary partial health for churn computation
      avgReward: computeAvgReward(recentEpisodes),
      firedPerQuery: computeFiredPerQuery(recentEpisodes),
      dormantPercent: computeDormantPercent(graph, recentEpisodes),
    } as HealthMetrics),
    packVersion,
    lastUpdateAt: Date.now(),
    totalEpisodes: recentEpisodes.length,
  };

  return health;
}
