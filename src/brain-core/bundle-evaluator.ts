/**
 * Mutation Bundle Evaluation
 *
 * Clusters individual mutation proposals into bundles and evaluates them
 * via comparative replay against recent episodes before promotion.
 */

import { randomUUID } from "node:crypto";
import type { BrainGraph } from "./graph.js";
import type {
  BundleEvaluationReason,
  BundleEvaluationVerdict,
  Episode,
  MutationBundleStatus,
  MutationProposal,
} from "./types.js";
import {
  applyShadowMutationProposalToState,
  createShadowCandidateState,
} from "./shadow-application.js";

/** Bundle of clustered mutations to evaluate together */
export interface MutationBundle {
  id: string;
  mutationIds: string[];
  proposals: MutationProposal[];
  bundleSize: number;
  status: MutationBundleStatus;
  baseScore: number | null;
  candidateScore: number | null;
  expectedGain: number;
  rejectionReason: string | null;
  createdAt: number;
  resolvedAt: number | null;
}

/** Configuration for bundle evaluation */
export interface BundleEvaluationConfig {
  /** Minimum mutations per bundle */
  minBundleSize: number;
  /** Maximum mutations per bundle */
  maxBundleSize: number;
  /** Minimum reward threshold for episodes to consider */
  minRewardThreshold: number;
  /** Maximum allowed context inflation ratio */
  maxContextInflation: number;
  /** Minimum improvement ratio to promote */
  minImprovementRatio: number;
}

/** Default configuration */
export const DEFAULT_BUNDLE_CONFIG: BundleEvaluationConfig = {
  minBundleSize: 3,
  maxBundleSize: 10,
  minRewardThreshold: 0.3,
  maxContextInflation: 1.5,
  minImprovementRatio: 1.05,
};

/**
 * Cluster mutations into bundles based on graph neighborhood.
 *
 * Connect mutations cluster by shared nodes.
 * Prune mutations cluster by source/target node relationships.
 * Inject mutations cluster by query similarity or shared fired nodes.
 */
export function clusterMutationsIntoBundles(
  proposals: MutationProposal[],
  config: BundleEvaluationConfig = DEFAULT_BUNDLE_CONFIG
): MutationBundle[] {
  if (proposals.length < config.minBundleSize) {
    return [];
  }

  const bundles: MutationBundle[] = [];
  const used = new Set<string>();

  // Build adjacency for connect mutations (shared nodes)
  const connectProposals = proposals.filter(p => p.kind === "connect");
  const pruneProposals = proposals.filter(p => p.kind === "prune");
  const injectProposals = proposals.filter(p => p.kind === "inject");

  // Cluster connect mutations - group by node neighborhood
  const connectBundles = clusterByNeighbor(connectProposals, config.maxBundleSize);
  bundles.push(...connectBundles);
  connectBundles.forEach(b => b.proposals.forEach(p => used.add(p.id)));

  // Cluster prune mutations
  const pruneBundles = clusterByNeighbor(pruneProposals, config.maxBundleSize);
  bundles.push(...pruneBundles);
  pruneBundles.forEach(b => b.proposals.forEach(p => used.add(p.id)));

  // Cluster inject mutations by query similarity (simple token overlap)
  const injectBundles = clusterByTokenOverlap(injectProposals, config.maxBundleSize);
  bundles.push(...injectBundles);
  injectBundles.forEach(b => b.proposals.forEach(p => used.add(p.id)));

  // Handle remaining orphans - either discard or create single-item bundles
  const orphans = proposals.filter(p => !used.has(p.id));
  for (const orphan of orphans) {
    bundles.push(createBundle([orphan], config));
  }

  // Filter to only bundles meeting minimum size
  return bundles.filter(b => b.proposals.length >= config.minBundleSize);
}

/**
 * Cluster proposals by shared node relationships
 */
function clusterByNeighbor(
  proposals: MutationProposal[],
  maxSize: number
): MutationBundle[] {
  if (proposals.length === 0) return [];

  const bundles: MutationBundle[] = [];
  const nodeToProposals = new Map<string, MutationProposal[]>();

  // Index proposals by involved nodes
  for (const proposal of proposals) {
    const nodes = getProposalNodes(proposal);
    for (const node of nodes) {
      const list = nodeToProposals.get(node) ?? [];
      list.push(proposal);
      nodeToProposals.set(node, list);
    }
  }

  // Cluster by connected components in the proposal graph
  const used = new Set<string>();
  const proposalAdj = new Map<string, Set<string>>();

  // Build adjacency
  for (const [node, propList] of nodeToProposals) {
    if (propList.length > 1) {
      for (let i = 0; i < propList.length; i++) {
        for (let j = i + 1; j < propList.length; j++) {
          const a = propList[i].id;
          const b = propList[j].id;
          if (!proposalAdj.has(a)) proposalAdj.set(a, new Set());
          if (!proposalAdj.has(b)) proposalAdj.set(b, new Set());
          proposalAdj.get(a)!.add(b);
          proposalAdj.get(b)!.add(a);
        }
      }
    }
  }

  // Find connected components
  for (const proposal of proposals) {
    if (used.has(proposal.id)) continue;

    const cluster: MutationProposal[] = [];
    const queue = [proposal];

    while (queue.length > 0 && cluster.length < maxSize) {
      const current = queue.shift()!;
      if (used.has(current.id)) continue;
      used.add(current.id);
      cluster.push(current);

      const neighbors = proposalAdj.get(current.id) ?? new Set();
      for (const neighborId of neighbors) {
        const neighbor = proposals.find(p => p.id === neighborId);
        if (neighbor && !used.has(neighbor.id)) {
          queue.push(neighbor);
        }
      }
    }

    if (cluster.length > 0) {
      bundles.push(createBundle(cluster, DEFAULT_BUNDLE_CONFIG));
    }
  }

  return bundles;
}

/**
 * Get nodes involved in a proposal
 */
function getProposalNodes(proposal: MutationProposal): string[] {
  const p = proposal.proposal as Record<string, unknown>;
  switch (proposal.kind) {
    case "connect":
      return [p.nodeA as string, p.nodeB as string];
    case "prune":
      return [p.source as string, p.target as string];
    case "inject":
      return (p.firedNodes as string[]) ?? [];
    default:
      return [];
  }
}

/**
 * Cluster inject mutations by token overlap
 */
function clusterByTokenOverlap(
  proposals: MutationProposal[],
  maxSize: number
): MutationBundle[] {
  if (proposals.length === 0) return [];

  const bundles: MutationBundle[] = [];
  const used = new Set<string>();

  for (const proposal of proposals) {
    if (used.has(proposal.id)) continue;

    const cluster: MutationProposal[] = [proposal];
    used.add(proposal.id);

    const p = proposal.proposal as { content?: string; queryText?: string };
    const query1 = (p.content ?? p.queryText ?? "").toLowerCase();
    const tokens1 = new Set(query1.split(/\s+/).filter(t => t.length > 2));

    // Find similar proposals
    for (const other of proposals) {
      if (used.has(other.id)) continue;
      if (cluster.length >= maxSize) break;

      const op = other.proposal as { content?: string; queryText?: string };
      const query2 = (op.content ?? op.queryText ?? "").toLowerCase();
      const tokens2 = new Set(query2.split(/\s+/).filter(t => t.length > 2));

      // Calculate Jaccard similarity
      let overlap = 0;
      for (const t of tokens1) {
        if (tokens2.has(t)) overlap++;
      }
      const union = tokens1.size + tokens2.size - overlap;
      const similarity = union > 0 ? overlap / union : 0;

      if (similarity > 0.3) {
        cluster.push(other);
        used.add(other.id);
      }
    }

    if (cluster.length >= DEFAULT_BUNDLE_CONFIG.minBundleSize) {
      bundles.push(createBundle(cluster, DEFAULT_BUNDLE_CONFIG));
    }
  }

  return bundles;
}

/**
 * Create a bundle from proposals
 */
function createBundle(
  proposals: MutationProposal[],
  config: BundleEvaluationConfig
): MutationBundle {
  const expectedGain = proposals.reduce((sum, p) => sum + (p.expectedGain ?? 0), 0);

  return {
    id: `mb_${randomUUID().slice(0, 8)}`,
    mutationIds: proposals.map(p => p.id),
    proposals,
    bundleSize: proposals.length,
    status: "pending",
    baseScore: null,
    candidateScore: null,
    expectedGain,
    rejectionReason: null,
    createdAt: Date.now(),
    resolvedAt: null,
  };
}

/**
 * Evaluate a bundle by comparative replay.
 *
 * Creates a clone of the graph, applies the mutations,
 * and compares retrieval quality on recent episodes.
 */
export async function evaluateBundle(
  bundle: MutationBundle,
  graph: BrainGraph,
  recentEpisodes: Episode[],
  config: BundleEvaluationConfig = DEFAULT_BUNDLE_CONFIG
): Promise<{
  baseScore: number;
  candidateScore: number;
  shouldPromote: boolean;
  rejectionReason: string | null;
  verdict: BundleEvaluationVerdict;
}> {
  // Filter episodes by reward threshold
  const validEpisodes = recentEpisodes.filter(
    ep => ep.reward !== null && ep.reward >= config.minRewardThreshold
  );
  const resolvedAt = Date.now();

  if (validEpisodes.length === 0) {
    const reason: BundleEvaluationReason = {
      code: "no_qualifying_episodes",
      summary: "no qualifying episodes for evaluation",
      details: {
        episodeCount: recentEpisodes.length,
        qualifyingEpisodeCount: 0,
        minRewardThreshold: config.minRewardThreshold,
      },
    };
    const verdict: BundleEvaluationVerdict = {
      bundleId: bundle.id,
      mutationIds: [...bundle.mutationIds],
      bundleSize: bundle.bundleSize,
      status: "rejected",
      baseScore: 0,
      candidateScore: 0,
      expectedGain: bundle.expectedGain,
      evaluatedEpisodeCount: recentEpisodes.length,
      qualifyingEpisodeCount: 0,
      improvementRatio: null,
      reason,
      createdAt: bundle.createdAt,
      resolvedAt,
    };
    return {
      baseScore: 0,
      candidateScore: 0,
      shouldPromote: false,
      rejectionReason: reason.summary,
      verdict,
    };
  }

  // Calculate base score (current graph)
  const baseScore = await calculateRetrievalScore(graph, validEpisodes);

  // Clone graph and apply mutations off-path with reversible shadow application.
  const shadowState = createShadowCandidateState(graph);

  for (const proposal of bundle.proposals) {
    applyShadowMutationProposalToState(shadowState, proposal);
  }

  const candidateGraph = shadowState.candidateGraph;

  // Calculate candidate score
  const candidateScore = await calculateRetrievalScore(candidateGraph, validEpisodes);

  const improvementRatio = calculateImprovementRatio(baseScore, candidateScore);

  // Check rejection criteria
  const outcome = checkBundleOutcome(
    baseScore,
    candidateScore,
    improvementRatio,
    config,
    validEpisodes
  );
  const shouldPromote = outcome.status === "promoted";
  const verdict: BundleEvaluationVerdict = {
    bundleId: bundle.id,
    mutationIds: [...bundle.mutationIds],
    bundleSize: bundle.bundleSize,
    status: outcome.status,
    baseScore,
    candidateScore,
    expectedGain: bundle.expectedGain,
    evaluatedEpisodeCount: recentEpisodes.length,
    qualifyingEpisodeCount: validEpisodes.length,
    improvementRatio,
    reason: outcome.reason,
    createdAt: bundle.createdAt,
    resolvedAt,
  };

  return {
    baseScore,
    candidateScore,
    shouldPromote,
    rejectionReason: shouldPromote ? null : outcome.reason.summary,
    verdict,
  };
}

/**
 * Calculate retrieval score for a graph on episodes.
 * 
 * Simple heuristic: reward-weighted retrieval quality.
 * Higher is better.
 */
async function calculateRetrievalScore(
  graph: BrainGraph,
  episodes: Episode[]
): Promise<number> {
  let totalScore = 0;

  for (const episode of episodes) {
    if (!episode.queryText || episode.firedNodes.length === 0) {
      continue;
    }

    // Simple retrieval: check how many fired nodes are reachable
    const queryNode = findOrCreateQueryNode(graph, episode.queryText);
    const reached = new Set<string>();
    
    // BFS from query node
    const queue = [queryNode];
    const visited = new Set<string>([queryNode]);
    
    while (queue.length > 0 && reached.size < 10) {
      const current = queue.shift()!;
      const edges = graph.getOutgoingEdges(current);
      
      for (const edge of edges) {
        if (edge.kind === "learned" && !visited.has(edge.target)) {
          visited.add(edge.target);
          reached.add(edge.target);
          queue.push(edge.target);
        }
      }
    }

    // Score: fraction of fired nodes reached, weighted by reward
    const recall = episode.firedNodes.filter(n => reached.has(n)).length / episode.firedNodes.length;
    totalScore += (episode.reward ?? 0.5) * recall;
  }

  return episodes.length > 0 ? totalScore / episodes.length : 0;
}

/**
 * Find or create a query node for retrieval
 */
function findOrCreateQueryNode(graph: BrainGraph, queryText: string): string {
  const hash = hashString(queryText);
  const nodeId = `__query_${hash.slice(0, 8)}`;
  
  if (!graph.getNode(nodeId)) {
    // Create transient query node (won't persist)
    // This is a simplification - real impl would use embedding similarity
  }
  
  return nodeId;
}

/**
 * Simple string hash
 */
function hashString(str: string): string {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash;
  }
  return Math.abs(hash).toString(36);
}


/**
 * Check rejection criteria
 */
function calculateImprovementRatio(
  baseScore: number,
  candidateScore: number,
): number | null {
  if (baseScore <= 0) {
    return null;
  }
  return candidateScore / baseScore;
}

function checkBundleOutcome(
  baseScore: number,
  candidateScore: number,
  improvementRatio: number | null,
  config: BundleEvaluationConfig,
  episodes: Episode[]
): {
  status: BundleEvaluationVerdict["status"];
  reason: BundleEvaluationReason;
} {
  // 1. Reject if candidate is worse than base
  if (candidateScore < baseScore) {
    return {
      status: "rejected",
      reason: {
        code: "candidate_regressed",
        summary: `candidate score ${candidateScore.toFixed(3)} worse than base ${baseScore.toFixed(3)}`,
        details: {
          baseScore,
          candidateScore,
          evaluatedEpisodeCount: episodes.length,
        },
      },
    };
  }

  // 2. Reject if improvement is too small
  if (baseScore > 0 && improvementRatio !== null && improvementRatio < config.minImprovementRatio) {
    return {
      status: "rejected",
      reason: {
        code: "insufficient_improvement",
        summary: `improvement ratio ${improvementRatio.toFixed(2)} below threshold ${config.minImprovementRatio}`,
        details: {
          baseScore,
          candidateScore,
          improvementRatio,
          minImprovementRatio: config.minImprovementRatio,
          evaluatedEpisodeCount: episodes.length,
        },
      },
    };
  }

  if (baseScore <= 0 && candidateScore <= baseScore) {
    return {
      status: "rejected",
      reason: {
        code: "insufficient_improvement",
        summary: "candidate did not improve over the zero baseline",
        details: {
          baseScore,
          candidateScore,
          improvementRatio,
          minImprovementRatio: config.minImprovementRatio,
          evaluatedEpisodeCount: episodes.length,
        },
      },
    };
  }

  // 3. Reject if context inflation is too high (not implemented - would need context size tracking)
  // This would check that added nodes/edges don't bloat context too much

  return {
    status: "promoted",
    reason: {
      code: "promoted",
      summary: baseScore > 0
        ? `candidate improved replay score from ${baseScore.toFixed(3)} to ${candidateScore.toFixed(3)}`
        : `candidate established positive replay score ${candidateScore.toFixed(3)} from a zero baseline`,
      details: {
        baseScore,
        candidateScore,
        improvementRatio,
        minImprovementRatio: config.minImprovementRatio,
        evaluatedEpisodeCount: episodes.length,
      },
    },
  };
}
