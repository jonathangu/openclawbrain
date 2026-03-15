/**
 * Structural graph mutations: split, merge, prune, connect, inject.
 *
 * All mutations are proposals validated via replay gate before promotion.
 * Mutations happen in the learned retrieval graph ONLY, never the LCM summary DAG.
 */

import { randomUUID } from "node:crypto";
import type { Episode, MutationProposal, BrainNode } from "./types.js";
import type { BrainGraph } from "./graph.js";
import { cosineSimilarity } from "./graph.js";

export interface BrainMutationPersistence {
  insertNode(node: BrainNode): void;
  insertEdge(edge: {
    source: string;
    target: string;
    kind: "learned";
    weight: number;
    prior: number;
    metadata: Record<string, unknown>;
    decayedAt: number;
    createdAt: number;
  }): void;
  deleteNode(id: string): void;
  deleteEdge(source: string, target: string, kind: string): void;
  resolveMutation(id: string, status: "promoted" | "rejected"): void;
}

export class BrainMutator {
  constructor(
    private persistence: BrainMutationPersistence,
    private graph: BrainGraph,
    private log: { info: (msg: string) => void },
  ) {}

  /**
   * Propose mutations based on episode patterns.
   * Called periodically by the trainer.
   */
  proposeMutations(recentEpisodes: Episode[]): MutationProposal[] {
    const proposals: MutationProposal[] = [];

    proposals.push(...this.proposePrunes());
    proposals.push(...this.proposeConnections(recentEpisodes));
    proposals.push(...this.proposeInjects(recentEpisodes));

    return proposals;
  }

  /**
   * Prune: edges dormant across many episodes.
   * Signal: weight has decayed near zero and no recent traversal.
   */
  private proposePrunes(): MutationProposal[] {
    const proposals: MutationProposal[] = [];

    for (const node of this.graph.getAllNodes()) {
      for (const edge of this.graph.getOutgoingEdges(node.id)) {
        if (edge.kind === "sibling" || edge.kind === "bridge") continue;
        if (Math.abs(edge.weight) < 0.05 && Math.abs(edge.weight - edge.prior) < 0.05) {
          proposals.push({
            id: `bm_${randomUUID().slice(0, 8)}`,
            kind: "prune",
            proposal: { source: edge.source, target: edge.target, edgeKind: edge.kind, weight: edge.weight },
            evidence: null,
            expectedGain: 0.01,
            status: "pending",
            createdAt: Date.now(),
            resolvedAt: null,
          });
        }
      }
    }

    return proposals.slice(0, 5); // Limit per tick
  }

  /**
   * Connect: successful episodes repeatedly bridge two regions.
   * Signal: nodes co-fire frequently but have no direct edge.
   */
  private proposeConnections(episodes: Episode[]): MutationProposal[] {
    const coFiring = new Map<string, number>();

    for (const ep of episodes) {
      if (ep.reward === null || ep.reward < 0.3) continue;
      for (let i = 0; i < ep.firedNodes.length; i++) {
        for (let j = i + 1; j < ep.firedNodes.length; j++) {
          const key = [ep.firedNodes[i], ep.firedNodes[j]].sort().join("↔");
          coFiring.set(key, (coFiring.get(key) ?? 0) + 1);
        }
      }
    }

    const proposals: MutationProposal[] = [];
    for (const [key, count] of coFiring) {
      if (count < 3) continue;
      const [a, b] = key.split("↔");
      if (this.graph.getEdge(a, b) || this.graph.getEdge(b, a)) continue;

      proposals.push({
        id: `bm_${randomUUID().slice(0, 8)}`,
        kind: "connect",
        proposal: { nodeA: a, nodeB: b, coFireCount: count },
        evidence: { episodeCount: count },
        expectedGain: count * 0.05,
        status: "pending",
        createdAt: Date.now(),
        resolvedAt: null,
      });
    }

    return proposals.slice(0, 3);
  }

  private proposeInjects(episodes: Episode[]): MutationProposal[] {
    const proposals: MutationProposal[] = [];
    const seenQueries = new Set<string>();

    for (const episode of episodes) {
      if (episode.reward === null || episode.reward < 0.6) {
        continue;
      }
      if (seenQueries.has(episode.queryText)) {
        continue;
      }
      seenQueries.add(episode.queryText);
      if (episode.firedNodes.length < 2) {
        continue;
      }

      proposals.push({
        id: `bm_${randomUUID().slice(0, 8)}`,
        kind: "inject",
        proposal: {
          nodeKind: "episode_anchor",
          content: episode.queryText,
          firedNodes: episode.firedNodes,
        },
        evidence: { episodeId: episode.id, reward: episode.reward },
        expectedGain: 0.05,
        status: "pending",
        createdAt: Date.now(),
        resolvedAt: null,
      });
      if (proposals.length >= 3) {
        break;
      }
    }

    return proposals;
  }

  /**
   * Apply a validated mutation to the graph and store.
   */
  private applyMutationToGraph(targetGraph: BrainGraph, proposal: MutationProposal): {
    insertedEdges?: Array<{
      source: string;
      target: string;
      kind: "learned";
      weight: number;
      prior: number;
      metadata: Record<string, unknown>;
      decayedAt: number;
      createdAt: number;
    }>;
    deletedEdges?: Array<{ source: string; target: string; kind: string }>;
    deletedNodes?: string[];
    insertedNodes?: BrainNode[];
  } {
    const p = proposal.proposal as Record<string, unknown>;
    const result: {
      insertedEdges?: Array<{
        source: string;
        target: string;
        kind: "learned";
        weight: number;
        prior: number;
        metadata: Record<string, unknown>;
        decayedAt: number;
        createdAt: number;
      }>;
      deletedEdges?: Array<{ source: string; target: string; kind: string }>;
      deletedNodes?: string[];
      insertedNodes?: BrainNode[];
    } = {};

    switch (proposal.kind) {
      case "prune": {
        targetGraph.removeEdge(p.source as string, p.target as string, p.edgeKind as any);
        result.deletedEdges = [{
          source: p.source as string,
          target: p.target as string,
          kind: p.edgeKind as string,
        }];
        break;
      }
      case "connect": {
        const now = Date.now();
        const edge = {
          source: p.nodeA as string,
          target: p.nodeB as string,
          kind: "learned" as const,
          weight: 0.5,
          prior: 0.5,
          metadata: { mutationId: proposal.id },
          decayedAt: now,
          createdAt: now,
        };
        targetGraph.addEdge(edge);
        result.insertedEdges = [edge];
        break;
      }
      case "inject": {
        const now = Date.now();
        const node: BrainNode = {
          id: `bn_${randomUUID().slice(0, 12)}`,
          kind: "episode_anchor",
          content: String(p.content ?? ""),
          embedding: null,
          sourceUri: null,
          trust: "scanner",
          tags: ["episode-anchor"],
          tokenCount: Math.ceil(String(p.content ?? "").length / 4),
          metadata: {
            mutationId: proposal.id,
            firedNodes: Array.isArray(p.firedNodes) ? p.firedNodes : [],
          },
          createdAt: now,
          updatedAt: now,
        };
        targetGraph.addNode(node);
        result.insertedNodes = [node];

        const firedNodes = Array.isArray(p.firedNodes) ? p.firedNodes.filter((value): value is string => typeof value === "string") : [];
        result.insertedEdges = firedNodes.map((firedNodeId) => {
          const edge = {
            source: node.id,
            target: firedNodeId,
            kind: "learned" as const,
            weight: 0.4,
            prior: 0.4,
            metadata: { mutationId: proposal.id },
            decayedAt: now,
            createdAt: now,
          };
          targetGraph.addEdge(edge);
          return edge;
        });
        break;
      }
    }

    return result;
  }

  applyToCandidateGraph(targetGraph: BrainGraph, proposal: MutationProposal): void {
    this.applyMutationToGraph(targetGraph, proposal);
  }

  applyMutation(proposal: MutationProposal): void {
    const result = this.applyMutationToGraph(this.graph, proposal);
    for (const edge of result.insertedEdges ?? []) {
      this.persistence.insertEdge(edge);
    }
    for (const node of result.insertedNodes ?? []) {
      this.persistence.insertNode(node);
    }
    for (const edge of result.deletedEdges ?? []) {
      this.persistence.deleteEdge(edge.source, edge.target, edge.kind);
    }
    for (const nodeId of result.deletedNodes ?? []) {
      this.persistence.deleteNode(nodeId);
    }

    this.persistence.resolveMutation(proposal.id, "promoted");
    this.log.info(`[brain] Applied ${proposal.kind}: ${proposal.id}`);
  }
}
