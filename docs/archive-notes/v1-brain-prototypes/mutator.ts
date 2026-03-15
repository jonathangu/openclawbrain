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
import type { BrainStore } from "./store.js";

export class BrainMutator {
  constructor(
    private store: BrainStore,
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
    proposals.push(...this.proposeMerges());

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

  /**
   * Merge: near-duplicate nodes that always co-fire.
   * Signal: cosine similarity > 0.9 and high co-fire rate.
   */
  private proposeMerges(): MutationProposal[] {
    const proposals: MutationProposal[] = [];
    const nodes = this.graph.getAllNodes().filter((n) => n.embedding);

    for (let i = 0; i < nodes.length && proposals.length < 3; i++) {
      for (let j = i + 1; j < nodes.length && proposals.length < 3; j++) {
        const sim = cosineSimilarity(nodes[i].embedding!, nodes[j].embedding!);
        if (sim > 0.92) {
          proposals.push({
            id: `bm_${randomUUID().slice(0, 8)}`,
            kind: "merge",
            proposal: { nodeA: nodes[i].id, nodeB: nodes[j].id, similarity: sim },
            evidence: null,
            expectedGain: 0.02,
            status: "pending",
            createdAt: Date.now(),
            resolvedAt: null,
          });
        }
      }
    }

    return proposals;
  }

  /**
   * Apply a validated mutation to the graph and store.
   */
  applyMutation(proposal: MutationProposal): void {
    const p = proposal.proposal as Record<string, unknown>;

    switch (proposal.kind) {
      case "prune": {
        this.graph.removeEdge(p.source as string, p.target as string, p.edgeKind as any);
        this.log.info(`[brain] Applied prune: ${p.source}→${p.target}`);
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
        this.graph.addEdge(edge);
        this.store.insertEdge(edge);
        this.log.info(`[brain] Applied connect: ${p.nodeA}↔${p.nodeB}`);
        break;
      }
      case "merge": {
        // Keep nodeA, redirect edges from nodeB to nodeA, delete nodeB
        const nodeB = this.graph.getNode(p.nodeB as string);
        if (nodeB) {
          for (const edge of this.graph.getIncomingEdges(nodeB.id)) {
            if (edge.source !== p.nodeA) {
              const redirected = { ...edge, target: p.nodeA as string };
              this.graph.addEdge(redirected);
            }
          }
          for (const edge of this.graph.getOutgoingEdges(nodeB.id)) {
            if (edge.target !== p.nodeA) {
              const redirected = { ...edge, source: p.nodeA as string };
              this.graph.addEdge(redirected);
            }
          }
          this.graph.removeNode(nodeB.id);
          this.store.deleteNode(nodeB.id);
          this.log.info(`[brain] Applied merge: ${p.nodeB} → ${p.nodeA}`);
        }
        break;
      }
    }

    this.store.resolveMutation(proposal.id, "promoted");
  }
}
