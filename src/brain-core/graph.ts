/**
 * In-memory knowledge graph for the brain's learned retrieval layer.
 *
 * Loaded from SQLite, provides adjacency queries, seed selection by
 * embedding similarity, action set computation, and inhibitory veto checks.
 */

import type {
  BrainNode,
  BrainEdge,
  EdgeKind,
  NodeKind,
  TraversalAction,
} from "./types.js";
import { START_NODE_ID } from "./types.js";

/**
 * Cosine similarity between two Float32Arrays.
 */
export function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length || a.length === 0) return 0;
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}

export class BrainGraph {
  private nodes: Map<string, BrainNode> = new Map();
  private outEdges: Map<string, BrainEdge[]> = new Map();
  private inEdges: Map<string, BrainEdge[]> = new Map();
  private seedWeights: Map<string, number> = new Map();
  private stopLocalWeights: Map<string, number> = new Map();

  addNode(node: BrainNode): void {
    this.nodes.set(node.id, node);
    if (!this.outEdges.has(node.id)) this.outEdges.set(node.id, []);
    if (!this.inEdges.has(node.id)) this.inEdges.set(node.id, []);
  }

  removeNode(nodeId: string): void {
    this.nodes.delete(nodeId);
    this.seedWeights.delete(nodeId);
    this.stopLocalWeights.delete(nodeId);
    // Remove all edges involving this node
    const out = this.outEdges.get(nodeId) ?? [];
    for (const edge of out) {
      const targetIn = this.inEdges.get(edge.target);
      if (targetIn) {
        const idx = targetIn.indexOf(edge);
        if (idx >= 0) targetIn.splice(idx, 1);
      }
    }
    this.outEdges.delete(nodeId);

    const inc = this.inEdges.get(nodeId) ?? [];
    for (const edge of inc) {
      const sourceOut = this.outEdges.get(edge.source);
      if (sourceOut) {
        const idx = sourceOut.indexOf(edge);
        if (idx >= 0) sourceOut.splice(idx, 1);
      }
    }
    this.inEdges.delete(nodeId);
  }

  getNode(nodeId: string): BrainNode | undefined {
    return this.nodes.get(nodeId);
  }

  getAllNodes(): BrainNode[] {
    return [...this.nodes.values()];
  }

  getAllEdges(): BrainEdge[] {
    return [...this.outEdges.values()].flatMap((edges) => edges);
  }

  getNodesByKind(kind: NodeKind): BrainNode[] {
    return [...this.nodes.values()].filter((n) => n.kind === kind);
  }

  addEdge(edge: BrainEdge): void {
    const out = this.outEdges.get(edge.source);
    if (out) {
      // Replace if same source/target/kind exists
      const idx = out.findIndex(
        (e) => e.target === edge.target && e.kind === edge.kind,
      );
      if (idx >= 0) out[idx] = edge;
      else out.push(edge);
    } else {
      this.outEdges.set(edge.source, [edge]);
    }

    const inc = this.inEdges.get(edge.target);
    if (inc) {
      const idx = inc.findIndex(
        (e) => e.source === edge.source && e.kind === edge.kind,
      );
      if (idx >= 0) inc[idx] = edge;
      else inc.push(edge);
    } else {
      this.inEdges.set(edge.target, [edge]);
    }
  }

  removeEdge(source: string, target: string, kind: EdgeKind): void {
    const out = this.outEdges.get(source);
    if (out) {
      const idx = out.findIndex((e) => e.target === target && e.kind === kind);
      if (idx >= 0) out.splice(idx, 1);
    }
    const inc = this.inEdges.get(target);
    if (inc) {
      const idx = inc.findIndex((e) => e.source === source && e.kind === kind);
      if (idx >= 0) inc.splice(idx, 1);
    }
  }

  getEdge(source: string, target: string): BrainEdge | undefined {
    const out = this.outEdges.get(source);
    if (!out) return undefined;
    return out.find((e) => e.target === target);
  }

  getOutgoingEdges(nodeId: string): BrainEdge[] {
    return this.outEdges.get(nodeId) ?? [];
  }

  getIncomingEdges(nodeId: string): BrainEdge[] {
    return this.inEdges.get(nodeId) ?? [];
  }

  getNeighbors(nodeId: string): string[] {
    const edges = this.outEdges.get(nodeId) ?? [];
    return [...new Set(edges.map((e) => e.target))];
  }

  getSeedWeight(nodeId: string): number {
    return this.seedWeights.get(nodeId) ?? 0;
  }

  getSeedWeights(nodeIds: string[]): Record<string, number> {
    const weights: Record<string, number> = {};
    for (const nodeId of nodeIds) {
      weights[nodeId] = this.getSeedWeight(nodeId);
    }
    return weights;
  }

  setSeedWeight(nodeId: string, weight: number): void {
    if (!this.nodes.has(nodeId)) {
      return;
    }
    this.seedWeights.set(nodeId, weight);
  }

  getStopLocalWeight(sourceNodeId: string | null): number {
    return this.stopLocalWeights.get(sourceNodeId ?? START_NODE_ID) ?? 0;
  }

  setStopLocalWeight(sourceNodeId: string | null, weight: number): void {
    const key = sourceNodeId ?? START_NODE_ID;
    if (key !== START_NODE_ID && !this.nodes.has(key)) {
      return;
    }
    this.stopLocalWeights.set(key, weight);
  }

  getAllSeedWeights(): Array<{ nodeId: string; weight: number }> {
    return [...this.seedWeights.entries()].map(([nodeId, weight]) => ({ nodeId, weight }));
  }

  getAllStopLocalWeights(): Array<{ sourceNodeId: string; weight: number }> {
    return [...this.stopLocalWeights.entries()].map(([sourceNodeId, weight]) => ({
      sourceNodeId,
      weight,
    }));
  }

  hasSeedWeights(): boolean {
    return this.seedWeights.size > 0;
  }

  clone(): BrainGraph {
    const clone = new BrainGraph();
    for (const node of this.nodes.values()) {
      clone.addNode({
        ...node,
        embedding: node.embedding ? new Float32Array(node.embedding) : null,
        tags: [...node.tags],
        metadata: { ...node.metadata },
      });
    }
    for (const edge of this.getAllEdges()) {
      clone.addEdge({
        ...edge,
        metadata: { ...edge.metadata },
      });
    }
    for (const { nodeId, weight } of this.getAllSeedWeights()) {
      clone.setSeedWeight(nodeId, weight);
    }
    for (const { sourceNodeId, weight } of this.getAllStopLocalWeights()) {
      clone.setStopLocalWeight(sourceNodeId, weight);
    }
    return clone;
  }

  /**
   * Seed selection: top-k nodes by cosine similarity to query embedding.
   * Linear scan — fine for <10K nodes.
   */
  seedByEmbedding(
    queryEmbedding: Float32Array,
    topK: number,
    threshold: number,
  ): Array<{ nodeId: string; score: number }> {
    const scored: Array<{ nodeId: string; score: number }> = [];

    for (const node of this.nodes.values()) {
      if (!node.embedding) continue;
      const score = cosineSimilarity(queryEmbedding, node.embedding);
      if (score >= threshold) {
        scored.push({ nodeId: node.id, score });
      }
    }

    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, topK);
  }

  /**
   * Compute action set at current state.
   * A(s) = { traverse(neighbor) for neighbor not in visited } ∪ { STOP }
   *
   * At seed phase (currentNodeId === null), uses provided seeds.
   */
  getActionSet(
    sourceNodeId: string | null,
    visited: Set<string>,
    options: {
      seeds?: Array<{ nodeId: string; score?: number }>;
      excludedTargets?: Set<string>;
    } = {},
  ): TraversalAction[] {
    const actions: TraversalAction[] = [];
    const excludedTargets = options.excludedTargets ?? new Set<string>();

    if (sourceNodeId === null) {
      // Seed phase: actions are the seed candidates
      if (options.seeds) {
        for (const seed of options.seeds) {
          if (!visited.has(seed.nodeId) && !excludedTargets.has(seed.nodeId)) {
            actions.push({ type: "traverse", targetNodeId: seed.nodeId, seedScore: seed.score });
          }
        }
      }
    } else {
      // Normal phase: neighbors via outgoing edges
      const neighbors = this.getNeighbors(sourceNodeId).sort();
      for (const neighborId of neighbors) {
        if (!visited.has(neighborId) && !excludedTargets.has(neighborId)) {
          actions.push({ type: "traverse", targetNodeId: neighborId });
        }
      }
    }

    // Local STOP is always available within an expansion.
    actions.push({ type: "stop_local" });
    return actions;
  }

  /**
   * Check if traversal from source to target is suppressed by an inhibitory edge.
   */
  isVetoed(sourceNodeId: string, targetNodeId: string): boolean {
    const edges = this.outEdges.get(sourceNodeId) ?? [];
    return edges.some(
      (e) => e.target === targetNodeId && (e.kind === "inhibitory" || e.weight < -0.5),
    );
  }

  getVetoReason(sourceNodeId: string, targetNodeId: string): string | null {
    const edges = this.outEdges.get(sourceNodeId) ?? [];
    const vetoEdge = edges.find(
      (e) => e.target === targetNodeId && (e.kind === "inhibitory" || e.weight < -0.5),
    );
    if (!vetoEdge) return null;
    if (vetoEdge.kind === "inhibitory") return "inhibitory edge";
    return `negative weight (${vetoEdge.weight.toFixed(3)})`;
  }

  /**
   * Nodes with no edges at all.
   */
  getOrphanNodes(): string[] {
    const orphans: string[] = [];
    for (const nodeId of this.nodes.keys()) {
      const outCount = (this.outEdges.get(nodeId) ?? []).length;
      const inCount = (this.inEdges.get(nodeId) ?? []).length;
      if (outCount === 0 && inCount === 0) {
        orphans.push(nodeId);
      }
    }
    return orphans;
  }

  /**
   * Nodes not present in any recent episode's fired list.
   */
  getDormantNodes(recentFiredNodeIds: Set<string>): string[] {
    const dormant: string[] = [];
    for (const nodeId of this.nodes.keys()) {
      if (!recentFiredNodeIds.has(nodeId)) {
        dormant.push(nodeId);
      }
    }
    return dormant;
  }

  nodeCount(): number {
    return this.nodes.size;
  }

  edgeCount(): number {
    let count = 0;
    for (const edges of this.outEdges.values()) {
      count += edges.length;
    }
    return count;
  }

  /**
   * Clear all nodes and edges.
   */
  clear(): void {
    this.nodes.clear();
    this.outEdges.clear();
    this.inEdges.clear();
    this.seedWeights.clear();
    this.stopLocalWeights.clear();
  }
}
