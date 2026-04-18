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
  PolicyParams,
  TrustLevel,
} from "./types.ts";
import type {
  ColdStartStopLabelV1,
  RouteCandidateV1,
  RouteDecisionRowV1,
} from "./cold-start-router-contracts.ts";

const START_NODE_ID = "__START__";
const DEFAULT_POLICY_PARAMS: PolicyParams = {
  temperature: 1.0,
  stopBias: -2.0,
  budgetPressure: 3.0,
  hopPressure: 2.0,
  frontierPressure: 1.5,
  branchOpportunityCost: 1.1,
  localRedundancyPenalty: 0.75,
  evidenceQualityBias: 0.45,
  edgeKindBias: {
    sibling: 0.0,
    semantic: 0.1,
    learned: 0.2,
    seed: 0.15,
    inhibitory: -10.0,
    bridge: 0.0,
  },
};

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
  private toolActionPriors: Map<string, Map<string, number>> = new Map();

  addNode(node: BrainNode): void {
    this.nodes.set(node.id, node);
    if (!this.outEdges.has(node.id)) this.outEdges.set(node.id, []);
    if (!this.inEdges.has(node.id)) this.inEdges.set(node.id, []);
  }

  removeNode(nodeId: string): void {
    this.nodes.delete(nodeId);
    this.seedWeights.delete(nodeId);
    this.stopLocalWeights.delete(nodeId);
    this.toolActionPriors.delete(nodeId);
    for (const [sourceNodeId, priors] of this.toolActionPriors.entries()) {
      priors.delete(nodeId);
      if (priors.size === 0) {
        this.toolActionPriors.delete(sourceNodeId);
      }
    }
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

  getToolActionPrior(sourceNodeId: string | null, toolNodeId: string): number {
    const key = sourceNodeId ?? START_NODE_ID;
    return this.toolActionPriors.get(key)?.get(toolNodeId) ?? 0;
  }

  setToolActionPrior(sourceNodeId: string | null, toolNodeId: string, weight: number): void {
    const key = sourceNodeId ?? START_NODE_ID;
    if (key !== START_NODE_ID && !this.nodes.has(key)) {
      return;
    }
    if (!this.nodes.has(toolNodeId)) {
      return;
    }
    const priors = this.toolActionPriors.get(key) ?? new Map<string, number>();
    priors.set(toolNodeId, weight);
    this.toolActionPriors.set(key, priors);
  }

  getToolActionPriorEntries(sourceNodeId: string | null): Array<{ toolNodeId: string; weight: number }> {
    const key = sourceNodeId ?? START_NODE_ID;
    const priors = this.toolActionPriors.get(key);
    if (!priors) {
      return [];
    }
    return [...priors.entries()]
      .map(([toolNodeId, weight]) => ({ toolNodeId, weight }))
      .sort((left, right) => left.toolNodeId.localeCompare(right.toolNodeId));
  }

  getAllToolActionPriors(): Array<{ sourceNodeId: string; toolNodeId: string; weight: number }> {
    return [...this.toolActionPriors.entries()]
      .flatMap(([sourceNodeId, priors]) => [...priors.entries()].map(([toolNodeId, weight]) => ({ sourceNodeId, toolNodeId, weight })))
      .sort((left, right) => {
        const bySource = left.sourceNodeId.localeCompare(right.sourceNodeId);
        if (bySource !== 0) {
          return bySource;
        }
        return left.toolNodeId.localeCompare(right.toolNodeId);
      });
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
    for (const { sourceNodeId, toolNodeId, weight } of this.getAllToolActionPriors()) {
      clone.setToolActionPrior(sourceNodeId, toolNodeId, weight);
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
    const actions = new Map<string, TraversalAction>();
    const excludedTargets = options.excludedTargets ?? new Set<string>();
    const addTraverseAction = (action: Extract<TraversalAction, { type: "traverse" }>): void => {
      if (visited.has(action.targetNodeId) || excludedTargets.has(action.targetNodeId)) {
        return;
      }
      if (!this.nodes.has(action.targetNodeId)) {
        return;
      }
      if (!actions.has(action.targetNodeId)) {
        actions.set(action.targetNodeId, action);
      }
    };

    if (sourceNodeId === null) {
      // Seed phase: actions are the seed candidates
      if (options.seeds) {
        for (const seed of options.seeds) {
          addTraverseAction({ type: "traverse", targetNodeId: seed.nodeId, ...(seed.score !== undefined ? { seedScore: seed.score } : {}) });
        }
      }
    } else {
      // Normal phase: neighbors via outgoing edges
      const neighbors = this.getNeighbors(sourceNodeId).sort();
      for (const neighborId of neighbors) {
        addTraverseAction({ type: "traverse", targetNodeId: neighborId });
      }
    }

    for (const { toolNodeId } of this.getToolActionPriorEntries(sourceNodeId)) {
      addTraverseAction({ type: "traverse", targetNodeId: toolNodeId });
    }

    // Local STOP is always available within an expansion.
    return [...actions.values(), { type: "stop_local" }];
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
    this.toolActionPriors.clear();
  }
}

export const COLD_START_ROUTER_LIVE_POLICY_INITIALIZER_CONTRACT_V1 =
  "cold_start_router_live_policy_initializer.v1" as const;

export const COLD_START_ROUTER_TOOL_ACTION_PRIORS_CONTRACT_V1 =
  "cold_start_router_tool_action_priors.v1" as const;

export interface ColdStartRouterLivePolicySeedWeightV1 {
  nodeId: string;
  positive: number;
  negative: number;
  support: number;
  weight: number;
}

export interface ColdStartRouterLivePolicyStopWeightV1 {
  sourceNodeId: string;
  positive: number;
  negative: number;
  support: number;
  weight: number;
}

export interface ColdStartRouterLivePolicyEdgeWeightV1 {
  sourceNodeId: string;
  targetNodeId: string;
  positive: number;
  negative: number;
  support: number;
  prior: number;
  weight: number;
}

export interface ColdStartRouterLivePolicySemanticClassSeedWeightV1 {
  semanticClass: string;
  positive: number;
  negative: number;
  support: number;
  weight: number;
}

export interface ColdStartRouterLivePolicySemanticClassEdgeWeightV1 {
  sourceBindingKey: string;
  targetSemanticClass: string;
  positive: number;
  negative: number;
  support: number;
  prior: number;
  weight: number;
}

export interface ColdStartRouterToolActionPriorV1 {
  sourceNodeId: string;
  toolNodeId: string;
  positive: number;
  negative: number;
  support: number;
  prior: number;
  weight: number;
}

export interface ColdStartRouterToolActionSetV1 {
  sourceNodeId: string;
  rowIds: string[];
  teacherToolNodeIds: string[];
  candidateIds: string[];
  candidates: RouteCandidateV1[];
  support: number;
}

export interface ColdStartRouterLivePolicyInitializerV1 {
  contract: typeof COLD_START_ROUTER_LIVE_POLICY_INITIALIZER_CONTRACT_V1;
  policyParams: PolicyParams;
  seedWeights: ColdStartRouterLivePolicySeedWeightV1[];
  semanticClassSeedWeights?: ColdStartRouterLivePolicySemanticClassSeedWeightV1[];
  stopLocalWeights: ColdStartRouterLivePolicyStopWeightV1[];
  edgeWeights: ColdStartRouterLivePolicyEdgeWeightV1[];
  semanticClassEdgeWeights?: ColdStartRouterLivePolicySemanticClassEdgeWeightV1[];
  toolActionPriors: ColdStartRouterToolActionPriorV1[];
  toolActionSets: ColdStartRouterToolActionSetV1[];
  skippedToolRowIds: string[];
  usedRowCount: number;
  traverseRowCount: number;
  toolRowCount: number;
}

export interface ColdStartRouterLivePolicyMaterializationV1 {
  graph: BrainGraph;
  policyParams: PolicyParams;
  sourceNodeId: string | null;
}

interface CountBucket {
  positive: number;
  negative: number;
  support: number;
}

const ACTIVATION_FIRST_CONTINUE_ROW_BONUS = 0.25;
const ACTIVATION_FIRST_HARD_NEGATIVE_BONUS = 0.15;
const MAX_TRAINING_ROW_WEIGHT = 2.5;
const RESUME_GATE_REPLAY_EVENT_CONTEXT_FALLBACK_EDGE_FLOOR = 0.4;
const RESUME_GATE_REPLAY_EVENT_CONTEXT_FALLBACK_EDGE_EXPERIMENT_V1 =
  "resume_gate_replay_event_context_fallback_edge_floor.v1";

function createCountBucket(): CountBucket {
  return { positive: 0, negative: 0, support: 0 };
}

function bumpCountBucket(bucket: CountBucket, isPositive: boolean, weight: number): void {
  if (isPositive) {
    bucket.positive += weight;
  } else {
    bucket.negative += weight;
  }
  bucket.support += weight;
}

function clampFinite(value: number, lower: number, upper: number): number {
  if (!Number.isFinite(value)) {
    return lower;
  }
  return Math.min(upper, Math.max(lower, value));
}

function activationFirstRowBonus(
  row: Pick<RouteDecisionRowV1, "outcome_gain" | "teacher_action" | "stop_label" | "hard_negatives">,
): number {
  if (row.teacher_action.kind !== "traverse" || row.stop_label !== "CONTINUE" || Number(row.outcome_gain) <= 0) {
    return 0;
  }
  return ACTIVATION_FIRST_CONTINUE_ROW_BONUS
    + (row.hard_negatives.length > 0 ? ACTIVATION_FIRST_HARD_NEGATIVE_BONUS : 0);
}

function toRowWeight(
  row: Pick<RouteDecisionRowV1, "outcome_gain" | "teacher_action" | "stop_label" | "hard_negatives">,
): number {
  const magnitude = Math.abs(Number(row.outcome_gain));
  if (!Number.isFinite(magnitude) || magnitude === 0) {
    return 0.25;
  }
  return clampFinite(magnitude + activationFirstRowBonus(row), 0.25, MAX_TRAINING_ROW_WEIGHT);
}

function calcLiveWeight(
  positive: number,
  negative: number,
  support: number,
  smoothing: number,
  supportDampening: number,
): number {
  const odds = Math.log((positive + smoothing) / (negative + smoothing));
  const supportScale = support / (support + supportDampening);
  return clampFinite(odds * supportScale, -8, 8);
}

function normalizeText(value: unknown, fallback = ""): string {
  if (typeof value !== "string") {
    return fallback;
  }
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : fallback;
}

function normalizeCursorSourceNodeId(cursorPath: readonly string[]): string {
  const reversed = [...cursorPath].reverse();
  for (const entry of reversed) {
    const normalized = normalizeText(entry);
    if (normalized.length > 0) {
      return normalized;
    }
  }
  return START_NODE_ID;
}

export function normalizeLivePolicySourceBindingKeyV1(sourceNodeId: string): string {
  const normalized = normalizeText(sourceNodeId, START_NODE_ID);
  if (normalized === "felt_resume_25" || normalized === "recorded_session_replay") {
    return "resume_replay_context";
  }
  return normalized;
}

function candidateTrust(candidate: RouteCandidateV1): TrustLevel {
  switch (normalizeText(candidate.authority, "").toLowerCase()) {
    case "human":
      return "human";
    case "runtime":
    case "docs":
    case "policy":
    case "operator_policy":
      return "teacher";
    case "self":
      return "self";
    case "scanner":
      return "scanner";
    default:
      return "scanner";
  }
}

function candidateNodeKind(candidate: RouteCandidateV1): NodeKind {
  switch (candidate.candidate_type) {
    case "tool":
      return "toolcard";
    case "graph_node":
      return "summary_bridge";
    case "trace":
      return "episode_anchor";
    case "issue":
    case "pr":
    case "file":
    case "repo_node":
      return "workflow";
    case "memory_node":
    case "doc_chunk":
    default:
      return "chunk";
  }
}

function ensureCandidateNode(graph: BrainGraph, candidate: RouteCandidateV1): void {
  if (graph.getNode(candidate.candidate_id)) {
    return;
  }

  const now = Date.now();
  graph.addNode({
    id: candidate.candidate_id,
    kind: candidateNodeKind(candidate),
    content: candidate.candidate_id,
    embedding: null,
    sourceUri: null,
    trust: candidateTrust(candidate),
    tags: [
      `candidate_type:${candidate.candidate_type}`,
      ...(candidate.candidate_type === "tool" ? ["action_kind:tool"] : []),
      ...(candidate.semantic_class ? [`semantic_class:${candidate.semantic_class}`] : []),
      ...(candidate.authority ? [`authority:${candidate.authority}`] : []),
      ...(candidate.freshness ? [`freshness:${candidate.freshness}`] : []),
    ],
    tokenCount: Number.isFinite(Number(candidate.token_cost ?? 0)) && Number(candidate.token_cost ?? 0) > 0
      ? Number(candidate.token_cost)
      : 0,
    metadata: {
      candidate_type: candidate.candidate_type,
      action_kind: candidate.candidate_type === "tool" ? "tool" : "traverse",
      semantic_class: candidate.semantic_class ?? null,
      authority: candidate.authority ?? null,
      freshness: candidate.freshness ?? null,
      score_hint: candidate.score_hint ?? null,
      token_cost: candidate.token_cost ?? null,
    },
    createdAt: now,
    updatedAt: now,
  });
}

function ensureSourceNode(graph: BrainGraph, sourceNodeId: string): void {
  if (sourceNodeId === START_NODE_ID || graph.getNode(sourceNodeId)) {
    return;
  }

  const now = Date.now();
  graph.addNode({
    id: sourceNodeId,
    kind: "workflow",
    content: sourceNodeId,
    embedding: null,
    sourceUri: null,
    trust: "scanner",
    tags: ["source_context"],
    tokenCount: 0,
    metadata: {},
    createdAt: now,
    updatedAt: now,
  });
}

function parsePositiveCandidateIds(row: RouteDecisionRowV1): Set<string> {
  const teacherAction = row.teacher_action;
  if (teacherAction.kind === "traverse") {
    return new Set(teacherAction.target_ids.map((targetId: string) => targetId.trim()).filter((targetId: string) => targetId.length > 0));
  }

  const toolTeacherAction = teacherAction.kind === "tool" ? teacherAction : null;
  const explicitToolMatch = row.candidate_set
    .filter((candidate: RouteCandidateV1) => candidate.candidate_type === "tool" && candidate.candidate_id === toolTeacherAction?.tool_name)
    .map((candidate: RouteCandidateV1) => candidate.candidate_id);
  if (explicitToolMatch.length > 0) {
    return new Set(explicitToolMatch);
  }

  const toolCandidates = row.candidate_set
    .filter((candidate: RouteCandidateV1) => candidate.candidate_type === "tool")
    .map((candidate: RouteCandidateV1) => candidate.candidate_id);
  return toolCandidates.length === 1 ? new Set(toolCandidates) : new Set();
}

function sortedCountEntries(counts: Map<string, CountBucket>): Array<{ key: string; bucket: CountBucket }> {
  return [...counts.entries()]
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([key, bucket]) => ({ key, bucket }));
}

function countBucketFromEntry(entry: {
  positive: number;
  negative: number;
  support: number;
}): CountBucket {
  return {
    positive: entry.positive,
    negative: entry.negative,
    support: entry.support,
  };
}

function seedCountMapFromEntries<T extends {
  positive: number;
  negative: number;
  support: number;
}>(entries: readonly T[], getKey: (entry: T) => string): Map<string, CountBucket> {
  const counts = new Map<string, CountBucket>();
  for (const entry of entries) {
    counts.set(getKey(entry), countBucketFromEntry(entry));
  }
  return counts;
}

function seedToolActionCountsFromInitializer(
  initializer: ColdStartRouterLivePolicyInitializerV1 | undefined,
): Map<string, Map<string, CountBucket>> {
  const counts = new Map<string, Map<string, CountBucket>>();
  for (const entry of initializer?.toolActionPriors ?? []) {
    const toolMap = counts.get(entry.sourceNodeId) ?? new Map<string, CountBucket>();
    toolMap.set(entry.toolNodeId, countBucketFromEntry(entry));
    counts.set(entry.sourceNodeId, toolMap);
  }
  return counts;
}

function seedToolActionSetsFromInitializer(
  initializer: ColdStartRouterLivePolicyInitializerV1 | undefined,
): Map<string, {
  rowIds: Set<string>;
  teacherToolNodeIds: Set<string>;
  candidateIds: Set<string>;
  candidates: Map<string, RouteCandidateV1>;
  support: number;
}> {
  const sets = new Map<string, {
    rowIds: Set<string>;
    teacherToolNodeIds: Set<string>;
    candidateIds: Set<string>;
    candidates: Map<string, RouteCandidateV1>;
    support: number;
  }>();

  for (const entry of initializer?.toolActionSets ?? []) {
    sets.set(entry.sourceNodeId, {
      rowIds: new Set(entry.rowIds),
      teacherToolNodeIds: new Set(entry.teacherToolNodeIds),
      candidateIds: new Set(entry.candidateIds),
      candidates: new Map(entry.candidates.map((candidate) => [candidate.candidate_id, cloneRouteCandidate(candidate)])),
      support: entry.support,
    });
  }

  return sets;
}

function seedGlobalStopCounts(
  stopLabelCounts: Record<ColdStartStopLabelV1, number> | undefined,
): CountBucket {
  if (!stopLabelCounts) {
    return createCountBucket();
  }
  const positive = (stopLabelCounts.STOP_LOCAL ?? 0) + (stopLabelCounts.STOP ?? 0);
  const negative = stopLabelCounts.CONTINUE ?? 0;
  return {
    positive,
    negative,
    support: positive + negative,
  };
}

function buildPolicyParams(params: {
  stopPositive: number;
  stopNegative: number;
  stopSupport: number;
  basePolicyParams?: PolicyParams;
}): PolicyParams {
  return {
    ...(params.basePolicyParams ?? DEFAULT_POLICY_PARAMS),
    stopBias: calcLiveWeight(params.stopPositive, params.stopNegative, params.stopSupport, 1, 2),
  };
}

function sortRouteCandidates(candidates: readonly RouteCandidateV1[]): RouteCandidateV1[] {
  return [...candidates].sort((left, right) => left.candidate_id.localeCompare(right.candidate_id));
}

function cloneRouteCandidate(candidate: RouteCandidateV1): RouteCandidateV1 {
  return {
    candidate_id: candidate.candidate_id,
    candidate_type: candidate.candidate_type,
    ...(candidate.semantic_class ? { semantic_class: candidate.semantic_class } : {}),
    ...(candidate.authority ? { authority: candidate.authority } : {}),
    ...(candidate.freshness ? { freshness: candidate.freshness } : {}),
    ...(candidate.token_cost !== undefined ? { token_cost: candidate.token_cost } : {}),
    ...(candidate.score_hint !== undefined ? { score_hint: candidate.score_hint } : {}),
  };
}

function isReplayLikeLivePriorFallbackCandidate(candidate: RouteCandidateV1): boolean {
  const authority = normalizeText(candidate.authority, "").toLowerCase();
  const freshness = normalizeText(candidate.freshness, "").toLowerCase();
  return authority === "recorded_session_replay" || freshness === "replay_eval";
}

function isReplayPreferredFallbackSemanticClass(semanticClass: string): boolean {
  return semanticClass === "feedback_context" || semanticClass === "event_context";
}

function resolveSemanticFallbackEdgeMaterialization(params: {
  candidate: RouteCandidateV1;
  entry: ColdStartRouterLivePolicySemanticClassEdgeWeightV1;
  sourceBindingKey: string;
  applyResumeGateReplaySemanticFallbackBoost: boolean;
}): { weight: number; metadata: Record<string, unknown> } {
  if (
    !params.applyResumeGateReplaySemanticFallbackBoost
    || params.sourceBindingKey !== "resume_replay_context"
    || !isReplayPreferredFallbackSemanticClass(normalizeText(params.candidate.semantic_class, ""))
    || !isReplayLikeLivePriorFallbackCandidate(params.candidate)
  ) {
    return {
      weight: params.entry.weight,
      metadata: {},
    };
  }

  const adjustedWeight = Math.max(
    params.entry.weight,
    RESUME_GATE_REPLAY_EVENT_CONTEXT_FALLBACK_EDGE_FLOOR,
  );
  if (Math.abs(adjustedWeight - params.entry.weight) < 1e-12) {
    return {
      weight: params.entry.weight,
      metadata: {},
    };
  }

  return {
    weight: adjustedWeight,
    metadata: {
      fallbackExperiment: RESUME_GATE_REPLAY_EVENT_CONTEXT_FALLBACK_EDGE_EXPERIMENT_V1,
      fallbackBaseWeight: params.entry.weight,
      fallbackAdjustedWeight: adjustedWeight,
      fallbackAppliedBoost: adjustedWeight - params.entry.weight,
    },
  };
}

export function buildColdStartRouterLivePolicyInitializerV1(params: {
  routeRows: RouteDecisionRowV1[];
  warmStartInitializer?: ColdStartRouterLivePolicyInitializerV1;
  warmStartStopLabelCounts?: Record<ColdStartStopLabelV1, number>;
}): ColdStartRouterLivePolicyInitializerV1 {
  const seedCounts = seedCountMapFromEntries(params.warmStartInitializer?.seedWeights ?? [], (entry) => entry.nodeId);
  const semanticClassSeedCounts = seedCountMapFromEntries(
    params.warmStartInitializer?.semanticClassSeedWeights ?? [],
    (entry) => entry.semanticClass,
  );
  const stopCounts = seedCountMapFromEntries(params.warmStartInitializer?.stopLocalWeights ?? [], (entry) => entry.sourceNodeId);
  const edgeCounts = seedCountMapFromEntries(
    params.warmStartInitializer?.edgeWeights ?? [],
    (entry) => `${entry.sourceNodeId}→${entry.targetNodeId}`,
  );
  const semanticClassEdgeCounts = seedCountMapFromEntries(
    params.warmStartInitializer?.semanticClassEdgeWeights ?? [],
    (entry) => `${entry.sourceBindingKey}→${entry.targetSemanticClass}`,
  );
  const toolActionCounts = seedToolActionCountsFromInitializer(params.warmStartInitializer);
  const toolActionSets = seedToolActionSetsFromInitializer(params.warmStartInitializer);
  const globalStopCounts = seedGlobalStopCounts(params.warmStartStopLabelCounts);
  const skippedToolRowIds = [...(params.warmStartInitializer?.skippedToolRowIds ?? [])];
  const priorUsedRowCount = params.warmStartInitializer?.usedRowCount ?? 0;
  let traverseRowCount = params.warmStartInitializer?.traverseRowCount ?? 0;
  let toolRowCount = params.warmStartInitializer?.toolRowCount ?? 0;

  for (const row of params.routeRows) {
    const rowWeight = toRowWeight(row);
    const sourceNodeId = normalizeCursorSourceNodeId(row.cursor_path);
    const sourceBindingKey = normalizeLivePolicySourceBindingKeyV1(sourceNodeId);
    const positiveCandidateIds = parsePositiveCandidateIds(row);
    const toolCandidates = row.candidate_set.filter((candidate) => candidate.candidate_type === "tool");
    const teacherToolNodeIds = row.teacher_action.kind === "tool"
      ? new Set<string>(positiveCandidateIds)
      : new Set<string>();

    if (toolCandidates.length > 0) {
      const setEntry = toolActionSets.get(sourceNodeId) ?? {
        rowIds: new Set<string>(),
        teacherToolNodeIds: new Set<string>(),
        candidateIds: new Set<string>(),
        candidates: new Map<string, RouteCandidateV1>(),
        support: 0,
      };
      setEntry.rowIds.add(row.row_id);
      setEntry.support += rowWeight;
      for (const candidate of toolCandidates) {
        setEntry.candidateIds.add(candidate.candidate_id);
        if (teacherToolNodeIds.has(candidate.candidate_id)) {
          setEntry.teacherToolNodeIds.add(candidate.candidate_id);
        }
        const existing = setEntry.candidates.get(candidate.candidate_id);
        if (!existing || (candidate.score_hint ?? Number.NEGATIVE_INFINITY) > (existing.score_hint ?? Number.NEGATIVE_INFINITY)) {
          setEntry.candidates.set(candidate.candidate_id, cloneRouteCandidate(candidate));
        }

        const sourceToolMap = toolActionCounts.get(sourceNodeId) ?? new Map<string, CountBucket>();
        const bucket = sourceToolMap.get(candidate.candidate_id) ?? createCountBucket();
        bumpCountBucket(bucket, teacherToolNodeIds.has(candidate.candidate_id), rowWeight);
        sourceToolMap.set(candidate.candidate_id, bucket);
        toolActionCounts.set(sourceNodeId, sourceToolMap);
      }
      toolActionSets.set(sourceNodeId, setEntry);
    }

    for (const candidate of row.candidate_set) {
      const bucket = seedCounts.get(candidate.candidate_id) ?? createCountBucket();
      bumpCountBucket(bucket, positiveCandidateIds.has(candidate.candidate_id), rowWeight);
      seedCounts.set(candidate.candidate_id, bucket);

      const semanticClass = normalizeText(candidate.semantic_class, "");
      if (semanticClass.length > 0) {
        const semanticBucket = semanticClassSeedCounts.get(semanticClass) ?? createCountBucket();
        bumpCountBucket(semanticBucket, positiveCandidateIds.has(candidate.candidate_id), rowWeight);
        semanticClassSeedCounts.set(semanticClass, semanticBucket);
      }
    }

    const stopBucket = stopCounts.get(sourceNodeId) ?? createCountBucket();
    const stopPositive = row.stop_label === "STOP_LOCAL" || row.stop_label === "STOP";
    bumpCountBucket(stopBucket, stopPositive, rowWeight);
    stopCounts.set(sourceNodeId, stopBucket);
    bumpCountBucket(globalStopCounts, stopPositive, rowWeight);

    if (positiveCandidateIds.size > 0) {
      for (const candidate of row.candidate_set) {
        const bucketKey = `${sourceNodeId}→${candidate.candidate_id}`;
        const edgeBucket = edgeCounts.get(bucketKey) ?? createCountBucket();
        bumpCountBucket(edgeBucket, positiveCandidateIds.has(candidate.candidate_id), rowWeight);
        edgeCounts.set(bucketKey, edgeBucket);

        const semanticClass = normalizeText(candidate.semantic_class, "");
        if (semanticClass.length > 0) {
          const semanticEdgeKey = `${sourceBindingKey}→${semanticClass}`;
          const semanticEdgeBucket = semanticClassEdgeCounts.get(semanticEdgeKey) ?? createCountBucket();
          bumpCountBucket(semanticEdgeBucket, positiveCandidateIds.has(candidate.candidate_id), rowWeight);
          semanticClassEdgeCounts.set(semanticEdgeKey, semanticEdgeBucket);
        }
      }
    }

    if (row.teacher_action.kind === "traverse") {
      traverseRowCount += 1;
    } else {
      toolRowCount += 1;
      skippedToolRowIds.push(row.row_id);
    }
  }

  const policyParams = buildPolicyParams({
    stopPositive: globalStopCounts.positive,
    stopNegative: globalStopCounts.negative,
    stopSupport: globalStopCounts.support,
    basePolicyParams: params.warmStartInitializer?.policyParams,
  });

  const toolActionPriors = [...toolActionCounts.entries()]
    .flatMap(([sourceNodeId, toolMap]) => [...toolMap.entries()].map(([toolNodeId, bucket]) => ({
      sourceNodeId,
      toolNodeId,
      positive: bucket.positive,
      negative: bucket.negative,
      support: bucket.support,
      prior: bucket.support > 0 ? bucket.positive / bucket.support : 0,
      weight: calcLiveWeight(bucket.positive, bucket.negative, bucket.support, 1, 2),
    })))
    .sort((left, right) => {
      const bySource = left.sourceNodeId.localeCompare(right.sourceNodeId);
      if (bySource !== 0) {
        return bySource;
      }
      return left.toolNodeId.localeCompare(right.toolNodeId);
    });

  const toolActionSetEntries = [...toolActionSets.entries()]
    .map(([sourceNodeId, setEntry]) => ({
      sourceNodeId,
      rowIds: [...setEntry.rowIds].sort(),
      teacherToolNodeIds: [...setEntry.teacherToolNodeIds].sort(),
      candidateIds: [...setEntry.candidateIds].sort(),
      candidates: sortRouteCandidates([...setEntry.candidates.values()].map(cloneRouteCandidate)),
      support: setEntry.support,
    }))
    .sort((left, right) => left.sourceNodeId.localeCompare(right.sourceNodeId));

  return {
    contract: COLD_START_ROUTER_LIVE_POLICY_INITIALIZER_CONTRACT_V1,
    policyParams,
    seedWeights: sortedCountEntries(seedCounts).map(({ key, bucket }) => ({
      nodeId: key,
      positive: bucket.positive,
      negative: bucket.negative,
      support: bucket.support,
      weight: calcLiveWeight(bucket.positive, bucket.negative, bucket.support, 1, 2),
    })),
    semanticClassSeedWeights: sortedCountEntries(semanticClassSeedCounts).map(({ key, bucket }) => ({
      semanticClass: key,
      positive: bucket.positive,
      negative: bucket.negative,
      support: bucket.support,
      weight: calcLiveWeight(bucket.positive, bucket.negative, bucket.support, 1, 2),
    })),
    stopLocalWeights: sortedCountEntries(stopCounts).map(({ key, bucket }) => ({
      sourceNodeId: key,
      positive: bucket.positive,
      negative: bucket.negative,
      support: bucket.support,
      weight: calcLiveWeight(bucket.positive, bucket.negative, bucket.support, 1, 2) - policyParams.stopBias,
    })),
    edgeWeights: sortedCountEntries(edgeCounts).map(({ key, bucket }) => {
      const [sourceNodeId, targetNodeId] = key.split("→", 2);
      return {
        sourceNodeId: sourceNodeId ?? START_NODE_ID,
        targetNodeId: targetNodeId ?? "",
        positive: bucket.positive,
        negative: bucket.negative,
        support: bucket.support,
        prior: 1,
        weight: calcLiveWeight(bucket.positive, bucket.negative, bucket.support, 1, 2),
      };
    }),
    semanticClassEdgeWeights: sortedCountEntries(semanticClassEdgeCounts).map(({ key, bucket }) => {
      const [sourceBindingKey, targetSemanticClass] = key.split("→", 2);
      return {
        sourceBindingKey: sourceBindingKey ?? START_NODE_ID,
        targetSemanticClass: targetSemanticClass ?? "",
        positive: bucket.positive,
        negative: bucket.negative,
        support: bucket.support,
        prior: 1,
        weight: calcLiveWeight(bucket.positive, bucket.negative, bucket.support, 1, 2),
      };
    }),
    toolActionPriors,
    toolActionSets: toolActionSetEntries,
    skippedToolRowIds,
    usedRowCount: priorUsedRowCount + params.routeRows.length,
    traverseRowCount,
    toolRowCount,
  };
}

export function materializeColdStartRouterLivePolicyGraphV1(params: {
  initializer: ColdStartRouterLivePolicyInitializerV1;
  row: RouteDecisionRowV1;
  applyResumeGateReplaySemanticFallbackBoost?: boolean;
}): ColdStartRouterLivePolicyMaterializationV1 {
  const graph = new BrainGraph();
  const sourceNodeIdRaw = normalizeCursorSourceNodeId(params.row.cursor_path);
  const sourceBindingKey = normalizeLivePolicySourceBindingKeyV1(sourceNodeIdRaw);
  const sourceNodeId = sourceNodeIdRaw === START_NODE_ID ? null : sourceNodeIdRaw;

  for (const candidate of params.row.candidate_set) {
    ensureCandidateNode(graph, candidate);
  }
  if (sourceNodeId) {
    ensureSourceNode(graph, sourceNodeId);
  }

  for (const toolActionSet of params.initializer.toolActionSets) {
    if (toolActionSet.sourceNodeId !== sourceNodeIdRaw) {
      continue;
    }
    for (const candidate of toolActionSet.candidates) {
      ensureCandidateNode(graph, candidate);
    }
  }

  for (const entry of params.initializer.seedWeights) {
    if (graph.getNode(entry.nodeId)) {
      graph.setSeedWeight(entry.nodeId, entry.weight);
    }
  }
  const semanticClassSeedWeightMap = new Map(
    (params.initializer.semanticClassSeedWeights ?? []).map((entry) => [entry.semanticClass, entry.weight]),
  );
  for (const candidate of params.row.candidate_set) {
    if (graph.getSeedWeight(candidate.candidate_id) !== 0) {
      continue;
    }
    const semanticClass = normalizeText(candidate.semantic_class, "");
    if (semanticClass.length === 0) {
      continue;
    }
    const semanticWeight = semanticClassSeedWeightMap.get(semanticClass);
    if (semanticWeight === undefined) {
      continue;
    }
    graph.setSeedWeight(candidate.candidate_id, semanticWeight);
  }
  for (const entry of params.initializer.stopLocalWeights) {
    graph.setStopLocalWeight(entry.sourceNodeId === START_NODE_ID ? null : entry.sourceNodeId, entry.weight);
  }
  for (const entry of params.initializer.edgeWeights) {
    if (!graph.getNode(entry.sourceNodeId) || !graph.getNode(entry.targetNodeId)) {
      continue;
    }
    graph.addEdge({
      source: entry.sourceNodeId,
      target: entry.targetNodeId,
      kind: "learned",
      weight: entry.weight,
      prior: entry.prior,
      metadata: {
        positive: entry.positive,
        negative: entry.negative,
        support: entry.support,
      },
      decayedAt: 0,
      createdAt: Date.now(),
    });
  }
  const semanticClassEdgeWeightMap = new Map(
    (params.initializer.semanticClassEdgeWeights ?? []).map((entry) => [`${entry.sourceBindingKey}→${entry.targetSemanticClass}`, entry]),
  );
  if (graph.getNode(sourceNodeIdRaw)) {
    for (const candidate of params.row.candidate_set) {
      if (graph.getEdge(sourceNodeIdRaw, candidate.candidate_id)) {
        continue;
      }
      const semanticClass = normalizeText(candidate.semantic_class, "");
      if (semanticClass.length === 0) {
        continue;
      }
      const entry = semanticClassEdgeWeightMap.get(`${sourceBindingKey}→${semanticClass}`);
      if (!entry) {
        continue;
      }
      const semanticFallbackEdge = resolveSemanticFallbackEdgeMaterialization({
        candidate,
        entry,
        sourceBindingKey,
        applyResumeGateReplaySemanticFallbackBoost: params.applyResumeGateReplaySemanticFallbackBoost ?? false,
      });
      graph.addEdge({
        source: sourceNodeIdRaw,
        target: candidate.candidate_id,
        kind: "learned",
        weight: semanticFallbackEdge.weight,
        prior: entry.prior,
        metadata: {
          positive: entry.positive,
          negative: entry.negative,
          support: entry.support,
          sourceBindingKey: entry.sourceBindingKey,
          targetSemanticClass: entry.targetSemanticClass,
          ...semanticFallbackEdge.metadata,
        },
        decayedAt: 0,
        createdAt: Date.now(),
      });
    }
  }

  for (const entry of params.initializer.toolActionPriors) {
    if (entry.sourceNodeId !== sourceNodeIdRaw) {
      continue;
    }
    graph.setToolActionPrior(entry.sourceNodeId === START_NODE_ID ? null : entry.sourceNodeId, entry.toolNodeId, entry.weight);
  }

  return {
    graph,
    policyParams: params.initializer.policyParams,
    sourceNodeId,
  };
}
