/**
 * Core type definitions for the OpenClawBrain learning layer.
 *
 * These types define the learned retrieval graph, traversal MDP,
 * episodes, labels, packs, mutations, and traces.
 *
 * Paper reference: Gu (2016), "Reinforcement Learning"
 * Lemma 6.1: ∂/∂ρ v_ρ(s_t) = E[z · Σ_{l=t}^{T} ∂logP_ρ(a_l|s_l)/∂ρ]
 */

// ═══════════════════════════════════════════
// Node & Edge Types
// ═══════════════════════════════════════════

export type NodeKind =
  | "chunk"            // Document/code fragment
  | "workflow"         // Multi-step procedure
  | "correction"       // Human-authored fix ("use X not Y")
  | "toolcard"         // When/how to use a specific tool
  | "episode_anchor"   // Pointer to a prior successful episode
  | "summary_bridge";  // Bridge to LCM summary DAG

export type EdgeKind =
  | "sibling"          // Same-document adjacency (prior = 1.0)
  | "semantic"         // Embedding cosine similarity (prior = cosine)
  | "learned"          // Created by learning (prior = 0.5)
  | "inhibitory"       // Suppresses traversal (weight < 0)
  | "bridge";          // Links brain node to LCM summary

export type TrustLevel = "human" | "scanner" | "teacher" | "self";

export type RewardSource = TrustLevel;

export interface BrainNode {
  id: string;
  kind: NodeKind;
  content: string;
  embedding: Float32Array | null;
  sourceUri: string | null;
  trust: TrustLevel;
  tags: string[];
  tokenCount: number;
  metadata: Record<string, unknown>;
  createdAt: number;
  updatedAt: number;
}

export interface BrainEdge {
  source: string;
  target: string;
  kind: EdgeKind;
  weight: number;       // Learned parameter ρ (signed; negative = suppress)
  prior: number;        // Immutable structural baseline
  metadata: Record<string, unknown>;
  decayedAt: number;
  createdAt: number;
}

// ═══════════════════════════════════════════
// Traversal MDP (paper-faithful)
// ═══════════════════════════════════════════

/**
 * State s_t in the MDP.
 * Paper: S = {s_0, s_1, s_2, ...}
 */
export interface TraversalState {
  currentNodeId: string | null;   // null at seed phase (t=0)
  queryEmbedding: Float32Array;
  visited: Set<string>;
  fired: string[];
  budgetRemaining: number;
  hopCount: number;
  maxHops: number;
}

/**
 * Action a_t in the MDP.
 * Paper: A(s) ⊂ {a_0, a_1, a_2, ...}
 * Our action set: A(s) = { traverse(neighbor) } ∪ { STOP }
 */
export type TraversalAction =
  | { type: "traverse"; targetNodeId: string }
  | { type: "stop" };

/**
 * One step of a recorded trajectory.
 * Captures the full candidate distribution for REINFORCE gradient computation.
 */
export interface TrajectoryStep {
  stateSnapshot: {
    currentNodeId: string | null;
    hopCount: number;
    budgetRemaining: number;
    visitedCount: number;
    firedCount: number;
  };
  candidates: Array<{
    action: TraversalAction;
    score: number;
    probability: number;
  }>;
  chosenAction: TraversalAction;
  chosenActionProbability: number;
  stopProbability: number;
}

// ═══════════════════════════════════════════
// Episodes & Labels
// ═══════════════════════════════════════════

/**
 * A complete episode: one traversal from seed to terminal state.
 * Paper: A game that ends in finite time with terminal reward z.
 */
export interface Episode {
  id: string;
  conversationId: number | null;
  queryText: string;
  queryEmbedding: Float32Array | null;
  trajectory: TrajectoryStep[];
  firedNodes: string[];
  vetoedNodes: string[];
  contextChars: number;
  reward: number | null;          // Terminal reward z ∈ [-1, +1]
  rewardSource: RewardSource | null;
  packVersion: number | null;
  createdAt: number;
}

/**
 * A reward label from one of four sources.
 * Human > self > scanner > teacher by trust ranking.
 */
export interface Label {
  id: string;
  episodeId: string;
  source: RewardSource;
  value: number;                  // z ∈ [-1, +1]
  confidence: number;             // [0, 1]
  reason: string | null;
  applied: boolean;
  createdAt: number;
}

// ═══════════════════════════════════════════
// Packs & Mutations
// ═══════════════════════════════════════════

export interface Pack {
  version: number;
  nodeCount: number;
  edgeCount: number;
  healthJson: string;
  promotedAt: number | null;
  rolledBack: boolean;
  createdAt: number;
}

export type MutationKind = "split" | "merge" | "prune" | "connect" | "inject";
export type MutationStatus = "pending" | "validated" | "promoted" | "rejected";

export interface MutationProposal {
  id: string;
  kind: MutationKind;
  proposal: unknown;
  evidence: unknown | null;
  expectedGain: number | null;
  status: MutationStatus;
  createdAt: number;
  resolvedAt: number | null;
}

// ═══════════════════════════════════════════
// Health Metrics
// ═══════════════════════════════════════════

export interface HealthMetrics {
  nodeCount: number;
  edgeCount: number;
  nodesByKind: Record<NodeKind, number>;
  edgesByKind: Record<EdgeKind, number>;
  firedPerQuery: number;
  dormantPercent: number;
  inhibitoryPercent: number;
  orphanCount: number;
  avgPathLength: number;
  avgReward: number;
  crossFileEdgePercent: number;
  churn: number;
  packVersion: number;
  lastUpdateAt: number;
  totalEpisodes: number;
}

// ═══════════════════════════════════════════
// Decision Traces
// ═══════════════════════════════════════════

export interface DecisionTrace {
  id: string;
  episodeId: string | null;
  packVersion: number | null;
  queryText: string;
  seedScores: Array<{ nodeId: string; score: number }>;
  trajectory: TrajectoryStep[];
  firedNodes: string[];
  vetoedNodes: string[];
  contextChars: number;
  footer: string;
  createdAt: number;
}

// ═══════════════════════════════════════════
// Configuration
// ═══════════════════════════════════════════

export interface BrainConfig {
  enabled: boolean;
  root: string;
  maxHops: number;
  servingTemperature: number;
  learningTemperature: number;
  budgetFraction: number;
  maxSeeds: number;
  semanticThreshold: number;
  learningRate: number;
  baselineAlpha: number;
  decayRate: number;
  trainerIntervalMs: number;
  teacherEnabled: boolean;
  mutationsEnabled: boolean;
  replayEpisodeCount: number;
  minFiredPerQuery: number;
  maxDormantPercent: number;
  maxOrphanCount: number;
  embeddingProvider: string;
  embeddingModel: string;
  embeddingBaseUrl: string;
  teacherProvider: string;
  teacherModel: string;
}

export const DEFAULT_BRAIN_CONFIG: BrainConfig = {
  enabled: true,
  root: "",
  maxHops: 8,
  servingTemperature: 0.1,
  learningTemperature: 1.0,
  budgetFraction: 0.3,
  maxSeeds: 10,
  semanticThreshold: 0.7,
  learningRate: 0.01,
  baselineAlpha: 0.1,
  decayRate: 0.995,
  trainerIntervalMs: 30_000,
  teacherEnabled: true,
  mutationsEnabled: true,
  replayEpisodeCount: 100,
  minFiredPerQuery: 1.0,
  maxDormantPercent: 0.3,
  maxOrphanCount: 10,
  embeddingProvider: "openai",
  embeddingModel: "",
  embeddingBaseUrl: "",
  teacherProvider: "",
  teacherModel: "",
};

// ═══════════════════════════════════════════
// Traversal Result (returned to assembler)
// ═══════════════════════════════════════════

export interface TraversalResult {
  fired: Array<{ nodeId: string; kind: NodeKind; content: string; tokenCount: number }>;
  vetoed: Array<{ nodeId: string; reason: string }>;
  episode: Episode;
  trace: DecisionTrace;
}

/**
 * The route_fn interface: query + candidate IDs → selected subset.
 */
export type RouteFn = (query: string, candidateIds: string[]) => Promise<string[]>;

// ═══════════════════════════════════════════
// Policy Parameters
// ═══════════════════════════════════════════

export interface PolicyParams {
  temperature: number;
  stopBias: number;
  budgetPressure: number;
  hopPressure: number;
  edgeKindBias: Record<EdgeKind, number>;
}

export const DEFAULT_POLICY_PARAMS: PolicyParams = {
  temperature: 1.0,
  stopBias: -2.0,
  budgetPressure: 3.0,
  hopPressure: 2.0,
  edgeKindBias: {
    sibling: 0.0,
    semantic: 0.1,
    learned: 0.2,
    inhibitory: -10.0,
    bridge: 0.0,
  },
};

// Trust rank ordering: human > self > scanner > teacher
export function trustRank(source: RewardSource): number {
  switch (source) {
    case "human": return 4;
    case "self": return 3;
    case "scanner": return 2;
    case "teacher": return 1;
  }
}
