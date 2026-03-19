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
  | "seed"             // Learned seed-head parameter from __START__
  | "inhibitory"       // Suppresses traversal (weight < 0)
  | "bridge";          // Links brain node to LCM summary

export const START_NODE_ID = "__START__";

export interface SeedWeightUpdate {
  kind: "seed";
  nodeId: string;
  delta: number;
}

export interface EdgeWeightUpdate {
  kind: "edge";
  source: string;
  target: string;
  delta: number;
}

export type PolicyWeightUpdate = SeedWeightUpdate | EdgeWeightUpdate;

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
  | { type: "traverse"; targetNodeId: string; seedScore?: number }
  | { type: "stop" };

export interface SeedScore {
  nodeId: string;
  priorScore: number;
  learnedSeedWeight: number;
  policyScore: number;
  probability: number;
  chosen: boolean;
}

export interface SeedWeight {
  nodeId: string;
  weight: number;
  updatedAt: number;
}

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
    priorScore?: number;
    learnedSeedWeight?: number;
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

export type BrainEvidenceKind =
  | "human_feedback"
  | "self_result"
  | "scanner_signal"
  | "teacher_review"
  | "teach_correction";

export interface BrainEvidence {
  id: string;
  episodeId: string;
  conversationId: number | null;
  source: RewardSource;
  kind: BrainEvidenceKind;
  value: number;
  confidence: number;
  reason: string | null;
  contentSnippet: string | null;
  metadata: Record<string, unknown>;
  resolved: boolean;
  createdAt: number;
}

export type BrainEvidenceResolution =
  | "promoted_to_label"
  | "discarded_missing_episode"
  | "discarded_lower_trust"
  | "discarded_duplicate";

export interface ResolvedLabel {
  id: string;
  evidenceId: string;
  episodeId: string;
  source: RewardSource;
  value: number;
  confidence: number;
  resolution: BrainEvidenceResolution;
  labelId: string | null;
  note: string | null;
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
export type MutationBundleStatus = "pending" | "evaluating" | "promoted" | "rejected";

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

export type ReplayGateReasonCode =
  | "no_episodes_to_replay"
  | "fired_per_query_below_min"
  | "dormant_percent_above_max"
  | "orphan_count_above_max"
  | "human_positive_route_regression"
  | "self_negative_route_unchanged"
  | "all_gates_passed";

export interface ReplayGateReason {
  code: ReplayGateReasonCode;
  summary: string;
  details: Record<string, unknown>;
}

export interface ReplayGateVerdict {
  passed: boolean;
  reason: ReplayGateReason;
  health: HealthMetrics;
  evaluatedEpisodeCount: number;
  humanPositiveEpisodeCount: number;
  selfNegativeEpisodeCount: number;
}

export type BundleEvaluationReasonCode =
  | "promoted"
  | "no_qualifying_episodes"
  | "candidate_regressed"
  | "insufficient_improvement";

export interface BundleEvaluationReason {
  code: BundleEvaluationReasonCode;
  summary: string;
  details: Record<string, unknown>;
}

export interface BundleEvaluationVerdict {
  bundleId: string;
  mutationIds: string[];
  bundleSize: number;
  status: Extract<MutationBundleStatus, "promoted" | "rejected">;
  baseScore: number;
  candidateScore: number;
  expectedGain: number;
  evaluatedEpisodeCount: number;
  qualifyingEpisodeCount: number;
  improvementRatio: number | null;
  reason: BundleEvaluationReason;
  createdAt: number;
  resolvedAt: number;
}

export interface MutationBundleRecord {
  id: string;
  mutationIds: string[];
  bundleSize: number;
  status: MutationBundleStatus;
  baseScore: number | null;
  candidateScore: number | null;
  expectedGain: number;
  rejectionReason: string | null;
  verdict: BundleEvaluationVerdict | null;
  createdAt: number;
  resolvedAt: number | null;
}

export interface PromotionRunVerdict {
  mode: "bundle" | "legacy";
  status: "promoted" | "rejected";
  summary: string;
  mutationCount: number;
  promotedMutationCount: number;
  rejectedMutationCount: number;
  bundleCount: number;
  promotedBundleCount: number;
  rejectedBundleCount: number;
  packPromotionTriggered: boolean;
  health: HealthMetrics | null;
  replayGate: ReplayGateVerdict | null;
  bundleVerdicts: BundleEvaluationVerdict[];
  createdAt: number;
}

export type LearningJournalEventType =
  | "mutation_proposed"
  | "bundle_evaluation_started"
  | "bundle_evaluation_completed"
  | "promotion_accepted"
  | "promotion_rejected";

export interface BundleEvaluationConfigSnapshot {
  minBundleSize: number;
  maxBundleSize: number;
  minRewardThreshold: number;
  maxContextInflation: number;
  minImprovementRatio: number;
}

export interface MutationProposedJournalPayload {
  mutationKind: MutationKind;
  expectedGain: number | null;
  proposal: unknown;
  evidence: unknown | null;
}

export interface BundleEvaluationStartedJournalPayload {
  mutationKinds: MutationKind[];
  bundleSize: number;
  expectedGain: number;
  candidateMutationCount: number;
  recentEpisodeIds: string[];
  config: BundleEvaluationConfigSnapshot;
}

export interface BundleEvaluationCompletedJournalPayload {
  mutationKinds: MutationKind[];
  bundleSize: number;
  expectedGain: number;
  qualifyingEpisodeIds: string[];
  baseScore: number;
  candidateScore: number;
  shouldPromote: boolean;
  rejectionReason: string | null;
}

export interface PromotionJournalPayload {
  gate: "bundle_evaluation" | "replay_gate";
  mutationKinds: MutationKind[];
  mutationCount: number;
  reason: string | null;
  baseScore: number | null;
  candidateScore: number | null;
  metadata: Record<string, unknown>;
}

export interface LearningJournalRecordBase<TEvent extends LearningJournalEventType, TPayload> {
  id: string;
  eventType: TEvent;
  mutationId: string | null;
  mutationIds: string[];
  bundleId: string | null;
  packVersion: number | null;
  payload: TPayload;
  createdAt: number;
}

export type MutationProposedJournalRecord = LearningJournalRecordBase<"mutation_proposed", MutationProposedJournalPayload>;
export type BundleEvaluationStartedJournalRecord = LearningJournalRecordBase<"bundle_evaluation_started", BundleEvaluationStartedJournalPayload>;
export type BundleEvaluationCompletedJournalRecord = LearningJournalRecordBase<"bundle_evaluation_completed", BundleEvaluationCompletedJournalPayload>;
export type PromotionAcceptedJournalRecord = LearningJournalRecordBase<"promotion_accepted", PromotionJournalPayload>;
export type PromotionRejectedJournalRecord = LearningJournalRecordBase<"promotion_rejected", PromotionJournalPayload>;

export type LearningJournalRecord =
  | MutationProposedJournalRecord
  | BundleEvaluationStartedJournalRecord
  | BundleEvaluationCompletedJournalRecord
  | PromotionAcceptedJournalRecord
  | PromotionRejectedJournalRecord;

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
  seedScores: SeedScore[];
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
  workerMode: "child" | "in_process";
  workerHeartbeatTimeoutMs: number;
  workerRestartDelayMs: number;
  teacherEnabled: boolean;
  autoUserCorrectionsEnabled: boolean;
  autoUserCorrectionsProvider: string;
  autoUserCorrectionsModel: string;
  autoUserCorrectionsMinConfidence: number;
  mutationsEnabled: boolean;
  replayEpisodeCount: number;
  minFiredPerQuery: number;
  maxDormantPercent: number;
  maxOrphanCount: number;
  shadowMode: boolean;
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
  workerMode: "child",
  workerHeartbeatTimeoutMs: 90_000,
  workerRestartDelayMs: 5_000,
  teacherEnabled: true,
  autoUserCorrectionsEnabled: false,
  autoUserCorrectionsProvider: "",
  autoUserCorrectionsModel: "",
  autoUserCorrectionsMinConfidence: 0.8,
  mutationsEnabled: true,
  replayEpisodeCount: 100,
  minFiredPerQuery: 1.0,
  maxDormantPercent: 0.3,
  maxOrphanCount: 10,
  shadowMode: false,
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
    seed: 0.15,
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
