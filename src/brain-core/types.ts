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

export interface StopLocalWeightUpdate {
  kind: "stop_local";
  sourceNodeId: string;
  delta: number;
}

export interface EdgeWeightUpdate {
  kind: "edge";
  source: string;
  target: string;
  delta: number;
}

export type PolicyWeightUpdate = SeedWeightUpdate | StopLocalWeightUpdate | EdgeWeightUpdate;

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
  sourceNodeId: string | null;    // null at seed phase (t=0 / __START__)
  queryEmbedding: Float32Array;
  frontier: string[];
  visited: Set<string>;
  fired: string[];
  budgetRemaining: number;
  initialBudget: number;
  reservedTokenCost: number;
  expansionCount: number;
  maxHops: number;
}

/**
 * Action a_t in the MDP.
 * Paper: A(s) ⊂ {a_0, a_1, a_2, ...}
 * Our action set: A(s) = { traverse(neighbor) } ∪ { stop_local }
 */
export type TraversalAction =
  | { type: "traverse"; targetNodeId: string; seedScore?: number }
  | { type: "stop_local" };

export interface SeedScore {
  nodeId: string;
  priorScore: number;
  learnedSeedWeight: number;
  initialPolicyScore: number;
  initialProbability: number;
  latestPolicyScore: number;
  latestProbability: number;
  selected: boolean;
  selectionSubstepIndex: number | null;
}

export interface SeedWeight {
  nodeId: string;
  weight: number;
  updatedAt: number;
}

export interface StopLocalWeight {
  sourceNodeId: string;
  weight: number;
  updatedAt: number;
}

/**
 * One local selection substep inside a source-node expansion.
 * Captures the full candidate distribution for REINFORCE gradient computation.
 */
export interface TrajectoryCandidate {
  action: TraversalAction;
  score: number;
  probability: number;
  priorScore?: number;
  learnedSeedWeight?: number;
}

export interface TrajectoryStateSnapshot {
  sourceNodeId: string | null;
  expansionIndex: number;
  selectionIndex: number;
  budgetRemaining: number;
  initialBudget: number;
  reservedTokenCost: number;
  maxHops: number;
  frontierSize: number;
  frontierNodeIds: string[];
  visitedCount: number;
  firedCount: number;
}

export interface TrajectorySubstep {
  stateSnapshot: {
    sourceNodeId: string | null;
    expansionIndex: number;
    selectionIndex: number;
    budgetRemaining: number;
    initialBudget: number;
    reservedTokenCost: number;
    maxHops: number;
    frontierSize: number;
    frontierNodeIds: string[];
    visitedCount: number;
    firedCount: number;
  };
  candidates: TrajectoryCandidate[];
  chosenAction: TraversalAction;
  chosenActionProbability: number;
  stopProbability: number;
}

export type TrajectoryStep = TrajectorySubstep;

export interface TrajectoryExpansion {
  sourceNodeId: string | null;
  expansionIndex: number;
  frontierBefore: string[];
  frontierAfter: string[];
  budgetBefore: number;
  budgetAfter: number;
  substeps: TrajectorySubstep[];
  selectedTargets: string[];
  acceptedTargets: string[];
  vetoedTargets: Array<{ targetNodeId: string; reason: string }>;
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
  trajectory: TrajectoryExpansion[];
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

export interface TraceSupervisionRecord {
  id: string;
  traceId: string;
  episodeId: string;
  conversationId: number | null;
  source: RewardSource;
  kind: BrainEvidenceKind;
  value: number;
  confidence: number;
  reason: string | null;
  contentSnippet: string | null;
  resolution: BrainEvidenceResolution;
  labelId: string | null;
  evidenceId: string | null;
  metadata: Record<string, unknown>;
  createdAt: number;
}

export interface PolicyGradientCandidateUpdateArtifact {
  version: 1;
  updateCount: number;
  candidatePackVersion: number;
  currentPackVersion: number | null;
  generatedAt: number;
  episodeIds: string[];
  traceIds: string[];
  supervisionIds: string[];
  teacherTraceIds: string[];
  rewardSources: Record<RewardSource, number>;
  episodeCount: number;
  traceCount: number;
  supervisionCount: number;
  teacherLabelCount: number;
  routeUpdateCount: number;
  seedUpdateCount: number;
  stopLocalUpdateCount?: number;
  edgeUpdateCount: number;
  baselineBefore: number;
  baselineAfter: number;
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

export interface DecisionTraceInjectedNodeSummary {
  nodeId: string;
  kind: NodeKind;
  trust: TrustLevel;
  sourceUri: string | null;
  tags: string[];
  tokenCount: number;
  contentPreview: string;
}

export interface DecisionRouteTrace {
  requestDigest: string;
  conversationId: number | null;
  activePackId: string | null;
  routerIdentity: string;
  candidateNodeIds: string[];
  selectedNodeIds: string[];
  selectedTraversalNodeIds: string[];
  selectedPathNodeIds: string[];
  selectedSeedNodeIds: string[];
  injectedNodeSummaries: DecisionTraceInjectedNodeSummary[];
  sourceSummary: {
    injectedCount: number;
    kinds: Partial<Record<NodeKind, number>>;
    trusts: Partial<Record<TrustLevel, number>>;
    sourceUris: string[];
  };
  selectionMetadata: {
    traceSliceVersion: number;
    queryChars: number;
    budgetChars: number;
    maxHops: number;
    maxFanoutPerNode: number;
    maxFrontierSize: number;
    seedCount: number;
    seedSelectionCount: number;
    candidateCount: number;
    hopCount: number;
    expansionCount: number;
    selectionSubstepCount: number;
    firedCount: number;
    vetoedCount: number;
    chosenSeedNodeId: string | null;
    selectedSeedNodeIds: string[];
    routeSelectionMs: number | null;
    embeddingMs: number | null;
    totalQueryMs: number | null;
    queryEmbeddingSource: "provided" | "runtime";
    // Post-formatting clip attribution is optional so legacy traces remain readable as unknown.
    maxContextChars?: number | null;
    queryBudgetChars?: number | null;
    injectedChars?: number | null;
    droppedChars?: number | null;
    contextClipped?: boolean | null;
  };
}

export interface DecisionTrace {
  id: string;
  episodeId: string | null;
  packVersion: number | null;
  queryText: string;
  seedScores: SeedScore[];
  trajectory: TrajectoryExpansion[];
  firedNodes: string[];
  vetoedNodes: string[];
  contextChars: number;
  footer: string;
  routeTrace?: DecisionRouteTrace | null;
  supervision?: TraceSupervisionRecord[];
  createdAt: number;
}

export interface BrainObservationToolResult {
  sourceRole: "assistant" | "tool";
  toolCallId: string | null;
  toolName: string | null;
  input: string | null;
  output: string | null;
  isError: boolean;
  excerpt: string | null;
}

export interface BrainObservationRouteMetadata {
  requestDigest: string | null;
  activePackId: string | null;
  routerIdentity: string | null;
  candidateNodeIds: string[];
  selectedNodeIds: string[];
  selectedTraversalNodeIds: string[];
  selectedPathNodeIds: string[];
  selectedSeedNodeIds: string[];
  sourceSummary: DecisionRouteTrace["sourceSummary"] | null;
  selectionMetadata: DecisionRouteTrace["selectionMetadata"] | null;
}

export type BrainObservationStatus =
  | "pending_followup"
  | "pending_teacher"
  | "completed";

export interface BrainObservationTeacherEvaluation {
  version: 2;
  observationId: string;
  episodeId: string;
  traceId: string | null;
  retrievalRelevance: number;
  agentUsage: number;
  outcomeSupport: number;
  finalScore: number;
  confidence: number;
  reason: string;
}

export interface BrainObservation {
  id: string;
  episodeId: string;
  conversationId: number | null;
  traceId: string | null;
  queryText: string;
  retrievedContext: DecisionTraceInjectedNodeSummary[];
  routeMetadata: BrainObservationRouteMetadata;
  assistantResponse: string;
  toolResults: BrainObservationToolResult[];
  followUpText: string | null;
  phase1Score: number | null;
  phase2Score: number | null;
  finalScore: number | null;
  confidence: number | null;
  reason: string | null;
  status: BrainObservationStatus;
  teacherEvaluation: BrainObservationTeacherEvaluation | null;
  createdAt: number;
  updatedAt: number;
  evaluatedAt: number | null;
}

// ═══════════════════════════════════════════
// Configuration
// ═══════════════════════════════════════════

export interface BrainConfig {
  enabled: boolean;
  root: string;
  maxHops: number;
  maxFanoutPerNode: number;
  maxFrontierSize: number;
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
  maxFanoutPerNode: 4,
  maxFrontierSize: 32,
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
