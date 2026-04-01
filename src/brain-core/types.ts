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
  pendingNodeIds?: string[];
  maxFrontierSize?: number;
}

/**
 * Action a_t in the MDP.
 * Paper: A(s) ⊂ {a_0, a_1, a_2, ...}
 * Our action set: A(s) = { traverse(neighbor) } ∪ { stop_local }
 */
export type TraversalAction =
  | { type: "traverse"; targetNodeId: string; seedScore?: number }
  | { type: "stop_local" };

export type TrajectoryStopTruth = "chosen" | "forced";

export type TrajectoryStopReason =
  | "policy_stop"
  | "fanout_cap"
  | "frontier_cap"
  | "selection_budget_exhausted"
  | "no_traversable_candidates"
  | "missing_target_node"
  | "interrupt";

export interface TrajectoryProposalOutcome {
  targetNodeId: string;
  outcome: "accepted" | "vetoed" | "dropped";
  reason: string;
}

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
  scoreBreakdown?: TrajectoryCandidateScoreBreakdown;
}

export interface TrajectoryPolicyStateSnapshot {
  effectiveBudgetRemaining: number;
  budgetUsedFraction: number;
  frontierBacklogPressure: number;
  frontierSaturation: number;
  frontierPressure: number;
  pressureLevel: number;
  remainingExpansionSlots: number;
  activeFrontierSize: number;
  pendingSelectionCount: number;
}

export interface TrajectoryCandidateScoreBreakdown {
  totalScore: number;
  pressureLevel?: number;
  edgeScore?: number;
  seedPrior?: number;
  learnedSeedWeight?: number;
  learnedStopWeight?: number;
  stopBias?: number;
  budgetPressureContribution?: number;
  hopPressureContribution?: number;
  frontierPressureContribution?: number;
  relevance?: number;
  kindBias?: number;
  evidenceQualityBonus?: number;
  opportunityCostPenalty?: number;
  tokenCostFraction?: number;
  downstreamOpportunityCount?: number;
  downstreamPressure?: number;
  redundancyPenalty?: number;
  redundancySimilarity?: number;
  redundancyPressureMultiplier?: number;
}

export interface TrajectoryStateSnapshot {
  sourceNodeId: string | null;
  expansionIndex: number;
  selectionIndex: number;
  budgetRemaining: number;
  initialBudget: number;
  reservedTokenCost: number;
  maxHops: number;
  maxFrontierSize?: number;
  frontierSize: number;
  frontierNodeIds: string[];
  visitedCount: number;
  firedCount: number;
  pendingSelectionCount?: number;
  pendingTargetNodeIds?: string[];
  policyState?: TrajectoryPolicyStateSnapshot;
}

export interface TrajectorySubstep {
  stateSnapshot: TrajectoryStateSnapshot;
  candidates: TrajectoryCandidate[];
  chosenAction: TraversalAction;
  chosenActionProbability: number;
  stopProbability: number;
  stopTruth?: TrajectoryStopTruth;
  stopReason?: TrajectoryStopReason;
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
  proposalOutcomes?: TrajectoryProposalOutcome[];
  terminationReason?: TrajectoryStopReason | null;
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

export interface PolicyGradientRouteUpdateContribution {
  updateKey: string;
  kind: PolicyWeightUpdate["kind"];
  sourceNodeId: string;
  targetNodeId: string | null;
  expansionIndex: number;
  selectionIndex: number;
  chosenActionProbability: number;
  delta: number;
  stopTruth: TrajectoryStopTruth | null;
  stopReason: TrajectoryStopReason | null;
}

export interface PolicyGradientRouteUpdateArtifact {
  updateKey: string;
  kind: PolicyWeightUpdate["kind"];
  sourceNodeId: string;
  targetNodeId: string | null;
  delta: number;
  previousWeight: number | null;
  nextWeight: number;
  contributionCount: number;
  contributions: PolicyGradientRouteUpdateContribution[];
}

export interface PolicyGradientSupervisionArtifact {
  supervisionId: string;
  traceId: string;
  source: RewardSource;
  kind: BrainEvidenceKind;
  value: number;
  confidence: number;
  reason: string | null;
  labelId: string | null;
  evidenceId: string | null;
  observationId: string | null;
  teacherTraceId: string | null;
  serveDecisionRecordId: string | null;
  selectionDigest: string | null;
  turnCompileEventId: string | null;
  activePackGraphChecksum: string | null;
  bindingMode: BrainObservationBindingMode | null;
  traceRequestDigest: string | null;
  traceSelectedNodeIds: string[];
  traceSelectedPathNodeIds: string[];
}

export interface PolicyGradientEpisodeUpdateArtifact {
  episodeId: string;
  observationIds: string[];
  traceIds: string[];
  supervisionIds: string[];
  reward: number;
  rewardSource: RewardSource | null;
  baselineBefore: number;
  baselineAfter: number;
  advantage: number;
  routeUpdateCount: number;
  seedUpdateCount: number;
  stopLocalUpdateCount: number;
  edgeUpdateCount: number;
  supervision: PolicyGradientSupervisionArtifact[];
  routeUpdates: PolicyGradientRouteUpdateArtifact[];
}

export interface PolicyGradientCandidateUpdateArtifactBase {
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

export interface PolicyGradientCandidateUpdateArtifactV1
  extends PolicyGradientCandidateUpdateArtifactBase {
  version: 1;
}

export interface PolicyGradientCandidateUpdateArtifactV2
  extends PolicyGradientCandidateUpdateArtifactBase {
  version: 2;
  observationIds: string[];
  observationCount: number;
  episodeUpdates: PolicyGradientEpisodeUpdateArtifact[];
}

export type PolicyGradientCandidateUpdateArtifact =
  | PolicyGradientCandidateUpdateArtifactV1
  | PolicyGradientCandidateUpdateArtifactV2;

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
  provenanceRef: string | null;
  sourceUri: string | null;
  tags: string[];
  tokenCount: number;
  contentPreview: string;
}

export type BrainPersistenceMode =
  | "redacted"
  | "redacted_with_operator_audit";

export interface DecisionTraceOperatorAudit {
  queryText: string;
  injectedNodeSummaries: DecisionTraceInjectedNodeSummary[];
}

export interface DecisionTraceBranchOutcome {
  sourceNodeId: string | null;
  expansionIndex: number;
  selectionSubstepCount: number;
  continued: boolean;
  selectedTargetIds: string[];
  acceptedTargetIds: string[];
  vetoedTargetIds: string[];
  droppedTargetIds: string[];
  stopTruth: TrajectoryStopTruth | null;
  stopReason: TrajectoryStopReason | null;
  terminationReason: TrajectoryStopReason | null;
  proof: string;
}

export interface DecisionTraceBranchOutcomeSummary {
  branchCount: number;
  continuingBranchCount: number;
  stoppedWithoutProgressCount: number;
  chosenStopBranchCount: number;
  forcedStopBranchCount: number;
  terminationReasons: Record<string, number> | null;
  detail: string;
}

export interface BrainCompileItemV1 {
  nodeId: string;
  provenanceRef: string | null;
  kind: NodeKind | null;
  trust: TrustLevel | null;
  tokenCount: number | null;
  preview: string | null;
  state: "selected" | "dropped" | "prefetched" | "compressed";
  reason: string | null;
  sourceUri?: string | null;
  stage?: BrainDropStage | null;
  fitStrategy?: BrainFitStrategy | null;
  compressionMode?: string | null;
}

export interface BrainCompileReportV1 {
  schemaVersion: 1;
  summary: string;
  traceId: string | null;
  episodeId: string | null;
  requestDigest: string | null;
  activePackId: string | null;
  routerIdentity: string | null;
  bindingMode: BrainObservationBindingMode | null;
  decision: {
    mode: string | null;
    brainDropReason: BrainDropReason | null;
    brainDropStage: BrainDropStage | null;
    queryInterrupted: boolean | null;
    interruptionStage: BrainInterruptionStage | null;
    interruptionReason: string | null;
    servedPartial: boolean | null;
  };
  timing: {
    compileElapsedMs: number | null;
    compileDeadlineMs: number | null;
    compileDeadlineHit: boolean | null;
    routeSelectionMs: number | null;
    totalQueryMs: number | null;
  };
  budget: {
    budgetFraction: number | null;
    budgetChars: number | null;
    queryBudgetChars: number | null;
    maxContextChars: number | null;
    injectedChars: number | null;
    droppedChars: number | null;
    contextClipped: boolean | null;
    fitStrategy: BrainFitStrategy | null;
  };
  counters: {
    candidateNodeCount: number;
    selectedNodeCount: number;
    selectedTraversalNodeCount: number;
    selectedSeedNodeCount: number;
    droppedNodeCount: number;
    prefetchedNodeCount: number;
    compressedNodeCount: number;
    droppedProposalCount: number;
    branchCount: number;
    continuingBranchCount: number;
    sourceRefCount: number;
  };
  reasons: {
    droppedNodeReasons: Record<string, number>;
    droppedProposalReasons: Record<string, number>;
    prefetchedReasons: Record<string, number>;
    compressionReasons: Record<string, number>;
    terminationReasons: Record<string, number>;
  };
  buckets: {
    selected: BrainCompileItemV1[];
    dropped: BrainCompileItemV1[];
    prefetched: BrainCompileItemV1[];
    compressed: BrainCompileItemV1[];
  };
  overflow: {
    selectedOverflowCount: number;
    droppedOverflowCount: number;
    prefetchedOverflowCount: number;
    compressedOverflowCount: number;
    reasonOverflowCount: number;
  };
  boundedness: {
    maxItemsPerBucket: number;
    maxReasonKeys: number;
    maxPreviewChars: number;
    maxSourceRefs: number;
  };
}

export type DecisionTraceSelectionMetadataV4 = DecisionRouteTrace["selectionMetadata"] & {
  traceSliceVersion: 4;
  compileReport?: BrainCompileReportV1 | null;
  compileReportSummary?: string | null;
};

export type BrainInterruptionStage =
  | "embedding"
  | "query"
  | "traversal"
  | "injection";

export interface BrainInterruptionMetadata {
  interrupted: true;
  stage: BrainInterruptionStage;
  reason: string;
  servedPartial: boolean;
}

/**
 * Accounting summary for bounded-anytime serving interruptions.
 *
 * Captures what was completed vs. dropped when a deadline forced
 * early termination, providing the truth surface needed to evaluate
 * serving quality under budget pressure.
 */
export interface InterruptionAccounting {
  /** Frontier node IDs that were queued but never expanded due to deadline. */
  droppedFrontierNodeIds: string[];
  /** Number of source-node expansions that completed before interruption. */
  completedExpansionCount: number;
  /** Maximum expansions allowed (maxHops). */
  maxExpansions: number;
  /** Budget chars consumed by fired nodes. */
  budgetUsed: number;
  /** Total budget chars available at query start. */
  budgetTotal: number;
  /** Fraction of budget consumed: budgetUsed / budgetTotal (0–1). */
  budgetUtilization: number;
  /** Number of proposal outcomes with outcome "dropped" across all expansions. */
  droppedProposalCount: number;
  /** Breakdown of dropped proposal reasons. */
  droppedProposalReasons: Record<string, number>;
}

export type BrainFitStrategy =
  | "structured_node_budget"
  | "legacy_raw_clip";

export type BrainFittingDropReason =
  | "omitted_for_max_context_chars";

export interface DecisionRouteTrace {
  persistenceMode?: BrainPersistenceMode | null;
  requestDigest: string;
  conversationId: number | null;
  activePackId: string | null;
  routerIdentity: string;
  candidateNodeIds: string[];
  selectedNodeIds: string[];
  selectedTraversalNodeIds: string[];
  selectedPathNodeIds: string[];
  selectedSeedNodeIds: string[];
  branchOutcomes: DecisionTraceBranchOutcome[];
  injectedNodeSummaries: DecisionTraceInjectedNodeSummary[];
  sourceSummary: {
    injectedCount: number;
    kinds: Partial<Record<NodeKind, number>>;
    trusts: Partial<Record<TrustLevel, number>>;
    sourceUris: string[];
    sourceRefs: string[];
  };
  operatorAudit?: DecisionTraceOperatorAudit | null;
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
    chosenStopCount?: number | null;
    forcedStopCount?: number | null;
    branchOutcomeSummary?: DecisionTraceBranchOutcomeSummary | null;
    droppedProposalCount?: number | null;
    droppedProposalReasons?: Record<string, number> | null;
    interruption?: BrainInterruptionMetadata | null;
    queryInterrupted?: boolean | null;
    interruptionStage?: BrainInterruptionStage | null;
    interruptionReason?: string | null;
    servedPartial?: boolean | null;
    compileElapsedMs?: number | null;
    compileDeadlineMs?: number | null;
    compileDeadlineHit?: boolean | null;
    brainDropReason?: BrainDropReason | null;
    brainDropStage?: BrainDropStage | null;
    // Post-formatting bounded-runtime metadata is optional so legacy traces
    // remain readable as unknown. `budgetFraction` is the runtime fraction
    // used to derive `queryBudgetChars`; `maxContextChars` is the final
    // injected-block cap.
    budgetFraction?: number | null;
    maxContextChars?: number | null;
    queryBudgetChars?: number | null;
    injectedChars?: number | null;
    droppedChars?: number | null;
    contextClipped?: boolean | null;
    fitStrategy?: BrainFitStrategy | null;
    retrievedNodeCount?: number | null;
    fittedNodeCount?: number | null;
    droppedNodeCount?: number | null;
    fittingDropReasons?: Partial<Record<BrainFittingDropReason, number>> | null;
    interruptionAccounting?: InterruptionAccounting | null;
    prefetch?: BrainPrefetchDecision | null;
    compileReport?: BrainCompileReportV1 | null;
    compileReportSummary?: string | null;
  };
}

export type BrainPrefetchState =
  | "scheduled"
  | "materialized"
  | "hit"
  | "miss"
  | "stale"
  | "invalidated"
  | "dropped";

export type BrainPrefetchKind =
  | "traversal"
  | "support_packet";

export type BrainPrefetchBudgetClass =
  | "tiny"
  | "small"
  | "standard"
  | "large"
  | "deadline_pressure";

export interface BrainPrefetchDecision {
  enabled: boolean;
  state: BrainPrefetchState;
  kind: BrainPrefetchKind | null;
  budgetClass: BrainPrefetchBudgetClass | null;
  key: string | null;
  queryDigest: string | null;
  activePackId: string | null;
  activePackVersion: number | null;
  summaryRoutingMode: string | null;
  prefetchMs: number | null;
  cacheAgeMs: number | null;
  invalidatedReason: string | null;
  reusedNodeCount: number | null;
  reusedChars: number | null;
  savingsChars: number | null;
}

export interface RecentPrefetchSummary {
  windowSize: number;
  sampleSize: number;
  histograms: {
    state: Record<BrainPrefetchState, number>;
    kind: Record<string, number>;
    budgetClass: Record<string, number>;
    summaryRoutingMode: Record<string, number>;
    invalidationReason: Record<string, number>;
  };
  hitRate: { count: number; rate: number | null };
  staleRate: { count: number; rate: number | null };
  invalidationRate: { count: number; rate: number | null };
  detail: string;
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

export type BrainObservationBindingMode =
  | "exact_decision_id"
  | "exact_selection_digest"
  | "turn_compile_event_id"
  | "trace_id"
  | "legacy_heuristic"
  | "unbound";

export interface BrainObservationServedArtifact {
  [key: string]: unknown;
}

export interface BrainObservationOperatorAudit {
  queryText: string;
  retrievedContext: DecisionTraceInjectedNodeSummary[];
  assistantResponse: string;
  toolResults: BrainObservationToolResult[];
  followUpText: string | null;
}

export interface BrainObservationRouteMetadata {
  requestDigest: string | null;
  activePackId: string | null;
  routerIdentity: string | null;
  persistenceMode?: BrainPersistenceMode | null;
  bindingMode: BrainObservationBindingMode | null;
  serveDecisionRecordId: string | null;
  selectionDigest: string | null;
  turnCompileEventId: string | null;
  decisionRecordedAt: string | null;
  activePackEventExportDigest: string | null;
  activePackGraphChecksum: string | null;
  activePackRouterChecksum: string | null;
  activePackBuiltAt: string | null;
  servedArtifact: BrainObservationServedArtifact | null;
  candidateNodeIds: string[];
  selectedNodeIds: string[];
  selectedTraversalNodeIds: string[];
  selectedPathNodeIds: string[];
  selectedSeedNodeIds: string[];
  sourceSummary: DecisionRouteTrace["sourceSummary"] | null;
  operatorAudit?: BrainObservationOperatorAudit | null;
  selectionMetadata: DecisionRouteTrace["selectionMetadata"] | null;
}

export type BrainDropReason =
  | "none"
  | "skip_uninitialized"
  | "skip_no_embedding"
  | "skip_budget_too_small"
  | "skip_no_query"
  | "skip_short_static_lookup"
  | "query_returned_no_nodes"
  | "shadow_mode"
  | "injection_cap_clipped"
  | "deadline_before_query"
  | "deadline_after_query"
  | "deadline_before_injection"
  | "assembly_fail_open";

export type BrainDropStage =
  | "decision"
  | "query"
  | "injection";

export type BrainObservationStatus =
  | "pending_followup"
  | "pending_teacher"
  | "completed";

export interface BrainObservationTeacherEvaluation {
  version: 2;
  observationId: string;
  episodeId: string;
  traceId: string | null;
  serveDecisionRecordId: string | null;
  selectionDigest: string | null;
  turnCompileEventId: string | null;
  decisionRecordedAt: string | null;
  activePackId: string | null;
  activePackEventExportDigest: string | null;
  activePackGraphChecksum: string | null;
  activePackRouterChecksum: string | null;
  activePackBuiltAt: string | null;
  bindingMode: BrainObservationBindingMode | null;
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

export type ContextFeedbackVerdict =
  | "helpful"
  | "irrelevant"
  | "harmful";

export type ContextFeedbackFocusAction =
  | "review_harmful_context"
  | "capture_follow_up"
  | "wait_for_teacher"
  | "increase_feedback_coverage"
  | "monitor";

export interface ContextFeedbackCoverageSummary {
  routeTraceCount: number;
  observationCount: number;
  completedObservationCount: number;
  supervisedTraceCount: number;
  unsupervisedTraceCount: number;
  observationCoverage: number;
  supervisionCoverage: number;
  pendingFollowupCount: number;
  pendingTeacherCount: number;
}

export interface ContextFeedbackLatestVerdict {
  traceId: string;
  episodeId: string;
  observationId: string | null;
  source: RewardSource;
  verdict: ContextFeedbackVerdict;
  score: number;
  confidence: number;
  reason: string | null;
  bindingMode: BrainObservationBindingMode | null;
  createdAt: number;
}

export interface ContextFeedbackSummary {
  scoreBands: {
    helpfulMin: number;
    harmfulMax: number;
  };
  verdictCounts: Record<ContextFeedbackVerdict, number>;
  coverage: ContextFeedbackCoverageSummary;
  latest: ContextFeedbackLatestVerdict | null;
  focus: {
    action: ContextFeedbackFocusAction;
    detail: string;
  };
  detail: string;
}

export type BrainContextUsefulnessVerdict =
  | "helpful"
  | "irrelevant"
  | "harmful";

export type BrainContextUsefulnessFollowUpClass =
  | "confirmation"
  | "neutral_ack"
  | "reask"
  | "correction"
  | "contradiction"
  | "missing";

export type BrainContextUsefulnessToolOutcomeClass =
  | "success"
  | "partial"
  | "failure"
  | "error"
  | "missing";

export type BrainContextUsefulnessRouteIntegrityClass =
  | "exact_full_serve"
  | "fallback"
  | "clipped_or_fail_open"
  | "unbound";

export type BrainContextUsefulnessTeacherAlignmentClass =
  | "calibration";

export interface BrainContextUsefulnessSignal {
  class: string;
  score: number;
  evidence: string[];
  detail: string | null;
}

export interface BrainContextUsefulnessSignalsV1 {
  followUp: BrainContextUsefulnessSignal & { class: BrainContextUsefulnessFollowUpClass };
  toolOutcome: BrainContextUsefulnessSignal & { class: BrainContextUsefulnessToolOutcomeClass };
  routeIntegrity: BrainContextUsefulnessSignal & { class: BrainContextUsefulnessRouteIntegrityClass };
  teacherAlignment: (BrainContextUsefulnessSignal & { class: BrainContextUsefulnessTeacherAlignmentClass }) | null;
  authorityGate: {
    blocked: boolean;
    reason: string | null;
  };
}

export interface BrainContextUsefulnessEvaluationV1 {
  version: 1;
  observationId: string;
  episodeId: string;
  traceId: string | null;
  conversationId: number | null;
  bindingMode: BrainObservationBindingMode | null;
  signals: BrainContextUsefulnessSignalsV1;
  finalScore: number;
  confidence: number;
  verdict: BrainContextUsefulnessVerdict;
  reason: string;
  computedAt: number;
}

export interface BrainContextUsefulnessRecord {
  id: string;
  observationId: string;
  episodeId: string;
  traceId: string | null;
  conversationId: number | null;
  bindingMode: BrainObservationBindingMode | null;
  followUpText: string | null;
  toolResults: BrainObservationToolResult[];
  signals: BrainContextUsefulnessSignalsV1;
  finalScore: number;
  confidence: number;
  verdict: BrainContextUsefulnessVerdict;
  reason: string | null;
  createdAt: number;
  updatedAt: number;
  evaluatedAt: number;
}

export interface ContextUsefulnessCoverageSummary {
  observationCount: number;
  readyObservationCount: number;
  scoredObservationCount: number;
  completedObservationCount: number;
  observationCoverage: number;
  readyCoverage: number;
  pendingFollowupCount: number;
  pendingTeacherCount: number;
}

export interface ContextUsefulnessLatestVerdict {
  observationId: string;
  episodeId: string;
  traceId: string | null;
  verdict: BrainContextUsefulnessVerdict;
  score: number;
  confidence: number;
  reason: string | null;
  bindingMode: BrainObservationBindingMode | null;
  createdAt: number;
}

export interface ContextUsefulnessSummary {
  scoreBands: {
    helpfulMin: number;
    harmfulMax: number;
  };
  verdictCounts: Record<BrainContextUsefulnessVerdict, number>;
  coverage: ContextUsefulnessCoverageSummary;
  latest: ContextUsefulnessLatestVerdict | null;
  detail: string;
}

export function resolveObservationBindingMode(params: {
  bindingMode?: BrainObservationBindingMode | null;
  serveDecisionRecordId?: string | null;
  selectionDigest?: string | null;
  activePackGraphChecksum?: string | null;
  turnCompileEventId?: string | null;
  traceId?: string | null;
}): BrainObservationBindingMode {
  if (params.bindingMode) {
    return params.bindingMode;
  }
  if (params.serveDecisionRecordId) {
    return "exact_decision_id";
  }
  if (params.selectionDigest && params.activePackGraphChecksum) {
    return "exact_selection_digest";
  }
  if (params.turnCompileEventId) {
    return "turn_compile_event_id";
  }
  if (params.traceId) {
    return "trace_id";
  }
  return "unbound";
}

// ═══════════════════════════════════════════
// Configuration
// ═══════════════════════════════════════════

export interface BrainConfig {
  enabled: boolean;
  root: string;
  maxCompileMs: number | null;
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
  persistRawSurfaces: boolean;
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
  maxCompileMs: null,
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
  persistRawSurfaces: false,
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
  interruption?: BrainInterruptionMetadata | null;
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
  frontierPressure: number;
  branchOpportunityCost: number;
  localRedundancyPenalty: number;
  evidenceQualityBias: number;
  edgeKindBias: Record<EdgeKind, number>;
}

export const DEFAULT_POLICY_PARAMS: PolicyParams = {
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

// Trust rank ordering: human > self > scanner > teacher
export function trustRank(source: RewardSource): number {
  switch (source) {
    case "human": return 4;
    case "self": return 3;
    case "scanner": return 2;
    case "teacher": return 1;
  }
}
