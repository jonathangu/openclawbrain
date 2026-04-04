/**
 * Teacher v3 proposal / lineage / compiled-artifact contracts.
 *
 * These are narrow, reusable type surfaces for off-path Teacher v3 work.
 * They intentionally do not wire live mutation behavior or storage.
 */

import type { EdgeKind, NodeKind } from "./types.js";

export type EvidenceSourceKind =
  | "user_turn"
  | "tool_trace"
  | "file"
  | "repo"
  | "summary"
  | "correction";

export type EvidenceAuthority = "user_explicit" | "raw_source" | "operator_policy";

export type EvidenceDerivation =
  | "summary_navigation"
  | "teacher_inference"
  | "teacher_compilation"
  | "teacher_lint"
  | "teacher_mutation_proposal"
  | "teacher_forgetting_proposal";

export type ProposalClass = "compiler" | "lint" | "mutation" | "forgetting" | "correction";

export type ProposalStatus =
  | "proposed"
  | "validated"
  | "shadow_scored"
  | "promotable"
  | "promoted"
  | "rejected"
  | "expired"
  | "rolled_back";

export type CompiledArtifactKind =
  | "concept_page"
  | "workflow_page"
  | "topic_index"
  | "map_of_territory"
  | "neighborhood_summary"
  | "cross_source_synthesis"
  | "stale_fact_watch"
  | "contradiction_report"
  | "provenance_gap_report";

export type CompiledArtifactStatus =
  | "draft"
  | "proposed"
  | "validated"
  | "promotable"
  | "promoted"
  | "rejected"
  | "expired"
  | "superseded";

/**
 * Retention state model for teacher-driven forgetting proposals.
 *
 * The progression is intentionally fail-closed: prefer retention, then
 * demotion, then archive, then tombstone, and only then hard delete.
 * Explicit user corrections are protected from hard deletion.
 */
export type RetentionState = "retained" | "demoted" | "archived" | "tombstoned" | "deleted";

export type RetentionTransitionKind = "retain" | "demote" | "archive" | "tombstone" | "hard_delete";

export interface RetentionTargetRefV1 {
  sourceId: string;
  sourceKind?: EvidenceSourceKind;
  authority: EvidenceAuthority;
  label?: string;
}

export interface RetentionTransitionDecisionV1 {
  from: RetentionState;
  to: RetentionState;
  via: RetentionTransitionKind;
  allowed: boolean;
  reason: string;
  guardrail?: "deny_hard_delete_user_explicit" | "requires_tombstoned_prestate";
}

export const RETENTION_STATE_TRANSITIONS: Record<RetentionState, readonly RetentionState[]> = {
  retained: ["retained", "demoted", "archived", "tombstoned"],
  demoted: ["demoted", "archived", "tombstoned"],
  archived: ["archived", "tombstoned"],
  tombstoned: ["tombstoned", "deleted"],
  deleted: ["deleted"],
} as const;

export function evaluateRetentionTransitionV1(params: {
  current: RetentionState;
  requested: RetentionTransitionKind;
  target: RetentionTargetRefV1;
}): RetentionTransitionDecisionV1 {
  const current = params.current;

  if (params.requested === "retain") {
    return {
      from: current,
      to: current,
      via: params.requested,
      allowed: true,
      reason: "retention no-op",
    };
  }

  if (params.requested === "hard_delete") {
    if (params.target.authority === "user_explicit") {
      return {
        from: current,
        to: current,
        via: params.requested,
        allowed: false,
        guardrail: "deny_hard_delete_user_explicit",
        reason: "teacher-driven forgetting may not hard-delete user_explicit correction memory",
      };
    }
    if (current !== "tombstoned") {
      return {
        from: current,
        to: current,
        via: params.requested,
        allowed: false,
        guardrail: "requires_tombstoned_prestate",
        reason: "hard delete requires a tombstoned pre-state",
      };
    }
    return {
      from: current,
      to: "deleted",
      via: params.requested,
      allowed: true,
      reason: "hard delete allowed after tombstone for non-user-explicit memory",
    };
  }

  if (current === "deleted") {
    return {
      from: current,
      to: current,
      via: params.requested,
      allowed: false,
      reason: "deleted is terminal and cannot be revised by teacher forgetting",
    };
  }

  const next =
    params.requested === "demote"
      ? "demoted"
      : params.requested === "archive"
        ? "archived"
        : "tombstoned";

  if (!RETENTION_STATE_TRANSITIONS[current].includes(next)) {
    return {
      from: current,
      to: current,
      via: params.requested,
      allowed: false,
      reason: `retention transition ${current} -> ${next} is not allowed`,
    };
  }

  return {
    from: current,
    to: next,
    via: params.requested,
    allowed: true,
    reason: `retention transition ${current} -> ${next}`,
  };
}

export interface EvidenceSpan {
  start: number;
  end: number;
}

/**
 * Evidence reference used across proposals, compiled artifacts, and proof bundles.
 *
 * `sourceId` should point at a durable source record.
 * `authority` describes the source substrate, not the proposal.
 * `derivation` stays optional because the same evidence may be reused in
 * multiple Teacher v3 lanes.
 */
export interface EvidenceRefV1 {
  evidenceId?: string;
  sourceKind: EvidenceSourceKind;
  sourceId: string;
  span?: EvidenceSpan;
  authority: EvidenceAuthority;
  derivation?: EvidenceDerivation;
  excerpt?: string;
  quote?: string;
  sourceHash?: string;
  digest?: string;
  capturedAt?: string;
  retrievedAt?: string;
}

export type EvidenceRef = EvidenceRefV1;

/**
 * Replay / dedupe lineage for a proposal instance.
 */
export interface ProposalLineageV1 {
  proposalClass: ProposalClass;
  basePackVersion?: number;
  baseGraphHash?: string;
  producerVersion: string;
  producerBuildId?: string;
  promptHash?: string;
  templateId?: string;
  scope: string;
  profile?: string;
  idempotencyKey: string;
  sourceBundleId?: string;
  parentProposalIds?: string[];
}

export type ProposalLineage = ProposalLineageV1;

export interface ProposalExpectedEffectV1 {
  retrieval?: "better" | "same" | "uncertain";
  truthRisk?: "low" | "medium" | "high";
  tokenBudget?: "lower" | "same" | "higher";
}

export interface TeacherProposalArtifactRefV1 {
  artifactId: string;
  kind: string;
  contentHash: string;
}

export interface TeacherProposalV1 {
  proposalId: string;
  proposalClass: ProposalClass;
  /** Compatibility alias for docs that still spell the outer lane as `lane`. */
  lane?: ProposalClass;
  status: ProposalStatus;
  lineage: ProposalLineageV1;
  subjectIds: string[];
  evidence: EvidenceRefV1[];
  counterevidence?: EvidenceRefV1[];
  payload: unknown;
  expectedEffect?: ProposalExpectedEffectV1;
  confidence: number;
  replaySuites: string[];
  rollbackKey: string;
  expiresAt?: string;
  createdAt: string;
  resolvedAt?: string;
  artifacts?: TeacherProposalArtifactRefV1[];
}

export type TeacherProposal = TeacherProposalV1;

export type TeacherProposalReplayGateDimensionName =
  | "truth_invariants"
  | "attribution_floor"
  | "boundedness"
  | "reversibility";

export interface TeacherProposalReplayGateDimensionV1 {
  name: TeacherProposalReplayGateDimensionName;
  summary: string;
  requirements: string[];
}

export interface TeacherProposalReplayGateV1 {
  proposalClass: ProposalClass;
  reviewMode: "shadow_only";
  dimensions: {
    truthInvariants: TeacherProposalReplayGateDimensionV1;
    attributionFloor: TeacherProposalReplayGateDimensionV1;
    boundedness: TeacherProposalReplayGateDimensionV1;
    reversibility: TeacherProposalReplayGateDimensionV1;
  };
}

const buildTeacherProposalReplayGate = (
  proposalClass: ProposalClass,
  focus: string,
): TeacherProposalReplayGateV1 => ({
  proposalClass,
  reviewMode: "shadow_only",
  dimensions: {
    truthInvariants: {
      name: "truth_invariants",
      summary: `${focus}: keep derived output subordinate to explicit authority.`,
      requirements: [
        "Explicit correction memory still outranks teacher synthesis.",
        "The live path stays read-only to the proposal.",
        "Evidence refs stay attached to any non-trivial claim.",
      ],
    },
    attributionFloor: {
      name: "attribution_floor",
      summary: `${focus}: every proposed change needs clear evidence coverage.`,
      requirements: [
        "Every proposal carries durable evidence refs.",
        "Source ids must be stable record ids, not display labels.",
        "Unattributed payload stays out of promotion.",
      ],
    },
    boundedness: {
      name: "boundedness",
      summary: `${focus}: keep the reviewable surface compact and inspectable.`,
      requirements: [
        "Proposal subject sets stay finite and small.",
        "Payloads avoid raw corpus dumps and unbounded excerpts.",
        "Replay fits inside a single review pass.",
      ],
    },
    reversibility: {
      name: "reversibility",
      summary: `${focus}: preserve rollback and replay identity.`,
      requirements: [
        "RollbackKey identifies the reversible path.",
        "Prior state remains recoverable for replay.",
        "Rejected or superseded proposals keep lineage.",
      ],
    },
  },
});

export const TEACHER_PROPOSAL_REPLAY_GATES_V1: Record<ProposalClass, TeacherProposalReplayGateV1> = {
  compiler: buildTeacherProposalReplayGate("compiler", "Compiler lane"),
  lint: buildTeacherProposalReplayGate("lint", "Lint lane"),
  mutation: buildTeacherProposalReplayGate("mutation", "Mutation lane"),
  forgetting: buildTeacherProposalReplayGate("forgetting", "Forgetting lane"),
  correction: buildTeacherProposalReplayGate("correction", "Correction lane"),
} as const;

export function describeTeacherProposalReplayGate(
  proposalClass: ProposalClass,
): TeacherProposalReplayGateV1 {
  return TEACHER_PROPOSAL_REPLAY_GATES_V1[proposalClass];
}

/**
 * Shadow-only structural mutation DSL for Teacher v3.
 *
 * These proposal shapes stay off the live graph path until replay-gated
 * promotion. The forgetting lane reuses the retention model below.
 */
export type TeacherMutationKindV1 =
  | "add_node"
  | "merge_nodes"
  | "split_node"
  | "add_edge"
  | "strengthen_edge"
  | "weaken_edge"
  | "add_inhibitory_edge"
  | "demote_node"
  | "archive_node"
  | "tombstone_node";

export interface TeacherMutationNodeDraftV1 {
  nodeId?: string;
  kind: NodeKind;
  content: string;
  sourceUri?: string | null;
  tags?: string[];
  metadata?: Record<string, unknown>;
}

export interface TeacherMutationEdgeRefV1 {
  sourceNodeId: string;
  targetNodeId: string;
  edgeKind: EdgeKind;
  weight?: number;
  prior?: number;
  metadata?: Record<string, unknown>;
}

export interface TeacherAddNodeMutationProposalV1 {
  mutationKind: "add_node";
  node: TeacherMutationNodeDraftV1;
  attachToNodeIds?: string[];
  rationale?: string;
}

export interface TeacherMergeNodesMutationProposalV1 {
  mutationKind: "merge_nodes";
  sourceNodeIds: [string, string, ...string[]];
  mergedNode?: TeacherMutationNodeDraftV1;
  preserveNodeIds?: string[];
  rationale?: string;
}

export interface TeacherSplitNodeMutationProposalV1 {
  mutationKind: "split_node";
  sourceNodeId: string;
  splitNodeDrafts: [TeacherMutationNodeDraftV1, TeacherMutationNodeDraftV1, ...TeacherMutationNodeDraftV1[]];
  preserveEdgeKinds?: EdgeKind[];
  rationale?: string;
}

export interface TeacherEdgeMutationProposalV1 {
  mutationKind:
    | "add_edge"
    | "strengthen_edge"
    | "weaken_edge"
    | "add_inhibitory_edge";
  edge: TeacherMutationEdgeRefV1;
  delta?: number;
  targetWeight?: number;
  rationale?: string;
}

export interface TeacherRetentionMutationProposalV1 {
  mutationKind: "demote_node" | "archive_node" | "tombstone_node";
  target: RetentionTargetRefV1;
  requestedTransition: Extract<RetentionTransitionKind, "demote" | "archive" | "tombstone">;
  rationale?: string;
}

export type TeacherMutationProposalPayloadV1 =
  | TeacherAddNodeMutationProposalV1
  | TeacherMergeNodesMutationProposalV1
  | TeacherSplitNodeMutationProposalV1
  | TeacherEdgeMutationProposalV1
  | TeacherRetentionMutationProposalV1;

export interface TeacherMutationProposalV1 extends TeacherProposalV1 {
  proposalClass: "mutation";
  lane?: "mutation";
  shadowOnly: true;
  payload: TeacherMutationProposalPayloadV1;
}

export type TeacherMutationKind = TeacherMutationKindV1;
export type TeacherMutationNodeDraft = TeacherMutationNodeDraftV1;
export type TeacherMutationEdgeRef = TeacherMutationEdgeRefV1;
export type TeacherMutationProposalPayload = TeacherMutationProposalPayloadV1;
export type TeacherMutationProposal = TeacherMutationProposalV1;

export interface CompiledArtifactClaimRefV1 {
  claimId: string;
  text: string;
  evidenceIds: string[];
  confidence: number;
  status: "supported" | "partial" | "uncertain";
}

export interface CompiledArtifactPromotionMetaV1 {
  promotedAt?: string;
  promotedPackId?: string;
  rejectedAt?: string;
  rejectedReason?: string;
  replaySuites?: string[];
  rollbackKey?: string;
}

export interface CompiledArtifactProvenanceV1 {
  producer: "teacher-v3" | string;
  producerVersion: string;
  promptHash?: string;
  runId?: string;
  basePackId?: string;
  baseGraphHash?: string;
  scope: string;
  idempotencyKey: string;
  sourceRoots?: string[];
  transformChain?: string[];
}

export interface CompiledArtifactMetaV1 {
  schemaVersion: 1;
  artifactId: string;
  kind: CompiledArtifactKind;
  title: string;
  status: CompiledArtifactStatus;
  packId: string;
  proposalId: string;
  proposalLane: "compiler";
  subjectIds: string[];
  evidence: EvidenceRefV1[];
  counterevidence?: EvidenceRefV1[];
  provenance: CompiledArtifactProvenanceV1;
  contentHash: string;
  markdownPath: string;
  metaPath: string;
  createdAt: string;
  updatedAt: string;
  expiresAt?: string;
  confidence: number;
  claims?: CompiledArtifactClaimRefV1[];
  promotion?: CompiledArtifactPromotionMetaV1;
  supersedesArtifactId?: string;
}

export type CompiledArtifactMeta = CompiledArtifactMetaV1;
