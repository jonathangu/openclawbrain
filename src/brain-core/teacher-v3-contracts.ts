/**
 * Teacher v3 proposal / lineage / compiled-artifact contracts.
 *
 * These are narrow, reusable type surfaces for off-path Teacher v3 work.
 * They intentionally do not wire live mutation behavior or storage.
 */

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
