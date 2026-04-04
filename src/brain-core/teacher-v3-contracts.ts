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

export const TEACHER_PROPOSAL_PROMOTABLE_CLASSES_V1 = ["compiler", "lint"] as const satisfies readonly ProposalClass[];

export const TEACHER_PROPOSAL_SHADOW_ONLY_CLASSES_V1 = [
  "mutation",
  "forgetting",
  "correction",
] as const satisfies readonly ProposalClass[];

export type TeacherProposalPromotableClassV1 = (typeof TEACHER_PROPOSAL_PROMOTABLE_CLASSES_V1)[number];

export type TeacherProposalShadowOnlyClassV1 = (typeof TEACHER_PROPOSAL_SHADOW_ONLY_CLASSES_V1)[number];

export type ProposalStatus =
  | "proposed"
  | "validated"
  | "shadow_scored"
  | "promotable"
  | "promoted"
  | "rejected"
  | "expired"
  | "rolled_back";

export interface TeacherProposalProofLinkV1 {
  refId: string;
  kind: string;
  path: string;
}

export type TeacherProposalReplayOutcomeResultV1 = "pass" | "warn" | "fail";

export type TeacherProposalReplayOutcomeSourceV1 = "proposal_record" | "proof_bundle" | "derived";

export interface TeacherProposalReplayOutcomeV1 {
  outcomeId: string;
  replaySuite: string;
  proposalClass: ProposalClass;
  reviewMode: "promotable" | "shadow_only";
  result: TeacherProposalReplayOutcomeResultV1;
  source: TeacherProposalReplayOutcomeSourceV1;
  summary: string;
  evidenceLinks?: TeacherProposalProofLinkV1[];
  counterevidenceLinks?: TeacherProposalProofLinkV1[];
  capturedAt: string;
  notes?: string[];
}

export interface TeacherProposalReplayOutcomeSummaryV1 {
  replayOutcomeCount: number;
  replaySuites: string[];
  resultCounts: Record<TeacherProposalReplayOutcomeResultV1, number>;
  reviewModeCounts: Record<"promotable" | "shadow_only", number>;
  sourceCounts: Record<TeacherProposalReplayOutcomeSourceV1, number>;
  summary: string;
}

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

export interface TeacherProposalProofBundleV1 {
  bundleId: string;
  proposalId: string;
  proposalClass: ProposalClass;
  status: "promoted" | "rolled_back";
  lineage: ProposalLineageV1;
  rollbackKey: string;
  replaySuites: string[];
  replayOutcomes?: TeacherProposalReplayOutcomeV1[];
  replayOutcomeSummary?: TeacherProposalReplayOutcomeSummaryV1;
  surfaceMap: TeacherV3ProofSurfaceRefV1[];
  evidenceLinks: TeacherProposalProofLinkV1[];
  counterevidenceLinks?: TeacherProposalProofLinkV1[];
  summary: string;
  createdAt: string;
  updatedAt?: string;
}

export interface TeacherProposalProofBundleSummaryV1 {
  bundleId: string;
  proposalId: string;
  proposalClass: ProposalClass;
  status: TeacherProposalProofBundleV1["status"];
  rollbackKey: string;
  replaySuites: string[];
  replayOutcomeSummary: TeacherProposalReplayOutcomeSummaryV1;
  surfaceIds: string[];
  surfaceCount: number;
  shippedSurfaceCount: number;
  targetSurfaceCount: number;
  evidenceLinkCount: number;
  counterevidenceLinkCount: number;
  summary: string;
}

export function summarizeTeacherProposalProofBundleV1(
  bundle: TeacherProposalProofBundleV1,
): TeacherProposalProofBundleSummaryV1 {
  const replayOutcomeSummary = bundle.replayOutcomeSummary ?? summarizeTeacherProposalReplayOutcomesV1(bundle.replayOutcomes ?? []);
  return {
    bundleId: bundle.bundleId,
    proposalId: bundle.proposalId,
    proposalClass: bundle.proposalClass,
    status: bundle.status,
    rollbackKey: bundle.rollbackKey,
    replaySuites: [...bundle.replaySuites],
    replayOutcomeSummary,
    surfaceIds: bundle.surfaceMap.map((surface) => surface.id),
    surfaceCount: bundle.surfaceMap.length,
    shippedSurfaceCount: bundle.surfaceMap.filter((surface) => surface.state === "shipped").length,
    targetSurfaceCount: bundle.surfaceMap.filter((surface) => surface.state === "target").length,
    evidenceLinkCount: bundle.evidenceLinks.length,
    counterevidenceLinkCount: bundle.counterevidenceLinks?.length ?? 0,
    summary: bundle.summary,
  };
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
  proofBundle?: TeacherProposalProofBundleV1;
  expiresAt?: string;
  createdAt: string;
  resolvedAt?: string;
  artifacts?: TeacherProposalArtifactRefV1[];
  replayGate?: TeacherProposalReplayGateV1;
  /** Target-state only: explicit canary rollout plan, defaulting to off. */
  canaryRollout?: TeacherCanaryRolloutPlanV1;
  /** Replay summary for candidate-pack evaluation, durable and inspectable. */
  replaySummary?: TeacherProposalReplaySummaryV1;
}

export type TeacherProposal = TeacherProposalV1;

export interface TeacherProposalReplayHealthSummaryV1 {
  firedPerQuery: number | null;
  dormantPercent: number | null;
  orphanCount: number | null;
}

export interface TeacherProposalReplayStateSnapshotV1 {
  phase: "before" | "after";
  surfaceState: "shipped" | "target";
  packVersion: number | null;
  packId: string | null;
  graphHash: string | null;
  nodeCount: number | null;
  edgeCount: number | null;
  health: TeacherProposalReplayHealthSummaryV1;
  notes: string[];
}

export interface TeacherCompilerReplaySummaryV1 {
  kind: "compiler";
  reviewMode: TeacherProposalReplayGateReviewModeV1;
  promotionDiscipline: "promotable";
  subjectCount: number;
  evidenceCount: number;
  counterevidenceCount: number;
  replaySuites: string[];
  candidatePackVersion: number | null;
  candidatePackId: string | null;
  candidateGraphHash: string | null;
  summary: string;
  notes: string[];
}

export interface TeacherLintReplaySummaryV1 {
  kind: "lint";
  reviewMode: TeacherProposalReplayGateReviewModeV1;
  promotionDiscipline: "promotable";
  subjectCount: number;
  evidenceCount: number;
  counterevidenceCount: number;
  replaySuites: string[];
  candidatePackVersion: number | null;
  candidatePackId: string | null;
  candidateGraphHash: string | null;
  summary: string;
  notes: string[];
}

export interface TeacherShadowReplaySummaryV1 {
  kind: Exclude<ProposalClass, "compiler" | "lint">;
  reviewMode: TeacherProposalReplayGateReviewModeV1;
  promotionDiscipline: "shadow_only";
  subjectCount: number;
  evidenceCount: number;
  counterevidenceCount: number;
  replaySuites: string[];
  candidatePackVersion: number | null;
  candidatePackId: string | null;
  candidateGraphHash: string | null;
  summary: string;
  notes: string[];
}

export type TeacherProposalClassReplaySummaryV1 =
  | TeacherCompilerReplaySummaryV1
  | TeacherLintReplaySummaryV1
  | TeacherShadowReplaySummaryV1;

export interface TeacherProposalReplaySummaryV1 {
  replayId: string;
  proposalId: string;
  proposalClass: ProposalClass;
  status: ProposalStatus;
  reviewMode: TeacherProposalReplayGateReviewModeV1;
  basePackVersion: number | null;
  baseGraphHash: string | null;
  candidatePackVersion: number | null;
  candidatePackId: string | null;
  candidateGraphHash: string | null;
  beforeScore: number;
  afterScore: number;
  scoreDelta: number;
  before: TeacherProposalReplayStateSnapshotV1;
  after: TeacherProposalReplayStateSnapshotV1;
  classSummary: TeacherProposalClassReplaySummaryV1;
  summary: string;
  createdAt: string;
  updatedAt: string;
}

export interface TeacherProposalLineageSummaryV1 extends ProposalLineageV1 {
  parentProposalIds: string[];
}

export interface TeacherProposalSummaryV1 {
  proposalId: string;
  proposalClass: ProposalClass;
  lane?: ProposalClass;
  status: ProposalStatus;
  lineage: TeacherProposalLineageSummaryV1;
  subjectIds: string[];
  subjectCount: number;
  evidenceIds: string[];
  evidenceCount: number;
  counterevidenceIds: string[];
  counterevidenceCount: number;
  replaySuites: string[];
  replaySuiteCount: number;
  rollbackKey: string;
  confidence: number;
  hasProofBundle: boolean;
  proofBundleId?: string;
  proofBundleStatus?: TeacherProposalProofBundleV1["status"];
  proofBundleReplayOutcomeSummary?: TeacherProposalReplayOutcomeSummaryV1;
  hasReplaySummary: boolean;
  replaySummary?: TeacherProposalReplaySummaryV1;
  createdAt: string;
  resolvedAt?: string;
}

export interface TeacherProposalDiffListV1 {
  added: string[];
  removed: string[];
}

export interface TeacherProposalDiffV1 {
  leftProposalId: string;
  rightProposalId: string;
  sameProposalClass: boolean;
  sameIdempotencyKey: boolean;
  sameRollbackKey: boolean;
  sameStatus: boolean;
  changedFields: string[];
  subjectIds: TeacherProposalDiffListV1;
  evidenceIds: TeacherProposalDiffListV1;
  counterevidenceIds: TeacherProposalDiffListV1;
  replaySuites: TeacherProposalDiffListV1;
  summary: string;
}

function canonicalEvidenceRefId(ref: EvidenceRefV1): string {
  return ref.evidenceId ?? ref.sourceId;
}

function uniqueStrings(values: string[]): string[] {
  return [...new Set(values)];
}

function diffStringLists(left: string[], right: string[]): TeacherProposalDiffListV1 {
  const rightSet = new Set(right);
  const leftSet = new Set(left);
  return {
    added: right.filter((value) => !leftSet.has(value)),
    removed: left.filter((value) => !rightSet.has(value)),
  };
}

function cloneProposalLineage(lineage: ProposalLineageV1): TeacherProposalLineageSummaryV1 {
  return {
    ...lineage,
    parentProposalIds: [...(lineage.parentProposalIds ?? [])],
  };
}

function cloneTeacherProposalReplaySummary(
  replaySummary: TeacherProposalReplaySummaryV1 | undefined,
): TeacherProposalReplaySummaryV1 | undefined {
  return replaySummary === undefined ? undefined : JSON.parse(JSON.stringify(replaySummary)) as TeacherProposalReplaySummaryV1;
}

export function summarizeTeacherProposalV1(
  proposal: TeacherProposalV1,
): TeacherProposalSummaryV1 {
  const evidenceIds = uniqueStrings(proposal.evidence.map(canonicalEvidenceRefId));
  const counterevidenceIds = uniqueStrings((proposal.counterevidence ?? []).map(canonicalEvidenceRefId));
  const proofBundle = proposal.proofBundle;
  const proofBundleReplayOutcomeSummary = proofBundle
    ? proofBundle.replayOutcomeSummary ?? summarizeTeacherProposalReplayOutcomesV1(proofBundle.replayOutcomes ?? [])
    : undefined;

  return {
    proposalId: proposal.proposalId,
    proposalClass: proposal.proposalClass,
    lane: proposal.lane,
    status: proposal.status,
    lineage: cloneProposalLineage(proposal.lineage),
    subjectIds: [...proposal.subjectIds],
    subjectCount: proposal.subjectIds.length,
    evidenceIds,
    evidenceCount: evidenceIds.length,
    counterevidenceIds,
    counterevidenceCount: counterevidenceIds.length,
    replaySuites: [...proposal.replaySuites],
    replaySuiteCount: proposal.replaySuites.length,
    rollbackKey: proposal.rollbackKey,
    confidence: proposal.confidence,
    hasProofBundle: proofBundle !== undefined,
    proofBundleId: proofBundle?.bundleId,
    proofBundleStatus: proofBundle?.status,
    proofBundleReplayOutcomeSummary,
    hasReplaySummary: proposal.replaySummary !== undefined,
    replaySummary: cloneTeacherProposalReplaySummary(proposal.replaySummary),
    createdAt: proposal.createdAt,
    resolvedAt: proposal.resolvedAt,
  };
}

export function diffTeacherProposalV1(
  left: TeacherProposalV1,
  right: TeacherProposalV1,
): TeacherProposalDiffV1 {
  const leftSummary = summarizeTeacherProposalV1(left);
  const rightSummary = summarizeTeacherProposalV1(right);
  const changedFields: string[] = [];

  const compareField = (field: string, leftValue: unknown, rightValue: unknown): void => {
    if (JSON.stringify(leftValue) !== JSON.stringify(rightValue)) {
      changedFields.push(field);
    }
  };

  compareField("proposalClass", leftSummary.proposalClass, rightSummary.proposalClass);
  compareField("lane", leftSummary.lane ?? null, rightSummary.lane ?? null);
  compareField("status", leftSummary.status, rightSummary.status);
  compareField("confidence", leftSummary.confidence, rightSummary.confidence);
  compareField("rollbackKey", leftSummary.rollbackKey, rightSummary.rollbackKey);
  compareField("replaySuites", leftSummary.replaySuites, rightSummary.replaySuites);
  compareField("subjectIds", leftSummary.subjectIds, rightSummary.subjectIds);
  compareField("evidenceIds", leftSummary.evidenceIds, rightSummary.evidenceIds);
  compareField("counterevidenceIds", leftSummary.counterevidenceIds, rightSummary.counterevidenceIds);
  compareField("proofBundleId", leftSummary.proofBundleId ?? null, rightSummary.proofBundleId ?? null);
  compareField("proofBundleStatus", leftSummary.proofBundleStatus ?? null, rightSummary.proofBundleStatus ?? null);
  compareField(
    "proofBundleReplayOutcomeSummary",
    leftSummary.proofBundleReplayOutcomeSummary ?? null,
    rightSummary.proofBundleReplayOutcomeSummary ?? null,
  );
  compareField("replaySummary", leftSummary.replaySummary ?? null, rightSummary.replaySummary ?? null);
  compareField("createdAt", leftSummary.createdAt, rightSummary.createdAt);
  compareField("resolvedAt", leftSummary.resolvedAt ?? null, rightSummary.resolvedAt ?? null);

  const lineageKeys: Array<keyof TeacherProposalLineageSummaryV1> = [
    "proposalClass",
    "basePackVersion",
    "baseGraphHash",
    "producerVersion",
    "producerBuildId",
    "promptHash",
    "templateId",
    "scope",
    "profile",
    "idempotencyKey",
    "sourceBundleId",
    "parentProposalIds",
  ];
  for (const key of lineageKeys) {
    compareField(`lineage.${String(key)}`, leftSummary.lineage[key], rightSummary.lineage[key]);
  }

  return {
    leftProposalId: leftSummary.proposalId,
    rightProposalId: rightSummary.proposalId,
    sameProposalClass: leftSummary.proposalClass === rightSummary.proposalClass,
    sameIdempotencyKey: leftSummary.lineage.idempotencyKey === rightSummary.lineage.idempotencyKey,
    sameRollbackKey: leftSummary.rollbackKey === rightSummary.rollbackKey,
    sameStatus: leftSummary.status === rightSummary.status,
    changedFields,
    subjectIds: diffStringLists(leftSummary.subjectIds, rightSummary.subjectIds),
    evidenceIds: diffStringLists(leftSummary.evidenceIds, rightSummary.evidenceIds),
    counterevidenceIds: diffStringLists(leftSummary.counterevidenceIds, rightSummary.counterevidenceIds),
    replaySuites: diffStringLists(leftSummary.replaySuites, rightSummary.replaySuites),
    summary: `${leftSummary.proposalClass} ${leftSummary.proposalId} → ${rightSummary.proposalId}: ${
      changedFields.length > 0 ? changedFields.join(", ") : "no material differences"
    }`,
  };
}


export type TeacherProposalReplayGateDimensionName =
  | "truth_invariants"
  | "attribution_floor"
  | "boundedness"
  | "reversibility";

export type TeacherProposalReplayGateReviewModeV1 = "promotable" | "shadow_only";

export interface TeacherProposalReplayGateDimensionV1 {
  name: TeacherProposalReplayGateDimensionName;
  summary: string;
  requirements: string[];
}

export interface TeacherProposalReplayGateV1 {
  proposalClass: ProposalClass;
  reviewMode: TeacherProposalReplayGateReviewModeV1;
  dimensions: {
    truthInvariants: TeacherProposalReplayGateDimensionV1;
    attributionFloor: TeacherProposalReplayGateDimensionV1;
    boundedness: TeacherProposalReplayGateDimensionV1;
    reversibility: TeacherProposalReplayGateDimensionV1;
  };
}

export const TEACHER_PROPOSAL_REVIEW_MODE_BY_CLASS_V1 = {
  compiler: "promotable",
  lint: "promotable",
  mutation: "shadow_only",
  forgetting: "shadow_only",
  correction: "shadow_only",
} as const satisfies Record<ProposalClass, TeacherProposalReplayGateReviewModeV1>;

export function describeTeacherProposalReplayGateReviewModeV1(
  proposalClass: ProposalClass,
): TeacherProposalReplayGateReviewModeV1 {
  return TEACHER_PROPOSAL_REVIEW_MODE_BY_CLASS_V1[proposalClass];
}

export function isTeacherProposalPromotableClassV1(
  proposalClass: ProposalClass,
): proposalClass is TeacherProposalPromotableClassV1 {
  return describeTeacherProposalReplayGateReviewModeV1(proposalClass) === "promotable";
}

const buildTeacherProposalReplayGate = (
  proposalClass: ProposalClass,
  focus: string,
): TeacherProposalReplayGateV1 => ({
  proposalClass,
  reviewMode: describeTeacherProposalReplayGateReviewModeV1(proposalClass),
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

export function summarizeTeacherProposalReplayOutcomesV1(
  outcomes: TeacherProposalReplayOutcomeV1[],
): TeacherProposalReplayOutcomeSummaryV1 {
  const resultCounts: TeacherProposalReplayOutcomeSummaryV1["resultCounts"] = {
    pass: 0,
    warn: 0,
    fail: 0,
  };
  const reviewModeCounts: TeacherProposalReplayOutcomeSummaryV1["reviewModeCounts"] = {
    promotable: 0,
    shadow_only: 0,
  };
  const sourceCounts: TeacherProposalReplayOutcomeSummaryV1["sourceCounts"] = {
    proposal_record: 0,
    proof_bundle: 0,
    derived: 0,
  };
  const replaySuites: string[] = [];

  for (const outcome of outcomes ?? []) {
    if (!outcome) {
      continue;
    }
    if (Object.prototype.hasOwnProperty.call(resultCounts, outcome.result)) {
      resultCounts[outcome.result] += 1;
    }
    if (Object.prototype.hasOwnProperty.call(reviewModeCounts, outcome.reviewMode)) {
      reviewModeCounts[outcome.reviewMode] += 1;
    }
    if (Object.prototype.hasOwnProperty.call(sourceCounts, outcome.source)) {
      sourceCounts[outcome.source] += 1;
    }
    replaySuites.push(outcome.replaySuite);
  }

  const uniqueReplaySuites = uniqueStrings(replaySuites);
  const replayOutcomeCount = outcomes.length;
  const summary = replayOutcomeCount === 0
    ? "No replay outcomes captured."
    : `Captured ${replayOutcomeCount} replay outcome${replayOutcomeCount === 1 ? "" : "s"} across ${uniqueReplaySuites.length} suite${uniqueReplaySuites.length === 1 ? "" : "s"} (${uniqueReplaySuites.join(", ") || "none"}); results pass=${resultCounts.pass}, warn=${resultCounts.warn}, fail=${resultCounts.fail}; review modes promotable=${reviewModeCounts.promotable}, shadow_only=${reviewModeCounts.shadow_only}; sources proposal_record=${sourceCounts.proposal_record}, proof_bundle=${sourceCounts.proof_bundle}, derived=${sourceCounts.derived}.`;

  return {
    replayOutcomeCount,
    replaySuites: uniqueReplaySuites,
    resultCounts,
    reviewModeCounts,
    sourceCounts,
    summary,
  };
}

export type TeacherCanaryRolloutSurfaceStateV1 = "shipped" | "target";

export interface TeacherCanaryRolloutPlanV1 {
  proposalClass: ProposalClass;
  surfaceState: TeacherCanaryRolloutSurfaceStateV1;
  rolloutMode: "off";
  enabled: false;
  candidatePackVersion?: number;
  candidatePackId?: string;
  shippedStateSummary: string;
  targetStateSummary: string;
  guardrails: string[];
}

const buildTeacherCanaryRolloutPlan = (
  proposalClass: ProposalClass,
  focus: string,
): TeacherCanaryRolloutPlanV1 => ({
  proposalClass,
  surfaceState: "target",
  rolloutMode: "off",
  enabled: false,
  shippedStateSummary: `${focus}: shipped runtime serves only promoted packs; no canary live rollout is shipped.`,
  targetStateSummary: `${focus}: the canary plan stays explicit, replayable, and off by default until a later tranche opts it in.`,
  guardrails: [
    "Keep the rollout plan target-state only until it is explicitly shipped.",
    "Default rolloutMode stays off.",
    "Do not use the canary plan to change live serving without separate replay and rollback proof.",
    "Bind any candidate pack by durable version or id, never by ad hoc display labels.",
  ],
});

export const TEACHER_CANARY_ROLLOUT_PLANS_V1: Record<ProposalClass, TeacherCanaryRolloutPlanV1> = {
  compiler: buildTeacherCanaryRolloutPlan("compiler", "Compiler lane"),
  lint: buildTeacherCanaryRolloutPlan("lint", "Lint lane"),
  mutation: buildTeacherCanaryRolloutPlan("mutation", "Mutation lane"),
  forgetting: buildTeacherCanaryRolloutPlan("forgetting", "Forgetting lane"),
  correction: buildTeacherCanaryRolloutPlan("correction", "Correction lane"),
} as const;

export function describeTeacherCanaryRolloutPlanV1(
  proposalClass: ProposalClass,
  candidatePackVersion?: number,
  candidatePackId?: string,
): TeacherCanaryRolloutPlanV1 {
  return {
    ...TEACHER_CANARY_ROLLOUT_PLANS_V1[proposalClass],
    candidatePackVersion,
    candidatePackId,
  };
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
  proofBundle?: TeacherProposalProofBundleV1;
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

export type TeacherV3ProofSurfaceState = "shipped" | "target";
export type TeacherV3ProofSurfacePhase = "before" | "after";
export type TeacherV3ProofSurfaceKind =
  | "runtime_truth"
  | "proof_truth"
  | "docs_truth"
  | "proposal_truth";

export interface TeacherV3ProofSurfaceRefV1 {
  id: string;
  state: TeacherV3ProofSurfaceState;
  phase: TeacherV3ProofSurfacePhase;
  kind: TeacherV3ProofSurfaceKind;
  source: string;
  note?: string;
}

export type TeacherV3ProofCheckKind = "token" | "latency" | "truth";
export type TeacherV3ProofCheckStatus = "pass" | "warn" | "fail";

export interface TeacherV3ProofCheckV1 {
  kind: TeacherV3ProofCheckKind;
  status: TeacherV3ProofCheckStatus;
  summary: string;
  evidenceSurfaceIds?: string[];
}

export type TeacherV3PublicationSafeArtifactKind =
  | "summary"
  | "status"
  | "metadata"
  | "surface-map"
  | "verdict";

export interface TeacherV3PublicationSafeArtifactV1 {
  artifactId: string;
  kind: TeacherV3PublicationSafeArtifactKind;
  path: string;
  redactions: string[];
  containsRawLogs: false;
}

export interface TeacherV3LiveProofRungV1 {
  rungId: "live-proof-rung-1";
  summary: string;
  beforeSurfaces: TeacherV3ProofSurfaceRefV1[];
  afterSurfaces: TeacherV3ProofSurfaceRefV1[];
  checks: TeacherV3ProofCheckV1[];
  publicationSafeArtifacts: TeacherV3PublicationSafeArtifactV1[];
  shippedStateNotes: string[];
  targetStateNotes: string[];
}

export interface TeacherV3LiveProofRungSummaryV1 {
  rungId: string;
  summary: string;
  before: {
    count: number;
    shippedCount: number;
    targetCount: number;
    ids: string[];
  };
  after: {
    count: number;
    shippedCount: number;
    targetCount: number;
    ids: string[];
  };
  checks: Record<TeacherV3ProofCheckKind, TeacherV3ProofCheckV1>;
  publicationSafeArtifacts: {
    count: number;
    ids: string[];
    kinds: TeacherV3PublicationSafeArtifactKind[];
    redactions: string[];
  };
}

function summarizeProofSurfaces(surfaces: TeacherV3ProofSurfaceRefV1[]) {
  return {
    count: surfaces.length,
    shippedCount: surfaces.filter((surface) => surface.state === "shipped").length,
    targetCount: surfaces.filter((surface) => surface.state === "target").length,
    ids: surfaces.map((surface) => surface.id),
  };
}

function getTeacherV3ProofCheck(
  checks: TeacherV3ProofCheckV1[],
  kind: TeacherV3ProofCheckKind,
): TeacherV3ProofCheckV1 {
  const check = checks.find((entry) => entry.kind === kind);
  if (!check) {
    throw new Error(`missing teacher v3 live-proof ${kind} check`);
  }
  return check;
}

function summarizePublicationSafeArtifacts(
  artifacts: TeacherV3PublicationSafeArtifactV1[],
): TeacherV3LiveProofRungSummaryV1["publicationSafeArtifacts"] {
  const unsafeArtifact = artifacts.find((artifact) => artifact.containsRawLogs !== false);
  if (unsafeArtifact) {
    throw new Error(`publication-safe artifact must not contain raw logs: ${unsafeArtifact.artifactId}`);
  }

  return {
    count: artifacts.length,
    ids: artifacts.map((artifact) => artifact.artifactId),
    kinds: artifacts.map((artifact) => artifact.kind),
    redactions: artifacts.flatMap((artifact) => artifact.redactions),
  };
}

export function summarizeTeacherV3LiveProofRungV1(
  rung: TeacherV3LiveProofRungV1,
): TeacherV3LiveProofRungSummaryV1 {
  if (rung.beforeSurfaces.length === 0) {
    throw new Error("teacher v3 live-proof rung requires before surfaces");
  }
  if (rung.afterSurfaces.length === 0) {
    throw new Error("teacher v3 live-proof rung requires after surfaces");
  }

  return {
    rungId: rung.rungId,
    summary: rung.summary,
    before: summarizeProofSurfaces(rung.beforeSurfaces),
    after: summarizeProofSurfaces(rung.afterSurfaces),
    checks: {
      token: getTeacherV3ProofCheck(rung.checks, "token"),
      latency: getTeacherV3ProofCheck(rung.checks, "latency"),
      truth: getTeacherV3ProofCheck(rung.checks, "truth"),
    },
    publicationSafeArtifacts: summarizePublicationSafeArtifacts(rung.publicationSafeArtifacts),
  };
}
