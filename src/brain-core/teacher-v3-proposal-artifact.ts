import { createHash } from "node:crypto";
import type {
  EvidenceRefV1,
  TeacherProposal,
  TeacherProposalProofLinkV1,
  TeacherProposalProofLinkageV1,
  TeacherProposalPromotableClassV1,
  TeacherProposalReplayHookV1,
  TeacherProposalReportArtifactV1,
} from "./teacher-v3-contracts.js";
import {
  describeTeacherProposalReplayGate,
  describeTeacherProposalReplayGateMatrixV1,
  isTeacherProposalPromotableClassV1,
  summarizeTeacherProposalProofBundleV1,
  summarizeTeacherProposalV1,
} from "./teacher-v3-contracts.js";

export const TEACHER_V3_PROPOSAL_ARTIFACT_CONTRACT = "teacher_v3_proposal_artifact.v1";

export interface BuildTeacherProposalReportArtifactInputV1 {
  proposal: TeacherProposal;
  artifactId?: string;
  recommendations?: string[];
}

function sha256Text(text: string): string {
  return `sha256:${createHash("sha256").update(text, "utf8").digest("hex")}`;
}

function normalizeText(value: string | null | undefined): string | null {
  if (typeof value !== "string") {
    return null;
  }
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
}

function uniqueStrings(values: string[]): string[] {
  return [...new Set(values)];
}

function toEvidenceLink(ref: EvidenceRefV1): TeacherProposalProofLinkV1 {
  return {
    refId: ref.evidenceId ?? ref.sourceId,
    kind: ref.derivation ?? ref.sourceKind,
    path: ref.sourceId,
  };
}

function replayCandidateLabel(proposal: TeacherProposal): string {
  const replaySummary = proposal.replaySummary;
  if (!replaySummary) {
    return "unbound";
  }
  return replaySummary.candidatePackId ?? String(replaySummary.candidatePackVersion ?? "unbound");
}

export function buildTeacherProposalReplayHookV1(
  proposal: TeacherProposal,
): TeacherProposalReplayHookV1 {
  const replaySummaryId = normalizeText(proposal.replaySummary?.replayId);
  const replaySuites = uniqueStrings([
    ...(proposal.replaySuites ?? []),
    ...(proposal.replaySuiteIds ?? []),
  ]);

  if (replaySummaryId) {
    return {
      replayReady: true,
      replaySummaryId,
      replaySuites,
      placeholder: false,
      summary: `Replay summary ${replaySummaryId} is attached for ${replaySuites.length > 0 ? replaySuites.join(", ") : "no named suite"} on candidate pack ${replayCandidateLabel(proposal)}.`,
    };
  }

  return {
    replayReady: false,
    replaySummaryId: null,
    replaySuites,
    placeholder: true,
    summary: replaySuites.length > 0
      ? `Replay summary is not attached yet; keep the report-only placeholder over ${replaySuites.join(", ")}.`
      : "Replay summary is not attached yet; keep this proposal report-only until replay hooks are added.",
  };
}

export function buildTeacherProposalProofLinkageV1(
  proposal: TeacherProposal,
): TeacherProposalProofLinkageV1 {
  const proofBundleSummary = proposal.proofBundle ? summarizeTeacherProposalProofBundleV1(proposal.proofBundle) : undefined;
  const replaySummaryId = normalizeText(proposal.replaySummary?.replayId);
  const rollbackBound = proposal.proofBundle ? proposal.proofBundle.rollbackKey === proposal.rollbackKey : false;
  const proofLinked = rollbackBound && proofBundleSummary !== undefined;

  return {
    rollbackKey: proposal.rollbackKey,
    proofBundleId: proofBundleSummary?.bundleId ?? null,
    proofBundleStatus: proofBundleSummary?.status ?? null,
    replaySummaryId,
    rollbackBound,
    proofLinked,
    surfaceIds: proofBundleSummary?.surfaceIds ?? [],
    summary: proofBundleSummary
      ? proofLinked
        ? `Rollback key ${proposal.rollbackKey} remains bound to proof bundle ${proofBundleSummary.bundleId} across ${proofBundleSummary.surfaceCount} surfaced item(s).`
        : `Proof bundle ${proofBundleSummary.bundleId} is attached, but rollback binding to ${proposal.rollbackKey} is incomplete.`
      : `No proof bundle is attached yet; rollback key ${proposal.rollbackKey} remains report-only until proof linkage lands.`,
  };
}

function buildRecommendations(params: {
  proposalClass: TeacherProposalPromotableClassV1;
  replayHook: TeacherProposalReplayHookV1;
  proofLinkage: TeacherProposalProofLinkageV1;
  recommendations?: string[];
}): string[] {
  if (Array.isArray(params.recommendations) && params.recommendations.length > 0) {
    return uniqueStrings(params.recommendations.map((value) => value.trim()).filter(Boolean)).slice(0, 4);
  }

  return uniqueStrings([
    params.replayHook.placeholder
      ? "Attach a candidate-pack replay summary or keep the replay placeholder explicit before any promotion step."
      : null,
    params.proofLinkage.proofLinked
      ? null
      : params.proofLinkage.proofBundleId
        ? "Repair rollback/proof binding before treating this proposal as more than a report-only review surface."
        : "Attach a proof bundle before promotion so rollback/proof linkage is reviewable.",
    params.proposalClass === "lint"
      ? "Keep lint findings report-only, bounded, and tied to cited evidence plus counterevidence."
      : "Keep compiler outputs derived and review-only until a later tranche materializes them into a separate promoted artifact path.",
  ].filter((value): value is string => Boolean(value))).slice(0, 4);
}

function buildArtifactSummary(params: {
  proposalId: string;
  proposalClass: TeacherProposalPromotableClassV1;
  replayHook: TeacherProposalReplayHookV1;
  proofLinkage: TeacherProposalProofLinkageV1;
}): string {
  const replayLabel = params.replayHook.replayReady
    ? `replay ${params.replayHook.replaySummaryId}`
    : "a replay placeholder";

  if (params.proofLinkage.proofLinked) {
    return `${params.proposalClass} proposal ${params.proposalId} stays report-only with ${replayLabel} and rollback/proof linkage to ${params.proofLinkage.proofBundleId}.`;
  }

  return `${params.proposalClass} proposal ${params.proposalId} stays report-only with ${replayLabel}; ${params.proofLinkage.summary}`;
}

export function buildTeacherProposalReportArtifactV1(
  input: BuildTeacherProposalReportArtifactInputV1,
): TeacherProposalReportArtifactV1 {
  const proposal = input.proposal;
  if (!isTeacherProposalPromotableClassV1(proposal.proposalClass)) {
    throw new Error(`teacher proposal artifact only supports compiler/lint report lanes; received ${proposal.proposalClass}`);
  }

  const artifactId = normalizeText(input.artifactId) ?? `teacher-v3-proposal-${proposal.proposalId}`;
  const proposalClass = proposal.proposalClass as TeacherProposalPromotableClassV1;
  const proposalSummary = summarizeTeacherProposalV1(proposal);
  const replayHook = buildTeacherProposalReplayHookV1(proposal);
  const proofLinkage = buildTeacherProposalProofLinkageV1(proposal);
  const proofBundleSummary = proposal.proofBundle ? summarizeTeacherProposalProofBundleV1(proposal.proofBundle) : undefined;
  const replayGate = proposal.replayGate ?? describeTeacherProposalReplayGate(proposalClass);
  const gateMatrix = proposal.proofBundle?.replayGateMatrix ?? describeTeacherProposalReplayGateMatrixV1({
    proposalId: proposal.proposalId,
    proposalClass,
    rollbackKey: proposal.rollbackKey,
    proofBundleId: proofBundleSummary?.bundleId,
    replaySummaryId: proposal.replaySummary?.replayId,
    replaySummary: proposal.replaySummary
      ? {
        status: proposal.replaySummary.status,
        reviewMode: proposal.replaySummary.reviewMode,
        beforeScore: proposal.replaySummary.beforeScore,
        afterScore: proposal.replaySummary.afterScore,
        scoreDelta: proposal.replaySummary.scoreDelta,
        candidatePackId: proposal.replaySummary.candidatePackId,
        candidatePackVersion: proposal.replaySummary.candidatePackVersion,
        summary: proposal.replaySummary.summary,
      }
      : null,
    replayOutcomeSummary: proofBundleSummary?.replayOutcomeSummary ?? null,
    replaySuites: proposal.replaySuites,
    surfaceMap: proposal.proofBundle?.surfaceMap ?? [],
    evidenceLinks: proposal.proofBundle?.evidenceLinks ?? proposal.evidence.map(toEvidenceLink),
    counterevidenceLinks: proposal.proofBundle?.counterevidenceLinks ?? (proposal.counterevidence ?? []).map(toEvidenceLink),
  });
  const recommendations = buildRecommendations({
    proposalClass,
    replayHook,
    proofLinkage,
    recommendations: input.recommendations,
  });
  const summary = buildArtifactSummary({
    proposalId: proposal.proposalId,
    proposalClass,
    replayHook,
    proofLinkage,
  });

  const baseArtifact: Omit<TeacherProposalReportArtifactV1, "artifactRef"> = {
    contract: TEACHER_V3_PROPOSAL_ARTIFACT_CONTRACT,
    artifactId,
    proposalId: proposal.proposalId,
    proposalClass,
    reviewMode: replayGate.reviewMode,
    reviewDiscipline: "report_only",
    status: proposal.status,
    summary,
    proposal: proposalSummary,
    replayGate,
    gateMatrix,
    evidenceRefs: [...proposal.evidence],
    counterevidenceRefs: [...(proposal.counterevidence ?? [])],
    replayHook,
    replaySummary: proposal.replaySummary,
    proofLinkage,
    proofBundleSummary,
    attachedArtifacts: [...(proposal.artifacts ?? [])],
    recommendations,
    createdAt: proposal.createdAt,
    updatedAt: proposal.freshnessTs ?? proposal.resolvedAt ?? proposal.createdAt,
  };

  return {
    ...baseArtifact,
    artifactRef: {
      artifactId,
      kind: "proposal_review_bundle",
      contentHash: sha256Text(JSON.stringify(baseArtifact)),
    },
  };
}

function renderEvidenceLines(evidenceRefs: EvidenceRefV1[]): string[] {
  if (evidenceRefs.length === 0) {
    return ["- none"];
  }
  return evidenceRefs.map((ref) =>
    `- \`${ref.evidenceId ?? ref.sourceId}\` -> \`${ref.sourceId}\`${ref.excerpt ? `: ${ref.excerpt}` : ""}`,
  );
}

export function renderTeacherProposalReportArtifactMarkdownV1(
  artifact: TeacherProposalReportArtifactV1,
): string {
  const lines = [
    "# Teacher v3 proposal artifact",
    "",
    `- artifact: \`${artifact.artifactId}\``,
    `- proposal: \`${artifact.proposalId}\` (${artifact.proposalClass}, ${artifact.status})`,
    `- review mode: **${artifact.reviewMode}**`,
    `- discipline: **${artifact.reviewDiscipline}**`,
    `- summary: ${artifact.summary}`,
    `- rollback key: \`${artifact.proofLinkage.rollbackKey}\``,
    `- proof bundle: \`${artifact.proofLinkage.proofBundleId ?? "missing"}\``,
    `- replay hook: ${artifact.replayHook.summary}`,
    "",
    "## Evidence refs",
    ...renderEvidenceLines(artifact.evidenceRefs),
    "",
    "## Counterevidence refs",
    ...renderEvidenceLines(artifact.counterevidenceRefs),
    "",
    "## Replay",
    `- replay summary id: \`${artifact.replayHook.replaySummaryId ?? "missing"}\``,
    `- placeholder: ${artifact.replayHook.placeholder ? "yes" : "no"}`,
    `- replay suites: ${artifact.replayHook.replaySuites.length > 0 ? artifact.replayHook.replaySuites.join(", ") : "none"}`,
    artifact.replaySummary ? `- replay summary: ${artifact.replaySummary.summary}` : null,
    `- gate matrix: ${artifact.gateMatrix.summary}`,
    "",
    "## Proof linkage",
    `- summary: ${artifact.proofLinkage.summary}`,
    `- rollback bound: ${artifact.proofLinkage.rollbackBound ? "yes" : "no"}`,
    `- proof linked: ${artifact.proofLinkage.proofLinked ? "yes" : "no"}`,
    `- surfaced ids: ${artifact.proofLinkage.surfaceIds.length > 0 ? artifact.proofLinkage.surfaceIds.join(", ") : "none"}`,
    "",
    "## Attached artifacts",
    ...(artifact.attachedArtifacts.length > 0
      ? artifact.attachedArtifacts.map((ref) => `- \`${ref.artifactId}\` (${ref.kind}) — ${ref.contentHash}`)
      : ["- none"]),
    "",
    "## Recommendations",
    ...artifact.recommendations.map((recommendation) => `- ${recommendation}`),
  ].filter((line): line is string => line !== null);

  return `${lines.join("\n")}\n`;
}
