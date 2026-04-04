import { createHash } from "node:crypto";
import type { BrainGraph } from "./graph.js";
import type { Pack } from "./types.js";
import type {
  ProposalClass,
  ProposalStatus,
  TeacherProposal,
  TeacherProposalClassReplaySummaryV1,
  TeacherProposalReplayHealthSummaryV1,
  TeacherProposalReplayStateSnapshotV1,
  TeacherProposalReplaySummaryV1,
} from "./teacher-v3-contracts.js";
import {
  describeTeacherProposalReplayGateReviewModeV1,
  isTeacherProposalPromotableClassV1,
} from "./teacher-v3-contracts.js";

export interface TeacherProposalReplayCandidatePackStateV1 {
  candidatePack: Pack | null;
  candidatePackId?: string | null;
  candidateGraph?: BrainGraph | null;
}

export interface TeacherProposalReplayInputV1 {
  proposal: TeacherProposal;
  candidateState?: TeacherProposalReplayCandidatePackStateV1;
  evaluatedAt?: string;
}

function clamp01(value: number): number {
  if (Number.isNaN(value) || !Number.isFinite(value)) {
    return 0;
  }
  return Math.min(1, Math.max(0, value));
}

function normalizeText(value: string | null | undefined): string | null {
  if (typeof value !== "string") {
    return null;
  }
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
}

function toFiniteNumber(value: unknown, fallback = 0): number {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : fallback;
}

function parseJson<T>(value: string | null | undefined, fallback: T): T {
  if (typeof value !== "string" || value.trim().length === 0) {
    return fallback;
  }
  try {
    return JSON.parse(value) as T;
  } catch {
    return fallback;
  }
}

function cloneJson<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

function summarizeHealth(pack: Pack | null): TeacherProposalReplayHealthSummaryV1 {
  if (!pack) {
    return {
      firedPerQuery: null,
      dormantPercent: null,
      orphanCount: null,
    };
  }

  const health = parseJson<Record<string, unknown>>(pack.healthJson, {});
  return {
    firedPerQuery: Number.isFinite(toFiniteNumber(health.firedPerQuery, NaN)) ? toFiniteNumber(health.firedPerQuery, NaN) : null,
    dormantPercent: Number.isFinite(toFiniteNumber(health.dormantPercent, NaN)) ? toFiniteNumber(health.dormantPercent, NaN) : null,
    orphanCount: Number.isFinite(toFiniteNumber(health.orphanCount, NaN)) ? toFiniteNumber(health.orphanCount, NaN) : null,
  };
}

function scoreHealthQuality(health: TeacherProposalReplayHealthSummaryV1): number {
  const firedPerQuery = health.firedPerQuery ?? 0;
  const dormantPercent = health.dormantPercent ?? 1;
  const orphanCount = health.orphanCount ?? 10;

  const firedScore = clamp01(firedPerQuery / 2);
  const dormantScore = clamp01(1 - dormantPercent);
  const orphanScore = clamp01(1 - orphanCount / 10);
  return clamp01((firedScore * 0.5) + (dormantScore * 0.25) + (orphanScore * 0.25));
}

function hashBrainGraphState(graph: BrainGraph | null | undefined): string | null {
  if (!graph) {
    return null;
  }

  const snapshot = {
    nodes: graph.getAllNodes()
      .map((node) => ({
        id: node.id,
        kind: node.kind,
        trust: node.trust,
        sourceUri: node.sourceUri,
        tags: [...node.tags].sort(),
        tokenCount: node.tokenCount,
        metadata: node.metadata,
        createdAt: node.createdAt,
        updatedAt: node.updatedAt,
      }))
      .sort((left, right) => left.id.localeCompare(right.id)),
    edges: graph.getAllEdges()
      .map((edge) => ({
        source: edge.source,
        target: edge.target,
        kind: edge.kind,
        weight: edge.weight,
        prior: edge.prior,
        metadata: edge.metadata,
        decayedAt: edge.decayedAt,
        createdAt: edge.createdAt,
      }))
      .sort((left, right) => {
        const bySource = left.source.localeCompare(right.source);
        if (bySource !== 0) return bySource;
        const byTarget = left.target.localeCompare(right.target);
        if (byTarget !== 0) return byTarget;
        return left.kind.localeCompare(right.kind);
      }),
    seedWeights: graph.getAllSeedWeights()
      .map((entry) => ({ nodeId: entry.nodeId, weight: entry.weight }))
      .sort((left, right) => left.nodeId.localeCompare(right.nodeId)),
    stopLocalWeights: graph.getAllStopLocalWeights()
      .map((entry) => ({ sourceNodeId: entry.sourceNodeId, weight: entry.weight }))
      .sort((left, right) => left.sourceNodeId.localeCompare(right.sourceNodeId)),
  };

  return `sha256:${createHash("sha256").update(JSON.stringify(snapshot), "utf8").digest("hex")}`;
}

function buildReplayStateSnapshot(params: {
  phase: "before" | "after";
  surfaceState: "shipped" | "target";
  packVersion: number | null;
  packId: string | null;
  graphHash: string | null;
  graph: BrainGraph | null | undefined;
  pack: Pack | null;
  notes: string[];
}): TeacherProposalReplayStateSnapshotV1 {
  const nodeCount = params.graph ? params.graph.getAllNodes().length : null;
  const edgeCount = params.graph ? params.graph.getAllEdges().length : null;
  return {
    phase: params.phase,
    surfaceState: params.surfaceState,
    packVersion: params.packVersion,
    packId: params.packId,
    graphHash: params.graphHash,
    nodeCount,
    edgeCount,
    health: summarizeHealth(params.pack),
    notes: params.notes,
  };
}

function scoreBaseReplay(proposal: TeacherProposal, reviewMode: ReturnType<typeof describeTeacherProposalReplayGateReviewModeV1>): number {
  const basePackScore = proposal.lineage.basePackVersion !== undefined && proposal.lineage.basePackVersion !== null ? 0.14 : 0;
  const baseGraphScore = normalizeText(proposal.lineage.baseGraphHash) ? 0.14 : 0;
  const evidenceScore = proposal.evidence.length > 0 ? 0.15 : 0;
  const rollbackScore = normalizeText(proposal.rollbackKey) ? 0.12 : 0;
  const replaySuitesScore = proposal.replaySuites.length > 0 ? 0.10 : 0;
  const reviewModeScore = reviewMode === "promotable" ? 0.14 : 0.06;
  const confidenceScore = clamp01(proposal.confidence) * 0.10;
  const subjectScore = proposal.subjectIds.length > 0 ? Math.min(0.15, 0.08 + (proposal.subjectIds.length * 0.02)) : 0.02;
  const counterevidenceScore = proposal.proposalClass === "lint"
    ? Math.min(0.10, (proposal.counterevidence?.length ?? 0) * 0.04)
    : 0.03;

  return clamp01(
    (
      basePackScore
      + baseGraphScore
      + evidenceScore
      + rollbackScore
      + replaySuitesScore
      + reviewModeScore
      + confidenceScore
      + subjectScore
      + counterevidenceScore
    ) * 0.75,
  );
}

function scoreCandidateReplay(params: {
  proposal: TeacherProposal;
  candidateState?: TeacherProposalReplayCandidatePackStateV1;
}): number {
  const candidatePack = params.candidateState?.candidatePack ?? null;
  const candidateGraph = params.candidateState?.candidateGraph ?? null;
  const graphHash = hashBrainGraphState(candidateGraph);

  if (!candidatePack && !candidateGraph) {
    return 0;
  }

  const packScore = candidatePack ? 0.12 : 0;
  const graphScore = candidateGraph ? 0.12 : 0;
  const hashScore = graphHash ? 0.06 : 0;
  const freshnessScore = candidatePack && params.proposal.lineage.basePackVersion !== undefined && params.proposal.lineage.basePackVersion !== null
    ? candidatePack.version >= params.proposal.lineage.basePackVersion
      ? 0.08
      : 0.02
    : 0.04;
  const divergenceScore = graphHash && params.proposal.lineage.baseGraphHash
    ? graphHash !== params.proposal.lineage.baseGraphHash
      ? 0.08
      : 0.03
    : graphHash
      ? 0.05
      : 0;
  const healthScore = candidatePack ? scoreHealthQuality(summarizeHealth(candidatePack)) * 0.16 : 0;
  const candidatePackBoundScore = params.candidateState?.candidatePackId ? 0.08 : (candidatePack ? 0.04 : 0);

  return clamp01(packScore + graphScore + hashScore + freshnessScore + divergenceScore + healthScore + candidatePackBoundScore);
}

function buildCompilerReplaySummary(params: {
  proposal: TeacherProposal;
  candidatePackVersion: number | null;
  candidatePackId: string | null;
  candidateGraphHash: string | null;
  beforeScore: number;
  afterScore: number;
  status: ProposalStatus;
}): TeacherProposalClassReplaySummaryV1 {
  return {
    kind: "compiler",
    reviewMode: "promotable",
    promotionDiscipline: "promotable",
    subjectCount: params.proposal.subjectIds.length,
    evidenceCount: params.proposal.evidence.length,
    counterevidenceCount: params.proposal.counterevidence?.length ?? 0,
    replaySuites: [...params.proposal.replaySuites],
    candidatePackVersion: params.candidatePackVersion,
    candidatePackId: params.candidatePackId,
    candidateGraphHash: params.candidateGraphHash,
    summary: params.status === "promotable"
      ? `Compiler replay is promotable on candidate pack ${params.candidatePackId ?? params.candidatePackVersion ?? "unbound"}; evidence-backed lineage stays intact and the candidate graph is distinct from base state.`
      : `Compiler replay stays shadow-scored on candidate pack ${params.candidatePackId ?? params.candidatePackVersion ?? "unbound"}; candidate binding or replay evidence is still incomplete.`,
    notes: [
      `basePackVersion=${params.proposal.lineage.basePackVersion ?? "none"}`,
      `baseGraphHash=${params.proposal.lineage.baseGraphHash ?? "none"}`,
      `beforeScore=${params.beforeScore.toFixed(3)}`,
      `afterScore=${params.afterScore.toFixed(3)}`,
      `scoreDelta=${(params.afterScore - params.beforeScore).toFixed(3)}`,
      params.status === "promotable"
        ? "compiler proposals may be evaluated as promotable, but they are not auto-promoted"
        : "promotion remains blocked until the replay summary binds real candidate state",
    ],
  };
}

function buildLintReplaySummary(params: {
  proposal: TeacherProposal;
  candidatePackVersion: number | null;
  candidatePackId: string | null;
  candidateGraphHash: string | null;
  beforeScore: number;
  afterScore: number;
  status: ProposalStatus;
}): TeacherProposalClassReplaySummaryV1 {
  return {
    kind: "lint",
    reviewMode: "promotable",
    promotionDiscipline: "promotable",
    subjectCount: params.proposal.subjectIds.length,
    evidenceCount: params.proposal.evidence.length,
    counterevidenceCount: params.proposal.counterevidence?.length ?? 0,
    replaySuites: [...params.proposal.replaySuites],
    candidatePackVersion: params.candidatePackVersion,
    candidatePackId: params.candidatePackId,
    candidateGraphHash: params.candidateGraphHash,
    summary: params.status === "promotable"
      ? `Lint replay is promotable on candidate pack ${params.candidatePackId ?? params.candidatePackVersion ?? "unbound"}; the bounded report-only review preserved counterevidence and replay discipline.`
      : `Lint replay stays shadow-scored on candidate pack ${params.candidatePackId ?? params.candidatePackVersion ?? "unbound"}; report-only evidence binding is still incomplete.`,
    notes: [
      `basePackVersion=${params.proposal.lineage.basePackVersion ?? "none"}`,
      `baseGraphHash=${params.proposal.lineage.baseGraphHash ?? "none"}`,
      `beforeScore=${params.beforeScore.toFixed(3)}`,
      `afterScore=${params.afterScore.toFixed(3)}`,
      `scoreDelta=${(params.afterScore - params.beforeScore).toFixed(3)}`,
      params.status === "promotable"
        ? "lint proposals may be evaluated as promotable, but they remain report-only until separately promoted"
        : "promotion remains blocked until the replay summary binds real candidate state",
    ],
  };
}

function buildShadowReplaySummary(params: {
  proposal: TeacherProposal;
  candidatePackVersion: number | null;
  candidatePackId: string | null;
  candidateGraphHash: string | null;
  beforeScore: number;
  afterScore: number;
  status: ProposalStatus;
}): TeacherProposalClassReplaySummaryV1 {
  return {
    kind: params.proposal.proposalClass as Exclude<ProposalClass, "compiler" | "lint">,
    reviewMode: "shadow_only",
    promotionDiscipline: "shadow_only",
    subjectCount: params.proposal.subjectIds.length,
    evidenceCount: params.proposal.evidence.length,
    counterevidenceCount: params.proposal.counterevidence?.length ?? 0,
    replaySuites: [...params.proposal.replaySuites],
    candidatePackVersion: params.candidatePackVersion,
    candidatePackId: params.candidatePackId,
    candidateGraphHash: params.candidateGraphHash,
    summary: `Shadow-only replay for ${params.proposal.proposalClass} remains non-promotable on candidate pack ${params.candidatePackId ?? params.candidatePackVersion ?? "unbound"}.`,
    notes: [
      `basePackVersion=${params.proposal.lineage.basePackVersion ?? "none"}`,
      `baseGraphHash=${params.proposal.lineage.baseGraphHash ?? "none"}`,
      `beforeScore=${params.beforeScore.toFixed(3)}`,
      `afterScore=${params.afterScore.toFixed(3)}`,
      `scoreDelta=${(params.afterScore - params.beforeScore).toFixed(3)}`,
      params.status === "promotable"
        ? "status remains promotable only as a review-mode label; promotion still requires the correct lane"
        : "shadow-only proposal classes stay off the promotable path",
    ],
  };
}

export function buildTeacherProposalReplaySummaryV1(
  input: TeacherProposalReplayInputV1,
): TeacherProposalReplaySummaryV1 {
  const proposal = input.proposal;
  const reviewMode = describeTeacherProposalReplayGateReviewModeV1(proposal.proposalClass);
  const candidatePack = input.candidateState?.candidatePack ?? null;
  const candidateGraph = input.candidateState?.candidateGraph ?? null;
  const candidatePackId = normalizeText(input.candidateState?.candidatePackId)
    ?? (candidatePack ? `pack_${candidatePack.version}` : null);
  const candidatePackVersion = candidatePack?.version ?? null;
  const candidateGraphHash = hashBrainGraphState(candidateGraph);
  const beforeScore = scoreBaseReplay(proposal, reviewMode);
  const candidateScore = scoreCandidateReplay({ proposal, candidateState: input.candidateState });
  const afterScore = clamp01(beforeScore + candidateScore);
  const status: ProposalStatus = isTeacherProposalPromotableClassV1(proposal.proposalClass)
    && candidatePack !== null
    && afterScore >= beforeScore
      ? "promotable"
      : "shadow_scored";
  const replayId = `treplay_${createHash("sha256")
    .update(JSON.stringify({
      proposalId: proposal.proposalId,
      proposalClass: proposal.proposalClass,
      basePackVersion: proposal.lineage.basePackVersion ?? null,
      baseGraphHash: proposal.lineage.baseGraphHash ?? null,
      candidatePackVersion,
      candidatePackId,
      candidateGraphHash,
      reviewMode,
    }), "utf8")
    .digest("hex")
    .slice(0, 12)}`;

  const before = buildReplayStateSnapshot({
    phase: "before",
    surfaceState: "shipped",
    packVersion: proposal.lineage.basePackVersion ?? null,
    packId: null,
    graphHash: proposal.lineage.baseGraphHash ?? null,
    graph: null,
    pack: null,
    notes: [
      "proposal evaluated against base lineage before candidate state replay",
      `reviewMode=${reviewMode}`,
    ],
  });
  const after = buildReplayStateSnapshot({
    phase: "after",
    surfaceState: "target",
    packVersion: candidatePackVersion,
    packId: candidatePackId,
    graphHash: candidateGraphHash,
    graph: candidateGraph,
    pack: candidatePack,
    notes: candidatePack !== null || candidateGraph !== null
      ? [
        "candidate state bound to a real pack / graph surface",
        candidatePackId ? `candidatePackId=${candidatePackId}` : "candidate pack id derived from pack version",
      ]
      : ["candidate state unavailable; replay remained synthetic"],
  });

  const classSummary = proposal.proposalClass === "compiler"
    ? buildCompilerReplaySummary({
      proposal,
      candidatePackVersion,
      candidatePackId,
      candidateGraphHash,
      beforeScore,
      afterScore,
      status,
    })
    : proposal.proposalClass === "lint"
      ? buildLintReplaySummary({
        proposal,
        candidatePackVersion,
        candidatePackId,
        candidateGraphHash,
        beforeScore,
        afterScore,
        status,
      })
      : buildShadowReplaySummary({
        proposal,
        candidatePackVersion,
        candidatePackId,
        candidateGraphHash,
        beforeScore,
        afterScore,
        status,
      });

  const candidateLabel = candidatePackId ?? candidatePackVersion ?? "unbound";
  return {
    replayId,
    proposalId: proposal.proposalId,
    proposalClass: proposal.proposalClass,
    status,
    reviewMode,
    basePackVersion: proposal.lineage.basePackVersion ?? null,
    baseGraphHash: proposal.lineage.baseGraphHash ?? null,
    candidatePackVersion,
    candidatePackId,
    candidateGraphHash,
    beforeScore,
    afterScore,
    scoreDelta: Number((afterScore - beforeScore).toFixed(6)),
    before,
    after,
    classSummary,
    summary: `${proposal.proposalClass} replay ${status === "promotable" ? "accepted" : "remained shadow-scored"} on ${candidateLabel}; before=${beforeScore.toFixed(3)} after=${afterScore.toFixed(3)} delta=${(afterScore - beforeScore).toFixed(3)}`,
    createdAt: input.evaluatedAt ?? new Date().toISOString(),
    updatedAt: input.evaluatedAt ?? new Date().toISOString(),
  };
}

export function cloneTeacherProposalReplaySummaryV1(
  replaySummary: TeacherProposalReplaySummaryV1 | null | undefined,
): TeacherProposalReplaySummaryV1 | undefined {
  return replaySummary ? cloneJson(replaySummary) : undefined;
}

export { hashBrainGraphState };
