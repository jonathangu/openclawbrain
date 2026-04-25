import type { BrainGraph } from "./graph.js";
import {
  applyShadowMutationProposalToState,
  createShadowCandidateState,
} from "./shadow-application.js";
import type { BrainNode, MutationProposal } from "./types.js";
import type {
  EvidenceRefV1,
  ProposalExpectedEffectV1,
  TeacherProposal,
  TeacherProposalReplaySummaryV1,
  TeacherProposalReplayStateSnapshotV1,
} from "./teacher-v3-contracts.js";
import {
  describeTeacherProposalReplayGate,
  describeTeacherProposalReplayGateReviewModeV1,
  normalizeTeacherProposalV1,
  summarizeTeacherProposalV1,
  type TeacherProposalReplayGateReviewModeV1,
} from "./teacher-v3-contracts.js";
import { hashBrainGraphState } from "./teacher-v3-replay.js";
import {
  summarizeTeacherMutationShadowReplayV1,
  type TeacherMutationShadowReplaySummaryV1,
} from "./teacher-v3-shadow-replay.js";

export const TEACHER_GRAPH_MAINTENANCE_LIFECYCLE_CONTRACT = "teacher_v3_graph_maintenance_lifecycle.v1";

export interface BuildTeacherAddEdgeGraphMaintenanceProposalInputV1 {
  proposalId: string;
  sourceNodeId: string;
  targetNodeId: string;
  evidence: EvidenceRefV1[];
  subjectIds?: string[];
  expectedEffect?: ProposalExpectedEffectV1;
  confidence?: number;
  replaySuites?: string[];
  rollbackKey: string;
  lineage: TeacherProposal["lineage"];
  createdAt: string;
}

export interface TeacherGraphMaintenanceReplayResultV1 {
  proposal: TeacherProposal;
  replaySummary: TeacherProposalReplaySummaryV1;
  shadowReplay: TeacherMutationShadowReplaySummaryV1;
  candidateGraph: BrainGraph;
}

export interface TeacherGraphMaintenanceLifecycleSummaryV1 {
  contract: typeof TEACHER_GRAPH_MAINTENANCE_LIFECYCLE_CONTRACT;
  proposalId: string;
  proposalClass: TeacherProposal["proposalClass"];
  proposalKind: string;
  lifecycleState: NonNullable<TeacherProposal["lifecycleState"]>;
  status: TeacherProposal["status"];
  safeClassMode: TeacherProposalReplayGateReviewModeV1;
  subjectIds: string[];
  evidenceIds: string[];
  expectedEffect: ProposalExpectedEffectV1 | null;
  replaySuites: string[];
  rollbackKey: string;
  replaySummaryId: string;
  replayOutcome: TeacherMutationShadowReplaySummaryV1["replayOutcome"];
  rollbackRestored: boolean;
  promotionBypass: false;
  liveSelfEditingEnabled: false;
  summary: string;
  boundary: string;
}

function readStringPayload(payload: unknown, key: string): string | null {
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    return null;
  }
  const value = (payload as Record<string, unknown>)[key];
  return typeof value === "string" && value.trim().length > 0 ? value.trim() : null;
}

function replaySnapshot(params: {
  phase: "before" | "after";
  graph: BrainGraph;
  graphHash: string | null;
  notes: string[];
}): TeacherProposalReplayStateSnapshotV1 {
  return {
    phase: params.phase,
    surfaceState: params.phase === "before" ? "shipped" : "target",
    packVersion: null,
    packId: null,
    graphHash: params.graphHash,
    nodeCount: params.graph.getAllNodes().length,
    edgeCount: params.graph.getAllEdges().length,
    health: {
      firedPerQuery: null,
      dormantPercent: null,
      orphanCount: null,
    },
    notes: params.notes,
  };
}

function nodeExists(graph: BrainGraph, nodeId: string): boolean {
  return graph.getNode(nodeId as BrainNode["id"]) !== undefined;
}

export function buildTeacherAddEdgeGraphMaintenanceProposalV1(
  input: BuildTeacherAddEdgeGraphMaintenanceProposalInputV1,
): TeacherProposal {
  const replaySuites = input.replaySuites && input.replaySuites.length > 0
    ? [...new Set(input.replaySuites)]
    : ["teacher-v3-graph-maintenance-shadow", "teacher-v3-rollback-smoke"];

  return normalizeTeacherProposalV1({
    schemaVersion: 1,
    proposalId: input.proposalId,
    proposalClass: "mutation",
    proposalKind: "add_edge",
    lane: "mutation",
    status: "proposed",
    lifecycleState: "proposed",
    safeClassMode: "shadow_only",
    lineage: {
      ...input.lineage,
      proposalClass: "mutation",
    },
    subjectIds: input.subjectIds ?? [input.sourceNodeId, input.targetNodeId],
    evidence: input.evidence,
    payload: {
      kind: "add_edge",
      sourceNodeId: input.sourceNodeId,
      targetNodeId: input.targetNodeId,
      edgeKind: "learned",
      safeClassMode: "shadow_only",
      liveSelfEditingEnabled: false,
    },
    expectedEffect: input.expectedEffect ?? {
      retrieval: "better",
      truthRisk: "low",
      tokenBudget: "same",
    },
    confidence: input.confidence ?? 0.8,
    replaySuites,
    replaySuiteIds: replaySuites,
    rollbackKey: input.rollbackKey,
    replayGate: describeTeacherProposalReplayGate("mutation"),
    createdAt: input.createdAt,
    freshnessTs: input.createdAt,
  });
}

export function assertTeacherGraphMaintenanceProposalSafetyV1(proposal: TeacherProposal): void {
  const normalized = normalizeTeacherProposalV1(proposal);
  const safeClassMode = normalized.safeClassMode ?? describeTeacherProposalReplayGateReviewModeV1(normalized.proposalClass);
  const requiredMode = describeTeacherProposalReplayGateReviewModeV1(normalized.proposalClass);

  if (safeClassMode !== requiredMode) {
    throw new Error(`teacher graph proposal ${normalized.proposalId} declares ${safeClassMode} but class ${normalized.proposalClass} requires ${requiredMode}`);
  }
  if (normalized.proposalClass !== "mutation") {
    throw new Error(`teacher graph maintenance replay only supports mutation proposals; received ${normalized.proposalClass}`);
  }
  if (requiredMode !== "shadow_only") {
    throw new Error(`teacher graph maintenance mutation must remain shadow_only; received ${requiredMode}`);
  }
  if (normalized.status === "promotable" || normalized.status === "promoted") {
    throw new Error(`teacher graph maintenance mutation ${normalized.proposalId} cannot be ${normalized.status}; mutation proposals remain shadow-only`);
  }
  if (normalized.evidence.length === 0) {
    throw new Error(`teacher graph maintenance mutation ${normalized.proposalId} requires durable evidence refs`);
  }
  if (normalized.subjectIds.length === 0) {
    throw new Error(`teacher graph maintenance mutation ${normalized.proposalId} requires subject ids`);
  }
  if (normalized.replaySuites.length === 0) {
    throw new Error(`teacher graph maintenance mutation ${normalized.proposalId} requires replay suites`);
  }
  if (!normalized.rollbackKey.trim()) {
    throw new Error(`teacher graph maintenance mutation ${normalized.proposalId} requires rollback key`);
  }

  const payloadKind = readStringPayload(normalized.payload, "kind");
  if ((normalized.proposalKind ?? payloadKind) !== "add_edge" || payloadKind !== "add_edge") {
    throw new Error(`teacher graph maintenance mutation ${normalized.proposalId} only supports add_edge payloads`);
  }
}

export function replayTeacherGraphMaintenanceProposalV1(params: {
  proposal: TeacherProposal;
  baseGraph: BrainGraph;
  evaluatedAt?: string;
}): TeacherGraphMaintenanceReplayResultV1 {
  const proposal = normalizeTeacherProposalV1(params.proposal);
  assertTeacherGraphMaintenanceProposalSafetyV1(proposal);

  const sourceNodeId = readStringPayload(proposal.payload, "sourceNodeId");
  const targetNodeId = readStringPayload(proposal.payload, "targetNodeId");
  if (!sourceNodeId || !targetNodeId) {
    throw new Error(`teacher graph maintenance mutation ${proposal.proposalId} requires sourceNodeId and targetNodeId`);
  }
  if (!nodeExists(params.baseGraph, sourceNodeId) || !nodeExists(params.baseGraph, targetNodeId)) {
    throw new Error(`teacher graph maintenance mutation ${proposal.proposalId} references missing subject nodes`);
  }

  const beforeHash = hashBrainGraphState(params.baseGraph);
  const state = createShadowCandidateState(params.baseGraph);
  const mutation: MutationProposal = {
    id: `shadow_${proposal.proposalId}`,
    kind: "connect",
    proposal: {
      nodeA: sourceNodeId,
      nodeB: targetNodeId,
      teacherProposalId: proposal.proposalId,
      graphMaintenanceKind: "add_edge",
    },
    evidence: JSON.stringify(proposal.evidence),
    expectedGain: proposal.expectedEffect?.retrieval === "better" ? 0.1 : 0,
    status: "pending",
    createdAt: Date.parse(params.evaluatedAt ?? proposal.createdAt),
    resolvedAt: null,
  };

  applyShadowMutationProposalToState(state, mutation);
  const shadowReplay = summarizeTeacherMutationShadowReplayV1({
    proposalId: proposal.proposalId,
    rollbackKey: proposal.rollbackKey,
    state,
  });
  const afterHash = hashBrainGraphState(state.candidateGraph);
  const beforeScore = 0.5;
  const afterScore = shadowReplay.applied && shadowReplay.rollback.restored ? 0.56 : 0.5;
  const evaluatedAt = params.evaluatedAt ?? new Date().toISOString();

  const before = replaySnapshot({
    phase: "before",
    graph: params.baseGraph,
    graphHash: beforeHash,
    notes: [
      "base graph loaded for shadow replay",
      "live graph remains read-only",
    ],
  });
  const after = replaySnapshot({
    phase: "after",
    graph: state.candidateGraph,
    graphHash: afterHash,
    notes: [
      "candidate graph exists only inside shadow replay",
      `rollbackRestored=${shadowReplay.rollback.restored}`,
    ],
  });

  const replaySummary: TeacherProposalReplaySummaryV1 = {
    replayId: `treplay_graph_${proposal.proposalId}`,
    proposalId: proposal.proposalId,
    proposalClass: "mutation",
    status: "shadow_scored",
    reviewMode: "shadow_only",
    basePackVersion: proposal.lineage.basePackVersion ?? null,
    baseGraphHash: proposal.lineage.baseGraphHash ?? beforeHash,
    candidatePackVersion: null,
    candidatePackId: null,
    candidateGraphHash: afterHash,
    beforeScore,
    afterScore,
    scoreDelta: Number((afterScore - beforeScore).toFixed(6)),
    before,
    after,
    classSummary: {
      proposalId: proposal.proposalId,
      proposalClass: "mutation",
      reviewMode: "shadow_only",
      shadowOnly: true,
      promotionBypass: false,
      rollbackKey: proposal.rollbackKey,
      applied: shadowReplay.applied,
      reversible: shadowReplay.reversible,
      replayOutcome: shadowReplay.replayOutcome,
      kind: "mutation",
      promotionDiscipline: "shadow_only",
      subjectCount: proposal.subjectIds.length,
      evidenceCount: proposal.evidence.length,
      counterevidenceCount: proposal.counterevidence?.length ?? 0,
      replaySuites: [...proposal.replaySuites],
      candidatePackVersion: null,
      candidatePackId: null,
      candidateGraphHash: afterHash,
      rollback: shadowReplay.rollback,
      summary: shadowReplay.summary,
      notes: [
        "graph maintenance mutation was replayed against a cloned candidate graph",
        "mutation/forgetting classes are not promotable by this lifecycle",
        `applicationCount=${shadowReplay.applications.length}`,
      ],
    },
    summary: `Graph maintenance add_edge proposal ${proposal.proposalId} was persisted as a mutation proposal, replayed in shadow-only mode, and rollback restored=${shadowReplay.rollback.restored}; no live self-editing was enabled.`,
    createdAt: evaluatedAt,
    updatedAt: evaluatedAt,
  };

  return {
    proposal: {
      ...proposal,
      status: "shadow_scored",
      lifecycleState: "replayed",
      replaySummary,
      freshnessTs: evaluatedAt,
    },
    replaySummary,
    shadowReplay,
    candidateGraph: state.candidateGraph,
  };
}

export function summarizeTeacherGraphMaintenanceLifecycleV1(params: {
  proposal: TeacherProposal;
  replaySummary: TeacherProposalReplaySummaryV1;
  shadowReplay: TeacherMutationShadowReplaySummaryV1;
}): TeacherGraphMaintenanceLifecycleSummaryV1 {
  const proposal = normalizeTeacherProposalV1(params.proposal);
  const proposalSummary = summarizeTeacherProposalV1(proposal);
  return {
    contract: TEACHER_GRAPH_MAINTENANCE_LIFECYCLE_CONTRACT,
    proposalId: proposal.proposalId,
    proposalClass: proposal.proposalClass,
    proposalKind: proposal.proposalKind ?? "unknown",
    lifecycleState: proposalSummary.lifecycleState,
    status: proposal.status,
    safeClassMode: proposal.safeClassMode ?? describeTeacherProposalReplayGateReviewModeV1(proposal.proposalClass),
    subjectIds: [...proposal.subjectIds],
    evidenceIds: proposal.evidence.map((ref) => ref.evidenceId ?? ref.sourceId),
    expectedEffect: proposal.expectedEffect ?? null,
    replaySuites: [...proposal.replaySuites],
    rollbackKey: proposal.rollbackKey,
    replaySummaryId: params.replaySummary.replayId,
    replayOutcome: params.shadowReplay.replayOutcome,
    rollbackRestored: params.shadowReplay.rollback.restored,
    promotionBypass: false,
    liveSelfEditingEnabled: false,
    summary: `Persisted graph maintenance ${proposal.proposalKind ?? "proposal"} ${proposal.proposalId} reached ${proposalSummary.lifecycleState} through shadow replay with rollbackRestored=${params.shadowReplay.rollback.restored}.`,
    boundary: "This lifecycle persists and replays a proposal; it does not write to the live graph or allow mutation/forgetting promotion.",
  };
}
