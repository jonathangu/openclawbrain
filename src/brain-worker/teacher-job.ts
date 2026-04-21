import { createHash } from "node:crypto";

import type { TeacherProposalReportArtifactV1 } from "../brain-core/teacher-v3-contracts.js";
import type { BrainWorkerJobResult } from "./jobs.js";

export type TeacherBatchFlowStep =
  | "observation_batch_bound"
  | "teacher_invoked"
  | "evaluation_recorded"
  | "reward_signal_emitted"
  | "reward_signal_applied";

export type TeacherBatchFlowLifecycleEventV1 = {
  contract: "openclawbrain_teacher_batch_flow_event.v1";
  version: 1;
  lookupKey: string;
  batchId: string;
  step: TeacherBatchFlowStep;
  status: "running" | "completed" | "failed";
  observationIds: string[];
  episodeIds: string[];
  conversationIds: number[];
  labelIds: string[];
  batchBudget: number;
  selectedCount: number;
  evaluatedCount: number;
  skippedCount: number;
  appliedLabelCount: number;
  batchCreatedAt: number;
  emittedAt: number;
  detail: string | null;
};

export type TeacherBatchFlowStateV1 = {
  lookupKey: string;
  batchId: string;
  observationIds: string[];
  episodeIds: string[];
  conversationIds: number[];
  labelIds: string[];
  batchBudget: number;
  selectedCount: number;
  evaluatedCount: number;
  skippedCount: number;
  appliedLabelCount: number;
  batchCreatedAt: number;
};

export function buildTeacherBatchLookupKey(params: {
  observations: Array<{ observationId: string; createdAt: number }>;
  batchBudget: number;
}): string {
  const sorted = [...params.observations].sort((a, b) => {
    if (a.createdAt !== b.createdAt) {
      return a.createdAt - b.createdAt;
    }
    return a.observationId.localeCompare(b.observationId);
  });
  const oldestCreatedAt = sorted[0]?.createdAt ?? 0;
  const firstObservationId = sorted[0]?.observationId ?? "none";
  const digest = createHash("sha256")
    .update(JSON.stringify({
      observationIds: sorted.map((entry) => entry.observationId),
      batchBudget: params.batchBudget,
    }))
    .digest("hex")
    .slice(0, 12);
  return `ocb-teacher-batch:${oldestCreatedAt}:${firstObservationId}:${sorted.length}:${digest}`;
}

export function createTeacherBatchFlowState(params: {
  observations: Array<{ observationId: string; episodeId: string; conversationId: number | null; createdAt: number }>;
  batchBudget: number;
}): TeacherBatchFlowStateV1 {
  const observationIds = params.observations.map((entry) => entry.observationId);
  const episodeIds = params.observations.map((entry) => entry.episodeId);
  const conversationIds = Array.from(new Set(
    params.observations
      .map((entry) => entry.conversationId)
      .filter((entry): entry is number => typeof entry === "number" && Number.isFinite(entry)),
  ));
  const batchCreatedAt = params.observations.reduce((min, entry) => Math.min(min, entry.createdAt), Number.POSITIVE_INFINITY);
  return {
    lookupKey: buildTeacherBatchLookupKey({
      observations: params.observations.map((entry) => ({
        observationId: entry.observationId,
        createdAt: entry.createdAt,
      })),
      batchBudget: params.batchBudget,
    }),
    batchId: `teacher-batch-${observationIds[0] ?? "none"}`,
    observationIds,
    episodeIds,
    conversationIds,
    labelIds: [],
    batchBudget: params.batchBudget,
    selectedCount: observationIds.length,
    evaluatedCount: 0,
    skippedCount: 0,
    appliedLabelCount: 0,
    batchCreatedAt: Number.isFinite(batchCreatedAt) ? batchCreatedAt : Date.now(),
  };
}

export function buildTeacherBatchFlowLifecycleEvent(params: {
  state: TeacherBatchFlowStateV1;
  step: TeacherBatchFlowStep;
  status?: TeacherBatchFlowLifecycleEventV1["status"];
  emittedAt?: number;
  detail?: string | null;
}): TeacherBatchFlowLifecycleEventV1 {
  return {
    contract: "openclawbrain_teacher_batch_flow_event.v1",
    version: 1,
    lookupKey: params.state.lookupKey,
    batchId: params.state.batchId,
    step: params.step,
    status: params.status ?? "running",
    observationIds: [...params.state.observationIds],
    episodeIds: [...params.state.episodeIds],
    conversationIds: [...params.state.conversationIds],
    labelIds: [...params.state.labelIds],
    batchBudget: params.state.batchBudget,
    selectedCount: params.state.selectedCount,
    evaluatedCount: params.state.evaluatedCount,
    skippedCount: params.state.skippedCount,
    appliedLabelCount: params.state.appliedLabelCount,
    batchCreatedAt: params.state.batchCreatedAt,
    emittedAt: params.emittedAt ?? Date.now(),
    detail: params.detail ?? null,
  };
}

export function teacherJobResult(changed: boolean, details?: Record<string, unknown>): BrainWorkerJobResult {
  return { job: "teacher", changed, details };
}

export function teacherProposalArtifactJobResult(
  artifact: TeacherProposalReportArtifactV1,
  details?: Record<string, unknown>,
): BrainWorkerJobResult {
  return teacherJobResult(false, {
    mode: "report_only",
    proposalId: artifact.proposalId,
    proposalClass: artifact.proposalClass,
    reviewMode: artifact.reviewMode,
    artifactId: artifact.artifactId,
    artifactKind: artifact.artifactRef.kind,
    artifactContentHash: artifact.artifactRef.contentHash,
    replayReady: artifact.replayHook.replayReady,
    proofLinked: artifact.proofLinkage.proofLinked,
    summary: artifact.summary,
    ...details,
  });
}
