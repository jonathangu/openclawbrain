import type { TeacherProposalReportArtifactV1 } from "../brain-core/teacher-v3-contracts.js";
import type { BrainWorkerJobResult } from "./jobs.js";

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
