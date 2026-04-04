#!/usr/bin/env node

import { createHash } from "node:crypto";
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { captureTeacherV3ReplayOutcomes } from "./teacher-v3-replay-outcomes.mjs";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..");
const workspaceRoot = path.resolve(repoRoot, "..");

export const TEACHER_V3_PROOF_BUNDLE_LAYOUT = {
  summary: "summary.md",
  status: "status.json",
  surfaceMap: "surface-map.json",
  proposalReport: "proposal-report.json",
  verdict: "verdict.json",
};

export const TEACHER_V3_PROOF_BUNDLE_CONTRACT = "teacher_v3_proof_bundle.v1";
export const TEACHER_V3_PROOF_STATUS_CONTRACT = "teacher_v3_proof_bundle_status.v1";
export const TEACHER_V3_PROOF_SURFACE_MAP_CONTRACT = "teacher_v3_surface_map.v1";
export const TEACHER_V3_PROPOSAL_REPORT_CONTRACT = "teacher_v3_proposal_report.v1";
export const TEACHER_V3_PROOF_VERDICT_CONTRACT = "teacher_v3_proof_bundle_verdict.v1";

export const DEFAULT_TEACHER_V3_PROOF_BUNDLE_PARENT = path.join(
  workspaceRoot,
  "artifacts",
  "teacher-v3-proof",
);

function sha256Text(text) {
  return `sha256:${createHash("sha256").update(String(text ?? ""), "utf8").digest("hex")}`;
}

function renderJson(value) {
  return `${JSON.stringify(value, null, 2)}\n`;
}

function ensureDir(dirPath) {
  if (!existsSync(dirPath)) {
    mkdirSync(dirPath, { recursive: true });
  }
}

function normalizeText(value) {
  if (typeof value !== "string") {
    return null;
  }
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
}

function normalizeBoolean(value) {
  return value === true ? true : value === false ? false : null;
}

function normalizeNumber(value) {
  return Number.isFinite(value) ? Number(value) : null;
}

function normalizeArray(value) {
  return Array.isArray(value) ? value.filter((item) => item !== null && item !== undefined) : [];
}

function relativeWorkspacePath(value) {
  const text = normalizeText(value);
  if (!text) {
    return null;
  }
  if (!path.isAbsolute(text)) {
    return text;
  }
  return path.relative(workspaceRoot, text);
}

function timestampToken(date = new Date()) {
  return date.toISOString().replace(/[-:]/g, "").replace(/\.\d{3}Z$/, "Z").replace("T", "-");
}

function summarizeRuntimeStatus(runtimeStatus) {
  const currentPackMetadata = runtimeStatus && typeof runtimeStatus.currentPackMetadata === "object"
    ? runtimeStatus.currentPackMetadata
    : null;
  const operatorHealth = runtimeStatus && typeof runtimeStatus.operatorHealth === "object"
    ? runtimeStatus.operatorHealth
    : null;
  const learningHealth = runtimeStatus && typeof runtimeStatus.learningHealth === "object"
    ? runtimeStatus.learningHealth
    : null;
  const lastReplayGateVerdict = runtimeStatus && typeof runtimeStatus.lastReplayGateVerdict === "object"
    ? runtimeStatus.lastReplayGateVerdict
    : null;
  const lastPromotionVerdict = runtimeStatus && typeof runtimeStatus.lastPromotionVerdict === "object"
    ? runtimeStatus.lastPromotionVerdict
    : null;
  const lastAssemblyDecision = runtimeStatus && typeof runtimeStatus.lastAssemblyDecision === "object"
    ? runtimeStatus.lastAssemblyDecision
    : null;
  const lastPrefetchDecision = runtimeStatus && typeof runtimeStatus.lastPrefetchDecision === "object"
    ? runtimeStatus.lastPrefetchDecision
    : null;

  return {
    serveState: normalizeText(runtimeStatus?.serveState),
    currentPackVersion: normalizeNumber(runtimeStatus?.currentPackVersion),
    currentPackPromotedAt: normalizeText(runtimeStatus?.currentPackPromotedAt),
    currentPackMetadata: {
      reason: normalizeText(currentPackMetadata?.reason),
      kind: normalizeText(currentPackMetadata?.kind),
      promotedAt: normalizeText(currentPackMetadata?.promotedAt),
      rolledBack: normalizeBoolean(currentPackMetadata?.rolledBack),
    },
    teacherConfigured: normalizeBoolean(runtimeStatus?.teacherConfigured),
    teacherProvider: normalizeText(runtimeStatus?.teacherProvider),
    teacherModel: normalizeText(runtimeStatus?.teacherModel),
    teacherConfigError: normalizeText(runtimeStatus?.teacherConfigError),
    operatorHealth: {
      status: normalizeText(operatorHealth?.status),
      detail: normalizeText(operatorHealth?.detail),
    },
    learningHealth: {
      status: normalizeText(learningHealth?.status),
      detail: normalizeText(learningHealth?.detail),
    },
    routeTraceCount: normalizeNumber(runtimeStatus?.routeTraceCount),
    supervisionCount: normalizeNumber(runtimeStatus?.supervisionCount),
    recentTraceCount: normalizeNumber(runtimeStatus?.recentTraceCount),
    pendingLabels: normalizeNumber(runtimeStatus?.pendingLabels),
    pendingObservations: normalizeNumber(runtimeStatus?.pendingObservations),
    mutationBacklog: runtimeStatus?.mutationBacklog && typeof runtimeStatus.mutationBacklog === "object"
      ? runtimeStatus.mutationBacklog
      : null,
    lastCompileReportSummary: normalizeText(runtimeStatus?.lastCompileReportSummary),
    lastAssemblyDecision: {
      summary: normalizeText(lastAssemblyDecision?.summary),
      verdict: normalizeText(lastAssemblyDecision?.verdict),
    },
    lastPrefetchDecision: {
      summary: normalizeText(lastPrefetchDecision?.summary),
      verdict: normalizeText(lastPrefetchDecision?.verdict),
    },
    lastPromotionReason: normalizeText(runtimeStatus?.lastPromotionReason),
    lastPromotionVerdict: {
      verdict: normalizeText(lastPromotionVerdict?.verdict),
      summary: normalizeText(lastPromotionVerdict?.summary),
    },
    lastReplayFailureReason: normalizeText(runtimeStatus?.lastReplayFailureReason),
    lastReplayGateVerdict: {
      verdict: normalizeText(lastReplayGateVerdict?.verdict),
      summary: normalizeText(lastReplayGateVerdict?.summary),
    },
  };
}

function summarizeOperatorProofTruth(operatorProof) {
  const verdict = operatorProof?.verdict && typeof operatorProof.verdict === "object"
    ? operatorProof.verdict
    : null;
  return {
    bundleDir: relativeWorkspacePath(operatorProof?.bundleDir),
    command: normalizeText(operatorProof?.command),
    summary: normalizeText(operatorProof?.summary),
    verdict: normalizeText(verdict?.verdict),
    severity: normalizeText(verdict?.severity),
    why: normalizeText(verdict?.why),
    runtimeLoadProofPath: relativeWorkspacePath(operatorProof?.runtimeLoadProofPath),
    runtimeLoadProofExists: normalizeBoolean(operatorProof?.runtimeLoadProofExists),
    stepCount: normalizeNumber(operatorProof?.stepCount),
    postBundleCount: normalizeNumber(operatorProof?.postBundleCount),
    missingProofCount: normalizeNumber(Array.isArray(verdict?.missingProofs) ? verdict.missingProofs.length : null),
  };
}

function summarizeDocsTruth(docsTruth) {
  return {
    path: relativeWorkspacePath(docsTruth?.path),
    title: normalizeText(docsTruth?.title),
    state: "shipped",
    summary: normalizeText(docsTruth?.summary),
  };
}

function buildTeacherV3ProposalSeed(input) {
  const bundleId = normalizeText(input.bundleId) ?? `teacher-v3-proof-${timestampToken(input.bundleStartedAt)}`;
  const runtimeStatus = summarizeRuntimeStatus(input.runtimeStatus ?? null);
  const operatorProof = summarizeOperatorProofTruth(input.operatorProof ?? null);
  const docsTruth = summarizeDocsTruth(input.docsTruth ?? null);
  const runtimeStatusCommand = normalizeText(input.runtimeStatusCommand) ?? "runtime-status";
  const operatorProofCommand = normalizeText(input.operatorProofCommand) ?? "openclawbrain proof";
  const subjectIds = normalizeArray(input.subjectIds).map((subject) => normalizeText(subject)).filter(Boolean);
  const proposalRecordSubjectIds = normalizeArray(input.proposalRecord?.subjectIds)
    .map((subject) => normalizeText(subject))
    .filter(Boolean);
  const proposalSubjectIds = subjectIds.length > 0
    ? subjectIds
    : proposalRecordSubjectIds.length > 0
      ? proposalRecordSubjectIds
      : ["surface:runtime-truth", "surface:proof-truth", "surface:docs-truth"];
  const proposalClass = normalizeText(input.proposalClass) ?? normalizeText(input.proposalRecord?.proposalClass) ?? "lint";
  const proposalLane = normalizeText(input.proposalLane) ?? normalizeText(input.proposalRecord?.proposalLane) ?? proposalClass;
  const producerVersion = normalizeText(input.producerVersion) ?? "openclawbrain-proof-capture";
  const proposalId = normalizeText(input.proposalId) ?? bundleId;
  const sourceBundleId = operatorProof.bundleDir ?? normalizeText(input.operatorProof?.bundleDir) ?? null;
  const proposalRecord = input.proposalRecord ?? null;
  const proposalRecordEvidence = normalizeArray(proposalRecord?.evidence);
  const proposalRecordCounterevidence = normalizeArray(proposalRecord?.counterevidence);
  const proposalRecordReplaySuites = normalizeArray(proposalRecord?.replaySuites);
  const idempotencyKey = sha256Text(JSON.stringify({
    bundleId,
    proposalId,
    proposalClass,
    proposalLane,
    producerVersion,
    basePackVersion: runtimeStatus.currentPackVersion,
    baseGraphHash: runtimeStatus.currentPackMetadata?.kind ?? null,
    runtimeStatusCommand,
    operatorProofCommand,
    runtimeLoadProofPath: operatorProof.runtimeLoadProofPath,
    docsTruthPath: docsTruth.path,
  }));

  return {
    recordSource: input.proposalRecord?.recordSource ?? "runtime-capture",
    bundleId,
    proposalId,
    proposalLane,
    proposalClass,
    status: normalizeText(input.proposalStatus) ?? normalizeText(input.proposalRecord?.status) ?? "validated",
    lineage: {
      proposalClass,
      basePackVersion: runtimeStatus.currentPackVersion,
      baseGraphHash: normalizeText(input.proposalRecord?.lineage?.baseGraphHash) ?? null,
      producerVersion,
      producerBuildId: normalizeText(input.proposalRecord?.lineage?.producerBuildId) ?? null,
      promptHash: normalizeText(input.proposalRecord?.lineage?.promptHash) ?? sha256Text(runtimeStatusCommand + operatorProofCommand + docsTruth.path),
      templateId: normalizeText(input.proposalRecord?.lineage?.templateId) ?? "teacher-v3-proof-bundle-v1",
      scope: normalizeText(input.proposalRecord?.lineage?.scope) ?? "runtime-proof-adoption",
      profile: normalizeText(input.proposalRecord?.lineage?.profile) ?? null,
      idempotencyKey,
      sourceBundleId,
      parentProposalIds: normalizeArray(input.proposalRecord?.lineage?.parentProposalIds)
        .map((parentId) => normalizeText(parentId))
        .filter(Boolean),
    },
    subjectIds: proposalSubjectIds,
    evidence: proposalRecordEvidence.length > 0
      ? proposalRecordEvidence
      : [
        {
          evidenceId: "runtime-status",
          sourceKind: "repo",
          sourceId: runtimeStatusCommand,
          authority: "raw_source",
          derivation: "teacher_inference",
          excerpt: "Runtime status snapshot captured for the Teacher v3 proof bundle.",
          sourceHash: sha256Text(JSON.stringify(runtimeStatus)),
        },
        {
          evidenceId: "operator-proof",
          sourceKind: "repo",
          sourceId: operatorProofCommand,
          authority: "raw_source",
          derivation: "teacher_compilation",
          excerpt: "Operator proof bundle captured alongside the runtime snapshot.",
          sourceHash: sha256Text(JSON.stringify(operatorProof)),
        },
        {
          evidenceId: "docs-truth",
          sourceKind: "file",
          sourceId: normalizeText(input.docsTruth?.path) ?? "docs/architecture/teacher-v3-proof.md",
          authority: "raw_source",
          derivation: "teacher_compilation",
          excerpt: "Teacher v3 proof-surface design note used as the docs truth anchor.",
          sourceHash: sha256Text(JSON.stringify(docsTruth)),
        },
      ],
    counterevidence: proposalRecordCounterevidence,
    payload: input.proposalRecord?.payload ?? {
      runtimeStatus,
      operatorProof,
      docsTruth,
      bundleArtifacts: [
        TEACHER_V3_PROOF_BUNDLE_LAYOUT.summary,
        TEACHER_V3_PROOF_BUNDLE_LAYOUT.status,
        TEACHER_V3_PROOF_BUNDLE_LAYOUT.surfaceMap,
        TEACHER_V3_PROOF_BUNDLE_LAYOUT.proposalReport,
        TEACHER_V3_PROOF_BUNDLE_LAYOUT.verdict,
      ],
    },
    confidence: Number.isFinite(proposalRecord?.confidence)
      ? Number(proposalRecord.confidence)
      : 0.92,
    replaySuites: proposalRecordReplaySuites.length > 0
      ? proposalRecordReplaySuites
      : [
        "runtime-status-surface",
        "operator-proof-surface",
        "teacher-v3-docs-truth",
      ],
    rollbackKey: normalizeText(input.proposalRecord?.rollbackKey) ?? `rollback:teacher-v3-proof-bundle:${bundleId}`,
    createdAt: normalizeText(input.proposalRecord?.createdAt) ?? input.bundleStartedAt,
    resolvedAt: normalizeText(input.proposalRecord?.resolvedAt) ?? input.bundleStartedAt,
    updatedAt: normalizeText(input.proposalRecord?.updatedAt) ?? input.bundleStartedAt,
    gate1Seam: {
      present: Boolean(input.proposalRecord?.gate1Seam?.present),
      recordSource: input.proposalRecord?.gate1Seam?.recordSource ?? "runtime-capture",
      note: input.proposalRecord?.gate1Seam?.note
        ?? "Gate 1 persistence is not wired in this branch; this bundle is seeded from runtime/proof inputs and can later be replaced by a stored proposal record.",
    },
    docsTruth,
    runtimeTruth: runtimeStatus,
    proofTruth: operatorProof,
    proposalReviewMode: proposalClass === "mutation" || proposalClass === "forgetting" || proposalClass === "correction"
      ? "shadow_only"
      : "promotable",
  };
}

function buildSurfaceMap(seed, bundlePaths) {
  const observedSurfaces = [
    {
      id: "runtime-truth",
      state: "shipped",
      kind: "runtime_truth",
      source: seed.runtimeStatusCommand,
      note: [
        "Live runtime snapshot captured from the shipped status surface.",
        seed.runtimeTruth.currentPackVersion !== null
          ? `currentPackVersion=${seed.runtimeTruth.currentPackVersion}`
          : null,
        seed.runtimeTruth.serveState ? `serveState=${seed.runtimeTruth.serveState}` : null,
      ].filter(Boolean).join("; "),
    },
    {
      id: "operator-proof-truth",
      state: "shipped",
      kind: "proof_truth",
      source: seed.operatorProofCommand,
      note: [
        "Host-anchored proof bundle emitted by the shipped proof capture path.",
        seed.proofTruth.verdict ? `verdict=${seed.proofTruth.verdict}` : null,
        seed.proofTruth.runtimeLoadProofExists !== null
          ? `runtimeLoadProof=${seed.proofTruth.runtimeLoadProofExists ? "present" : "missing"}`
          : null,
      ].filter(Boolean).join("; "),
    },
    {
      id: "docs-truth",
      state: "shipped",
      kind: "docs_truth",
      source: seed.docsTruth.path,
      note: "Teacher v3 proof-surface design note that keeps shipped truth distinct from target-state proposal surfaces.",
    },
  ];

  const bundleArtifacts = [
    {
      id: "teacher-v3-proof-summary",
      state: "target",
      kind: "proposal_truth",
      source: bundlePaths.summary,
      note: "bounded human summary",
    },
    {
      id: "teacher-v3-proof-status",
      state: "target",
      kind: "proposal_truth",
      source: bundlePaths.status,
      note: "thin machine status",
    },
    {
      id: "teacher-v3-proof-surface-map",
      state: "target",
      kind: "proposal_truth",
      source: bundlePaths.surfaceMap,
      note: "shipped-vs-target inventory",
    },
    {
      id: "teacher-v3-proof-proposal-report",
      state: "target",
      kind: "proposal_truth",
      source: bundlePaths.proposalReport,
      note: "machine-readable proposal report",
    },
    {
      id: "teacher-v3-proof-verdict",
      state: "target",
      kind: "proposal_truth",
      source: bundlePaths.verdict,
      note: "review verdict",
    },
  ];

  return {
    contract: TEACHER_V3_PROOF_SURFACE_MAP_CONTRACT,
    bundleId: seed.bundleId,
    observedSurfaces,
    bundleArtifacts,
    counts: {
      observedSurfaceCount: observedSurfaces.length,
      shippedSurfaceCount: observedSurfaces.filter((surface) => surface.state === "shipped").length,
      targetSurfaceCount: bundleArtifacts.length,
      totalSurfaceCount: observedSurfaces.length + bundleArtifacts.length,
    },
  };
}

function buildProposalReport(seed, surfaceMap, bundlePaths, replayCapture) {
  const publicationSafeArtifacts = [
    {
      artifactId: "teacher-v3-proof-summary",
      kind: "summary",
      path: bundlePaths.summary,
      redactions: ["raw stdout/stderr", "secret-bearing values"],
      containsRawLogs: false,
    },
    {
      artifactId: "teacher-v3-proof-status",
      kind: "status",
      path: bundlePaths.status,
      redactions: ["raw stdout/stderr", "secret-bearing values"],
      containsRawLogs: false,
    },
    {
      artifactId: "teacher-v3-proof-surface-map",
      kind: "surface-map",
      path: bundlePaths.surfaceMap,
      redactions: ["raw source payloads"],
      containsRawLogs: false,
    },
    {
      artifactId: "teacher-v3-proof-proposal-report",
      kind: "metadata",
      path: bundlePaths.proposalReport,
      redactions: ["raw stdout/stderr", "secret-bearing values"],
      containsRawLogs: false,
    },
    {
      artifactId: "teacher-v3-proof-verdict",
      kind: "verdict",
      path: bundlePaths.verdict,
      redactions: ["raw stdout/stderr", "secret-bearing values"],
      containsRawLogs: false,
    },
  ];

  const runtimeTruth = seed.runtimeTruth;
  const proofTruth = seed.proofTruth;
  const docsTruth = seed.docsTruth;
  const recommendations = normalizeArray(seed.proposalRecord?.recommendations).length > 0
    ? normalizeArray(seed.proposalRecord.recommendations)
    : [
      seed.gate1Seam.present
        ? "Preserve the persisted proposal record seam and load it directly once Gate 1 lands."
        : "Add Gate 1 proposal persistence so the synthetic runtime-capture record can be replaced by a stored proposal row.",
      "Keep the bundle publication-safe and bounded; never spill raw logs into the target-state artifacts.",
      "Thread candidate-state replay and rollback binding in the next tranche before considering canary activation.",
    ];

  return {
    contract: TEACHER_V3_PROPOSAL_REPORT_CONTRACT,
    bundleId: seed.bundleId,
    proposal: {
      proposalId: seed.proposalId,
      proposalLane: seed.proposalLane,
      proposalClass: seed.proposalClass,
      status: seed.status,
      reviewMode: seed.proposalReviewMode,
      lineage: seed.lineage,
      subjectIds: seed.subjectIds,
      rollbackKey: seed.rollbackKey,
      replaySuites: seed.replaySuites,
      replayOutcomeCount: replayCapture.summary.replayOutcomeCount,
      confidence: seed.confidence,
      recordSource: seed.recordSource,
    },
    replayGate: {
      proposalClass: seed.proposalClass,
      reviewMode: seed.proposalReviewMode,
      dimensions: {
        truthInvariants: {
          name: "truth_invariants",
          summary: "Keep derived output subordinate to explicit authority.",
          requirements: [
            "Explicit correction memory still outranks teacher synthesis.",
            "The live path stays read-only to the proposal.",
            "Evidence refs stay attached to any non-trivial claim.",
          ],
        },
        attributionFloor: {
          name: "attribution_floor",
          summary: "Every proposed change needs clear evidence coverage.",
          requirements: [
            "Every proposal carries durable evidence refs.",
            "Source ids must be stable record ids, not display labels.",
            "Unattributed payload stays out of promotion.",
          ],
        },
        boundedness: {
          name: "boundedness",
          summary: "Keep the reviewable surface compact and inspectable.",
          requirements: [
            "Proposal subject sets stay finite and small.",
            "Payloads avoid raw corpus dumps and unbounded excerpts.",
            "Replay fits inside a single review pass.",
          ],
        },
        reversibility: {
          name: "reversibility",
          summary: "Preserve rollback and replay identity.",
          requirements: [
            "RollbackKey identifies the reversible path.",
            "Prior state remains recoverable for replay.",
            "Rejected or superseded proposals keep lineage.",
          ],
        },
      },
    },
    runtimeTruth,
    proofTruth,
    docsTruth,
    surfaceMap: surfaceMap.observedSurfaces,
    surfaceCounts: surfaceMap.counts,
    evidenceLinks: seed.evidence,
    counterevidenceLinks: seed.counterevidence,
    recommendations,
    gate1Seam: seed.gate1Seam,
    publicationSafeArtifacts,
    createdAt: seed.createdAt,
    updatedAt: seed.updatedAt ?? seed.resolvedAt,
    runtimeTruthSummary: {
      currentPackVersion: runtimeTruth.currentPackVersion,
      serveState: runtimeTruth.serveState,
      teacherConfigured: runtimeTruth.teacherConfigured,
      operatorHealthStatus: runtimeTruth.operatorHealth.status,
      learningHealthStatus: runtimeTruth.learningHealth.status,
      routeTraceCount: runtimeTruth.routeTraceCount,
      supervisionCount: runtimeTruth.supervisionCount,
      lastReplayGateVerdict: runtimeTruth.lastReplayGateVerdict.verdict,
      lastPromotionVerdict: runtimeTruth.lastPromotionVerdict.verdict,
    },
    proofTruthSummary: {
      verdict: proofTruth.verdict,
      severity: proofTruth.severity,
      runtimeLoadProofExists: proofTruth.runtimeLoadProofExists,
      postBundleCount: proofTruth.postBundleCount,
      stepCount: proofTruth.stepCount,
    },
    replayOutcomes: replayCapture.outcomes,
    replayOutcomeSummary: replayCapture.summary,
  };
}

function buildStatusReport(seed, surfaceMap, proposalReport) {
  return {
    contract: TEACHER_V3_PROOF_STATUS_CONTRACT,
    bundleId: seed.bundleId,
    proposalId: seed.proposalId,
    proposalLane: seed.proposalLane,
    proposalClass: seed.proposalClass,
    proposalStatus: seed.status,
    reviewMode: seed.proposalReviewMode,
    surfaceCounts: surfaceMap.counts,
    runtimeTruth: proposalReport.runtimeTruthSummary,
    proofTruth: proposalReport.proofTruthSummary,
    docsTruth: proposalReport.docsTruth,
    replayOutcomeSummary: proposalReport.replayOutcomeSummary,
    gate1Seam: seed.gate1Seam,
    recommendations: proposalReport.recommendations,
    publicationSafeArtifacts: proposalReport.publicationSafeArtifacts.map((artifact) => ({
      artifactId: artifact.artifactId,
      kind: artifact.kind,
      path: artifact.path,
      redactions: artifact.redactions,
      containsRawLogs: artifact.containsRawLogs,
    })),
    createdAt: seed.createdAt,
    updatedAt: seed.updatedAt ?? seed.resolvedAt ?? seed.createdAt,
  };
}

function buildVerdictReport(seed, statusReport, proposalReport) {
  const proofSeverity = proposalReport.proofTruthSummary.severity;
  const runtimeReady = statusReport.runtimeTruth.currentPackVersion !== null
    || statusReport.runtimeTruth.serveState !== null
    || statusReport.runtimeTruth.teacherConfigured !== null;
  const proofReady = proposalReport.proofTruthSummary.runtimeLoadProofExists !== null;
  const blocking = !runtimeReady || !proofReady;

  return {
    contract: TEACHER_V3_PROOF_VERDICT_CONTRACT,
    bundleId: seed.bundleId,
    verdict: blocking ? "rejected" : "reviewable",
    severity: blocking ? "blocking" : proofSeverity === "blocking" ? "warn" : "info",
    why: blocking
      ? "runtime or proof truth could not be summarized into a bounded bundle"
      : seed.gate1Seam.present
        ? `runtime, proof, and docs truth were summarized; Gate 1 persistence is already wired so the record can be loaded from storage (${proposalReport.replayOutcomeSummary.summary})`
        : `runtime, proof, and docs truth were summarized; Gate 1 persistence is still pending so the bundle remains a derived review surface (${proposalReport.replayOutcomeSummary.summary})`,
    reviewMode: seed.proposalReviewMode,
    gate1Seam: seed.gate1Seam,
    replayOutcomeSummary: proposalReport.replayOutcomeSummary,
    blockers: blocking ? [
      !runtimeReady ? "missing runtime truth summary" : null,
      !proofReady ? "missing proof truth summary" : null,
    ].filter(Boolean) : [],
    targetStateOnly: !seed.gate1Seam.present,
    createdAt: seed.createdAt,
    updatedAt: seed.updatedAt ?? seed.resolvedAt ?? seed.createdAt,
  };
}

function buildSummaryMarkdown(seed, statusReport, verdictReport, bundlePaths) {
  const lines = [
    "# Teacher v3 proof bundle",
    "",
    `- bundle: \`${seed.bundleId}\``,
    `- proposal: \`${seed.proposalId}\` (${seed.proposalClass}, ${seed.status})`,
    `- review mode: **${seed.proposalReviewMode}**`,
    `- verdict: **${verdictReport.verdict}**`,
    `- severity: **${verdictReport.severity}**`,
    `- runtime truth: \`${seed.runtimeStatusCommand}\``,
    `- proof truth: \`${seed.operatorProofCommand}\``,
    `- docs truth: \`${seed.docsTruth.path}\``,
    "",
    "## Surface counts",
    `- shipped surfaces: ${statusReport.surfaceCounts.shippedSurfaceCount}`,
    `- target bundle artifacts: ${statusReport.surfaceCounts.targetSurfaceCount}`,
    `- total referenced surfaces: ${statusReport.surfaceCounts.totalSurfaceCount}`,
    "",
    "## Replay outcomes",
    `- captured outcomes: ${statusReport.replayOutcomeSummary.replayOutcomeCount}`,
    `- replay suites: ${statusReport.replayOutcomeSummary.replaySuites.length > 0 ? statusReport.replayOutcomeSummary.replaySuites.join(", ") : "none"}`,
    `- results: pass=${statusReport.replayOutcomeSummary.resultCounts.pass}, warn=${statusReport.replayOutcomeSummary.resultCounts.warn}, fail=${statusReport.replayOutcomeSummary.resultCounts.fail}`,
    `- review modes: promotable=${statusReport.replayOutcomeSummary.reviewModeCounts.promotable}, shadow_only=${statusReport.replayOutcomeSummary.reviewModeCounts.shadow_only}`,
    `- sources: proposal_record=${statusReport.replayOutcomeSummary.sourceCounts.proposal_record}, proof_bundle=${statusReport.replayOutcomeSummary.sourceCounts.proof_bundle}, derived=${statusReport.replayOutcomeSummary.sourceCounts.derived}`,
    "",
    "## Gate 1 seam",
    `- present: ${seed.gate1Seam.present ? "yes" : "no"}`,
    `- record source: ${seed.gate1Seam.recordSource}`,
    `- note: ${seed.gate1Seam.note}`,
    "",
    "## Publication-safe artifacts",
    ...proposalReportArtifactsLines(bundlePaths),
    "",
    "## Recommendations",
    ...statusReport.recommendations.map((item) => `- ${item}`),
  ];

  return `${lines.join("\n")}\n`;
}

function proposalReportArtifactsLines(bundlePaths) {
  return [
    `- \`${bundlePaths.summary}\` — bounded human summary`,
    `- \`${bundlePaths.status}\` — thin machine status`,
    `- \`${bundlePaths.surfaceMap}\` — shipped-vs-target inventory`,
    `- \`${bundlePaths.proposalReport}\` — machine-readable proposal report`,
    `- \`${bundlePaths.verdict}\` — review verdict`,
  ];
}

export function resolveTeacherV3ProofOutputDir({ outputDir = null, bundleStartedAt = new Date() } = {}) {
  if (typeof outputDir === "string" && outputDir.trim().length > 0) {
    return path.resolve(outputDir);
  }
  const startedAt = bundleStartedAt instanceof Date ? bundleStartedAt : new Date(bundleStartedAt);
  return path.join(
    DEFAULT_TEACHER_V3_PROOF_BUNDLE_PARENT,
    `teacher-v3-proof-${timestampToken(Number.isNaN(startedAt.getTime()) ? new Date() : startedAt)}`,
  );
}

export function buildTeacherV3ProofBundle(input) {
  const bundleStartedAt = normalizeText(input.bundleStartedAt) ?? new Date().toISOString();
  const outputDir = path.resolve(input.outputDir ?? resolveTeacherV3ProofOutputDir({ bundleStartedAt }));
  const bundleId = normalizeText(input.bundleId) ?? path.basename(outputDir);
  const bundlePaths = {
    summary: relativeWorkspacePath(path.join(outputDir, TEACHER_V3_PROOF_BUNDLE_LAYOUT.summary)),
    status: relativeWorkspacePath(path.join(outputDir, TEACHER_V3_PROOF_BUNDLE_LAYOUT.status)),
    surfaceMap: relativeWorkspacePath(path.join(outputDir, TEACHER_V3_PROOF_BUNDLE_LAYOUT.surfaceMap)),
    proposalReport: relativeWorkspacePath(path.join(outputDir, TEACHER_V3_PROOF_BUNDLE_LAYOUT.proposalReport)),
    verdict: relativeWorkspacePath(path.join(outputDir, TEACHER_V3_PROOF_BUNDLE_LAYOUT.verdict)),
  };

  const seed = buildTeacherV3ProposalSeed({
    ...input,
    bundleId,
    bundleStartedAt,
  });
  const replayCapture = captureTeacherV3ReplayOutcomes({
    bundleId,
    proposalId: seed.proposalId,
    proposalClass: seed.proposalClass,
    reviewMode: seed.proposalReviewMode,
    replaySuites: seed.replaySuites,
    proofVerdict: seed.proofTruth,
    replayOutcomes: input.proposalRecord?.replayOutcomes,
    bundleStartedAt,
  });
  const surfaceMap = buildSurfaceMap(seed, bundlePaths);
  const proposalReport = buildProposalReport(seed, surfaceMap, bundlePaths, replayCapture);
  const statusReport = buildStatusReport(seed, surfaceMap, proposalReport);
  const verdictReport = buildVerdictReport(seed, statusReport, proposalReport);
  const summaryMarkdown = buildSummaryMarkdown(seed, statusReport, verdictReport, bundlePaths);

  return {
    bundleId,
    bundleStartedAt,
    outputDir,
    bundlePaths,
    seed,
    surfaceMap,
    proposalReport,
    statusReport,
    verdictReport,
    summaryMarkdown,
    files: {
      [TEACHER_V3_PROOF_BUNDLE_LAYOUT.summary]: summaryMarkdown,
      [TEACHER_V3_PROOF_BUNDLE_LAYOUT.status]: renderJson(statusReport),
      [TEACHER_V3_PROOF_BUNDLE_LAYOUT.surfaceMap]: renderJson(surfaceMap),
      [TEACHER_V3_PROOF_BUNDLE_LAYOUT.proposalReport]: renderJson(proposalReport),
      [TEACHER_V3_PROOF_BUNDLE_LAYOUT.verdict]: renderJson(verdictReport),
    },
  };
}

export function writeTeacherV3ProofBundle(outputDir, bundle) {
  ensureDir(outputDir);
  for (const [fileName, content] of Object.entries(bundle.files)) {
    writeFileSync(path.join(outputDir, fileName), content, "utf8");
  }
  return {
    outputDir,
    writtenFiles: Object.keys(bundle.files).map((fileName) => path.join(outputDir, fileName)),
  };
}

export function buildTeacherV3ProofBundleDigest(bundle) {
  return {
    bundleId: bundle.bundleId,
    files: Object.fromEntries(
      Object.entries(bundle.files).map(([fileName, content]) => [fileName, sha256Text(content)]),
    ),
  };
}
